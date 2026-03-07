"""
Aegis-X FreqNet Forensic Tool
-----------------------------
Frequency domain deepfake detection using F3Net architecture.

Architecture:
    - Dual-stream ResNet-50 (Spatial + Frequency)
    - DCT Conv2d frequency preprocessing (frozen basis)
    - FAD-style fusion with band analysis
    - Dual-mode calibration (Z-score + ratio fallback)

Constraints & Mitigations:
    - VRAM: Sequential face processing + no_grad + detach()
    - Drift: Proper device tracking via register_buffer()
    - Leaks: Hook cleanup via try/finally
    - Schema: ToolResult compliant with Day 11/12
"""

import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult
from core.config import AegisConfig
from utils.vram_manager import VRAMLifecycleManager
from utils.logger import setup_logger

from .freqnet.preprocessor import DCTPreprocessor, SpatialPreprocessor
from .freqnet.fad_hook import FADHook, BandAnalysis
from .freqnet.calibration import CalibrationManager

# Threshold imports
try:
    from utils.thresholds import FREQNET_FAKE_THRESHOLD
except ImportError:
    FREQNET_FAKE_THRESHOLD = 0.60

logger = setup_logger(__name__)


class FreqNetDual(nn.Module):
    """
    Dual-stream F3Net architecture.
    
    Spatial Stream: ResNet-50 with ImageNet weights
    Frequency Stream: ResNet-50 with modified conv1 (64-channel DCT input)
    Fusion: Concat + Linear layers
    """
    
    def __init__(self):
        super().__init__()
        
        # Spatial stream (standard RGB input)
        self.spatial_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.spatial_backbone.fc = nn.Identity()  # Remove classifier
        self.spatial_features = 2048
        
        # Frequency stream (64-channel DCT input)
        self.freq_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # FIX: Replace conv1 to accept 64-channel DCT input
        self.freq_backbone.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.freq_backbone.fc = nn.Identity()
        self.freq_features = 2048
        
        # FAD-style fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.spatial_features + self.freq_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Raw logit (sigmoid applied externally)
        )
        
        # Preprocessors
        self.spatial_preprocessor = SpatialPreprocessor()
        self.freq_preprocessor = DCTPreprocessor()
        
        # Hook point: after DCT conv, before freq backbone
        # We'll register hook on freq_preprocessor output
        self.dct_output = None
    
    def forward(self, rgb: torch.Tensor, return_dct: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through both streams.
        
        Args:
            rgb: (B, 3, H, W) RGB tensor in [0, 1] range
            return_dct: If True, return DCT coefficients for band analysis
        
        Returns:
            logit: (B, 1) raw classification logit
            dct_coeffs: (B, 64, H/8, W/8) DCT coefficients (if return_dct=True)
        """
        # Spatial stream
        spatial_norm = self.spatial_preprocessor(rgb)
        spatial_features = self.spatial_backbone(spatial_norm)  # (B, 2048)
        
        # Frequency stream
        dct_coeffs = self.freq_preprocessor(rgb)  # (B, 64, 28, 28)
        freq_features = self.freq_backbone(dct_coeffs)  # (B, 2048)
        
        # Fusion
        combined = torch.cat([spatial_features, freq_features], dim=1)  # (B, 4096)
        logit = self.fusion(combined)  # (B, 1)
        
        if return_dct:
            return logit, dct_coeffs
        return logit, None


class FreqNetTool(BaseForensicTool):
    @property
    def tool_name(self) -> str:
        return "run_freqnet"
    
    def __init__(self):
        super().__init__()
        self.device = None
        self.has_sigmoid = False
        self.calibration_manager = CalibrationManager()
        self.requires_gpu = True
    
    def setup(self):
        """Tool-specific setup."""
        # Load calibration (dual-mode)
        self.calibration_manager.load()
        logger.info(f"FreqNetTool setup complete (mode: {self.calibration_manager.mode})")
        return True
    
    def _load_model(self) -> torch.nn.Module:
        """Load FreqNet model with proper weight handling."""
        model = FreqNetDual()
        
        # Try to load custom FreqNet weights
        config = AegisConfig()
        weight_path = getattr(config.models, 'freqnet_weights', None)
        
        ROOT_DIR = Path(__file__).parent.parent.parent
        LOCAL_WEIGHT_PATH = ROOT_DIR / "models" / "freqnet" / "freqnet_f3net.pth"
        
        if LOCAL_WEIGHT_PATH.exists():
            weight_path = str(LOCAL_WEIGHT_PATH)
        
        if weight_path and os.path.exists(weight_path):
            try:
                state_dict = torch.load(weight_path, map_location='cpu', weights_only=True)
                
                # Handle nested state dicts
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                
                # Load with strict=False (our head may differ)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"✅ FreqNet weights loaded from {weight_path}")
            except Exception as e:
                logger.warning(f"Failed to load FreqNet weights: {e}. Using ImageNet backbone.")
        else:
            logger.info("⚠️  No FreqNet weights found. Using ImageNet backbone + random head.")
        
        # Check for sigmoid in final layer
        last_layer = model.fusion[-1]
        self.has_sigmoid = isinstance(last_layer, nn.Sigmoid)
        
        model.eval()
        return model
    
    def _safe_expand_crop(self, face_image: np.ndarray, landmarks: np.ndarray, 
                          expansion: float = 1.1) -> np.ndarray:
        """
        FIX #11: Safe crop expansion with boundary handling.
        """
        h_img, w_img, _ = face_image.shape
        
        # Get face bbox from landmarks
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        # Convert to absolute if normalized
        if landmarks.max() <= 1.0:
            x_min, x_max = x_min * w_img, x_max * w_img
            y_min, y_max = y_min * h_img, y_max * h_img
        
        w = x_max - x_min
        h = y_max - y_min
        cx, cy = x_min + w / 2, y_min + h / 2
        
        # Expand
        new_w, new_h = w * expansion, h * expansion
        
        # FIX #11: Clamp to image boundaries
        x1 = max(0, int(cx - new_w / 2))
        y1 = max(0, int(cy - new_h / 2))
        x2 = min(w_img, int(cx + new_w / 2))
        y2 = min(h_img, int(cy + new_h / 2))
        
        crop = face_image[y1:y2, x1:x2]
        
        # Resize to 224×224
        if crop.shape[0] > 0 and crop.shape[1] > 0:
            crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Fallback to original if crop failed
            crop = cv2.resize(face_image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
        
        return crop
    
    def _run_inference(self, input_data: dict) -> ToolResult:
        start_time = time.time()
        
        # Check for tracked faces
        tracked_faces = input_data.get("tracked_faces", [])
        if not tracked_faces:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.0,
                confidence=0.0,
                details={},
                error=True,
                error_msg="No tracked faces found.",
                execution_time=time.time() - start_time,
                evidence_summary="FreqNet detector: No tracked faces found."
            )
        
        with VRAMLifecycleManager(self._load_model) as model:
            # Get device from model
            self.device = next(model.parameters()).device
            
            best_score = 0.0
            best_band_analysis = None
            all_analyses = []
            
            for face in tracked_faces:
                face_crop = face.get("face_crop_224")
                landmarks = face.get("landmarks")
                
                if face_crop is None or landmarks is None:
                    continue
                
                if isinstance(landmarks, list):
                    landmarks = np.array(landmarks)
                
                # Normalize landmarks if needed
                if landmarks.max() > 1.0 and landmarks.max() <= 224.0:
                    landmarks = landmarks / 224.0
                
                # FIX #11: Safe crop expansion
                try:
                    crop_expanded = self._safe_expand_crop(face_crop, landmarks, 1.1)
                except Exception as e:
                    logger.warning(f"Crop expansion failed: {e}. Using original crop.")
                    crop_expanded = cv2.resize(face_crop, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                
                # Convert to tensor
                tensor = torch.from_numpy(crop_expanded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                tensor = tensor.to(self.device)
                
                # FIX #4 & #12: no_grad for inference, hook with detach()
                with torch.no_grad():
                    # Create hook for band analysis
                    hook = FADHook(self.calibration_manager.get_data())
                    
                    try:
                        # Register hook on frequency preprocessor
                        hook.register(model.freq_preprocessor)
                        
                        # Forward pass
                        logit, dct_coeffs = model(tensor, return_dct=True)
                        
                        # Apply sigmoid if needed
                        if not self.has_sigmoid:
                            score = torch.sigmoid(logit).item()
                        else:
                            score = logit.item()
                        
                        # Band analysis
                        band_analysis = hook.analyze()
                        all_analyses.append(band_analysis)
                        
                        if score > best_score:
                            best_score = score
                            best_band_analysis = band_analysis
                    
                    finally:
                        # FIX #7 & #12: Always remove hook
                        hook.remove()
                
                # FIX #4: Per-face VRAM cleanup
                del tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Aggregate results across faces
            if all_analyses:
                # Use best score and its corresponding band analysis
                anomaly_info = best_band_analysis.interpretation if best_band_analysis else ""
            else:
                anomaly_info = "No faces analyzed successfully"
            
            # Compute confidence based on score and anomaly detection
            if best_band_analysis and best_band_analysis.anomaly_detected:
                confidence = min(1.0, best_score + 0.2)
            else:
                confidence = max(0.5, 1.0 - abs(best_score - 0.5))
            
            # Evidence summary
            if best_score > FREQNET_FAKE_THRESHOLD:
                if best_band_analysis and best_band_analysis.anomaly_detected:
                    summary = (f"FreqNet detector: frequency anomaly detected ({anomaly_info}). "
                               f"Score: {best_score:.2f}. Consistent with synthetic generation.")
                else:
                    summary = (f"FreqNet detector: high fake score ({best_score:.2f}) but no specific "
                               f"frequency anomaly. May indicate advanced synthesis.")
            else:
                summary = (f"FreqNet detector: no frequency anomalies detected (score: {best_score:.2f}). "
                           f"Consistent with authentic content.")
            
            # Build details dict
            details = {
                "band_analysis": {
                    "base_ratio": best_band_analysis.base_ratio if best_band_analysis else 0.0,
                    "mid_ratio": best_band_analysis.mid_ratio if best_band_analysis else 0.0,
                    "high_ratio": best_band_analysis.high_ratio if best_band_analysis else 0.0,
                } if best_band_analysis else {},
                "z_scores": {
                    "base": best_band_analysis.z_base,
                    "mid": best_band_analysis.z_mid,
                    "high": best_band_analysis.z_high,
                } if best_band_analysis and best_band_analysis.z_base is not None else {},
                "anomaly_detected": best_band_analysis.anomaly_detected if best_band_analysis else False,
                "anomaly_type": best_band_analysis.anomaly_type if best_band_analysis else None,
                "calibration_mode": self.calibration_manager.mode,
                "faces_analyzed": len(all_analyses),
            }
            
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=best_score,
                confidence=confidence,
                details=details,
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary=summary
            )