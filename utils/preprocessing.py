"""Face preprocessing and patching leveraging MediaPipe Face Mesh.

Extracts standardized face crops and precise anatomical patches according
to Aegis-X Phase 1 specifications.
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.config import PreprocessingConfig
from utils.video import extract_frames, is_video_file
from utils.image import load_image, is_image

# Added logger to prevent silent failures
from utils.logger import setup_logger
logger = setup_logger(__name__)

@dataclass
class PreprocessResult:
    """Standardized output payload from the MediaPipe face preprocessing pipeline."""
    has_face: bool
    landmarks: Optional[np.ndarray] = None
    face_crop_224: Optional[np.ndarray] = None
    face_crop_380: Optional[np.ndarray] = None
    patch_left_periorbital: Optional[np.ndarray] = None
    patch_right_periorbital: Optional[np.ndarray] = None
    patch_nasolabial_left: Optional[np.ndarray] = None
    patch_nasolabial_right: Optional[np.ndarray] = None
    patch_hairline_band: Optional[np.ndarray] = None
    patch_chin_jaw: Optional[np.ndarray] = None
    # WARNING: Holding all frames in RAM is an OOM risk. Consider dropping this downstream.
    frames_30fps: Optional[List[np.ndarray]] = None 
    selected_frame_index: int = 0
    selected_frame_sharpness: float = 0.0
    original_media_type: str = "image"


class Preprocessor:
    """MediaPipe-based robust face landmark extraction and patching class."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
    def close(self):
        """Releases MediaPipe C++ resources to prevent memory leaks."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

    def __del__(self):
        """Ensures resources are freed when the object is garbage collected."""
        self.close()
        
    def _get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Runs MediaPipe Face Mesh and returns 478 landmarks scaled to image bounds."""
        # Safety check for empty arrays
        if image is None or image.size == 0:
            return None
            
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
            
        h, w = image.shape[:2]
        face_landmarks = results.multi_face_landmarks[0]
        
        coords = np.zeros((478, 2), dtype=np.float32)
        for i, lm in enumerate(face_landmarks.landmark):
            coords[i] = [lm.x * w, lm.y * h]
            
        nose = coords[1]
        jaw_l = coords[234]
        jaw_r = coords[454]
        for node in [nose, jaw_l, jaw_r]:
            if not (0 <= node[0] < w and 0 <= node[1] < h):
                return None
                
        return coords
        
    def _crop_align(self, image: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray:
        """Extracts bounding box covering 478 landmarks with 20% margin, resizes with Lanczos4."""
        h, w = image.shape[:2]
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        margin_x = box_w * 0.2
        margin_y = box_h * 0.2
        
        x1 = max(0, int(x_min - margin_x))
        y1 = max(0, int(y_min - margin_y))
        x2 = min(w, int(x_max + margin_x))
        y2 = min(h, int(y_max + margin_y))
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros((size, size, 3), dtype=np.uint8)
            
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
        
    def _extract_native_patches(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple:
        """Extracts 6 anatomical patches directly mapping to CLIP adapter Stage 0."""
        h, w = image.shape[:2]
        size = self.config.native_patch_size
        
        patches_def = {
            "left_periorbital": [33, 133, 160, 159, 158, 144],
            "right_periorbital": [263, 362, 385, 386, 387, 373],
            "nasolabial_left": [92, 205, 216, 206],
            "nasolabial_right": [322, 425, 436, 426],
            "hairline_band": [10, 338, 297, 332, 284, 103, 67],
            "chin_jaw": [172, 136, 150, 149, 176, 148, 152, 377, 400, 379, 365]
        }
        
        results = {}
        for name, nodes in patches_def.items():
            pts = landmarks[nodes]
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            box_w = x_max - x_min
            box_h = y_max - y_min
            
            margin_x = box_w * 0.2
            margin_y = box_h * 0.2
            
            x1 = max(0, int(x_min - margin_x))
            y1 = max(0, int(y_min - margin_y))
            x2 = min(w, int(x_max + margin_x))
            y2 = min(h, int(y_max + margin_y))
            
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                results[name] = np.zeros((size, size, 3), dtype=np.uint8)
            else:
                results[name] = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
                
        return (
            results["left_periorbital"],
            results["right_periorbital"],
            results["nasolabial_left"],
            results["nasolabial_right"],
            results["hairline_band"],
            results["chin_jaw"]
        )

    def _select_sharpest_frame(self, frames: List[np.ndarray]) -> Tuple[int, float, Optional[np.ndarray]]:
        """
        Fix: Implements Dynamic Quality Snipe. Re-evaluates face bounds on sampled frames 
        to prevent static bounding-box drift when the subject moves.
        Returns: (best_idx, best_sharpness, best_landmarks)
        """
        if not frames:
            return 0, 0.0, None
            
        num_frames = len(frames)
        num_samples = min(num_frames, self.config.quality_snipe_samples)
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        
        best_idx = 0
        best_sharpness = -1.0
        best_landmarks = None
        
        for idx in indices:
            frame = frames[idx]
            
            # Dynamically detect landmarks to ensure we are cropping the actual face, not the wall
            lm = self._get_landmarks(frame)
            if lm is None:
                continue
                
            x_min, y_min = np.min(lm, axis=0)
            x_max, y_max = np.max(lm, axis=0)
            
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, int(x_min)), max(0, int(y_min))
            cx2, cy2 = min(w, int(x_max)), min(h, int(y_max))
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            if face_crop.size == 0:
                continue
                
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > best_sharpness:
                best_sharpness = float(sharpness)
                best_idx = int(idx)
                best_landmarks = lm
                
        return best_idx, max(0.0, best_sharpness), best_landmarks

    def process_media(self, path: Path) -> PreprocessResult:
        """End-to-end processing pipeline for images or videos."""
        path_str = str(path)
        result = PreprocessResult(has_face=False)
        
        try:
            if is_video_file(path_str):
                result.original_media_type = "video"
                frames = extract_frames(path_str, self.config.max_video_frames, self.config.extract_fps)
                if not frames:
                    return result
                    
                result.frames_30fps = frames
                
                # Use the new dynamic snipe to find the sharpest actual face
                best_idx, best_sharp, final_landmarks = self._select_sharpest_frame(frames)
                
                if final_landmarks is None:
                    # No face found in any of the sampled frames
                    return result
                    
                result.selected_frame_index = best_idx
                result.selected_frame_sharpness = best_sharp
                target_image = frames[best_idx]
                
            elif is_image(path_str):
                result.original_media_type = "image"
                image = load_image(path)
                result.frames_30fps = [image]
                result.selected_frame_index = 0
                
                final_landmarks = self._get_landmarks(image)
                if final_landmarks is None:
                    return result
                    
                target_image = image
                
            else:
                return result
                
            # If we reach here, we have a valid face image and aligned landmarks
            result.has_face = True
            result.landmarks = final_landmarks
            
            result.face_crop_224 = self._crop_align(target_image, final_landmarks, self.config.face_crop_size)
            result.face_crop_380 = self._crop_align(target_image, final_landmarks, self.config.sbi_crop_size)
            
            patches = self._extract_native_patches(target_image, final_landmarks)
            result.patch_left_periorbital = patches[0]
            result.patch_right_periorbital = patches[1]
            result.patch_nasolabial_left = patches[2]
            result.patch_nasolabial_right = patches[3]
            result.patch_hairline_band = patches[4]
            result.patch_chin_jaw = patches[5]
            
            return result
            
        except Exception as e:
            # Fix: Log the exception instead of silently burying it
            logger.error(f"Preprocessing failed for {path_str}: {e}", exc_info=True)
            return PreprocessResult(has_face=False)