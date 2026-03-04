"""Video I/O utilities for Aegis-X.

Provides robust, memory-safe functions for extracting video frames using hardware
acceleration (NVDEC via torchcodec) when available, falling back to OpenCV.
"""
# on line 53 you can change "cpu" to "cuda" to use GPU acceleration
import math
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)

# Attempt to import torchcodec and torch for hardware-accelerated video decoding
TORCHCODEC_AVAILABLE = False
try:
    import torch
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except ImportError:
    pass

# Known valid video extensions
VALID_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}

def is_video_file(path: str) -> bool:
    """Checks if the given file path has a valid video extension."""
    return Path(path).suffix.lower() in VALID_VIDEO_EXTENSIONS

def _calculate_scale(width: int, height: int, max_width: int = 1280) -> Optional[Tuple[int, int]]:
    """Calculates dimensions to downscale a frame preserving aspect ratio."""
    if width <= max_width:
        return None
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    return (new_width, new_height)

def extract_frames(video_path: str, max_frames: int = 300, target_fps: int = 30) -> List[np.ndarray]:
    """Safely extracts a sequence of RGB frames from a video file."""
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return []

    # Fast path: TorchCodec GPU Decoding
    if TORCHCODEC_AVAILABLE:
        try:
            # Fix: Explicitly assign CUDA device to actually leverage NVDEC
            device = "cpu" if torch.cuda.is_available() else "gpu"
            decoder = VideoDecoder(video_path, device=device)
            
            # Fetch metadata with safe null-checks
            total_frames = decoder.metadata.num_frames
            source_fps = decoder.metadata.average_fps
            
            if total_frames is None or source_fps is None or total_frames <= 0 or source_fps <= 0:
                 raise ValueError("Invalid metadata returned by TorchCodec.")
                 
            total_frames = int(total_frames)
            source_fps = float(source_fps)
                 
            # Compute temporal indices
            fps_ratio = source_fps / target_fps
            skip_interval = max(1.0, fps_ratio)
            
            # Generate frame indices to extract
            indices = []
            current_index = 0.0
            while current_index < total_frames and len(indices) < max_frames:
                idx = int(round(current_index))
                if idx < total_frames:
                    indices.append(idx)
                current_index += skip_interval
                
            if not indices:
                return []
                
            # Fix: Batch extraction leveraging NVDEC in chunks to genuinely prevent OOM
            frames_list = []
            batch_size = 32 
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Returns a PyTorch tensor usually shape (N, C, H, W)
                frames_tensor = decoder.get_frames_at(indices=batch_indices).data
                
                # Convert to CPU numpy array
                frames_np = frames_tensor.cpu().numpy()
                
                # Torchcodec defaults to NCHW. Let's strictly permute.
                if frames_np.ndim == 4 and frames_np.shape[1] == 3:
                    frames_np = np.transpose(frames_np, (0, 2, 3, 1))

                for frame in frames_np:
                    h, w, c = frame.shape
                    
                    # Check for memory limits
                    scale_dims = _calculate_scale(w, h)
                    if scale_dims:
                        frame = cv2.resize(frame, scale_dims, interpolation=cv2.INTER_AREA)
                        
                    frames_list.append(frame)
                
            return frames_list
        except Exception as e:
            logger.warning(f"TorchCodec extraction failed: {e}. Falling back to OpenCV CPU decode.")

    # Fallback path: OpenCV CPU Decoding
    return _extract_cv2(video_path, max_frames, target_fps)

def _extract_cv2(video_path: str, max_frames: int, target_fps: int) -> List[np.ndarray]:
    """Fallback frame extraction method utilizing traditional OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file via cv2: {video_path}")
        return []
        
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    # Fix: Added NaN check, as OpenCV routinely returns NaN for corrupted metadata
    if source_fps <= 0 or math.isnan(source_fps):
         source_fps = 30.0 # Blind fallback

    fps_ratio = source_fps / target_fps
    skip_interval = max(1.0, fps_ratio)
    
    frames_list = []
    frame_idx = 0
    current_target = 0.0
    
    while cap.isOpened() and len(frames_list) < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
            
        if frame_idx >= int(round(current_target)):
            h, w = frame_bgr.shape[:2]
            scale_dims = _calculate_scale(w, h)
            if scale_dims:
                frame_bgr = cv2.resize(frame_bgr, scale_dims, interpolation=cv2.INTER_AREA)
                
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_list.append(frame_rgb)
            current_target += skip_interval
            
        frame_idx += 1
        
    cap.release()
    return frames_list

def get_video_duration(path: Path) -> float:
    """Gets the duration of the video file in seconds."""
    video_path_str = str(path)
    
    if TORCHCODEC_AVAILABLE:
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            decoder = VideoDecoder(video_path_str, device=device)
            
            # Fix: Safely fetch the native duration first
            duration = decoder.metadata.duration_seconds
            if duration is not None and duration > 0:
                return float(duration)
                
            num_frames = decoder.metadata.num_frames
            fps = decoder.metadata.average_fps
            if num_frames and fps and float(fps) > 0:
                return float(int(num_frames) / float(fps))
        except Exception:
            pass # Fallback to OpenCV
            
    # OpenCV Fallback
    cap = cv2.VideoCapture(video_path_str)
    if not cap.isOpened():
        return 0.0
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    
    if fps > 0:
        return float(frame_count / fps)
        
    return 0.0