"""Face preprocessing and patching leveraging MediaPipe Face Mesh.

Extracts standardized face crops and precise anatomical patches according
to Aegis-X Phase 1 specifications. Support tracking via CPU-SORT.
"""
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from scipy.optimize import linear_sum_assignment

from core.config import PreprocessingConfig
from utils.video import extract_frames, is_video_file
from utils.image import load_image, is_image
from utils.logger import setup_logger

logger = setup_logger(__name__)

# --- CPU SORT IMPLEMENTATION ---
def iou_batch(bb_test, bb_gt):
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.zeros((len(bb_test), len(bb_gt)))
    bb_test = np.expand_dims(bb_test, 1)
    bb_gt = np.expand_dims(bb_gt, 0)
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
    return wh / np.maximum(1e-6, area_test + area_gt - wh)

class KalmanBoxTracker:
    # Fix: Removed global class-level counter to ensure thread safety
    def __init__(self, bbox, track_id):
        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.transitionMatrix = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                                             [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], np.float32)
        self.kf.statePost = self.convert_bbox_to_z(bbox)
        self.kf.errorCovPost = np.eye(7, dtype=np.float32) * 1.0
        
        # Assign passed-in ID
        self.id = track_id
        self.time_since_update = 0
        self.hit_streak = 0

    def convert_bbox_to_z(self, bbox):
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        return np.array([x, y, w*h, w/h], dtype=np.float32).reshape((4, 1))

    def convert_x_to_bbox(self, x):
        w = np.sqrt(x[2] * x[3]) if x[2]*x[3] > 0 else 0
        h = x[2] / w if w > 0 else 0
        return [float(x[0]-w/2.), float(x[1]-h/2.), float(x[0]+w/2.), float(x[1]+h/2.)]

    def predict(self):
        if (self.kf.statePost[6] + self.kf.statePost[2]) <= 0:
            self.kf.statePost[6] *= 0.0
        self.kf.predict()
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.convert_x_to_bbox(self.kf.statePre)

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

class SortTracker:
    def __init__(self, iou_threshold=0.3):
        self.trackers = []
        self.frame_count = 0
        self.iou_threshold = iou_threshold
        # Fix: Instance-level counter ensures thread-isolation across parallel video jobs
        self.id_count = 0 

    def update(self, dets=None):
        if dets is None:
            dets = np.empty((0, 4))
            
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t,:] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate(dets, trks, self.iou_threshold)

        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])
        for i in unmatched_dets:
            # Increment instance counter and pass safely to new tracker
            self.id_count += 1
            self.trackers.append(KalmanBoxTracker(dets[i], self.id_count))

        ret = []
        for trk in self.trackers:
            d = trk.convert_x_to_bbox(trk.kf.statePost)
            if trk.time_since_update < 1 and (trk.hit_streak >= 1 or self.frame_count <= 1):
                ret.append([d[0], d[1], d[2], d[3], trk.id])
        return np.array(ret)

    def associate(self, detections, trackers, iou_threshold):
        if len(trackers) == 0:
            return np.empty((0,2),dtype=int), np.arange(len(detections)), []
        iou_matrix = iou_batch(detections, trackers)
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = np.asarray(linear_sum_assignment(-iou_matrix)).T
        else:
            matched_indices = np.empty((0,2),dtype=int)

        unmatched_dets = [d for d in range(len(detections)) if d not in matched_indices[:,0]]
        unmatched_trks = [t for t in range(len(trackers)) if t not in matched_indices[:,1]]

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_dets.append(m[0])
                unmatched_trks.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        matches = np.concatenate(matches, axis=0) if len(matches) > 0 else np.empty((0,2),dtype=int)
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

@dataclass
class TrackedFace:
    """Represents a single detected identity tracked across multiple frames."""
    identity_id: int
    landmarks: np.ndarray
    trajectory_bboxes: Dict[int, Tuple[int, int, int, int]]
    face_crop_224: Optional[np.ndarray] = None
    face_crop_380: Optional[np.ndarray] = None
    patch_left_periorbital: Optional[np.ndarray] = None
    patch_right_periorbital: Optional[np.ndarray] = None
    patch_nasolabial_left: Optional[np.ndarray] = None
    patch_nasolabial_right: Optional[np.ndarray] = None
    patch_hairline_band: Optional[np.ndarray] = None
    patch_chin_jaw: Optional[np.ndarray] = None

@dataclass
class PreprocessResult:
    """Standardized output payload from the MediaPipe face preprocessing pipeline."""
    has_face: bool
    tracked_faces: List[TrackedFace] = field(default_factory=list)
    frames_30fps: Optional[List[np.ndarray]] = None 
    selected_frame_index: int = 0
    selected_frame_sharpness: float = 0.0
    original_media_type: str = "image"
    
class Preprocessor:
    """MediaPipe-based robust face landmark extraction and patching class."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        max_faces = getattr(config, 'max_subjects_to_analyze', 2)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=max_faces,
            min_detection_confidence=0.5
        )
        self.tracker = SortTracker()
        
    def close(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

    def __del__(self):
        self.close()
        
    def _get_landmarks(self, image: np.ndarray) -> Optional[List[np.ndarray]]:
        """Runs MediaPipe Face Mesh, returns list of 478 landmarks sorted by area."""
        if image is None or image.size == 0:
            return None
            
        results = self.face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None
            
        h, w = image.shape[:2]
        faces_data = []
        for face_landmarks in results.multi_face_landmarks:
            coords = np.zeros((478, 2), dtype=np.float32)
            for i, lm in enumerate(face_landmarks.landmark):
                coords[i] = [lm.x * w, lm.y * h]
                
            nose = coords[1]
            jaw_l = coords[234]
            jaw_r = coords[454]
            valid = True
            for node in [nose, jaw_l, jaw_r]:
                if not (0 <= node[0] < w and 0 <= node[1] < h):
                    valid = False
                    break
            
            if valid:
                x_min, y_min = np.min(coords, axis=0)
                x_max, y_max = np.max(coords, axis=0)
                area = (x_max - x_min) * (y_max - y_min)
                faces_data.append((area, coords))
                
        faces_data.sort(key=lambda x: x[0], reverse=True)
        if not faces_data:
            return None
        return [data[1] for data in faces_data]
        
    def _crop_align(self, image: np.ndarray, landmarks: np.ndarray, size: int) -> np.ndarray:
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

    def _select_sharpest_frame(self, frames: List[np.ndarray], face_rects: Dict[int, Tuple[int, int, int, int]]) -> Tuple[int, float]:
        if not frames:
            return 0, 0.0
            
        num_frames = len(frames)
        num_samples = min(num_frames, getattr(self.config, 'quality_snipe_samples', 5))
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        
        best_idx = 0
        best_sharpness = -1.0
        
        last_rect = list(face_rects.values())[0] if face_rects else (0, 0, frames[0].shape[1], frames[0].shape[0])
        
        for idx in indices:
            frame = frames[idx]
            x1, y1, x2, y2 = face_rects.get(idx, last_rect)
            last_rect = (x1, y1, x2, y2)
            
            h, w = frame.shape[:2]
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(w, x2), min(h, y2)
            
            face_crop = frame[cy1:cy2, cx1:cx2]
            if face_crop.size == 0:
                continue
                
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > best_sharpness:
                best_sharpness = float(sharpness)
                best_idx = int(idx)
                
        return best_idx, max(0.0, best_sharpness)

    def process_media(self, path: Path) -> PreprocessResult:
        """End-to-end processing pipeline implementing tracking and patching."""
        path_str = str(path)
        result = PreprocessResult(has_face=False)
        
        # Reset tracker state entirely per video
        self.tracker = SortTracker()
        min_res = getattr(self.config, 'min_face_resolution', 64)
        
        try:
            if is_video_file(path_str):
                result.original_media_type = "video"
                frames = extract_frames(path_str, self.config.max_video_frames, self.config.extract_fps)
                if not frames:
                    return result
                result.frames_30fps = frames
                
                established_tracks: Dict[int, TrackedFace] = {}
                
                for i, frame in enumerate(frames):
                    lm_list = self._get_landmarks(frame)
                    dets = []
                    
                    if lm_list:
                        for lm in lm_list:
                            x_min, y_min = np.min(lm, axis=0)
                            x_max, y_max = np.max(lm, axis=0)
                            
                            w = x_max - x_min
                            h = y_max - y_min
                            
                            # Apply the resolution gate
                            if w >= min_res and h >= min_res:
                                dets.append([x_min, y_min, x_max, y_max])
                                
                        # FIX: Fallback logic for the "Sole Distant Subject" edge case
                        # If the gate filtered out everyone, but a face DID exist, keep the largest one
                        if not dets and lm_list:
                            largest_lm = lm_list[0] # Landmarks are pre-sorted by area descending
                            x_min, y_min = np.min(largest_lm, axis=0)
                            x_max, y_max = np.max(largest_lm, axis=0)
                            dets.append([x_min, y_min, x_max, y_max])
                            
                    dets = np.array(dets) if dets else np.empty((0, 4))
                    tracked_items = self.tracker.update(dets)
                    
                    for item in tracked_items:
                        x1, y1, x2, y2, trk_id = item
                        trk_id = int(trk_id)
                        if trk_id not in established_tracks:
                            established_tracks[trk_id] = TrackedFace(
                                identity_id=trk_id,
                                landmarks=np.zeros((478, 2)),
                                trajectory_bboxes={}
                            )
                        established_tracks[trk_id].trajectory_bboxes[i] = (int(x1), int(y1), int(x2), int(y2))
                
                if not established_tracks:
                    return result

                primary_id = list(established_tracks.keys())[0]
                primary_track = established_tracks[primary_id]
                
                winning_idx, best_sharpness = self._select_sharpest_frame(frames, primary_track.trajectory_bboxes)
                result.selected_frame_index = winning_idx
                result.selected_frame_sharpness = best_sharpness
                
                target_image = frames[winning_idx]
                
                final_landmarks_list = self._get_landmarks(target_image)
                if final_landmarks_list is None:
                    return result

                result.has_face = True
                
                final_dets = []
                for lm in final_landmarks_list:
                    x_min, y_min = np.min(lm, axis=0)
                    x_max, y_max = np.max(lm, axis=0)
                    final_dets.append([x_min, y_min, x_max, y_max])
                final_dets = np.array(final_dets)

                trk_ids = []
                trk_boxes = []
                for trk_id, track_obj in established_tracks.items():
                    if winning_idx in track_obj.trajectory_bboxes:
                        trk_ids.append(trk_id)
                        trk_boxes.append(track_obj.trajectory_bboxes[winning_idx])

                if len(trk_boxes) > 0 and len(final_dets) > 0:
                    trk_boxes = np.array(trk_boxes)
                    matches, _, _ = self.tracker.associate(final_dets, trk_boxes, iou_threshold=0.1)

                    for m in matches:
                        det_idx, trk_idx = m[0], m[1]
                        trk_id = trk_ids[trk_idx]
                        track_obj = established_tracks[trk_id]
                        lm = final_landmarks_list[det_idx]

                        track_obj.landmarks = lm
                        track_obj.face_crop_224 = self._crop_align(target_image, lm, self.config.face_crop_size)
                        track_obj.face_crop_380 = self._crop_align(target_image, lm, self.config.sbi_crop_size)
                        
                        patches = self._extract_native_patches(target_image, lm)
                        track_obj.patch_left_periorbital = patches[0]
                        track_obj.patch_right_periorbital = patches[1]
                        track_obj.patch_nasolabial_left = patches[2]
                        track_obj.patch_nasolabial_right = patches[3]
                        track_obj.patch_hairline_band = patches[4]
                        track_obj.patch_chin_jaw = patches[5]
                        
                        result.tracked_faces.append(track_obj)
                
            elif is_image(path_str):
                result.original_media_type = "image"
                image = load_image(path)
                result.frames_30fps = [image]
                result.selected_frame_index = 0
                
                final_landmarks_list = self._get_landmarks(image)
                if final_landmarks_list is None:
                    return result
                    
                result.has_face = True
                for i, lm in enumerate(final_landmarks_list):
                    # For static images, we bypass the SORT tracker completely.
                    track_obj = TrackedFace(
                        identity_id=i+1,
                        landmarks=lm,
                        trajectory_bboxes={0: (0,0,0,0)} 
                    )
                    track_obj.face_crop_224 = self._crop_align(image, lm, self.config.face_crop_size)
                    track_obj.face_crop_380 = self._crop_align(image, lm, self.config.sbi_crop_size)
                    
                    patches = self._extract_native_patches(image, lm)
                    track_obj.patch_left_periorbital = patches[0]
                    track_obj.patch_right_periorbital = patches[1]
                    track_obj.patch_nasolabial_left = patches[2]
                    track_obj.patch_nasolabial_right = patches[3]
                    track_obj.patch_hairline_band = patches[4]
                    track_obj.patch_chin_jaw = patches[5]
                    result.tracked_faces.append(track_obj)
            else:
                return result
                
            return result
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {path_str}: {e}", exc_info=True)
            return PreprocessResult(has_face=False)