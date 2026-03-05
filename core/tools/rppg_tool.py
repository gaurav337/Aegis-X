import time
import numpy as np
from typing import Dict, Any, Tuple

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult


class RPPGTool(BaseForensicTool):
    """Tool for detecting biological liveness signals using POS rPPG algorithm and Spectral Coherence."""

    @property
    def tool_name(self) -> str:
        return "run_rppg"

    def setup(self) -> None:
        self._debug = False

    def _extract_roi(
        self, frame: np.ndarray, current_bbox: tuple, relative_box: tuple
    ) -> np.ndarray:
        x1, y1, x2, y2 = current_bbox
        w = x2 - x1
        h = y2 - y1
        rx_min, ry_min, rx_max, ry_max = relative_box
        fx1 = int(x1 + (w * rx_min))
        fy1 = int(y1 + (h * ry_min))
        fx2 = int(x1 + (w * rx_max))
        fy2 = int(y1 + (h * ry_max))
        frame_h, frame_w = frame.shape[:2]
        fx1 = max(0, min(fx1, frame_w - 1))
        fy1 = max(0, min(fy1, frame_h - 1))
        fx2 = max(0, min(fx2, frame_w))
        fy2 = max(0, min(fy2, frame_h))
        if fx1 >= fx2 or fy1 >= fy2:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        return frame[fy1:fy2, fx1:fx2]

    def _get_facial_rois(self, landmarks: np.ndarray) -> Dict[str, tuple]:
        rois = {
            "forehead": (0.2, 0.05, 0.8, 0.25),
            "left_cheek": (0.1, 0.5, 0.4, 0.85),
            "right_cheek": (0.6, 0.5, 0.9, 0.85),
        }
        if len(landmarks.shape) == 2 and landmarks.shape[0] == 478:
            face_min_x, face_max_x = np.min(landmarks[:, 0]), np.max(landmarks[:, 0])
            face_min_y, face_max_y = np.min(landmarks[:, 1]), np.max(landmarks[:, 1])
            face_w = face_max_x - face_min_x
            face_h = face_max_y - face_min_y
            if face_w > 0 and face_h > 0:
                def _rel(pts):
                    return (
                        (np.min(pts[:, 0]) - face_min_x) / face_w,
                        (np.min(pts[:, 1]) - face_min_y) / face_h,
                        (np.max(pts[:, 0]) - face_min_x) / face_w,
                        (np.max(pts[:, 1]) - face_min_y) / face_h,
                    )
                rois["forehead"] = _rel(landmarks[[10, 338, 297, 332, 284, 103, 67]])
                rois["left_cheek"] = _rel(landmarks[[50, 205, 207, 215, 138, 135, 210]])
                rois["right_cheek"] = _rel(landmarks[[280, 425, 427, 435, 367, 364, 430]])
        return rois

    def _extract_pos_signal(
        self, frames: list, trajectory: dict, relative_roi: tuple
    ) -> np.ndarray:
        """Extracts the 1D POS pulse signal for a specific ROI.
        Returns (H, temporal_std) or (None, 0.0) on failure.
        temporal_std = std of raw green channel over time (quality indicator).
        """
        rgb_means = []
        last_known_box = None

        for f_idx, frame in enumerate(frames):
            if f_idx in trajectory:
                curr_box = trajectory[f_idx]
                last_known_box = curr_box
            elif last_known_box is not None:
                curr_box = last_known_box
            else:
                continue

            roi = self._extract_roi(frame, curr_box, relative_roi)

            # Darkness occlusion guard on first valid frame
            if len(rgb_means) == 0 and roi.size > 0:
                gray_roi = np.mean(roi, axis=2)
                if np.mean(gray_roi) < 50.0:
                    return None, 0.0

            if roi.size == 0:
                if rgb_means:
                    rgb_means.append(rgb_means[-1])
                else:
                    rgb_means.append(np.array([128.0, 128.0, 128.0]))
            else:
                spatial_mean = np.mean(roi, axis=(0, 1))
                rgb_means.append(spatial_mean)

        rgb_matrix = np.array(rgb_means, dtype=np.float64)
        if len(rgb_matrix) < 90:
            return None, 0.0

        # ── Quality metric: temporal std of green channel ──
        # This tells us if there's ANY temporal variation to work with
        green_temporal_std = float(np.std(rgb_matrix[:, 1]))

        mean_rgb = np.maximum(np.mean(rgb_matrix, axis=0), 1.0)
        Cn = rgb_matrix / mean_rgb
        pos_weights = np.array([[0, 1, -1], [-2, 1, 1]])
        S = pos_weights @ Cn.T
        std_s0 = np.std(S[0])
        std_s1 = np.std(S[1]) + 1e-7
        h = S[0] + (std_s0 / std_s1) * S[1]
        h_mean = np.mean(h)
        h_std = np.std(h) + 1e-7
        H = (h - h_mean) / h_std
        return np.nan_to_num(H), green_temporal_std

    def _calculate_signal_metrics(
        self, signal_1d: np.ndarray, fps: float = 30.0
    ) -> Dict[str, float]:
        """Compute frequency-domain metrics for a single ROI pulse signal.
        
        Returns dict with:
            peak_hz:                 Dominant frequency in cardiac band
            snr_db:                  Traditional SNR (for logging)
            spectral_concentration:  Peak power / median power in cardiac band
                                     (scale-invariant, robust to compression)
        """
        # Flatline fast path
        if np.std(signal_1d) < 1e-5:
            return {"peak_hz": 0.0, "snr_db": -100.0, "spectral_concentration": 0.0}

        signal_centered = signal_1d.copy() - np.mean(signal_1d)
        windowed = signal_centered * np.hanning(len(signal_centered))

        n_fft = 2048
        psd = np.abs(np.fft.rfft(windowed, n=n_fft)) ** 2
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)

        # Cardiac band: 0.7 – 2.5 Hz (42–150 BPM)
        band_mask = (freqs >= 0.7) & (freqs <= 2.5)
        band_psd = psd[band_mask]
        band_freqs = freqs[band_mask]

        if len(band_psd) == 0 or np.sum(band_psd) == 0:
            return {"peak_hz": 0.0, "snr_db": -100.0, "spectral_concentration": 0.0}

        peak_idx = int(np.argmax(band_psd))
        peak_hz = float(band_freqs[peak_idx])

        # ── Spectral Concentration (key metric) ──
        # How much does the peak stand out above the median?
        # Noise → flat → concentration ≈ 1.0
        # Real pulse → sharp peak → concentration >> 3.0
        # Robust to overall signal level (compression-invariant)
        median_power = float(np.median(band_psd))
        peak_power = float(band_psd[peak_idx])
        spectral_concentration = peak_power / (median_power + 1e-10)

        # ── Traditional SNR (for logging/debug only) ──
        start_idx = max(0, peak_idx - 2)
        end_idx = min(len(band_psd), peak_idx + 3)
        signal_power = np.sum(band_psd[start_idx:end_idx])
        noise_mask = (freqs >= 0.4)
        total_valid = np.sum(psd[noise_mask])
        noise_power = max(total_valid - signal_power, 1e-10)
        signal_power = max(signal_power, 1e-10)
        snr_db = float(10 * np.log10(signal_power / noise_power))

        return {
            "peak_hz": peak_hz,
            "snr_db": snr_db,
            "spectral_concentration": spectral_concentration,
        }

    def _evaluate_liveness(
        self,
        h_forehead: np.ndarray,
        h_left: np.ndarray,
        h_right: np.ndarray,
        quality_stds: list,
    ) -> dict:
        """Multi-stage liveness evaluation using flatline detection,
        quality gate, spectral concentration, and pairwise coherence."""

        roi_labels = ["Forehead", "L_Cheek", "R_Cheek"]
        signals = [h_forehead, h_left, h_right]

        # ══════════════════════════════════════════
        #  STAGE 0: FLATLINE DETECTION
        #  Must come BEFORE quality gate.
        #  Static/synthetic frames have zero temporal variance
        #  AND zero spectral content — that's a strong fake signal,
        #  not a "quality problem".
        # ══════════════════════════════════════════
        all_flat = all(np.std(s) < 1e-5 for s in signals)
        if all_flat:
            return {
                "label": "NO_PULSE",
                "score": 1.0,
                "confidence": 0.90,
                "interpretation": (
                    "Biological liveness failed: All facial regions show zero temporal "
                    "variation — complete flatline. No blood flow modulation detected. "
                    "Strong indicator of synthetic or static media."
                ),
            }

        # ══════════════════════════════════════════
        #  STAGE 1: QUALITY GATE
        #  Is the video analyzable?
        #  (Only reached if signal is NOT flatline)
        # ══════════════════════════════════════════
        MIN_TEMPORAL_STD = 0.3
        analyzable_count = sum(1 for s in quality_stds if s >= MIN_TEMPORAL_STD)

        if analyzable_count < 2:
            return {
                "label": "ABSTAIN",
                "score": 0.5,
                "confidence": 0.0,
                "interpretation": (
                    "rPPG abstained: Video quality too low for biological signal extraction. "
                    f"Temporal color variance ({max(quality_stds):.2f}) is below the minimum "
                    "threshold needed for pulse detection. Common in heavily compressed video. "
                    "Tool cannot make a liveness determination."
                ),
            }

        # ══════════════════════════════════════════
        #  STAGE 2: SPECTRAL ANALYSIS
        # ══════════════════════════════════════════
        metrics = [self._calculate_signal_metrics(s) for s in signals]

        if getattr(self, "_debug", False):
            for i, label in enumerate(roi_labels):
                m = metrics[i]
                print(
                    f"       [DEBUG rPPG] {label}: "
                    f"SNR={m['snr_db']:+.2f} dB  "
                    f"SC={m['spectral_concentration']:.1f}x  "
                    f"peak={m['peak_hz']:.3f} Hz ({m['peak_hz']*60:.0f} BPM)  "
                    f"green_std={quality_stds[i]:.2f}"
                )

        SC_USABLE = 3.0
        good_mask = [m["spectral_concentration"] >= SC_USABLE for m in metrics]
        n_good = sum(good_mask)

        # ── FLATLINE: No ROI has any spectral peak ──
        if n_good == 0:
            max_sc = max(m["spectral_concentration"] for m in metrics)
            if max_sc < 1.5:
                return {
                    "label": "NO_PULSE",
                    "score": 1.0,
                    "confidence": 0.90,
                    "interpretation": (
                        "Biological liveness failed: No cardiac peak detected in any facial region. "
                        f"Spectral concentration is flat (max {max_sc:.1f}x), indicating complete "
                        "absence of blood flow modulation. Consistent with synthetic or static media."
                    ),
                }
            else:
                return {
                    "label": "AMBIGUOUS",
                    "score": 0.5,
                    "confidence": 0.0,
                    "interpretation": (
                        "rPPG inconclusive: Weak spectral peaks detected but none strong enough "
                        "for reliable liveness determination."
                    ),
                }

        # ── INSUFFICIENT: Only 1 usable ROI ──
        if n_good == 1:
            return {
                "label": "AMBIGUOUS",
                "score": 0.5,
                "confidence": 0.0,
                "interpretation": (
                    "rPPG inconclusive: Only one facial region yielded a usable signal. "
                    "Cannot verify cross-region coherence."
                ),
            }

        # ══════════════════════════════════════════
        #  STAGE 3: PAIRWISE COHERENCE
        #  Instead of requiring ALL good ROIs to agree,
        #  find the BEST pair that agrees.
        #  This handles: one noisy ROI with a spurious peak,
        #  or one ROI corrupted by motion while others are clean.
        # ══════════════════════════════════════════
        COHERENCE_THRESHOLD_HZ = 0.15  # ~9 BPM tolerance

        good_indices = [i for i, g in enumerate(good_mask) if g]
        good_peaks = [metrics[i]["peak_hz"] for i in good_indices]
        good_scs = [metrics[i]["spectral_concentration"] for i in good_indices]

        best_pair = None
        best_pair_diff = float("inf")

        for a in range(len(good_indices)):
            for b in range(a + 1, len(good_indices)):
                diff = abs(good_peaks[a] - good_peaks[b])
                if diff < best_pair_diff:
                    best_pair_diff = diff
                    best_pair = (good_indices[a], good_indices[b])

        if best_pair_diff <= COHERENCE_THRESHOLD_HZ:
            # Count how many ROIs agree with the coherent pair
            pair_avg_hz = (metrics[best_pair[0]]["peak_hz"] + metrics[best_pair[1]]["peak_hz"]) / 2
            n_coherent = sum(
                1 for i in good_indices
                if abs(metrics[i]["peak_hz"] - pair_avg_hz) <= COHERENCE_THRESHOLD_HZ
            )
            avg_bpm = pair_avg_hz * 60

            if n_coherent >= 3:
                conf = 0.95
            elif n_coherent == 2:
                conf = 0.70
            else:
                conf = 0.50

            return {
                "label": "PULSE_PRESENT",
                "score": 0.0,
                "confidence": conf,
                "interpretation": (
                    f"Biological liveness confirmed: Synchronous cardiac pulse detected "
                    f"across {n_coherent}/3 facial regions at ~{avg_bpm:.0f} BPM "
                    f"(best pair spread: {best_pair_diff*60:.1f} BPM). "
                    f"Cross-region coherence verified."
                ),
            }

        # ── INCOHERENT: Every pair disagrees ──
        conf = 0.90 if n_good == 3 else 0.65
        return {
            "label": "INCOHERENT",
            "score": 1.0,
            "confidence": conf,
            "interpretation": (
                "Spoof detected: Pulse frequencies are unsynchronized across all pairs of "
                f"facial regions (best pair spread: {best_pair_diff*60:.1f} BPM). "
                "This physical impossibility indicates a screen replay or synthetic face swap."
            ),
        }

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:
        start_time = time.time()

        if "frames_30fps" not in input_data or "tracked_faces" not in input_data:
            return ToolResult(
                tool_name=self.tool_name,
                success=False,
                score=0.5,
                confidence=0.0,
                details={},
                error=True,
                error_msg="Missing 'frames_30fps' or 'tracked_faces' in input_data",
                execution_time=time.time() - start_time,
                evidence_summary="Missing required input data.",
            )

        frames = input_data["frames_30fps"]
        tracked_faces = input_data["tracked_faces"]

        if len(frames) < 90:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.5,
                confidence=0.0,
                details={},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="Neutral result: insufficient frames (< 90) for rPPG analysis.",
            )

        face_results = []

        for face in tracked_faces:
            if "trajectory_bboxes" not in face or "landmarks" not in face:
                continue

            trajectory = face["trajectory_bboxes"]
            landmarks = face["landmarks"]
            rois = self._get_facial_rois(landmarks)

            # Extract signals + quality metrics for each ROI
            h_forehead, std_f = self._extract_pos_signal(frames, trajectory, rois["forehead"])
            h_left, std_l = self._extract_pos_signal(frames, trajectory, rois["left_cheek"])
            h_right, std_r = self._extract_pos_signal(frames, trajectory, rois["right_cheek"])

            if h_forehead is None or h_left is None or h_right is None:
                face_results.append(
                    (0.5, 0.0, "Ambiguous: One or more facial regions occluded or tracking failed.")
                )
                continue

            liveness = self._evaluate_liveness(
                h_forehead, h_left, h_right,
                quality_stds=[std_f, std_l, std_r],
            )
            face_results.append(
                (liveness["score"], liveness["confidence"], liveness["interpretation"])
            )

        if not face_results:
            return ToolResult(
                tool_name=self.tool_name,
                success=True,
                score=0.5,
                confidence=0.0,
                details={},
                error=False,
                error_msg=None,
                execution_time=time.time() - start_time,
                evidence_summary="All tracked faces yielded ambiguous tracking or were occluded.",
            )

        best = sorted(face_results, key=lambda x: x[1], reverse=True)[0]

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=best[0],
            confidence=best[1],
            details={},
            error=False,
            error_msg=None,
            execution_time=time.time() - start_time,
            evidence_summary=best[2],
        )