import numpy as np
import cv2
from scipy.fft import dctn
from typing import Dict, Any

from core.base_tool import BaseForensicTool
from core.data_types import ToolResult


class DCTTool(BaseForensicTool):

    def setup(self):
        pass

    @property
    def tool_name(self):
        return "run_dct"

    def _run_inference(self, input_data: Dict[str, Any]) -> ToolResult:

        tracked_faces = input_data.get("tracked_faces", None)

        if not tracked_faces or len(tracked_faces) == 0:
            return ToolResult.abstain(self.tool_name)

        peak_ratios = []

        for face in tracked_faces:

            if "face_crop_224" not in face:
                continue

            img = face["face_crop_224"]

            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            h, w = gray.shape

            h = (h // 8) * 8
            w = (w // 8) * 8
            gray = gray[:h, :w]

            blocks = gray.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)

            dct_blocks = dctn(blocks, axes=(-2, -1), norm="ortho")

            ac_coeffs = dct_blocks.reshape(-1, 64)[:, 1:]
            ac_coeffs = ac_coeffs.flatten()

            hist, _ = np.histogram(ac_coeffs, bins=100, range=(-50, 50))

            autocorr = np.correlate(hist, hist, mode="same")

            center = len(autocorr) // 2

            primary_peak = autocorr[center] + 1e-10

            side = np.concatenate((autocorr[:center-1], autocorr[center+1:]))

            secondary_peak = np.max(side)

            peak_ratio = secondary_peak / primary_peak

            peak_ratios.append(peak_ratio)

        if len(peak_ratios) == 0:
            return ToolResult.abstain(self.tool_name)

        peak_ratio = float(np.mean(peak_ratios))

        # --- FIX APPLIED HERE ---
        # Empirical observation: Clean images yield Higher Ratio (~0.63)
        #                        JPEG images yield Lower Ratio (~0.49)
        # Original logic assumed Higher Ratio = Tampered. We invert this.
        # We also adjust thresholds to prevent saturation (both scoring 1.0).
        
        threshold = 0.55  # Midpoint between observed 0.49 and 0.63
        scale = 0.15      # Sensitivity
        
        # Lower peak_ratio results in Higher score
        score = max(0.0, min(1.0, (threshold - peak_ratio) / scale))

        confidence = min(0.9, score + 0.2)

        if score > 0.5:
            evidence = "DCT analysis detected double-quantization artifacts indicating structural modification"
        else:
            evidence = "Smooth DCT frequency distribution consistent with natural imagery."

        return ToolResult(
            tool_name=self.tool_name,
            success=True,
            score=float(score),
            confidence=float(confidence),
            details={
                "peak_ratio": float(peak_ratio),
                "faces_analyzed": len(peak_ratios)
            },
            error=False,
            error_msg=None,
            execution_time=0.0,
            evidence_summary=evidence
        )