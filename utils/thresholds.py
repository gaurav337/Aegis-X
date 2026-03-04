"""Single source of truth for all numeric thresholds used across the Aegis-X system.

This module defines all constants and thresholds utilized by the various forensic
tools and the orchestration engine to ensure consistent evaluation.
"""

# Verdict thresholds
REAL_THRESHOLD = 0.15
FAKE_THRESHOLD = 0.85

# Early stop thresholds
EARLY_STOP_CONFIDENCE = 0.85
MIN_WEIGHT_FOR_STOP = 0.40

# CLIP-specific thresholds
CLIP_FAKE_THRESHOLD = 0.65
CLIP_ATTN_CROSS_THRESHOLD = 0.25
CLIP_PATCH_REPORT_THRESHOLD = 0.65

# SBI-specific thresholds
SBI_FAKE_THRESHOLD = 0.60
SBI_GRADCAM_REGION_THRESHOLD = 0.40
SBI_SKIP_CLIP_THRESHOLD = 0.70
SBI_BLIND_SPOT_THRESHOLD = 0.30

# FreqNet-specific thresholds
FREQNET_FAKE_THRESHOLD = 0.65
FREQNET_Z_THRESHOLD = 1.5

# rPPG-specific thresholds
RPPG_PULSE_PRESENT_THRESHOLD = 0.70
RPPG_NO_PULSE_THRESHOLD = 0.30
RPPG_SNR_THRESHOLD = 3.0
RPPG_MIN_FRAMES = 90

# DCT-specific thresholds
DCT_DOUBLE_QUANT_COMPRESSION_THRESHOLD = 0.70

# Geometry-specific thresholds
# Recalibrated from 0.15: MediaPipe jaw nodes 234/454 sit near ear tragus —
# wider denominator requires higher threshold to maintain equivalent angular sensitivity
GEOMETRY_YAW_SKIP_THRESHOLD = 0.18

# Illumination-specific thresholds
ILLUMINATION_DIFFUSE_THRESHOLD = 0.05

# Ensemble discounts
SBI_COMPRESSION_DISCOUNT = 0.40
FREQNET_COMPRESSION_DISCOUNT = 0.50

# Ensemble weights
WEIGHT_CLIP = 0.30
WEIGHT_SBI = 0.20
WEIGHT_FREQNET = 0.20
WEIGHT_RPPG = 0.15
WEIGHT_DCT = 0.10
WEIGHT_GEOMETRY = 0.03
WEIGHT_ILLUMINATION = 0.02
WEIGHT_CORNEAL = 0.03
