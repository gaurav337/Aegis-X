# Aegis-X

Aegis-X is an advanced forensic system utilizing a hybrid AI and physics ensemble to detect deepfakes, synthetic media, and manipulated content.

## Architecture Updates (Phase 1)

### Day 1: Scaffolding & Configuration
- Set up foundational configuration in `core/config.py` using dataclasses.
- Centralized detection thresholds and ensemble weights in `utils/thresholds.py`.
- Developed robust, thread-safe logging mechanism via `utils/logger.py`.
- Defined base exception structures for pipeline robustness in `core/exceptions.py`.

### Day 2: Tooling Interfaces
- Created unified interface for forensic tools `BaseForensicTool` (`core/base_tool.py`).
- Defined canonical interface and payload definitions like `ToolResult` (`core/data_types.py`).

### Day 3: Media I/O & Hardware Acceleration
- Developed robust media extraction capabilities inside `utils/video.py` and `utils/image.py`.
- Implemented **TorchCodec (NVDEC)** hardware decoding path for faster batch processing and fallback to OpenCV.
- Enforced RGB normalizations and safe memory batching to avoid OOM scenarios.

### Day 4: Face Landmark & Patching (MediaPipe)
- Migrated to **MediaPipe Face Mesh (478 points)** (`utils/preprocessing.py`) explicitly capturing iris and pupil nodes.
- Extracts precise, native-resolution anatomical patches (6 key diagnostic regions: left/right periorbital, left/right nasolabial folds, hairline band, chin/jaw contour) targeting 224x224 patch inputs via **Lanczos4** interpolation for frequency artifact preservation.
- Implemented **Dynamic Quality Snipe Filter** to hunt early video frames and re-evaluate subject bounds to extract the absolute sharpest (highest Laplacian variance) crop minimizing motion blur.

### Day 5: VRAM Lifecycle Management
- Developed `utils/vram_manager.py` with device auto-detection prioritizing **TPU (XLA)** -> **CUDA** -> **MPS** -> **CPU**.
- Implemented `VRAMLifecycleManager` context manager with a global acceleration lock and deterministic memory flushing.
- Enforced a strict **del + .to("cpu") + empty_cache() + gc.collect()** sequence to ensure models are purged from VRAM immediately after use, staying within the system's 600MB peak limit.

## Tooling Implementation (Phase 2)

### Day 6: C2PA Provenance Analysis
- Integrated `c2pa-python` to verify Content Credentials on media files (`core/tools/c2pa_tool.py`).
- Implemented robust error handling for missing metadata or libraries, defaulting to a neutral score (0.0 confidence) ensuring pipelines do not crash on unsigned media.
- Verifies cryptographic signatures and recovers precise timestamps/issuers from the JUMBF claim box.

### Day 7: Multi-ROI POS rPPG Liveness Detection
- Implemented `core/tools/rppg_tool.py` relying on a Plane Orthogonal to Skin-tone (POS) algorithm windowed over 90+ tracked face spatial frames.
- Re-anchors tracking via MediaPipe nodes to extract three distinct ROIs (Forehead, Left Cheek, Right Cheek) and integrates a gray-scale variance guardrail to instantly abort ambiguous scans with hair occlusion.
- Evaluates SNR, cardiac peak bands (0.7-2.5 Hz), and **Spectral Coherence** across the three ROIs to calculate composite signal confidence without exposing integer internal BPM readings.
- Flags flatlined signals (Score 1.0, NO_PULSE) lacking physiological variation perfectly with synthetic datasets, and flags incoherent localized noise (screen replay spoof).

#### rPPG Spectral Coherence Pipeline
```text
Input: Raw video frames + face tracking data
                    │
    ┌───────────────┴───────────────┐
    │   3 ROIs × POS Algorithm      │
    │   (Forehead, L_Cheek, R_Cheek)│
    └───────────────┬───────────────┘
                    │
    ┌───────────────┴───────────────┐
    │   4-Stage Decision Pipeline   │
    │                               │
    │   0. Flatline → FAKE          │
    │   1. Quality Gate → ABSTAIN   │
    │   2. Spectral Concentration   │
    │   3. Pairwise Coherence       │
    └───────────────┬───────────────┘
                    │
    Output: score (0.0=real, 1.0=fake)
            confidence (0.0–0.95)
            human-readable interpretation
```
*Compression-robust. No ML model needed. Pure signal processing.*

## Running Tests
Run the daily unit tests using `python tests_files/test_dayX.py`. Or run `pytest` (when integrated) at the repository root.
