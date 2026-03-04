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

## Running Tests
Run the daily unit tests using `python tests_files/test_dayX.py`. Or run `pytest` (when integrated) at the repository root.
