# 🛡️ Aegis-X — Complete Implementation Plan

> **Project:** Aegis-X: Agentic Multi-Modal Forensic Engine
> **Total Estimated Timeline:** 10 Weeks (50 Working Days)
> **Start Date:** March 2026
> **Tech Stack:** Python 3.11, PyTorch, llama-cpp-python, transformers, scipy, dlib, OpenCV, Streamlit, Gradio, FastAPI
> **Current State:** 2 of ~40+ files exist ([readme.md](file:///d:/Aegis-X/readme.md), [rppg.py](file:///d:/Aegis-X/rppg.py)). Everything else must be built from scratch.

---

## 📋 Project Audit — What Exists vs. What's Missing

| Category | Status | Details |
|:---------|:-------|:--------|
| [readme.md](file:///d:/Aegis-X/readme.md) | ✅ Complete | 1,792-line architecture blueprint |
| [rppg.py](file:///d:/Aegis-X/rppg.py) | ✅ Complete | POS algorithm, SNR, HR stability, [check_pulse()](file:///d:/Aegis-X/rppg.py#94-168) |
| `requirements.txt` | ❌ Missing | No dependency file |
| `config.yaml` | ❌ Missing | No configuration |
| `.env` | ❌ Missing | No environment variables |
| `main.py` | ❌ Missing | No CLI entry point |
| `app.py` | ❌ Missing | No Streamlit UI |
| `gradio_app.py` | ❌ Missing | No Gradio UI |
| `core/agent.py` | ❌ Missing | No agent loop |
| `core/llm.py` | ❌ Missing | No LLM controller |
| `core/memory.py` | ❌ Missing | No memory system |
| `core/tools/*` | ❌ Missing | No tool implementations (7 tools) |
| `core/prompts/*` | ❌ Missing | No prompt templates |
| `utils/*` | ❌ Missing | No preprocessing, video, audio, visualization |
| `scripts/*` | ❌ Missing | No model management scripts |
| `tests/*` | ❌ Missing | No tests |

---

## Phase Overview

| Phase | Title | Weeks | Days | Focus |
|:------|:------|:------|:-----|:------|
| **1** | Foundation & Infrastructure | Week 1–2 | Day 1–10 | Project setup, config, utilities, device management |
| **2** | Forensic Tool Suite | Week 3–4 | Day 11–20 | All 7 forensic tools + tool registry |
| **3** | Agent Core & LLM Brain | Week 5–6 | Day 21–30 | Agent loop, LLM integration, prompts, memory |
| **4** | CLI & Web Interfaces | Week 7–8 | Day 31–40 | main.py CLI, Streamlit app, Gradio app |
| **5** | Testing, Polish & Deploy | Week 9–10 | Day 41–50 | Unit tests, integration tests, Docker, docs |

---

## 📅 PHASE 1 — Foundation & Infrastructure (Week 1 to Week 2)

**Goal:** Set up the complete project skeleton, dependency management, configuration system, device detection, and all utility modules that every other component depends on.

**Dependencies:** None — this is the foundational phase.

---

### 🗓️ Week 1

#### ✅ Day 1 — Project Skeleton & Dependency Management

- **Objective:** Create the full directory structure, `requirements.txt`, `.env`, `config.yaml`, `.gitignore`, and verify the Python environment works.
- **Files to Create/Modify:**
  - `requirements.txt` — All Python dependencies with pinned versions
  - `.env` — Environment variable defaults
  - `config.yaml` — Full YAML configuration matching README spec
  - `.gitignore` — Ignore models/, logs/, __pycache__, .env, venv/
  - `core/__init__.py` — Core package init
  - `core/tools/__init__.py` — Tools package init
  - `core/prompts/__init__.py` — Prompts package init
  - `utils/__init__.py` — Utils package init
  - `scripts/__init__.py` — Scripts package init
  - `tests/__init__.py` — Tests package init
  - `memory/` — Create directory (empty, populated at runtime)
  - `logs/` — Create directory
- **AI Prompt to Execute This Day's Work:**

```text
Context: I am building Aegis-X, an agentic multi-modal deepfake forensic engine. The project currently has only `readme.md` and `rppg.py` at the root `d:\Aegis-X\`. I need to create the full project skeleton.

Task: Create all directories, config files, and package init files for the Aegis-X project.

Files to create (all under d:\Aegis-X\):

1. `requirements.txt` — Include these exact dependencies with version pins:
   - torch>=2.1.0
   - torchvision>=0.16.0
   - torchaudio>=2.1.0
   - numpy>=1.24.0,<2.0.0
   - scipy>=1.11.0
   - opencv-python>=4.8.0
   - Pillow>=10.0.0
   - dlib>=19.24.0
   - transformers>=4.36.0
   - timm>=0.9.12
   - openai-whisper>=20231117
   - c2pa-python>=0.4.0
   - llama-cpp-python>=0.2.50
   - pyyaml>=6.0
   - python-dotenv>=1.0.0
   - streamlit>=1.30.0
   - gradio>=4.10.0
   - matplotlib>=3.8.0
   - seaborn>=0.13.0
   - tqdm>=4.66.0
   - loguru>=0.7.0
   - click>=8.1.0
   - rich>=13.7.0
   - pytest>=7.4.0
   - pytest-cov>=4.1.0

2. `.env` — With these defaults:
   AEGIS_MODEL_DIR=./models
   AEGIS_MINICPM_PATH=./models/minicpm-v-2.6-Q4_K_M.gguf
   AEGIS_AIMV2_PATH=./models/aimv2-large
   AEGIS_DLIB_LANDMARKS=./models/shape_predictor_68_face_landmarks.dat
   AEGIS_DEVICE=auto
   AEGIS_CONFIDENCE_THRESHOLD=0.9
   AEGIS_MAX_ITERATIONS=10
   AEGIS_MEMORY_PATH=./memory/cases.json
   AEGIS_ENABLE_MEMORY=true
   AEGIS_LOG_LEVEL=INFO
   AEGIS_LOG_FILE=./logs/aegis.log

3. `config.yaml` — Full configuration with sections for: agent (confidence_threshold: 0.9, max_iterations: 10, enable_memory: true), models (minicpm path/context_length/temperature, aimv2 path/device, whisper model_size/language), tools (c2pa enabled, rppg min_frames/fps/bpm_range, reflection deviation_threshold, entropy anomaly_threshold, lipsync sync_threshold, dct enabled, artifacts enabled), output (format: json, include_heatmaps: true, include_reasoning_trace: true).

4. `.gitignore` — Ignore: models/, logs/*.log, memory/*.json, __pycache__/, *.pyc, .env, venv/, *.egg-info/, dist/, build/, .pytest_cache/

5. Create empty `__init__.py` files in: core/, core/tools/, core/prompts/, utils/, scripts/, tests/

6. Create empty directories: memory/, logs/, models/

Code Requirements:
- requirements.txt must have each package on its own line
- config.yaml must use proper YAML syntax with comments explaining each section
- .env must use KEY=value format
- All __init__.py files should have a module docstring

Output: A fully structured project directory that can be installed with `pip install -r requirements.txt`.

Testing:
- Run `python -c "import yaml; yaml.safe_load(open('config.yaml'))"` to verify config.yaml is valid
- Run `python -c "from dotenv import load_dotenv; load_dotenv()"` to verify .env
- Verify all directories exist with `dir /s /b *.py`
```

- **Verification:** Run `pip install -r requirements.txt` in the venv. Verify all directories exist. Parse config.yaml with Python.
- **Expected Output:** Full project directory tree with all packages, config files, and empty init modules.

---

#### ✅ Day 2 — Device Management & Logging System

- **Objective:** Build the device detection/management system (CUDA/ROCm/MPS/CPU auto-detection) and the structured logging system used by all components.
- **Files to Create/Modify:**
  - `utils/device.py` — Device detection, VRAM query, model loading strategy selection
  - `utils/logger.py` — Structured logging with loguru, file + console output, log levels
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X project at d:\Aegis-X\ has its skeleton set up with requirements.txt, config.yaml, .env, and all package directories with __init__.py files. The rppg.py file exists at the root.

Task: Create the device management module and logging system.

Files to create:

1. `utils/device.py` — Device detection and management module with these exact functions:

   class DeviceManager:
       """Manages GPU/CPU device selection and VRAM-aware model loading."""

       def __init__(self, preferred_device: str = "auto"):
           """Initialize with preferred device: 'auto', 'cuda', 'mps', 'cpu'."""

       def detect_device(self) -> str:
           """Auto-detect best available device. Priority: cuda > mps > cpu.
           Returns torch device string."""

       def get_vram_mb(self) -> int:
           """Return available VRAM in MB. Returns 0 for CPU-only."""

       def get_loading_strategy(self) -> str:
           """Based on VRAM, return 'sequential' (<4GB), 'hybrid' (4-12GB), or 'concurrent' (12+GB)."""

       def to_device(self, tensor_or_model):
           """Move a tensor or model to the selected device."""

       @property
       def device(self) -> torch.device:
           """Return the torch.device object."""

       @property
       def device_str(self) -> str:
           """Return device as string: 'cuda:0', 'mps', or 'cpu'."""

   Implementation details:
   - Use torch.cuda.is_available(), torch.backends.mps.is_available()
   - For VRAM: torch.cuda.get_device_properties(0).total_memory
   - Handle edge cases: no GPU, multiple GPUs (use first), ROCm (appears as CUDA in PyTorch)
   - Log device selection with loguru
   - Include a get_device_info() -> dict function returning full device diagnostics

2. `utils/logger.py` — Logging setup module:

   import sys
   from loguru import logger
   from pathlib import Path

   def setup_logger(log_level: str = "INFO", log_file: str = "./logs/aegis.log") -> None:
       """Configure loguru for Aegis-X.
       - Console output with colorized, human-readable format
       - File output with structured JSON format
       - Rotation: 10MB per file, keep 5 backups
       - Filter: suppress noisy third-party loggers (transformers, torch)
       """

   def get_logger(name: str):
       """Return a contextualized logger with the given module name."""

   Implementation:
   - Remove default loguru handler, add custom ones
   - Console format: "{time:HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}"
   - File format: JSON with timestamp, level, module, message, extra
   - Create the logs/ directory if it doesn't exist
   - Add a log_step(step_num, action, tool, result) convenience function for agent trace logging

Code Requirements:
- Use type hints everywhere
- Handle ImportError for torch gracefully (CPU-only fallback)
- Include docstrings for all public methods
- Both files should be importable independently

Output: Two utility modules that all other Aegis-X components will import.

Testing:
- Run: python -c "from utils.device import DeviceManager; dm = DeviceManager(); print(dm.device_str, dm.get_loading_strategy())"
- Run: python -c "from utils.logger import setup_logger, get_logger; setup_logger(); log = get_logger('test'); log.info('Aegis-X logger working')"
- Verify logs/aegis.log is created with the test message
```

- **Verification:** Import both modules. Device detection should print correct device. Logger should create log file.
- **Expected Output:** `utils/device.py` and `utils/logger.py` fully functional.

---

#### ✅ Day 3 — Configuration Loader & Model Manager

- **Objective:** Build the configuration loading system (YAML + .env + CLI overrides) and the model path manager that verifies models exist and provides download instructions.
- **Files to Create/Modify:**
  - `core/config.py` — Configuration loader with validation, merging, and defaults
  - `scripts/check_models.py` — Verify all required models are present
  - `scripts/download_models.py` — Download missing models from HuggingFace
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has config.yaml, .env, utils/device.py, utils/logger.py. The project needs a config loader that merges YAML config + .env + CLI args, and model management scripts.

Task: Create the configuration system and model management scripts.

Files to create:

1. `core/config.py`:

   from dataclasses import dataclass, field
   from typing import Optional, List
   from pathlib import Path

   @dataclass
   class ModelConfig:
       minicpm_path: str = "./models/minicpm-v-2.6-Q4_K_M.gguf"
       minicpm_context_length: int = 8192
       minicpm_temperature: float = 0.7
       aimv2_path: str = "./models/aimv2-large"
       aimv2_device: str = "auto"
       whisper_model_size: str = "small.en"
       whisper_language: str = "en"
       dlib_landmarks_path: str = "./models/shape_predictor_68_face_landmarks.dat"
       efficientnet_path: str = "./models/efficientnet_b4_faceforensics.pth"

   @dataclass
   class AgentConfig:
       confidence_threshold: float = 0.9
       max_iterations: int = 10
       enable_memory: bool = True
       memory_path: str = "./memory/cases.json"

   @dataclass
   class ToolConfig:
       c2pa_enabled: bool = True
       rppg_min_frames: int = 30
       rppg_fps: int = 30
       rppg_bpm_range: tuple = (50, 120)
       reflection_deviation_threshold: float = 15.0
       entropy_anomaly_threshold: float = 0.7
       lipsync_sync_threshold: float = 0.7
       dct_enabled: bool = True
       artifacts_enabled: bool = True

   @dataclass
   class OutputConfig:
       format: str = "json"
       include_heatmaps: bool = True
       include_reasoning_trace: bool = True

   @dataclass
   class AegisConfig:
       agent: AgentConfig = field(default_factory=AgentConfig)
       models: ModelConfig = field(default_factory=ModelConfig)
       tools: ToolConfig = field(default_factory=ToolConfig)
       output: OutputConfig = field(default_factory=OutputConfig)
       device: str = "auto"
       log_level: str = "INFO"
       log_file: str = "./logs/aegis.log"

   def load_config(config_path: str = "config.yaml", env_path: str = ".env", cli_overrides: dict = None) -> AegisConfig:
       """Load config from YAML, overlay .env values, then CLI overrides.
       Priority: CLI > .env > YAML > defaults."""

   def validate_config(config: AegisConfig) -> List[str]:
       """Validate config values. Return list of warning messages."""

2. `scripts/check_models.py`:
   - Check each model path from config exists
   - Print a rich table showing: Model Name | Expected Path | Status (✅/❌) | Size
   - Use rich library for pretty console output
   - Exit with code 1 if any required model is missing
   - Show download instructions for missing models

3. `scripts/download_models.py`:
   - Use huggingface_hub to download: MiniCPM-V 2.6 GGUF, AIMv2-Large, Whisper-small
   - Use urllib for dlib shape predictor (from dlib.net)
   - Show progress bars with tqdm
   - Skip models that already exist
   - Handle network errors gracefully
   - Accept --model flag to download specific model only

Code Requirements:
- load_config must handle missing config.yaml gracefully (use defaults)
- All paths should be resolved relative to project root
- Use python-dotenv for .env loading
- Include comprehensive error messages

Output: Config system and model scripts.

Testing:
- python -c "from core.config import load_config; c = load_config(); print(c.agent.confidence_threshold)"
- python scripts/check_models.py (should show all models as missing with ❌)
- python scripts/download_models.py --help (should show usage)
```

- **Verification:** Config loads without error. `check_models.py` runs and shows missing models. `download_models.py --help` shows options.
- **Expected Output:** Configuration system ready, model management scripts working.

---

#### ✅ Day 4 — Video & Audio Preprocessing Utilities

- **Objective:** Build video frame extraction, face detection/alignment, and audio extraction utilities that all forensic tools depend on.
- **Files to Create/Modify:**
  - `utils/video.py` — Video loading, frame extraction, FPS detection
  - `utils/audio.py` — Audio track extraction from video files
  - `utils/preprocessing.py` — Face detection with dlib, face alignment, ROI cropping
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has config system (core/config.py), device manager (utils/device.py), logger (utils/logger.py). Now I need the preprocessing pipeline that extracts frames, detects faces, and extracts audio.

Task: Create the three preprocessing utility modules.

Files to create:

1. `utils/video.py`:

   import cv2
   import numpy as np
   from pathlib import Path
   from typing import Tuple, Optional, Generator

   class VideoLoader:
       """Load and extract frames from video files."""

       def __init__(self, video_path: str):
           """Open video file, read metadata (fps, frame_count, resolution, duration)."""

       @property
       def fps(self) -> float: ...
       @property
       def frame_count(self) -> int: ...
       @property
       def resolution(self) -> Tuple[int, int]: ...
       @property
       def duration_seconds(self) -> float: ...

       def extract_frames(self, max_frames: int = 300, sample_rate: int = 1) -> np.ndarray:
           """Extract frames as numpy array (N, H, W, 3) in RGB.
           max_frames: cap total frames extracted.
           sample_rate: take every Nth frame."""

       def stream_frames(self, batch_size: int = 30) -> Generator[np.ndarray, None, None]:
           """Yield batches of frames for memory-efficient processing."""

       def extract_frame_at(self, timestamp_sec: float) -> np.ndarray:
           """Extract single frame at given timestamp."""

       def get_metadata(self) -> dict:
           """Return dict with fps, frame_count, resolution, duration, codec."""

       def close(self): ...
       def __enter__(self): ...
       def __exit__(self, *args): ...

   def is_video_file(path: str) -> bool:
       """Check if file is a supported video format (.mp4, .avi, .mov, .mkv, .webm)."""

   def is_image_file(path: str) -> bool:
       """Check if file is a supported image format (.jpg, .jpeg, .png, .bmp, .webp)."""

   def load_image(path: str) -> np.ndarray:
       """Load image as RGB numpy array."""

2. `utils/audio.py`:

   import subprocess
   import tempfile
   from pathlib import Path

   def extract_audio(video_path: str, output_path: str = None, sample_rate: int = 16000) -> str:
       """Extract audio track from video using ffmpeg.
       Returns path to extracted WAV file.
       If output_path is None, use tempfile."""

   def has_audio_track(video_path: str) -> bool:
       """Check if video file contains an audio stream using ffprobe."""

   def get_audio_duration(audio_path: str) -> float:
       """Get duration of audio file in seconds."""

   Implementation: Use subprocess to call ffmpeg. Handle ffmpeg not found with clear error.

3. `utils/preprocessing.py`:

   import cv2
   import numpy as np
   import dlib
   from pathlib import Path
   from typing import List, Tuple, Optional

   class FaceDetector:
       """Detect and align faces using dlib."""

       def __init__(self, landmarks_path: str = None):
           """Load dlib face detector and shape predictor.
           landmarks_path: path to shape_predictor_68_face_landmarks.dat"""
           # Use dlib.get_frontal_face_detector() for detection
           # Use dlib.shape_predictor(landmarks_path) for landmarks

       def detect_faces(self, frame: np.ndarray) -> List[dlib.rectangle]:
           """Detect all faces in a frame. Returns list of bounding boxes."""

       def get_landmarks(self, frame: np.ndarray, face_rect: dlib.rectangle) -> np.ndarray:
           """Get 68 facial landmarks as (68, 2) numpy array."""

       def crop_face(self, frame: np.ndarray, face_rect: dlib.rectangle, margin: float = 0.3) -> np.ndarray:
           """Crop face region with margin. Returns RGB numpy array."""

       def align_face(self, frame: np.ndarray, landmarks: np.ndarray, output_size: int = 224) -> np.ndarray:
           """Align face using eye landmarks. Returns aligned face crop."""

       def get_forehead_roi(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
           """Extract forehead region (landmarks 19-24) for rPPG analysis."""

       def get_eye_regions(self, frame: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
           """Extract left and right eye regions for reflection analysis."""

       def extract_face_sequence(self, frames: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
           """Process a sequence of frames: detect face in first frame, track through sequence.
           Returns: (face_crops array, landmarks list)"""

   Implementation:
   - If dlib model not found, raise FileNotFoundError with download instructions
   - Handle no face detected (return empty list)
   - Face alignment: use affine transform based on eye centers
   - Forehead ROI: use convex hull of landmarks 17-26
   - All outputs in RGB format

Code Requirements:
- All functions must handle edge cases (no face, corrupt video, etc.)
- Use loguru for logging within each module
- Type hints on all functions
- VideoLoader must be usable as context manager

Output: Three utility modules used by all forensic tools.

Testing:
- python -c "from utils.video import VideoLoader, is_video_file; print(is_video_file('test.mp4'))"
- python -c "from utils.audio import has_audio_track; print('audio module loaded')"
- python -c "from utils.preprocessing import FaceDetector; print('preprocessing module loaded')"
  (FaceDetector init will fail without dlib model - that's expected)
```

- **Verification:** All three modules import without errors. `is_video_file` and `is_image_file` return correct booleans.
- **Expected Output:** Complete preprocessing pipeline ready for forensic tools.

---

#### ✅ Day 5 — Visualization Utilities & Data Models

- **Objective:** Build the visualization module (heatmaps, annotated frames, report charts) and the core data models (ToolResult, AgentState, ForensicReport) that define the data flow.
- **Files to Create/Modify:**
  - `utils/visualization.py` — Heatmap overlays, annotated frames, report generation
  - `core/models.py` — Pydantic/dataclass models for agent state, tool results, verdicts
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has config, device, logger, video, audio, preprocessing utilities. Now I need data models that define the types flowing through the agent, and visualization tools for reports.

Task: Create the data model definitions and visualization module.

Files to create:

1. `core/models.py` — Data models for the entire agent system:

   from dataclasses import dataclass, field
   from typing import List, Dict, Optional, Any
   from enum import Enum
   from datetime import datetime

   class Verdict(Enum):
       REAL = "REAL"
       FAKE = "FAKE"
       INCONCLUSIVE = "INCONCLUSIVE"

   class ToolName(Enum):
       CHECK_C2PA = "check_c2pa"
       RUN_RPPG = "run_rppg"
       RUN_REFLECTION = "run_reflection"
       RUN_ENTROPY = "run_entropy"
       RUN_DCT = "run_dct"
       RUN_LIPSYNC = "run_lipsync"
       RUN_ARTIFACTS = "run_artifacts"
       GENERATE_REPORT = "generate_report"
       ESCALATE_TO_HUMAN = "escalate_to_human"

   @dataclass
   class ToolResult:
       tool_name: str
       output: Dict[str, Any]
       confidence_delta: float  # how much this changed agent confidence
       execution_time_ms: float
       success: bool
       error: Optional[str] = None

   @dataclass
   class AgentStep:
       step_number: int
       reasoning: str          # LLM's explanation for choosing this tool
       selected_tool: str
       tool_result: Optional[ToolResult] = None
       confidence_before: float = 0.5
       confidence_after: float = 0.5
       timestamp: str = ""

   @dataclass
   class AgentState:
       media_path: str
       media_type: str         # "video" or "image"
       current_confidence: float = 0.5
       verdict: Optional[Verdict] = None
       steps: List[AgentStep] = field(default_factory=list)
       tool_results: List[ToolResult] = field(default_factory=list)
       tools_used: List[str] = field(default_factory=list)
       tools_skipped: List[str] = field(default_factory=list)
       face_detected: bool = False
       has_audio: bool = False
       metadata: Dict[str, Any] = field(default_factory=dict)
       start_time: str = ""
       end_time: str = ""

   @dataclass
   class ForensicReport:
       verdict: Verdict
       confidence: float
       reasoning: str
       key_evidence: List[str]
       tool_results: List[ToolResult]
       decision_trace: List[AgentStep]
       tools_used: List[str]
       tools_skipped: List[str]
       total_time_ms: float
       media_path: str
       media_type: str
       timestamp: str
       metadata: Dict[str, Any] = field(default_factory=dict)

       def to_dict(self) -> dict: ...
       def to_json(self, path: str = None) -> str: ...

   @dataclass
   class AnalysisConfig:
       confidence_threshold: float = 0.9
       max_iterations: int = 10
       enable_memory: bool = True
       device: str = "auto"
       skip_tools: List[str] = field(default_factory=list)

2. `utils/visualization.py`:

   import numpy as np
   import cv2
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt

   def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
       """Overlay a heatmap on a frame. Returns annotated RGB frame."""

   def annotate_face_landmarks(frame: np.ndarray, landmarks: np.ndarray, color: tuple = (0, 255, 0)) -> np.ndarray:
       """Draw 68 facial landmarks on frame."""

   def draw_verdict_overlay(frame: np.ndarray, verdict: str, confidence: float) -> np.ndarray:
       """Draw verdict badge (REAL/FAKE/INCONCLUSIVE) with confidence bar on frame."""

   def plot_rppg_signal(bvp_signal: np.ndarray, fs: int = 30, save_path: str = None) -> Optional[np.ndarray]:
       """Plot the BVP signal as a waveform chart. Returns image as numpy array or saves to file."""

   def plot_entropy_heatmap(entropy_map: np.ndarray, save_path: str = None) -> Optional[np.ndarray]:
       """Plot the patch entropy map. Returns image as numpy array or saves to file."""

   def plot_dct_spectrum(dct_data: np.ndarray, save_path: str = None) -> Optional[np.ndarray]:
       """Plot DCT frequency spectrum. Returns image as numpy array or saves to file."""

   def create_report_summary_image(report: dict, save_path: str = None) -> np.ndarray:
       """Create a visual summary card with verdict, confidence, key metrics."""

   Implementation:
   - Use matplotlib with Agg backend (no GUI required)
   - All visualizations return numpy arrays (RGB) unless save_path given
   - Use professional color scheme: green for REAL, red for FAKE, amber for INCONCLUSIVE
   - Handle edge cases (empty data, None values)

Code Requirements:
- core/models.py must be import-ready with no external dependencies beyond stdlib
- visualization.py must use Agg backend for headless rendering
- All to_dict/to_json methods must handle Enum serialization

Testing:
- python -c "from core.models import AgentState, Verdict; s = AgentState('test.mp4', 'video'); print(s.current_confidence)"
- python -c "from utils.visualization import overlay_heatmap; print('visualization loaded')"
```

- **Verification:** Import both modules. Create an `AgentState` and `ForensicReport` instance. Call `to_json()`.
- **Expected Output:** All data models and visualization tools defined. These are the "types" used everywhere.

---

### 🗓️ Week 2

#### ✅ Day 6 — Base Tool Class & Tool Registry

- **Objective:** Create the abstract base class for all forensic tools and the registry that the agent uses to discover and invoke tools.
- **Files to Create/Modify:**
  - `core/tools/base.py` — Abstract `ForensicTool` base class
  - `core/tools/registry.py` — `ToolRegistry` class for tool discovery and invocation
  - `core/tools/__init__.py` — Update with public exports
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has config, device, logger, video/audio/preprocessing utils, and data models (core/models.py with ToolResult, AgentState, ForensicReport, ToolName). Now I need the base tool class and registry.

Task: Create the forensic tool base class and registry system.

Files to create:

1. `core/tools/base.py`:

   from abc import ABC, abstractmethod
   from typing import Dict, Any, Optional
   from core.models import ToolResult
   import time

   class ForensicTool(ABC):
       """Abstract base class for all Aegis-X forensic tools."""

       name: str = ""               # e.g., "check_c2pa"
       description: str = ""         # Human-readable description
       compute_type: str = "cpu"     # "cpu" or "gpu"
       cost: float = 1.0             # Relative compute cost (1-5)
       requires_face: bool = False
       requires_audio: bool = False
       requires_video: bool = False  # True = needs multiple frames

       @abstractmethod
       def execute(self, **kwargs) -> Dict[str, Any]:
           """Run the forensic analysis. Returns raw output dict."""

       def run(self, **kwargs) -> ToolResult:
           """Wrapper that handles timing, error catching, and ToolResult creation."""
           start = time.time()
           try:
               output = self.execute(**kwargs)
               elapsed = (time.time() - start) * 1000
               return ToolResult(
                   tool_name=self.name,
                   output=output,
                   confidence_delta=output.get("confidence_delta", 0.0),
                   execution_time_ms=round(elapsed, 2),
                   success=True
               )
           except Exception as e:
               elapsed = (time.time() - start) * 1000
               return ToolResult(
                   tool_name=self.name,
                   output={},
                   confidence_delta=0.0,
                   execution_time_ms=round(elapsed, 2),
                   success=False,
                   error=str(e)
               )

       @abstractmethod
       def is_available(self) -> bool:
           """Check if required models/dependencies are available."""

       def get_description_for_llm(self) -> str:
           """Return a description string for the LLM planner to understand this tool."""
           return f"{self.name}: {self.description} (compute: {self.compute_type}, cost: {self.cost})"

2. `core/tools/registry.py`:

   from typing import Dict, List, Optional
   from core.tools.base import ForensicTool
   from core.models import ToolResult

   class ToolRegistry:
       """Registry of all available forensic tools."""

       def __init__(self):
           self._tools: Dict[str, ForensicTool] = {}

       def register(self, tool: ForensicTool) -> None:
           """Register a forensic tool."""

       def get(self, name: str) -> Optional[ForensicTool]:
           """Get tool by name."""

       def list_tools(self) -> List[str]:
           """List all registered tool names."""

       def list_available_tools(self) -> List[str]:
           """List only tools whose dependencies are met."""

       def get_tool_descriptions(self) -> str:
           """Get all tool descriptions formatted for the LLM prompt."""

       def execute_tool(self, name: str, **kwargs) -> ToolResult:
           """Execute a tool by name with given kwargs."""

       def get_tools_by_compute(self, compute_type: str) -> List[ForensicTool]:
           """Get tools filtered by compute type ('cpu' or 'gpu')."""

       def get_tools_for_context(self, has_face: bool, has_audio: bool, is_video: bool) -> List[ForensicTool]:
           """Return only tools applicable to current media context."""

3. Update `core/tools/__init__.py`:
   from core.tools.base import ForensicTool
   from core.tools.registry import ToolRegistry
   __all__ = ["ForensicTool", "ToolRegistry"]

Code Requirements:
- ForensicTool.run() must catch ALL exceptions and return a failed ToolResult
- Registry must prevent duplicate registrations
- get_tool_descriptions() output is used directly in LLM prompts
- Include logging for tool registration and execution

Testing:
- python -c "from core.tools import ForensicTool, ToolRegistry; r = ToolRegistry(); print(r.list_tools())"
```

- **Verification:** Import and instantiate `ToolRegistry`. Verify empty registry returns empty list.
- **Expected Output:** Tool abstraction layer ready. All 7 forensic tools will subclass `ForensicTool`.

---

#### ✅ Day 7 — C2PA Provenance Tool & rPPG Tool

- **Objective:** Implement the first two forensic tools: C2PA content credential verification and rPPG pulse detection (wrap existing rppg.py).
- **Files to Create/Modify:**
  - `core/tools/c2pa_tool.py` — C2PA content credential verification
  - `core/tools/rppg_tool.py` — Wrap rppg.py into ForensicTool interface
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has the tool base class (core/tools/base.py) and registry (core/tools/registry.py). The existing rppg.py at the project root contains the complete POS algorithm with extract_pos_signal(), compute_snr(), compute_hr_stability(), and check_pulse() functions.

Task: Create two forensic tool implementations.

Files to create:

1. `core/tools/c2pa_tool.py`:

   from core.tools.base import ForensicTool

   class C2PATool(ForensicTool):
       name = "check_c2pa"
       description = "Verify C2PA Content Credentials (cryptographic provenance). Checks if media was signed by a trusted camera or editing software."
       compute_type = "cpu"
       cost = 0.5  # Very cheap — just crypto verification
       requires_face = False
       requires_audio = False
       requires_video = False

       def execute(self, file_path: str, **kwargs) -> dict:
           """Check C2PA signature.
           Returns: {valid: bool, signer: str|None, timestamp: str|None, trust_chain: list, confidence_delta: float}
           - If valid C2PA signature found: confidence_delta = +0.4 (strong evidence of authenticity)
           - If no signature: confidence_delta = 0.0 (absence of proof ≠ proof of absence)
           - If invalid/tampered signature: confidence_delta = -0.3 (suspicious)
           """
           # Use c2pa-python library: import c2pa
           # Try c2pa.Reader(file_path)
           # Handle: no manifest, valid manifest, invalid manifest
           # Handle c2pa not installed gracefully

       def is_available(self) -> bool:
           try:
               import c2pa
               return True
           except ImportError:
               return False

2. `core/tools/rppg_tool.py`:

   from core.tools.base import ForensicTool
   import numpy as np

   class RPPGTool(ForensicTool):
       name = "run_rppg"
       description = "Extract biological pulse (rPPG) from facial video. Detects real heartbeat signal to verify physical presence of a living person."
       compute_type = "cpu"
       cost = 2.0  # Moderate — signal processing
       requires_face = True
       requires_audio = False
       requires_video = True  # Needs temporal sequence

       def execute(self, frames: np.ndarray, fps: int = 30, **kwargs) -> dict:
           """Run rPPG analysis on face video frames.
           frames: (N, H, W, 3) numpy array of CROPPED FACE region
           fps: video frame rate
           Returns: check_pulse() output + confidence_delta
           - PULSE_PRESENT: confidence_delta = +0.3
           - NO_PULSE: confidence_delta = -0.3
           - AMBIGUOUS: confidence_delta = -0.05
           - INSUFFICIENT_DATA: confidence_delta = 0.0
           """
           # Import from root rppg.py: from rppg import check_pulse
           # Call check_pulse(frames, fps)
           # Map verdict to confidence_delta
           # Add confidence_delta to the returned dict

       def is_available(self) -> bool:
           try:
               from rppg import check_pulse
               return True
           except ImportError:
               return False

Code Requirements:
- C2PA tool must handle: c2pa library not installed, file not found, no manifest, valid manifest, invalid manifest
- rPPG tool must handle: too few frames, frame shape errors, all-black frames
- Both tools must return confidence_delta in their output dict
- Import rppg.py from project root (add sys.path if needed or use relative import)
- Include detailed logging of tool execution

Testing:
- python -c "from core.tools.c2pa_tool import C2PATool; t = C2PATool(); print(t.name, t.is_available())"
- python -c "from core.tools.rppg_tool import RPPGTool; t = RPPGTool(); print(t.name, t.is_available())"
- python -c "from core.tools.rppg_tool import RPPGTool; import numpy as np; t = RPPGTool(); r = t.run(frames=np.random.randint(0,255,(100,64,64,3),dtype=np.uint8)); print(r.output)"
```

- **Verification:** Both tools load. rPPG tool can process random frames and return a ToolResult. C2PA reports availability.
- **Expected Output:** First two tools in the registry, both functional.

---

#### ✅ Day 8 — Entropy Tool & Artifact Detection Tool

- **Objective:** Implement the AIMv2 entropy analysis tool and the EfficientNet-B4 spatial artifact detection tool.
- **Files to Create/Modify:**
  - `core/tools/entropy_tool.py` — AIMv2-based patch entropy analysis
  - `core/tools/artifacts_tool.py` — EfficientNet-B4 deepfake artifact detection
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has the tool framework (base.py, registry.py) and two implemented tools (c2pa_tool.py, rppg_tool.py). The data models are in core/models.py. Device manager is in utils/device.py. Config is in core/config.py.

Task: Create the entropy analysis and artifact detection tools.

Files to create:

1. `core/tools/entropy_tool.py`:

   from core.tools.base import ForensicTool
   import numpy as np

   class EntropyTool(ForensicTool):
       name = "run_entropy"
       description = "Analyze patch-level prediction entropy using AIMv2. Detects generative artifacts through statistical anomalies in image patches. Real faces have uniform entropy; generated faces show high-entropy patches at boundaries."
       compute_type = "gpu"
       cost = 3.5
       requires_face = True
       requires_audio = False
       requires_video = False

       def __init__(self, model_path: str = "./models/aimv2-large", device: str = "auto"):
           self.model_path = model_path
           self.device_str = device
           self._model = None
           self._processor = None

       def _load_model(self):
           """Lazy-load AIMv2-Large model and processor."""
           # from transformers import AutoModel, AutoImageProcessor
           # self._processor = AutoImageProcessor.from_pretrained(self.model_path)
           # self._model = AutoModel.from_pretrained(self.model_path).to(device)

       def execute(self, face_crop: np.ndarray, **kwargs) -> dict:
           """Compute per-patch prediction entropy.
           face_crop: (H, W, 3) RGB numpy array of aligned face
           Returns: {anomaly_score: float, heatmap: np.ndarray, hotspot_regions: list, confidence_delta: float}
           - anomaly_score > 0.7: confidence_delta = -0.25 (evidence of fake)
           - anomaly_score < 0.3: confidence_delta = +0.15 (evidence of real)
           - else: confidence_delta = -0.05
           """
           # Lazy load model
           # Preprocess face_crop using processor
           # Run model forward pass with output_hidden_states=True
           # Extract patch features from last hidden state (skip CLS token)
           # Compute per-patch entropy: -(p * log(p)).sum(dim=-1)
           # Reshape to spatial grid (16x16 for 224x224 / 14 patch)
           # Compute anomaly_score: fraction of patches exceeding mean + 2*std
           # Identify hotspot regions (top-left, top-right, etc.)
           # Map anomaly_score to confidence_delta

       def is_available(self) -> bool:
           from pathlib import Path
           try:
               import transformers
               return Path(self.model_path).exists()
           except ImportError:
               return False

2. `core/tools/artifacts_tool.py`:

   from core.tools.base import ForensicTool
   import numpy as np

   class ArtifactsTool(ForensicTool):
       name = "run_artifacts"
       description = "Detect spatial deepfake artifacts using EfficientNet-B4 trained on FaceForensics++. Identifies GAN fingerprints, blending boundaries, and texture anomalies."
       compute_type = "gpu"
       cost = 3.0
       requires_face = True
       requires_audio = False
       requires_video = False

       def __init__(self, model_path: str = "./models/efficientnet_b4_faceforensics.pth", device: str = "auto"):
           self.model_path = model_path
           self.device_str = device
           self._model = None

       def _load_model(self):
           """Lazy-load EfficientNet-B4 model.
           If fine-tuned weights exist at model_path, load them.
           Otherwise, load pretrained timm model as feature extractor."""
           # import timm
           # self._model = timm.create_model('tf_efficientnet_b4', pretrained=True)
           # If model_path exists: load state dict with custom head
           # Else: use pretrained with threshold-based anomaly detection

       def execute(self, face_crop: np.ndarray, **kwargs) -> dict:
           """Detect spatial artifacts in face crop.
           face_crop: (H, W, 3) RGB numpy array
           Returns: {artifact_score: float, artifact_regions: list, confidence_delta: float}
           - artifact_score > 0.8: confidence_delta = -0.3 (strong fake evidence)
           - artifact_score < 0.3: confidence_delta = +0.1 (evidence of real)
           - else: confidence_delta = -0.1
           """
           # Lazy load model
           # Resize face to 380x380 (EfficientNet-B4 input)
           # Normalize with ImageNet stats
           # Run inference, get prediction
           # If using feature extractor: compute anomaly score from feature statistics
           # Map to confidence_delta

       def is_available(self) -> bool:
           try:
               import timm
               return True
           except ImportError:
               return False

Code Requirements:
- Both tools MUST use lazy loading (_load_model called on first execute)
- After execution, model stays loaded (cache for hybrid/concurrent strategy)
- Add unload_model() method to both for sequential strategy
- Device selection: use DeviceManager if device="auto"
- Handle model not downloaded gracefully (raise clear error with download instructions)
- Use torch.no_grad() for all inference

Testing:
- python -c "from core.tools.entropy_tool import EntropyTool; t = EntropyTool(); print(t.name, t.is_available())"
- python -c "from core.tools.artifacts_tool import ArtifactsTool; t = ArtifactsTool(); print(t.name, t.is_available())"
```

- **Verification:** Both tools import. `is_available()` returns correct status based on model presence.
- **Expected Output:** Two GPU-based forensic tools ready.

---

#### ✅ Day 9 — DCT Tool, Reflection Tool & Lip-sync Tool

- **Objective:** Implement the remaining three forensic tools: DCT frequency analysis, corneal reflection analysis, and audio-visual lip-sync verification.
- **Files to Create/Modify:**
  - `core/tools/dct_tool.py` — DCT frequency domain forensics
  - `core/tools/reflection_tool.py` — Corneal glint consistency analysis
  - `core/tools/lipsync_tool.py` — Whisper + dlib lip-sync verification
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has 4 forensic tools implemented (c2pa, rppg, entropy, artifacts). Now I need the final 3 tools: DCT, reflection, and lipsync.

Task: Create the remaining three forensic tool implementations.

Files to create:

1. `core/tools/dct_tool.py`:
   - name = "run_dct"
   - description = "Analyze DCT frequency spectrum for compression artifacts. Detects double quantization and grid artifacts that survive social media re-compression."
   - compute_type = "cpu", cost = 1.5
   - execute(image: np.ndarray) -> dict with {grid_artifacts: bool, score: float, double_quantization: bool, confidence_delta: float}
   - Uses scipy.fft.dctn on 8x8 blocks
   - Implements the analyze_dct_artifacts logic from the README
   - score > 0.7: confidence_delta = -0.2 (fake evidence)
   - score < 0.3: confidence_delta = +0.1 (real evidence)

2. `core/tools/reflection_tool.py`:
   - name = "run_reflection"
   - description = "Analyze corneal reflections for environmental consistency. Real eyes reflect the same environment; fakes often have inconsistent or missing reflections."
   - compute_type = "cpu", cost = 2.0, requires_face = True
   - execute(frame: np.ndarray, landmarks: np.ndarray) -> dict with {deviation_angle: float, left_reflection: dict, right_reflection: dict, consistent: bool, confidence_delta: float}
   - Extract eye regions using landmarks 36-41 (left) and 42-47 (right)
   - Detect bright spots (reflections) in each eye using threshold + contour detection
   - Compare reflection positions between eyes
   - deviation_angle > 15°: confidence_delta = -0.15 (suspicious)
   - deviation_angle < 5°: confidence_delta = +0.1 (consistent)

3. `core/tools/lipsync_tool.py`:
   - name = "run_lipsync"
   - description = "Verify audio-visual lip synchronization using Whisper speech recognition and dlib mouth landmarks. Detects dubbed or AI-generated speech overlaid on existing video."
   - compute_type = "gpu", cost = 4.0, requires_face = True, requires_audio = True, requires_video = True
   - execute(frames: np.ndarray, audio_path: str, fps: int, landmarks_sequence: list) -> dict with {sync_score: float, phoneme_matches: list, confidence_delta: float}
   - Use Whisper to transcribe audio with word-level timestamps
   - Extract mouth openness from landmarks (48-67) per frame
   - Correlate mouth movement with speech activity windows
   - sync_score > 0.7: confidence_delta = +0.1 (lip sync matches)
   - sync_score < 0.4: confidence_delta = -0.2 (lip sync mismatch)

Code Requirements for all three:
- Subclass ForensicTool
- Lazy model loading where applicable (lipsync needs Whisper)
- All return confidence_delta
- Handle edge cases (no eyes visible, no audio, etc.)
- Include unload_model() for memory management

Testing:
- python -c "from core.tools.dct_tool import DCTTool; t = DCTTool(); print(t.name)"
- python -c "from core.tools.reflection_tool import ReflectionTool; t = ReflectionTool(); print(t.name)"
- python -c "from core.tools.lipsync_tool import LipsyncTool; t = LipsyncTool(); print(t.name)"
```

- **Verification:** All three tools import. DCT tool can process a random grayscale image.
- **Expected Output:** All 7 forensic tools implemented. The Forensic Tool Suite is complete.

---

#### ✅ Day 10 — Tool Integration Test & Phase 1 Wrap-up

- **Objective:** Register all 7 tools in the registry, test each individually with synthetic data, write the first integration test, and verify the entire Phase 1 infrastructure.
- **Files to Create/Modify:**
  - `core/tools/__init__.py` — Update with all tool imports and a `create_default_registry()` factory
  - `tests/test_tools.py` — Unit tests for each tool
  - `tests/conftest.py` — Shared pytest fixtures
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has ALL infrastructure complete: config, device, logger, video/audio/preprocessing utils, data models, visualization, and 7 forensic tools (c2pa, rppg, entropy, artifacts, dct, reflection, lipsync). All tools subclass ForensicTool and are registered via ToolRegistry.

Task: Wire everything together and write tests.

Files to create/modify:

1. Update `core/tools/__init__.py`:
   - Import all 7 tools
   - Create function: def create_default_registry(config: AegisConfig) -> ToolRegistry
   - This function instantiates all tools with paths from config and registers them
   - Export: ForensicTool, ToolRegistry, create_default_registry, and all tool classes

2. `tests/conftest.py`:
   - Create fixtures: sample_video_frames (100 random RGB frames 64x64), sample_image (random 224x224 RGB), sample_grayscale (random 256x256 gray), mock_landmarks (68 random points), temp_config (AegisConfig with defaults)

3. `tests/test_tools.py`:
   - test_c2pa_no_file: C2PA with nonexistent file returns failed result
   - test_rppg_random_frames: rPPG with random data returns valid dict structure
   - test_rppg_insufficient_frames: <30 frames returns INSUFFICIENT_DATA
   - test_dct_random_image: DCT with random grayscale returns score
   - test_entropy_tool_loads: EntropyTool can be instantiated
   - test_artifacts_tool_loads: ArtifactsTool can be instantiated
   - test_reflection_tool_loads: ReflectionTool can be instantiated
   - test_lipsync_tool_loads: LipsyncTool can be instantiated
   - test_registry_all_tools: create_default_registry returns registry with 7+ tools
   - test_tool_result_structure: Every tool returns ToolResult with required fields

Testing:
- pytest tests/test_tools.py -v
- python -c "from core.tools import create_default_registry; from core.config import load_config; r = create_default_registry(load_config()); print(r.list_tools())"
```

- **Verification:** `pytest tests/test_tools.py -v` passes. Registry lists all 7 tools. Each tool produces valid ToolResult.
- **Expected Output:** Phase 1 complete. All infrastructure, utilities, and forensic tools ready for the agent core.

---

# 🛡️ Aegis-X — Implementation Plan — Part 2

> Continuation from Part 1. Covers **Phase 2** (Days 11–20) and **Phase 3** (Days 21–30).

---

## 📅 PHASE 2 — Agent Core & LLM Brain (Week 3 to Week 4)

**Goal:** Build the LLM controller (MiniCPM-V integration), the ReAct reasoning loop, prompt templates, memory/experience system, and the agent synthesis pipeline that ties everything together.

**Dependencies:** Phase 1 complete (all tools, registry, config, utils, data models).

---

### 🗓️ Week 3

#### ✅ Day 11 — LLM Controller Interface

- **Objective:** Build the LLM interface that loads MiniCPM-V 2.6 via llama-cpp-python and provides structured reasoning capabilities.
- **Files to Create/Modify:**
  - `core/llm.py` — LLM controller class with prompt execution and structured output parsing
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has the complete tool suite (7 tools), registry, config (core/config.py with ModelConfig.minicpm_path), device manager (utils/device.py), and data models (core/models.py). The LLM brain is MiniCPM-V 2.6 GGUF, loaded via llama-cpp-python.

Task: Create the LLM controller that loads and queries MiniCPM-V.

File to create: `core/llm.py`

   from typing import Dict, Any, Optional, List
   from pathlib import Path

   class LLMController:
       """Interface to MiniCPM-V 2.6 for agent reasoning."""

       def __init__(self, model_path: str, context_length: int = 8192,
                    temperature: float = 0.7, device: str = "auto"):
           """Load MiniCPM-V GGUF model using llama-cpp-python.
           - Use llama_cpp.Llama with n_ctx=context_length
           - Set n_gpu_layers=-1 for full GPU offload, 0 for CPU
           - Handle model not found with clear error + download instructions
           """
           self.model_path = model_path
           self._model = None
           self.context_length = context_length
           self.temperature = temperature

       def _load_model(self):
           """Lazy-load the GGUF model."""
           # from llama_cpp import Llama
           # self._model = Llama(
           #     model_path=self.model_path,
           #     n_ctx=self.context_length,
           #     n_gpu_layers=-1,  # offload all to GPU
           #     verbose=False,
           # )

       def query(self, prompt: str, max_tokens: int = 1024,
                 temperature: float = None) -> str:
           """Send a text prompt and get a text response."""

       def query_structured(self, prompt: str, schema: dict = None) -> Dict[str, Any]:
           """Send a prompt expecting JSON output. Parse and return dict.
           If parsing fails, retry once with a fix-JSON prompt.
           schema: optional JSON schema for guided generation."""

       def plan_next_tool(self, agent_state_summary: str,
                          available_tools: str) -> Dict[str, Any]:
           """Ask the LLM to select the next forensic tool.
           Returns: {tool: str, reasoning: str}
           Uses the planning prompt template."""

       def synthesize_verdict(self, tool_results_summary: str) -> Dict[str, Any]:
           """Ask the LLM to generate final verdict from all evidence.
           Returns: {verdict: str, confidence: float, reasoning: str, key_evidence: list}
           Uses the synthesis prompt template."""

       def interpret_result(self, tool_name: str, tool_output: str) -> str:
           """Ask the LLM to interpret a tool's output in forensic context.
           Returns natural language interpretation."""

       def unload(self):
           """Free model from memory."""
           del self._model
           self._model = None

       def is_loaded(self) -> bool:
           return self._model is not None

   Code Requirements:
   - Lazy loading: model loaded on first query() call
   - All query methods must handle: model not loaded, generation timeout, JSON parse errors
   - query_structured must attempt to extract JSON from response even if mixed with text
   - Use regex to find JSON in response: re.search(r'\{.*\}', response, re.DOTALL)
   - Log every query and response (truncated to 200 chars)
   - Temperature override per-call
   - If llama-cpp-python not installed, raise ImportError with install command

   Testing:
   - python -c "from core.llm import LLMController; llm = LLMController('./models/test.gguf'); print('LLM controller loaded')"
   (Will not load model since file doesn't exist, but class should instantiate)
```

- **Verification:** `LLMController` instantiates. Methods are callable (will fail without model, which is expected).
- **Expected Output:** LLM interface ready for agent integration.

---

#### ✅ Day 12 — Prompt Templates (ReAct, Planning, Synthesis)

- **Objective:** Create all prompt templates that the LLM uses for reasoning: the ReAct loop prompt, tool planning prompt, result interpretation prompt, and final synthesis prompt.
- **Files to Create/Modify:**
  - `core/prompts/react.py` — ReAct (Reason+Act) loop prompt template
  - `core/prompts/planning.py` — Tool selection/planning prompt template
  - `core/prompts/synthesis.py` — Final verdict synthesis prompt template
  - `core/prompts/__init__.py` — Export all prompt builders
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has LLMController (core/llm.py), all tools and registry, data models. The agent needs prompt templates to guide LLM reasoning.

Task: Create three prompt template modules.

Files to create:

1. `core/prompts/react.py`:

   REACT_SYSTEM_PROMPT = """You are Aegis-X, an AI forensic agent analyzing media for deepfakes.
   You have access to forensic tools. For each step:
   1. THINK: Analyze current evidence and decide what to investigate next
   2. ACT: Select one tool to run
   3. OBSERVE: Interpret the tool's output

   Available tools:
   {tool_descriptions}

   Current analysis state:
   {agent_state}

   Rules:
   - Start with cheap tools (C2PA, DCT) before expensive ones (entropy, artifacts)
   - If confidence > {confidence_threshold}, stop and synthesize verdict
   - If biological checks fail, try alternative tools
   - Never run the same tool twice
   - Prefer CPU tools before GPU tools when signals are weak
   """

   def build_react_prompt(tool_descriptions: str, agent_state: str,
                          confidence_threshold: float = 0.9) -> str: ...

   def build_step_prompt(step_number: int, previous_results: list,
                         current_confidence: float, available_tools: list) -> str: ...

2. `core/prompts/planning.py`:

   PLANNING_PROMPT = """Given the current forensic analysis state, select the NEXT tool to run.

   CURRENT STATE:
   - Media: {media_type} ({media_path})
   - Face detected: {face_detected}
   - Audio present: {has_audio}
   - Current confidence: {confidence}
   - Tools already used: {tools_used}
   - Previous results summary: {results_summary}

   AVAILABLE TOOLS (not yet used):
   {available_tools}

   Select the best next tool. Consider:
   1. Information value (which tool will most change confidence?)
   2. Compute cost (prefer cheaper tools)
   3. Prerequisites (face required? audio required?)
   4. Current evidence gaps

   Respond in JSON: {{"tool": "tool_name", "reasoning": "why this tool next"}}
   """

   def build_planning_prompt(agent_state) -> str: ...

3. `core/prompts/synthesis.py`:

   SYNTHESIS_PROMPT = """You are a forensic analyst. Based on the following tool results,
   provide a verdict (REAL, FAKE, or INCONCLUSIVE) with confidence and natural-language reasoning.

   Tool Results:
   {tool_results}

   Rules:
   1. Ground every claim in a specific tool output
   2. If signals conflict, explain the conflict
   3. If confidence < 0.5, recommend human review
   4. Never claim certainty — use probabilistic language

   Respond in JSON:
   {{"verdict": "REAL|FAKE|INCONCLUSIVE",
     "confidence": 0.0-1.0,
     "reasoning": "...",
     "key_evidence": ["...", "..."]}}
   """

   def build_synthesis_prompt(tool_results: list) -> str: ...
   def format_tool_results_for_prompt(tool_results: list) -> str: ...

4. Update `core/prompts/__init__.py` with exports.

Code Requirements:
- All prompts use {variable} substitution (str.format)
- Prompt strings must be clear, concise, and tested for LLM compliance
- format_tool_results_for_prompt must summarize each tool result in 2-3 lines
- Include token count estimation helper

Testing:
- python -c "from core.prompts import build_planning_prompt; print('prompts loaded')"
- python -c "from core.prompts.synthesis import build_synthesis_prompt; p = build_synthesis_prompt([{'tool': 'run_rppg', 'output': 'NO_PULSE'}]); print(p[:200])"
```

- **Verification:** All prompts render correctly with test data. No missing variables in format strings.
- **Expected Output:** Complete prompt template system.

---

#### ✅ Day 13 — Agent Loop (Core Agent)

- **Objective:** Build the central agent loop that implements the Observe→Think→Act→Update→Decide cycle, integrating the LLM controller, tool registry, and state management.
- **Files to Create/Modify:**
  - `core/agent.py` — Main `Agent` class with the reasoning loop
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has: LLM controller (core/llm.py), prompt templates (core/prompts/), tool registry with 7 tools (core/tools/), data models (core/models.py with AgentState, AgentStep, ToolResult, ForensicReport, Verdict), config (core/config.py), device (utils/device.py), preprocessing (utils/preprocessing.py), video (utils/video.py), audio (utils/audio.py).

Task: Create the core Agent class.

File to create: `core/agent.py`

   from typing import Optional, Callable
   from core.models import AgentState, AgentStep, ForensicReport, Verdict, AnalysisConfig, ToolResult
   from core.llm import LLMController
   from core.tools.registry import ToolRegistry
   from core.config import AegisConfig

   class Agent:
       """Aegis-X Forensic Agent — the central reasoning loop."""

       def __init__(self, config: AegisConfig = None, analysis_config: AnalysisConfig = None):
           """Initialize agent with config.
           - Load config if not provided
           - Initialize DeviceManager
           - Initialize ToolRegistry with create_default_registry
           - Initialize LLMController with model path from config
           - Initialize Memory (if enabled)
           """

       def analyze(self, media_path: str, on_step: Callable = None) -> ForensicReport:
           """Main entry point — analyze a media file.
           1. Detect media type (video/image)
           2. Extract frames (if video)
           3. Detect faces
           4. Check for audio
           5. Initialize AgentState
           6. Run agent loop
           7. Generate ForensicReport
           """

       def _run_agent_loop(self, state: AgentState) -> AgentState:
           """The core Observe→Think→Act→Update→Decide loop.
           while state.current_confidence < threshold and iteration < max:
               1. Build planning prompt from state
               2. Ask LLM to select next tool
               3. Execute selected tool
               4. Ask LLM to interpret result
               5. Update confidence (prior + confidence_delta)
               6. Record AgentStep
               7. Check early stopping conditions
           """

       def _select_next_tool(self, state: AgentState) -> str:
           """Use LLM to select the next tool. Fallback to heuristic if LLM fails."""

       def _heuristic_tool_selection(self, state: AgentState) -> str:
           """Fallback tool selection without LLM:
           Priority: check_c2pa → run_rppg → run_entropy → run_artifacts → run_dct → run_reflection → run_lipsync"""

       def _execute_tool(self, tool_name: str, state: AgentState) -> ToolResult:
           """Prepare inputs for the tool and execute it."""

       def _update_confidence(self, state: AgentState, tool_result: ToolResult) -> float:
           """Bayesian-like confidence update: new = old + delta, clamped to [0, 1]."""

       def _should_stop(self, state: AgentState) -> bool:
           """Check stopping conditions:
           1. Confidence >= threshold → stop with verdict
           2. Confidence <= (1 - threshold) → stop with opposite verdict
           3. Max iterations reached → stop with current best
           4. All applicable tools exhausted → stop
           """

       def _generate_report(self, state: AgentState) -> ForensicReport:
           """Use LLM synthesis prompt to generate final verdict and reasoning.
           Falls back to rule-based verdict if LLM fails."""

       def _rule_based_verdict(self, state: AgentState) -> dict:
           """Fallback verdict without LLM:
           confidence > 0.7 → REAL, confidence < 0.3 → FAKE, else INCONCLUSIVE"""

   Code Requirements:
   - analyze() must work end-to-end: file in → ForensicReport out
   - Handle: video files, image files, missing faces, missing audio
   - on_step callback called after each agent step with AgentStep object
   - Log every step with loguru
   - Timing: track total analysis time
   - Error resilience: if one tool fails, skip and try next
   - If LLM not available, use heuristic fallback for everything

   Testing:
   - python -c "from core.agent import Agent; a = Agent(); print('Agent initialized')"
   (Will warn about missing models but should not crash)
```

- **Verification:** Agent class instantiates. The `analyze` method signature matches the API spec from README.
- **Expected Output:** The core agent loop — the heart of Aegis-X.

---

#### ✅ Day 14 — Memory & Experience System

- **Objective:** Build the persistent memory system that stores past cases, artifact patterns, and failure analysis. The agent uses this for experience-based reasoning.
- **Files to Create/Modify:**
  - `core/memory.py` — Memory system with short-term (current case) and long-term (persistent) storage
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has the Agent class (core/agent.py) with the reasoning loop. The README describes a memory system with short-term (tool results, confidence history, decision trace for current case) and long-term (previous cases, artifact patterns, failure analysis stored in JSON).

Task: Create the memory system.

File to create: `core/memory.py`

   from typing import List, Dict, Optional, Any
   from pathlib import Path
   import json
   from datetime import datetime

   class ShortTermMemory:
       """In-memory state for the current analysis case."""
       def __init__(self):
           self.tool_results: List[dict] = []
           self.confidence_history: List[float] = [0.5]
           self.decision_trace: List[dict] = []

       def add_result(self, tool_name: str, result: dict, confidence: float): ...
       def get_summary(self) -> str: ...
       def get_last_n_results(self, n: int = 3) -> List[dict]: ...
       def clear(self): ...

   class LongTermMemory:
       """Persistent memory stored as JSON files."""

       def __init__(self, cases_path: str = "./memory/cases.json",
                    patterns_path: str = "./memory/patterns.json"):
           self.cases_path = Path(cases_path)
           self.patterns_path = Path(patterns_path)
           self._cases: List[dict] = []
           self._patterns: List[dict] = []
           self._load()

       def _load(self): """Load from JSON files if they exist."""
       def _save(self): """Save to JSON files."""

       def store_case(self, case: dict):
           """Store a completed analysis case.
           case: {media_hash, verdict, confidence, tools_used, key_patterns, timestamp}"""

       def find_similar_cases(self, patterns: List[str], limit: int = 5) -> List[dict]:
           """Find past cases with similar artifact patterns."""

       def add_pattern(self, pattern_name: str, indicators: dict, source_case: str):
           """Record a new artifact pattern for future matching."""

       def get_failure_cases(self, limit: int = 10) -> List[dict]:
           """Return cases where the agent was uncertain or wrong."""

       def get_statistics(self) -> dict:
           """Return memory statistics: total cases, verdict distribution, etc."""

   class AgentMemory:
       """Combined memory interface used by the Agent."""
       def __init__(self, config):
           self.short_term = ShortTermMemory()
           self.long_term = LongTermMemory(config.agent.memory_path) if config.agent.enable_memory else None

       def get_context_for_planning(self) -> str:
           """Return combined memory context for LLM planning prompts."""

       def save_case(self, report): """Store completed case in long-term memory."""
       def recall_patterns(self, current_results) -> str: """Find matching patterns."""

   Code Requirements:
   - Long-term memory files created lazily (first write)
   - JSON files use pretty-print with indent=2
   - Memory is optional (controlled by config.agent.enable_memory)
   - Media files are referenced by SHA-256 hash, never stored
   - Pattern matching uses simple keyword overlap
   - Limit long-term memory to 1000 most recent cases (FIFO)

   Testing:
   - python -c "from core.memory import AgentMemory; from core.config import load_config; m = AgentMemory(load_config()); print('memory loaded')"
```

- **Verification:** AgentMemory initializes. Short-term memory stores and retrieves results. Long-term memory creates JSON files on first save.
- **Expected Output:** Memory system operational.

---

#### ✅ Day 15 — Agent Integration & End-to-End Flow

- **Objective:** Wire the agent, LLM, memory, and all tools together. Test the complete flow from file input to report output (using mocked LLM responses for testing without GPU models).
- **Files to Create/Modify:**
  - `core/agent.py` — Update with memory integration and fallback strategies
  - `core/__init__.py` — Export Agent, AnalysisConfig
  - `tests/test_agent.py` — Agent unit tests with mocked LLM
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X has Agent (core/agent.py), LLMController (core/llm.py), AgentMemory (core/memory.py), ToolRegistry with 7 tools, all prompts, all utils. I need to wire everything together and ensure the agent works end-to-end.

Task: Complete the agent integration and write tests.

Files to modify/create:

1. Update `core/agent.py`:
   - Integrate AgentMemory into agent loop
   - Before planning each step, add memory context to prompt
   - After each analysis, call memory.save_case()
   - Add _prepare_tool_inputs() method that maps tool_name to correct kwargs:
     * check_c2pa → {file_path: state.media_path}
     * run_rppg → {frames: face_frames, fps: video_fps}
     * run_entropy → {face_crop: aligned_face}
     * run_artifacts → {face_crop: aligned_face}
     * run_dct → {image: grayscale_frame}
     * run_reflection → {frame: frame, landmarks: landmarks}
     * run_lipsync → {frames: face_frames, audio_path: audio, fps: fps, landmarks_sequence: landmarks_list}

2. Update `core/__init__.py`:
   from core.agent import Agent
   from core.models import AnalysisConfig, ForensicReport, Verdict
   __all__ = ["Agent", "AnalysisConfig", "ForensicReport", "Verdict"]

3. Create `tests/test_agent.py`:
   - test_agent_init: Agent initializes without errors
   - test_agent_heuristic_fallback: Agent selects tools in correct priority without LLM
   - test_agent_confidence_update: Confidence updates correctly with deltas
   - test_agent_early_stopping: Agent stops when confidence exceeds threshold
   - test_agent_max_iterations: Agent stops at max iterations
   - test_agent_missing_face: Agent skips face-required tools when no face
   - test_rule_based_verdict: Correct verdicts at different confidence levels
   - Mock LLMController for all tests (don't require actual model)

Testing:
- pytest tests/test_agent.py -v
- python -c "from core import Agent; a = Agent(); print('Aegis-X Agent ready')"
```

- **Verification:** All agent tests pass. Agent can be imported from `core` package.
- **Expected Output:** Fully wired agent — the brain of Aegis-X is functional.

---

### 🗓️ Week 4

#### ✅ Day 16–17 — Agent Robustness & Edge Cases (2 days)

- **Objective:** Harden the agent against edge cases: corrupt files, no face, no audio, GPU OOM, tool crashes, LLM failures. Implement the reward heuristics system.
- **Files to Create/Modify:**
  - `core/agent.py` — Add error recovery, fallback paths, reward system
  - `core/reward.py` — Reward heuristics for tool selection optimization
  - `tests/test_edge_cases.py` — Edge case tests

*(Detailed prompt follows same format as above — covers conditional autonomy from README: face not detected → audio-only, rPPG fails → try reflection, biological checks fail → neural analysis + escalate, all inconclusive → mandatory human review. Reward system: +1 per 0.1 confidence gain, -1 for no evidence, -0.5 for high GPU cost, +5 for early verdict, -2 for escalation.)*

---

#### ✅ Day 18–19 — Escalation System & Report Generation (2 days)

- **Objective:** Build the human escalation system (flag ambiguous cases for review) and the comprehensive report generator that produces JSON, HTML, and PDF reports.
- **Files to Create/Modify:**
  - `core/tools/escalation_tool.py` — The `escalate_to_human()` meta-tool
  - `core/tools/report_tool.py` — The `generate_report()` meta-tool
  - `utils/report_generator.py` — Report formatting (JSON, HTML with embedded heatmaps)

*(Detailed prompt: escalation triggers at confidence 0.5-0.9, report includes verdict, confidence, reasoning, key_evidence, tool results, decision trace, heatmap images, timing data. JSON output matches README API spec.)*

---

#### ✅ Day 20 — Phase 2 Integration Test

- **Objective:** Run a complete end-to-end test of the agent with all components. Create a synthetic test case and verify the agent produces a correct verdict.
- **Files to Create/Modify:**
  - `tests/test_integration.py` — Full integration test
  - `scripts/run_demo.py` — Demo script that creates a synthetic test and runs the agent

```text
Task: Write integration tests and a demo script.

1. `tests/test_integration.py`:
   - test_full_pipeline_synthetic: Create synthetic video frames (random noise), run Agent.analyze(), verify ForensicReport structure
   - test_full_pipeline_image: Create synthetic image, run analyze in image mode
   - test_report_json_output: Verify report serializes to valid JSON
   - test_memory_persistence: Run two analyses, verify second has memory context

2. `scripts/run_demo.py`:
   - Generate synthetic test video (100 frames of noise)
   - Save as temp file
   - Run Agent.analyze()
   - Print report to console with rich formatting
   - Test WITHOUT real models (uses heuristic fallback)

Testing:
- pytest tests/test_integration.py -v
- python scripts/run_demo.py
```

- **Verification:** Integration tests pass. Demo script runs and produces a formatted report.
- **Expected Output:** Phase 2 complete. Agent core, LLM, memory, prompts, and all tools fully integrated.

---

## 📅 PHASE 3 — CLI & Web Interfaces (Week 5 to Week 6)

**Goal:** Build all three user interfaces: CLI (main.py with Click), Streamlit web app, and Gradio web app. Add batch processing support.

**Dependencies:** Phase 2 complete (functional Agent class).

---

### 🗓️ Week 5

#### ✅ Day 21–22 — CLI Entry Point (2 days)

- **Objective:** Build `main.py` with Click framework providing all CLI commands from the README.
- **Files to Create/Modify:**
  - `main.py` — CLI entry point with Click commands
- **Key Features:** `--input`, `--output`, `--verbose`, `--mode`, `--confidence-threshold`, `--max-iterations`, `--skip-c2pa`, `--skip-audio`, `--cpu-only`, `--device`, `--input-dir`, `--output-dir`, `--pattern`.

---

#### ✅ Day 23–24 — Streamlit Web Interface (2 days)

- **Objective:** Build `app.py` with Streamlit: file upload, real-time analysis progress, verdict display with heatmaps, agent trace visualization.
- **Files to Create/Modify:**
  - `app.py` — Full Streamlit interface
- **Key Features:** File upload (video/image), real-time agent step display, verdict with confidence bar, interactive heatmap viewer, download report button, dark theme.

---

#### ✅ Day 25–26 — Gradio Web Interface (2 days)

- **Objective:** Build `gradio_app.py` with Gradio: tab-based interface, file upload, real-time analysis.
- **Files to Create/Modify:**
  - `gradio_app.py` — Full Gradio interface
- **Key Features:** Tabbed interface (Analyze / Batch / Settings), file upload, progress display, verdict display, JSON report download.

---

### 🗓️ Week 6

#### ✅ Day 27–28 — Batch Processing & API Module (2 days)

- **Objective:** Build batch processing for directories and the Python API module (`aegis_x` package import).
- **Files to Create/Modify:**
  - `core/batch.py` — Batch processing with progress bars, parallel execution
  - `aegis_x/__init__.py` — Public API: `from aegis_x import Agent, AnalysisConfig`
  - `setup.py` or `pyproject.toml` — Package configuration

---

#### ✅ Day 29 — Polish CLI & Web UIs

- **Objective:** Polish all three interfaces: error messages, loading states, progress indicators, keyboard shortcuts, help text.

---

#### ✅ Day 30 — Phase 3 Integration Test

- **Objective:** Test all three interfaces end-to-end. Create automated tests for CLI.
- **Files to Create/Modify:**
  - `tests/test_cli.py` — CLI integration tests
  - `tests/test_web.py` — Basic web UI smoke tests

---

# 🛡️ Aegis-X — Implementation Plan — Part 3

> Continuation from Part 2. Covers **Phase 4** (Days 31–40) and **Phase 5** (Days 41–50), plus **Master Checklist**.

---

## 📅 PHASE 4 — Testing & Hardening (Week 7 to Week 8)

**Goal:** Comprehensive test suite (unit, integration, performance), benchmarking system, error handling hardening, and cross-platform validation.

**Dependencies:** Phase 3 complete (all interfaces working).

---

### 🗓️ Week 7

#### ✅ Day 31 — Unit Tests for All Utilities

- **Objective:** Write comprehensive unit tests for every utility module.
- **Files to Create/Modify:**
  - `tests/test_video.py` — Tests for `utils/video.py`
  - `tests/test_audio.py` — Tests for `utils/audio.py`
  - `tests/test_preprocessing.py` — Tests for `utils/preprocessing.py`
  - `tests/test_visualization.py` — Tests for `utils/visualization.py`
  - `tests/test_config.py` — Tests for `core/config.py`
  - `tests/test_device.py` — Tests for `utils/device.py`
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X at d:\Aegis-X\ has all utility modules implemented. I need comprehensive unit tests for each.

Task: Create unit tests for all utility modules.

Files to create:

1. `tests/test_video.py`:
   - test_is_video_file_valid: .mp4, .avi, .mov, .mkv, .webm return True
   - test_is_video_file_invalid: .txt, .jpg, .pdf return False
   - test_is_image_file_valid: .jpg, .jpeg, .png, .bmp, .webp return True
   - test_load_image_nonexistent: raises FileNotFoundError
   - test_video_loader_nonexistent: raises FileNotFoundError
   - test_video_loader_context_manager: with statement works

2. `tests/test_audio.py`:
   - test_has_audio_nonexistent: returns False for nonexistent file
   - test_extract_audio_nonexistent: raises error for nonexistent file

3. `tests/test_preprocessing.py`:
   - test_face_detector_init_no_model: raises FileNotFoundError with instructions
   - test_face_detector_detect_no_face: empty frame returns empty list

4. `tests/test_visualization.py`:
   - test_overlay_heatmap_shape: output shape matches input
   - test_draw_verdict_overlay: returns valid numpy array
   - test_plot_rppg_signal: returns image or saves to file

5. `tests/test_config.py`:
   - test_load_default_config: loads without config file
   - test_config_values: default values match expected
   - test_config_from_yaml: loads from config.yaml
   - test_validate_config: catches invalid values

6. `tests/test_device.py`:
   - test_device_detection: returns valid device string
   - test_vram_non_negative: VRAM >= 0
   - test_loading_strategy: returns valid strategy string

Code Requirements:
- Use pytest fixtures from conftest.py
- Each test should be independent (no shared state)
- Use tmp_path fixture for file operations
- Mock GPU operations for CI compatibility

Testing:
- pytest tests/ -v --tb=short
```

- **Verification:** All unit tests pass. Coverage >80% for utility modules.
- **Expected Output:** Solid test foundation for all utilities.

---

#### ✅ Day 32 — Unit Tests for All Forensic Tools

- **Objective:** Write detailed unit tests for each of the 7 forensic tools, testing both success and failure cases.
- **Files to Create/Modify:**
  - `tests/test_tools_detailed.py` — Comprehensive tool tests
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X has 7 forensic tools: c2pa_tool, rppg_tool, entropy_tool, artifacts_tool, dct_tool, reflection_tool, lipsync_tool. All subclass ForensicTool.

Task: Create detailed tests for each tool.

File to create: `tests/test_tools_detailed.py`

Tests for each tool:

C2PATool:
- test_c2pa_nonexistent_file: Returns failed ToolResult
- test_c2pa_no_manifest: Regular file returns no signature
- test_c2pa_result_structure: Output has required keys
- test_c2pa_run_wrapper: .run() returns ToolResult with timing

RPPGTool:
- test_rppg_valid_frames: 100 frames produces result
- test_rppg_insufficient_frames: <90 frames returns INSUFFICIENT_DATA
- test_rppg_random_noise: Random noise returns NO_PULSE or AMBIGUOUS
- test_rppg_constant_color: All-white frames return NO_PULSE
- test_rppg_result_has_confidence_delta: Output includes confidence_delta
- test_rppg_result_has_metrics: Output includes snr_db, heart_rate_bpm

DCTTool:
- test_dct_random_image: Processes random grayscale image
- test_dct_result_structure: Output has grid_artifacts, score
- test_dct_small_image: Handles images smaller than 8x8
- test_dct_score_range: Score is between 0 and 1

EntropyTool:
- test_entropy_instantiation: Creates without error
- test_entropy_is_available: Returns bool based on model presence

ArtifactsTool:
- test_artifacts_instantiation: Creates without error
- test_artifacts_is_available: Returns bool based on timm availability

ReflectionTool:
- test_reflection_instantiation: Creates without error
- test_reflection_no_landmarks: Handles missing landmarks gracefully

LipsyncTool:
- test_lipsync_instantiation: Creates without error
- test_lipsync_no_audio: Handles missing audio gracefully

Registry:
- test_registry_register_all: All 7 tools register
- test_registry_get_by_name: Correct tool returned
- test_registry_tool_descriptions: Non-empty string
- test_registry_filter_by_context: Correct tools for face/audio/video combos

Code Requirements:
- Use numpy random data for frame inputs
- Mock GPU-dependent tools (entropy, artifacts, lipsync)
- Test .run() wrapper catches exceptions
- Verify ToolResult fields: tool_name, output, success, execution_time_ms

Testing:
- pytest tests/test_tools_detailed.py -v
```

- **Verification:** All tool tests pass. Tools handle edge cases gracefully.
- **Expected Output:** Comprehensive tool test coverage.

---

#### ✅ Day 33 — Agent & Memory Tests

- **Objective:** Write comprehensive tests for the agent loop, memory system, and prompt templates.
- **Files to Create/Modify:**
  - `tests/test_agent_detailed.py` — Agent loop tests
  - `tests/test_memory.py` — Memory system tests
  - `tests/test_prompts.py` — Prompt template tests

*(Tests cover: agent initialization with/without config, heuristic tool selection ordering, confidence clamping to [0,1], early stopping at threshold, max iteration limit, face-required tool skipping, audio-required tool skipping, memory case storage/retrieval, pattern matching, JSON persistence, prompt variable substitution, token estimation.)*

---

#### ✅ Day 34 — Error Handling & Resilience

- **Objective:** Add robust error handling throughout the codebase: graceful degradation, retry logic, timeout handling, and user-friendly error messages.
- **Files to Create/Modify:**
  - `core/errors.py` — Custom exception classes
  - `core/agent.py` — Update with comprehensive try/catch
  - `utils/retry.py` — Retry decorator with exponential backoff
- **AI Prompt to Execute This Day's Work:**

```text
Context: Aegis-X needs hardened error handling for production use.

Task: Create error handling infrastructure.

Files to create:

1. `core/errors.py`:
   class AegisError(Exception): """Base exception."""
   class ModelNotFoundError(AegisError): """Model file not found. Include download instructions."""
   class MediaError(AegisError): """Cannot read/process media file."""
   class ToolError(AegisError): """Tool execution failed."""
   class LLMError(AegisError): """LLM query failed."""
   class ConfigError(AegisError): """Invalid configuration."""
   class InsufficientDataError(AegisError): """Not enough data for analysis."""
   class GPUError(AegisError): """GPU/CUDA error."""

   Each exception class includes:
   - Descriptive message
   - Suggested fix in self.suggestion attribute
   - Error code (e.g., "E001", "E002")

2. `utils/retry.py`:
   def retry(max_attempts=3, backoff_factor=2, exceptions=(Exception,)):
       """Decorator that retries function on specified exceptions."""

   def timeout(seconds=60):
       """Decorator that raises TimeoutError if function takes too long.
       Note: uses threading on Windows (signal.alarm not available)."""

3. Update core/agent.py error handling:
   - Wrap each tool execution in try/except ToolError
   - Wrap LLM queries in try/except LLMError with fallback
   - Wrap file operations in try/except MediaError
   - Log all errors with full traceback at DEBUG level
   - Show user-friendly message at ERROR level

Testing:
- python -c "from core.errors import ModelNotFoundError; raise ModelNotFoundError('test', suggestion='run download_models.py')"
- pytest tests/ -v (all existing tests still pass)
```

- **Verification:** Error classes instantiate correctly. Retry decorator works. Agent handles tool failures without crashing.
- **Expected Output:** Production-grade error handling.

---

#### ✅ Day 35 — Logging, Metrics & Monitoring

- **Objective:** Add structured logging throughout the codebase, performance metrics collection, and a simple monitoring dashboard.
- **Files to Create/Modify:**
  - `core/metrics.py` — Performance metrics collector (timing, memory, tool usage stats)
  - Update all modules to use structured logging consistently

*(Metrics tracked: per-tool execution time, VRAM usage per tool, agent loop iterations, early stopping rate, tool skip rate, confidence convergence speed.)*

---

### 🗓️ Week 8

#### ✅ Day 36–37 — Benchmark Suite (2 days)

- **Objective:** Create a benchmarking system that measures detection accuracy, inference speed, and compute efficiency.
- **Files to Create/Modify:**
  - `scripts/benchmark.py` — Benchmark runner
  - `scripts/generate_synthetic.py` — Generate synthetic test cases for benchmarking
  - `tests/benchmarks/` — Benchmark test directory

```text
Task: Create the benchmarking infrastructure.

1. `scripts/generate_synthetic.py`:
   - Generate synthetic "real" videos: smooth color gradients with subtle temporal variation (simulating pulse)
   - Generate synthetic "fake" videos: random noise, static frames, abrupt color changes
   - Generate synthetic images with and without GAN-like artifacts
   - Output to tests/benchmarks/data/ directory
   - Generate 50 "real" and 50 "fake" samples

2. `scripts/benchmark.py`:
   - Load synthetic test cases
   - Run Agent.analyze() on each (using heuristic mode, no LLM required)
   - Measure: accuracy, false positive rate, false negative rate
   - Measure: average inference time, tools used per case
   - Measure: early stopping rate
   - Output results as rich table + JSON file
   - Compare against baseline (all-tools pipeline)

Testing:
- python scripts/generate_synthetic.py --output tests/benchmarks/data/ --count 20
- python scripts/benchmark.py --data tests/benchmarks/data/ --output benchmarks_result.json
```

- **Verification:** Benchmark runs on synthetic data. Results match expected performance characteristics.
- **Expected Output:** Repeatable benchmark suite for measuring improvements.

---

#### ✅ Day 38 — Cross-Platform Testing

- **Objective:** Verify Aegis-X works on Windows, Linux (via WSL/Docker), and macOS. Fix platform-specific issues.
- **Files to Create/Modify:**
  - `scripts/test_platform.py` — Platform compatibility test
  - Fix any path separator, subprocess, or device detection issues

---

#### ✅ Day 39 — Security & Privacy Audit

- **Objective:** Verify no network calls during analysis, validate file handling security, and ensure GDPR compliance.
- **Files to Create/Modify:**
  - `core/privacy.py` — Privacy guardrails (file hash computation, no telemetry, tmp file cleanup)
  - `tests/test_privacy.py` — Privacy verification tests

```text
Task: Implement privacy safeguards.

1. `core/privacy.py`:
   def compute_file_hash(path: str) -> str: """SHA-256 hash of file for reference without storing content."""
   def verify_no_network(): """Monkey-patch socket to verify no network calls during analysis."""
   def cleanup_temp_files(temp_dir: str): """Remove all temporary files created during analysis."""
   def sanitize_report(report: dict) -> dict: """Remove any file paths or sensitive info from report."""

2. `tests/test_privacy.py`:
   - test_no_network_calls: Run analysis with network blocked, verify success
   - test_temp_cleanup: Verify no temp files remain after analysis
   - test_report_no_absolute_paths: Report contains no system paths
```

- **Verification:** Privacy tests pass. No network calls detected during analysis.
- **Expected Output:** Verified offline, privacy-preserving system.

---

#### ✅ Day 40 — Phase 4 Integration Test

- **Objective:** Run the full test suite, generate coverage report, and verify all components work together.
- **Files to Create/Modify:**
  - `pytest.ini` or `pyproject.toml [tool.pytest]` — Pytest configuration
  - `scripts/run_all_tests.py` — Master test runner

```text
- pytest tests/ -v --cov=core --cov=utils --cov-report=html
- Verify coverage > 70% for core/, > 80% for utils/
- All tests green
- Generate HTML coverage report in htmlcov/
```

---

## 📅 PHASE 5 — Deployment & Documentation (Week 9 to Week 10)

**Goal:** Docker packaging, model management automation, REST API server, final documentation, and release preparation.

**Dependencies:** Phase 4 complete (all tests passing).

---

### 🗓️ Week 9

#### ✅ Day 41 — Docker Containerization

- **Objective:** Create Dockerfile and docker-compose for one-command deployment with pre-configured environment.
- **Files to Create/Modify:**
  - `Dockerfile` — Multi-stage build with CUDA support
  - `docker-compose.yml` — Compose config with GPU passthrough
  - `.dockerignore` — Ignore unnecessary files

```text
Task: Create Docker deployment.

1. `Dockerfile`:
   - Base: nvidia/cuda:11.8-runtime-ubuntu22.04
   - Install Python 3.11, ffmpeg, cmake, system deps
   - Copy requirements.txt, install dependencies
   - Copy source code
   - Set ENTRYPOINT to python main.py
   - Expose ports 8501 (Streamlit), 7860 (Gradio), 8000 (API)
   - Multi-stage: builder stage for compilation, runtime stage for slim image

2. `docker-compose.yml`:
   services:
     aegis-x:
       build: .
       runtime: nvidia
       volumes: ./models:/app/models, ./memory:/app/memory
       ports: 8501, 7860, 8000
     aegis-x-cpu:
       build: .
       volumes: same
       ports: same
       environment: AEGIS_DEVICE=cpu

3. `.dockerignore`: venv/, __pycache__/, .git/, logs/, tests/, *.md
```

- **Verification:** `docker build -t aegis-x .` completes. `docker compose up aegis-x-cpu` starts.
- **Expected Output:** Docker deployment ready.

---

#### ✅ Day 42 — Model Management Automation

- **Objective:** Polish model download, verification, and update scripts. Add integrity checks with checksums.
- **Files to Create/Modify:**
  - `scripts/download_models.py` — Update with progress bars, checksums, resume support
  - `scripts/check_models.py` — Update with SHA-256 verification
  - `scripts/update_models.py` — Check for newer model versions

---

#### ✅ Day 43 — REST API Server

- **Objective:** Build a FastAPI REST API server for programmatic access.
- **Files to Create/Modify:**
  - `api/server.py` — FastAPI application
  - `api/routes.py` — API routes
  - `api/schemas.py` — Request/response schemas

```text
Task: Create FastAPI REST API.

Endpoints:
- POST /analyze — Upload file, return ForensicReport
- POST /analyze/url — Analyze file from URL
- GET /status/{job_id} — Check async analysis status
- GET /health — Health check
- GET /models — List loaded models and their status
- GET /tools — List available tools

Features:
- Async analysis with job queue
- File upload (multipart form)
- JSON response matching ForensicReport schema
- OpenAPI documentation at /docs
- Rate limiting
- CORS support
```

---

#### ✅ Day 44 — API Testing & Documentation

- **Objective:** Test the REST API and generate API documentation.
- **Files to Create/Modify:**
  - `tests/test_api.py` — API endpoint tests
  - `docs/api.md` — API documentation

---

#### ✅ Day 45 — Performance Optimization

- **Objective:** Profile and optimize critical paths: model loading, frame extraction, batch processing.
- **Focus Areas:**
  - Lazy loading optimization (preload hint system)
  - Frame extraction with hardware decoding (if available)
  - Batch face detection for multi-frame sequences
  - Memory management for sequential model loading strategy

---

### 🗓️ Week 10

#### ✅ Day 46–47 — Final Documentation (2 days)

- **Objective:** Update README, create contributing guide, add inline code documentation, generate API docs.
- **Files to Create/Modify:**
  - `README.md` — Update with actual paths, working commands, real screenshots
  - `CONTRIBUTING.md` — Contribution guidelines
  - `docs/architecture.md` — Detailed architecture documentation
  - `docs/tools.md` — Individual tool documentation
  - `LICENSE` — MIT license file

---

#### ✅ Day 48 — Release Preparation

- **Objective:** Create release artifacts: PyPI package, GitHub release, changelog.
- **Files to Create/Modify:**
  - `pyproject.toml` — Package metadata for PyPI
  - `CHANGELOG.md` — Version 1.0.0 changelog
  - `scripts/release.py` — Release automation script

---

#### ✅ Day 49 — End-to-End Acceptance Test

- **Objective:** Run the complete system on real-world test data. Verify all interfaces work.
- **Tests:**
  - CLI: `python main.py --input test_video.mp4 --verbose --output report.json`
  - Streamlit: `streamlit run app.py` — upload file, get verdict
  - Gradio: `python gradio_app.py` — upload file, get verdict
  - API: `uvicorn api.server:app` — POST /analyze with test file
  - Docker: `docker compose up` — verify all services start
  - Batch: `python main.py --input-dir ./test_data/ --output-dir ./reports/`

---

#### ✅ Day 50 — Final Polish & v1.0.0 Release

- **Objective:** Fix any remaining issues, tag v1.0.0, push to GitHub.
- **Tasks:**
  - Run full test suite one final time
  - Fix any remaining lint issues (`flake8`, `black`)
  - Tag git release: `git tag v1.0.0`
  - Push to GitHub
  - Create GitHub Release with changelog
  - Update README badges with real CI status

---

## 📋 MASTER CHECKLIST

### Phase 1 — Foundation & Infrastructure (Week 1–2)

| Day | Title | Key Deliverable | Status |
|:----|:------|:----------------|:-------|
| 1 | Project Skeleton & Dependencies | `requirements.txt`, `config.yaml`, `.env`, `.gitignore`, all `__init__.py` | ⬜ |
| 2 | Device Management & Logging | `utils/device.py`, `utils/logger.py` | ⬜ |
| 3 | Config Loader & Model Manager | `core/config.py`, `scripts/check_models.py`, `scripts/download_models.py` | ⬜ |
| 4 | Video & Audio Preprocessing | `utils/video.py`, `utils/audio.py`, `utils/preprocessing.py` | ⬜ |
| 5 | Visualization & Data Models | `utils/visualization.py`, `core/models.py` | ⬜ |
| 6 | Base Tool Class & Registry | `core/tools/base.py`, `core/tools/registry.py` | ⬜ |
| 7 | C2PA & rPPG Tools | `core/tools/c2pa_tool.py`, `core/tools/rppg_tool.py` | ⬜ |
| 8 | Entropy & Artifact Tools | `core/tools/entropy_tool.py`, `core/tools/artifacts_tool.py` | ⬜ |
| 9 | DCT, Reflection & Lipsync Tools | `core/tools/dct_tool.py`, `core/tools/reflection_tool.py`, `core/tools/lipsync_tool.py` | ⬜ |
| 10 | Tool Integration Test | `tests/test_tools.py`, updated `core/tools/__init__.py` | ⬜ |

### Phase 2 — Agent Core & LLM Brain (Week 3–4)

| Day | Title | Key Deliverable | Status |
|:----|:------|:----------------|:-------|
| 11 | LLM Controller Interface | `core/llm.py` | ⬜ |
| 12 | Prompt Templates | `core/prompts/react.py`, `planning.py`, `synthesis.py` | ⬜ |
| 13 | Agent Loop (Core Agent) | `core/agent.py` | ⬜ |
| 14 | Memory & Experience System | `core/memory.py` | ⬜ |
| 15 | Agent Integration & E2E Flow | Updated `core/agent.py`, `tests/test_agent.py` | ⬜ |
| 16–17 | Agent Robustness & Edge Cases | `core/reward.py`, `tests/test_edge_cases.py` | ⬜ |
| 18–19 | Escalation & Report Generation | `core/tools/escalation_tool.py`, `core/tools/report_tool.py`, `utils/report_generator.py` | ⬜ |
| 20 | Phase 2 Integration Test | `tests/test_integration.py`, `scripts/run_demo.py` | ⬜ |

### Phase 3 — CLI & Web Interfaces (Week 5–6)

| Day | Title | Key Deliverable | Status |
|:----|:------|:----------------|:-------|
| 21–22 | CLI Entry Point | `main.py` with all Click commands | ⬜ |
| 23–24 | Streamlit Web Interface | `app.py` | ⬜ |
| 25–26 | Gradio Web Interface | `gradio_app.py` | ⬜ |
| 27–28 | Batch Processing & API Module | `core/batch.py`, `aegis_x/__init__.py`, `pyproject.toml` | ⬜ |
| 29 | Polish CLI & Web UIs | UI refinements across all interfaces | ⬜ |
| 30 | Phase 3 Integration Test | `tests/test_cli.py`, `tests/test_web.py` | ⬜ |

### Phase 4 — Testing & Hardening (Week 7–8)

| Day | Title | Key Deliverable | Status |
|:----|:------|:----------------|:-------|
| 31 | Unit Tests for Utilities | `tests/test_video.py`, `test_audio.py`, etc. | ⬜ |
| 32 | Unit Tests for Tools | `tests/test_tools_detailed.py` | ⬜ |
| 33 | Agent & Memory Tests | `tests/test_agent_detailed.py`, `test_memory.py`, `test_prompts.py` | ⬜ |
| 34 | Error Handling & Resilience | `core/errors.py`, `utils/retry.py` | ⬜ |
| 35 | Logging, Metrics & Monitoring | `core/metrics.py` | ⬜ |
| 36–37 | Benchmark Suite | `scripts/benchmark.py`, `scripts/generate_synthetic.py` | ⬜ |
| 38 | Cross-Platform Testing | `scripts/test_platform.py` | ⬜ |
| 39 | Security & Privacy Audit | `core/privacy.py`, `tests/test_privacy.py` | ⬜ |
| 40 | Phase 4 Integration Test | Full test suite pass + coverage report | ⬜ |

### Phase 5 — Deployment & Documentation (Week 9–10)

| Day | Title | Key Deliverable | Status |
|:----|:------|:----------------|:-------|
| 41 | Docker Containerization | `Dockerfile`, `docker-compose.yml` | ⬜ |
| 42 | Model Management Automation | Updated `scripts/download_models.py`, `check_models.py` | ⬜ |
| 43 | REST API Server | `api/server.py`, `api/routes.py`, `api/schemas.py` | ⬜ |
| 44 | API Testing & Documentation | `tests/test_api.py`, `docs/api.md` | ⬜ |
| 45 | Performance Optimization | Profiling + optimizations | ⬜ |
| 46–47 | Final Documentation | Updated `README.md`, `CONTRIBUTING.md`, `docs/` | ⬜ |
| 48 | Release Preparation | `pyproject.toml`, `CHANGELOG.md` | ⬜ |
| 49 | End-to-End Acceptance Test | All interfaces verified | ⬜ |
| 50 | Final Polish & v1.0.0 Release | Git tag, GitHub Release | ⬜ |

---

## 📊 File Creation Summary

| Category | Files | Count |
|:---------|:------|:------|
| **Root** | `main.py`, `app.py`, `gradio_app.py`, `requirements.txt`, `.env`, `config.yaml`, `.gitignore`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `pyproject.toml`, `CHANGELOG.md`, `CONTRIBUTING.md`, `LICENSE` | 14 |
| **core/** | `__init__.py`, `agent.py`, `llm.py`, `memory.py`, `config.py`, `models.py`, `errors.py`, `reward.py`, `metrics.py`, `privacy.py`, `batch.py` | 11 |
| **core/tools/** | `__init__.py`, `base.py`, `registry.py`, `c2pa_tool.py`, `rppg_tool.py`, `entropy_tool.py`, `artifacts_tool.py`, `dct_tool.py`, `reflection_tool.py`, `lipsync_tool.py`, `escalation_tool.py`, `report_tool.py` | 12 |
| **core/prompts/** | `__init__.py`, `react.py`, `planning.py`, `synthesis.py` | 4 |
| **utils/** | `__init__.py`, `device.py`, `logger.py`, `video.py`, `audio.py`, `preprocessing.py`, `visualization.py`, `retry.py`, `report_generator.py` | 9 |
| **scripts/** | `__init__.py`, `check_models.py`, `download_models.py`, `update_models.py`, `benchmark.py`, `generate_synthetic.py`, `run_demo.py`, `run_all_tests.py`, `test_platform.py`, `release.py` | 10 |
| **tests/** | `__init__.py`, `conftest.py`, `test_tools.py`, `test_tools_detailed.py`, `test_agent.py`, `test_agent_detailed.py`, `test_integration.py`, `test_edge_cases.py`, `test_memory.py`, `test_prompts.py`, `test_video.py`, `test_audio.py`, `test_preprocessing.py`, `test_visualization.py`, `test_config.py`, `test_device.py`, `test_cli.py`, `test_web.py`, `test_privacy.py`, `test_api.py` | 20 |
| **api/** | `server.py`, `routes.py`, `schemas.py` | 3 |
| **docs/** | `api.md`, `architecture.md`, `tools.md` | 3 |
| **TOTAL** | | **~86 files** |

---

> **Note:** This plan breaks the 10-week effort into standalone daily tasks. Each day's AI prompt is self-contained and can be executed independently. The prompts reference exact file paths, function signatures, class names, and expected outputs. When you're ready to start Day 1, just copy-paste the prompt and go!
