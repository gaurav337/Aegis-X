# Aegis-X — v2

Aegis-X is a **Python** repository that contains the building blocks for a media and deepfake forensics pipeline, including:

- A tool interface (`BaseForensicTool`) and a tool registry (`ToolRegistry`)
- Multiple forensic tools across CPU (physics / heuristics) and GPU (pre-trained neural networks) phases
- An "agent" orchestration layer with early-stopping logic
- A SQLite-backed memory system implementation for storing/querying prior cases
- Utilities for preprocessing, video/image I/O, VRAM lifecycle management, logging, and an Ollama client

> **v2 Note:** The three randomly-initialised GPU tools from v1 (SigLIP adapter, FreqNetDual, SBI random head) have been replaced with publicly available pre-trained checkpoints. No training is required to run the full pipeline. See [Architecture Changes (v2)](#architecture-changes-v2) below.

---

## Installation

### Requirements

- Python **>= 3.10** is required for full compatibility with the GPU dependencies.
- CUDA-capable GPU with **>= 3 GB VRAM** (for GPU tools)

### Setup Virtual Environments

This project uses two separate virtual environments to isolate heavy GPU dependencies from the main web application.

1. **Create the virtual environments**:
```bash
python3.10 -m venv venv_main
python3.10 -m venv venv_gpu
```

2. **Install requirements for the main application**:
```bash
source venv_main/bin/activate
pip install -r requirements-main.txt
deactivate
```

3. **Install requirements for the GPU models**:
```bash
source venv_gpu/bin/activate
pip install -r requirements-gpu.txt
deactivate
```

> **Note**: If you are not using the dual environment setup, you can simply run `pip install -r requirements.txt`.

### Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
cp .env.example .env
```

| Variable | Default | Purpose |
|---|---|---|
| `AEGIS_MODEL_DIR` | `models/` | Root directory for all model weight files |
| `AEGIS_DEVICE` | `auto` | `cuda`, `cpu`, or `auto` |
| `OLLAMA_ENDPOINT` | `http://localhost:11434` | Local Ollama LLM server |
| `AEGIS_VRAM_THRESHOLD` | `3.5` | Minimum free VRAM (GB) to run GPU tools |

---

## One-Time Model Setup

After installing Python dependencies, run the weight downloader once:

```bash
# Activate the GPU environment first if using the dual setup
source venv_gpu/bin/activate

# Downloads CLIP-ViT-L/14, UnivFD probe, XceptionNet, CNNDetect (~5.5 GB total)
python downloads/download_new_weights.py

# If CLIP backbone is already present locally:
python downloads/download_new_weights.py --skip-clip

# Verify existing files without downloading:
python downloads/download_new_weights.py --check-only

deactivate
```

**SBI weights** must be downloaded separately from `mapooon/SelfBlendedImages` (GitHub) and placed at `models/sbi/efficientnet_b4.pth`.

---

## Running the Application

To run the web interface, activate the main environment and start the Streamlit app:
```bash
source venv_main/bin/activate
streamlit run run_web.py
```

---

## Project Structure

```text
core/
├── base_tool.py            # Abstract base class for all tools
├── agent.py                # Orchestration: CPU → GPU → Ensemble → LLM
├── early_stopping.py       # Evidential subjective logic gating
├── memory.py               # SQLite-backed memory system implementation
├── forensic_summary.py     # Builds grounded prompts for Ollama LLM
├── config.py               # Typed configuration dataclasses
├── data_types.py           # ToolResult and related contracts
├── exceptions.py           # Custom exception hierarchy
└── tools/
    ├── registry.py             # Tool manifest and dispatch
    ├── c2pa_tool.py            # [CPU] C2PA provenance check
    ├── rppg_tool.py            # [CPU] Remote photoplethysmography
    ├── dct_tool.py             # [CPU] JPEG double-quantisation (plus dct_tool2.py)
    ├── geometry_tool.py        # [CPU] Anthropometric ratio checks
    ├── illumination_tool.py    # [CPU] Lighting consistency
    ├── corneal_tool.py         # [CPU] Corneal catchlight divergence
    ├── univfd_tool.py          # [GPU] UnivFD — generative AI detector  ← NEW v2
    ├── xception_tool.py        # [GPU] XceptionNet — face-swap detector  ← NEW v2
    ├── sbi_tool.py             # [GPU] Self-blended images boundary detector
    ├── freqnet_tool.py         # [GPU] CNNDetect + FADHook frequency analysis  ← REWRITTEN v2
    └── freqnet/
        ├── preprocessor.py     # DCTPreprocessor + SpatialPreprocessor (pure math)
        ├── fad_hook.py         # FADHook — DCT band energy analysis (pure math)
        └── calibration.py      # CalibrationManager for Z-score baselines

utils/
├── vram_manager.py         # GPU memory lifecycle (synchronize → del → gc → cache)  ← UPDATED v2
├── preprocessing.py        # MediaPipe face extraction, frame sampling
├── ensemble.py             # Weighted score aggregation
├── thresholds.py           # Central constants registry
├── video.py / image.py     # Media I/O helpers
├── ollama_client.py        # HTTP client for local Ollama LLM
└── logger.py               # Structured logging setup

downloads/
├── download_new_weights.py     # One-time v2 weight downloader  ← NEW v2
├── download_model.py           # Generic model downloader
├── download_freqnet_models.py  # Legacy FreqNet downloader
└── siglip.py                   # (Deprecated — SigLIP no longer used)

test_data/                      # SQLite databases used for testing/development
├── test_memory.db
├── sig_test.db
└── import_test.db
```

---

## Tool Manifest (v2)

All tools run sequentially. GPU tools unload completely before the next one loads. Peak VRAM is the single largest model (~1.8 GB for UnivFD), never additive.

| Phase | Tool | Weight | Trust | GPU | Threat Target | Checkpoint Source |
|---|---|---|---|---|---|---|
| CPU | `check_c2pa` | 0.05 | 2 | No | C2PA provenance | — (library) |
| CPU | `run_rppg` | 0.06 | 2 | No | Biological liveness | — (math) |
| CPU | `run_dct` | 0.07 | 1 | No | JPEG double-quant | — (math) |
| CPU | `run_geometry` | 0.18 | 3 | No | Anthropometric ratios | — (math) |
| CPU | `run_illumination` | 0.05 | 1 | No | Lighting consistency | — (math) |
| CPU | `run_corneal` | 0.07 | 2 | No | Catchlight divergence | — (math) |
| GPU | `run_univfd` | 0.15 | 2 | Yes | Generative AI (FLUX / Midjourney / SD / DALL-E) | `openai/clip-vit-large-patch14` + `ojha-group/UnivFD` |
| GPU | `run_xception` | 0.10 | 2 | Yes | Face-swap / reenactment (DeepFaceLab, FaceSwap, Face2Face) | `HongguLiu/Deepfake-Detection` |
| GPU | `run_sbi` | 0.18 | 3 | Yes | Blend boundary seam (face compositing) | `mapooon/SelfBlendedImages` |
| GPU | `run_freqnet` | 0.09 | 1 | Yes | DCT frequency spectrum (GAN / diffusion noise floor) | `PeterWang4158/CNNDetect` |

**Weight sum: 1.00**

---

## Architecture Changes (v2)

### What was replaced

| v1 component | Problem | v2 replacement |
|---|---|---|
| `siglip_adapter_tool.py` — custom 3-stage adapter (Dynamic Spatial Pooling + Cross-Patch Attention + LSE head) | Randomly initialised — required weeks of fine-tuning on 100K+ images | `univfd_tool.py` — CLIP-ViT-L/14 + 4 KB linear probe (pre-trained by Ojha et al. CVPR 2023) |
| `FreqNetDual` in `freqnet_tool.py` — dual ResNet-50 with custom 64-channel DCT conv1 | F3Net checkpoint architecture mismatch — `load_state_dict()` fails | `_CNNDetect` in new `freqnet_tool.py` — standard ResNet-50 + `Linear(2048,1)`, Wang et al. CVPR 2020 |
| No XceptionNet tool | No coverage for legacy FaceForensics-style face-swaps | `xception_tool.py` — Xception pretrained on FaceForensics++ |

### What was preserved (untouched)

- All six CPU tools (`c2pa`, `rppg`, `dct`, `geometry`, `illumination`, `corneal`)
- `core/tools/freqnet/` subpackage — `DCTPreprocessor`, `FADHook`, `CalibrationManager` are pure math and unchanged
- `sbi_tool.py` — architecture already matched official SBI weights
- `core/agent.py` (Orchestration logic)
- `utils/ensemble.py`
- `core/early_stopping.py`
- `core/forensic_summary.py`
- `core/memory.py`

### VRAM lifecycle improvement

`utils/vram_manager.py` now calls `torch.cuda.synchronize()` before `del model`. Without this, PyTorch's asynchronous CUDA execution can still be in-flight when Python drops the model reference, causing silent CUDA errors or race conditions.

---

## Pipeline Flow

```text
Media input (image or video)
        │
        ▼
Preprocessor (MediaPipe face extraction, frame sampling)
        │
        ▼
CPU Phase — runs all 6 tools sequentially, zero VRAM
  ├── C2PA           → provenance / short-circuit if hardware-signed
  ├── rPPG           → biological liveness (video only)
  ├── DCT            → JPEG double-quantisation artefacts
  ├── Geometry       → anthropometric ratio violations
  ├── Illumination   → lighting direction consistency
  └── Corneal        → catchlight divergence in both eyes
        │
        ▼
Early Stopping check (Evidential Subjective Logic)
  → HALT if C2PA hardware-signed, or mathematically locked REAL/FAKE
        │
        ▼
GPU Phase — tools execute and unload one at a time
  ├── UnivFD         → CLIP-ViT-L/14 + linear probe (generative AI)
  ├── XceptionNet    → FaceForensics++ weights (face-swap/reenactment)
  ├── SBI            → EfficientNet-B4 + GradCAM (blend boundary)
  └── FreqNet        → CNNDetect ResNet-50 + FADHook DCT stats
        │
        ▼
Ensemble scoring (weighted average, conflict detection, C2PA override)
        │
        ▼
LLM synthesis (Ollama / Phi-3 Mini — explains evidence in natural language)
        │
        ▼
Verdict: REAL / FAKE / INCONCLUSIVE + evidence summary
```

---

## Configuration Notes

- There is **no** `config.yaml` currently in the repository root.
- Configuration is driven by `core/config.py`, `utils/thresholds.py`, and environment variables (`.env`).
- Thresholds and ensemble weights are centralised in `utils/thresholds.py`.
- The tool registry (`core/tools/registry.py`) is the single source of truth for tool weights, trust tiers, and GPU flags.

---

## Benchmarking Datasets

To evaluate the pipeline against the modern threat landscape:

| Dataset | Threat class | Tool(s) targeted |
|---|---|---|
| GenImage | Pure text-to-image AI (Midjourney, SD, DALL-E) | `run_univfd` |
| ArtiFact | Multi-generator with real-world compression | `run_univfd`, `run_freqnet` |
| FaceForensics++ | Face-swap and reenactment | `run_xception`, `run_sbi` |
| ForgeryNet | Surgical face edits, morphs | `run_sbi` |
| WildDeepfake | Internet-scraped deepfakes | Full pipeline |
| DiffusionForensics | Diffusion outputs with heavy compression | `run_freqnet` |

---

## Development Notes

- `NOTES.md` — per-sprint design decisions and bug fix log
- `diagnostics_day14.py` — run to verify all tools import and initialise correctly
- `day18_test_results.txt` — recorded test outputs from v1 integration testing

---

## License

No `LICENSE` file is currently present. If you intend the project to be open-source, consider adding a license file (MIT / Apache-2.0 / GPL) and updating this section.
