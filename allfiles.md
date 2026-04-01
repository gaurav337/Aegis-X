# Aegis-X — Complete Repository Documentation (allfiles.md) — v2

> Generated automatically. Includes full README, every file, every class, every function —
> what it does, what it **cannot** do, and how to counter its limitations.
>
> **v2 Update:** Three GPU tools were replaced or rewritten. See [Architecture Changes (v2)](#architecture-changes-v2) and the updated tool entries below. All CPU tool documentation is unchanged.

---

## Part 1 — Original README

*(See README.md — reproduced above for completeness in the repo.)*

---

## Architecture Changes (v2)

### Summary of changes

| File | Action | What changed |
|---|---|---|
| `core/tools/univfd_tool.py` | **NEW** | Replaces `siglip_adapter_tool.py`. CLIP-ViT-L/14 + UnivFD linear probe. |
| `core/tools/xception_tool.py` | **NEW** | No prior equivalent. XceptionNet pretrained on FaceForensics++. |
| `core/tools/freqnet_tool.py` | **REWRITTEN** | FreqNetDual removed. CNNDetect ResNet-50 + existing FADHook. |
| `utils/vram_manager.py` | **UPDATED** | Added `torch.cuda.synchronize()` before `del model`. Explicit device pinning. |
| `core/tools/registry.py` | **UPDATED** | Tool manifest rebalanced. `run_siglip_adapter` → `run_univfd` + `run_xception`. |
| `downloads/download_new_weights.py` | **NEW** | One-time weight downloader for all v2 checkpoints. |
| `core/tools/siglip_adapter_tool.py` | **DELETED** | Replaced by `univfd_tool.py`. |

### Files completely unchanged

All six CPU tools, `sbi_tool.py`, the `core/tools/freqnet/` subpackage (DCTPreprocessor, FADHook, CalibrationManager), `core/agent.py` (one manual edit required — see below), `utils/ensemble.py`, `core/early_stopping.py`, `core/forensic_summary.py`, `utils/preprocessing.py`.

### Manual edit required in agent.py

The SBI-skip gate in `agent.py._run_gpu_phase()` must be updated to reference `run_univfd` instead of the removed `run_siglip_adapter`:

```python
# BEFORE (v1 — broken, references removed tool)
siglip_result = completed_results.get("run_siglip_adapter")
if siglip_result and siglip_result.score > 0.70:
    continue  # skip run_sbi

# AFTER (v2 — correct)
univfd_result = completed_results.get("run_univfd")
if univfd_result and univfd_result.fake_score > 0.70:
    logger.info("Skipping SBI — UnivFD strong generative-AI signal")
    continue  # skip run_sbi
```

---

## Repository Contents (v2)

Top-level files/directories:

- `.env.example`
- `.gitignore`
- `NOTES.md`
- `README.md`  *(updated for v2)*
- `__init__.py`
- `day18_test_results.txt`
- `diagnostics_day14.py`
- `pyproject.toml`
- `requirements.txt`  *(add `timm>=0.9.0`)*
- `core/`
- `utils/`
- `downloads/`
- `test_data/`

---

## Tool Manifest (v2) — weights sum to 1.00

| Tool name | Weight | Category | Trust Tier | GPU? |
|---|---|---|---|---|
| `check_c2pa` | 0.05 | PROVENANCE | 2 | No |
| `run_rppg` | 0.06 | BIOLOGICAL | 2 | No |
| `run_dct` | 0.07 | FREQUENCY | 1 | No |
| `run_geometry` | 0.18 | GEOMETRIC | 3 | No |
| `run_illumination` | 0.05 | PHYSICS | 1 | No |
| `run_corneal` | 0.07 | BIOLOGICAL | 2 | No |
| `run_univfd` | 0.15 | SEMANTIC | 2 | Yes |
| `run_xception` | 0.10 | FACE_SWAP | 2 | Yes |
| `run_sbi` | 0.18 | GENERATIVE | 3 | Yes |
| `run_freqnet` | 0.09 | FREQUENCY | 1 | Yes |

---

## Part 2 — Core Infrastructure

*(Unchanged from v1. See original allfiles.md Part 2 for full documentation of `core/data_types.py`, `core/exceptions.py`, `core/config.py`, `core/base_tool.py`, `core/agent.py`, `core/early_stopping.py`, `core/forensic_summary.py`.)*

---

## Part 3 — CPU Forensic Tools

*(Unchanged from v1. See original allfiles.md Part 3 for full documentation of `c2pa_tool.py`, `dct_tool.py`, `geometry_tool.py`, `illumination_tool.py`, `corneal_tool.py`.)*

---

## Part 4 — GPU Forensic Tools (v2 — Updated)

---

### File: `core/tools/univfd_tool.py` — UnivFD Generative-AI Detection Tool {#univfd}

**Status: NEW in v2. Replaces `siglip_adapter_tool.py`.**

#### What It Does

Runs a face crop through the frozen **CLIP-ViT-L/14** vision encoder (Google / OpenAI), then applies a **pre-trained linear probe** (768 → 1 logit) to score deepfake probability. The probe was trained by Ojha et al. (CVPR 2023) on CLIP features of real vs AI-generated images. No custom training required.

Reference: *"Towards Universal Fake Image Detection Exploiting Spurious Artifacts"*, Ojha et al., CVPR 2023.

#### Architecture

| Stage | Component | Output | Precision |
|---|---|---|---|
| 1 | CLIPVisionModelWithProjection (frozen) | (1, 768) | FP16 |
| 2 | L2 normalise | (1, 768) | FP32 (cast after backbone) |
| 3 | `_LinearProbe`: Linear(768, 1) | (1, 1) logit | FP32 |
| 4 | Sigmoid | scalar [0, 1] | FP32 |

#### `class UnivFDTool(BaseForensicTool)`

| Method | What it does |
|---|---|
| `tool_name` | Returns `"run_univfd"` |
| `setup()` | Checks for backbone dir and probe path; logs warnings if absent; sets runtime handles to None |
| `_load_model()` | Loads CLIP backbone (FP16, frozen); loads probe with 3-format key detection (PyTorch / sklearn / raw); moves both to `cuda:0` |
| `unload()` | CPU offload → `synchronize()` → `del` → `gc.collect()` → `empty_cache()` |
| `_crop_to_tensor(crop, device)` | Uses `CLIPImageProcessor` for correct 224×224 normalisation (CLIP stats, not ImageNet) |
| `_score_single_crop(crop, device)` | FP16 backbone → FP32 cast → L2 norm → linear probe → sigmoid |
| `_run_inference(input_data)` | Iterates tracked_faces; TTA (original + H-flip, max-pool); worst-face policy |

#### Pre-trained weights

- **Backbone:** `models/clip-vit-large-patch14/` (HuggingFace snapshot, ~3.5 GB)
- **Probe:** `models/univfd/probe.pth` (ojha-group/UnivFD GitHub releases, ~4 KB)

#### VRAM budget

| State | VRAM |
|---|---|
| Loaded (FP16 backbone + FP32 probe) | ~1.8 GB |
| After `unload()` | 0 MB |

#### ✅ What It CAN Detect

- **FLUX 1.x, Midjourney v5/v6, Stable Diffusion 3, DALL-E 3** — CLIP features capture the generalised "vibes" of AI generation
- **GAN faces** (StyleGAN, InsightFace) — mid-layer CLIP features encode unnatural texture statistics
- **Any generator not seen during probe training** — CLIP's universal representations generalise well

#### ❌ What It CANNOT Detect

- **Face-swap / reenactment** (DeepFaceLab, FaceSwap, Face2Face) — no blend boundary signal. XceptionTool and SBITool handle these.
- **Adversarial examples crafted against CLIP-ViT** — targeted perturbations can fool the ViT backbone
- **CPU-only machines** — falls back to abstention (score=0.0, confidence=0.0)

#### 🔧 Backward-compat shim

`result.details["siglip_score"]` mirrors `result.fake_score` so that any `ensemble.py` code reading the old `siglip_score` key continues to work during the transition.

---

### File: `core/tools/xception_tool.py` — XceptionNet Face-Swap Detector {#xception}

**Status: NEW in v2. No prior equivalent.**

#### What It Does

Runs a face crop through **XceptionNet** (Chollet 2017), pretrained on **FaceForensics++** (Rossler et al. ICCV 2019). XceptionNet is the standard academic baseline for detecting legacy face-swap and reenactment deepfakes. Scores `softmax[:, 1]` = P(fake).

#### `class XceptionTool(BaseForensicTool)`

| Method | What it does |
|---|---|
| `tool_name` | Returns `"run_xception"` |
| `setup()` | Checks for weight file at `models/xception/xception_deepfake.pth`; logs warning if absent |
| `_load_model()` | `timm.create_model('xception', num_classes=2)`; loads checkpoint with `_remap_keys()`; falls back to ImageNet backbone + zero head if load fails |
| `_remap_keys(state_dict)` | Strips `module.` / `model.` prefixes; renames `last_linear.*` → `fc.*` for HongguLiu→timm compatibility |
| `unload()` | CPU offload → synchronize → del → gc → cache clear |
| `_score_crop(crop, device)` | Resize to 299×299 (Lanczos4); ImageNet normalise; FP16 forward; softmax[:, 1] |
| `_run_inference(input_data)` | Iterates tracked_faces; TTA (original + H-flip, max-pool); worst-face policy |

#### Pre-trained weights

- `models/xception/xception_deepfake.pth` (HongguLiu/Deepfake-Detection GitHub releases, ~80 MB)

#### VRAM budget

| State | VRAM |
|---|---|
| Loaded (FP16) | ~350 MB |
| After `unload()` | 0 MB |

#### ✅ What It CAN Detect

- **DeepFaceLab, FaceSwap, DeepFakes** — direct training objective on FaceForensics++
- **Face2Face, NeuralTextures** — reenactment artefacts in texture statistics
- **FaceApp, Reface, ZAO** — consumer-grade compositing artefacts

#### ❌ What It CANNOT Detect

- **Fully-synthetic generative AI** (Midjourney, SD, DALL-E) — not in FaceForensics++ training distribution; UnivFDTool covers this
- **Very high-quality professional face swaps** with feathered masks
- **CPU-only machines** — abstains

---

### File: `core/tools/sbi_tool.py` — Self-Blended Images Blend Boundary Detector {#sbi}

*(Unchanged from v1. Documentation unchanged. Weights still required from `mapooon/SelfBlendedImages`.)*

**Status in v2:** Architecture unchanged. Official EfficientNet-B4 weights from `mapooon/SelfBlendedImages` must be placed at `models/sbi/efficientnet_b4.pth`. The tool will then function correctly — no code changes needed.

---

### File: `core/tools/freqnet_tool.py` — Frequency Domain Detection Tool {#freqnet}

**Status: REWRITTEN in v2. Same filename. `FreqNetDual` class removed. `_CNNDetect` class added.**

#### What It Does

Dual-signal frequency detector:

1. **Neural stream:** `_CNNDetect` (Wang et al. CVPR 2020) — standard ResNet-50 backbone + Linear(2048, 1), pretrained on ProGAN, generalises across GAN families via shared low-level spectral statistics.
2. **Statistical stream:** `DCTPreprocessor` → `FADHook` — pure-math BT.709 DCT-II transform + frequency band energy analysis (low / mid / high via JPEG zigzag ordering). Zero weights, zero VRAM beyond temp tensors.

Final score = `0.70 × neural + 0.30 × fad`. Falls back to `1.0 × fad` if CNNDetect weights are absent.

#### `class FreqNetTool(BaseForensicTool)` — v2

| Method | What it does |
|---|---|
| `tool_name` | Returns `"run_freqnet"` |
| `setup()` | Loads `CalibrationManager`; checks for CNNDetect weight file; sets runtime handles |
| `_load_model()` | Builds `_CNNDetect`; loads `cnndetect_resnet50.pth` with `_remap_cnndetect_keys()`; falls back to ImageNet backbone if load fails |
| `_remap_cnndetect_keys(ckpt)` | Strips `module.` / `model.` prefixes; renames `fc.*` → `classifier.*` |
| `unload()` | CPU offload → synchronize → del → gc → cache clear |
| `_run_fad_analysis(crop, device)` | Instantiates `DCTPreprocessor` + `FADHook`; hooks `dct_pre._dct_conv`; runs forward pass; calls `fad_hook.analyze()`; computes fad_score from band energy excesses |
| `_run_inference(input_data)` | Expands crop 10%; neural stream (CNNDetect FP16); statistical stream (FADHook); fuses scores; returns worst-face result |

#### `class _CNNDetect(nn.Module)` — NEW in v2

```python
class _CNNDetect(nn.Module):
    def __init__(self):
        backbone = resnet50(weights=None)
        self.features    = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self.classifier  = nn.Linear(2048, 1, bias=True)

    def forward(self, x):   # x: (B, 3, 224, 224) FP16
        feat = self.features(x).flatten(start_dim=1)  # (B, 2048)
        return self.classifier(feat)                   # (B, 1) logit
```

#### What was REMOVED (v1 FreqNetDual)

- `FreqNetDual` class — dual ResNet-50 with modified `conv1` accepting 64-channel DCT input
- Frequency stream: separate ResNet backbone with custom first layer
- Fusion: `concat(2048+2048) → Linear(4096→512) → ReLU → Dropout → Linear(512→1)`
- F3Net weight loading (architecture mismatch — F3Net is a completely different model)

#### What was PRESERVED (unchanged)

- `DCTPreprocessor` (imported from `core/tools/freqnet/preprocessor.py`) — pure math
- `SpatialPreprocessor` (imported from same) — ImageNet normalisation
- `FADHook` (imported from `core/tools/freqnet/fad_hook.py`) — band energy analysis
- `CalibrationManager` (imported from `core/tools/freqnet/calibration.py`) — Z-score baseline

#### Pre-trained weights

- `models/freqnet/cnndetect_resnet50.pth` (PeterWang4158/CNNDetect GitHub releases, ~100 MB)

#### VRAM budget

| State | VRAM |
|---|---|
| CNNDetect loaded (FP16) | ~400 MB |
| FADHook-only mode (no neural) | ~50 MB (temp tensors only) |
| After `unload()` | 0 MB |

#### ✅ What It CAN Detect

- **GAN-generated faces** — irregular high-frequency patterns (CNNDetect + FADHook)
- **Diffusion model outputs** — mid-band energy elevation from denoising steps (FADHook)
- **JPEG-compressed GAN outputs** — frequency fingerprint partially survives compression

#### ❌ What It CANNOT Detect

- **Post-processed deepfakes** (heavy Gaussian blur, median filter) — frequency fingerprint destroyed
- **Frequency-domain adversarial attacks** — published attack vectors exist
- **Real images with unusual camera sensors** (heavy sharpening) — may score as fake

---

## Part 5 — Updated Utility: VRAM Manager {#vram}

### File: `utils/vram_manager.py` — VRAM Lifecycle Manager (v2)

**Status: UPDATED. Two additions. Same external API.**

#### The key change — `_cleanup()` sequence

```python
# v2 _cleanup() — correct order for async CUDA safety
model.cpu()                          # 1. Move tensors off GPU
torch.cuda.synchronize(device)       # 2. Wait for ALL pending CUDA kernels ← NEW
del model                            # 3. Drop Python reference
gc.collect()                         # 4. Run Python GC
torch.cuda.empty_cache()             # 5. Return blocks to CUDA memory pool
```

Without step 2, PyTorch's asynchronous execution model means the GPU may still be mid-computation when Python attempts to free the memory, causing silent CUDA errors or OOM collisions on the next model load.

#### New helpers in v2

| Name | What it does |
|---|---|
| `_get_free_vram_gb(device_id)` | Returns currently free VRAM in GB |
| `_get_total_vram_gb(device_id)` | Returns total VRAM in GB |
| `log_vram_status(tag, device_id)` | Logs current VRAM usage to DEBUG level |

#### `VRAMLifecycleManager` — new parameter

```python
# v2 — optional VRAM pre-flight check
with VRAMLifecycleManager(loader_fn, device_id=0, min_vram_gb=2.0) as model:
    result = model(inputs)
# Raises RuntimeError before load if free VRAM < min_vram_gb
```

---

## Part 6 — FreqNet Subpackage: `core/tools/freqnet/`

*(All files in this subpackage are UNCHANGED from v1. Full documentation preserved below.)*

### File: `core/tools/freqnet/preprocessor.py` — DCT Preprocessor

*(Unchanged. See original allfiles.md for full documentation.)*

`DCTPreprocessor`: BT.709 luma → DCT-II conv2d (64 frozen basis functions) → log1p compression → (B, 64, 28, 28).
`SpatialPreprocessor`: ImageNet normalisation with registered buffers.

### File: `core/tools/freqnet/fad_hook.py` — FAD Hook

*(Unchanged. See original allfiles.md for full documentation.)*

`FADHook`: Forward hook capturing (B, 64, 28, 28) DCT tensors. Analyses energy in 3 frequency bands (base / mid / high) using JPEG zigzag ordering. Returns `BandAnalysis` dataclass with energy ratios and Z-scores.

### File: `core/tools/freqnet/calibration.py` — Calibration Manager

*(Unchanged. See original allfiles.md for full documentation.)*

`CalibrationManager`: Loads `freqnet_fad_baseline.pt` for Z-score mode; falls back to hardcoded natural-image statistics (base=0.70, mid=0.25, high=0.05) if file absent.

---

## Part 7 — Downloads

### File: `downloads/download_new_weights.py` — v2 Weight Downloader (NEW)

One-time script to download all four v2 checkpoint files.

| Target | Source | Size |
|---|---|---|
| `models/clip-vit-large-patch14/` | `openai/clip-vit-large-patch14` via HuggingFace `snapshot_download` | ~3.5 GB |
| `models/univfd/probe.pth` | ojha-group/UnivFD GitHub releases | ~4 KB |
| `models/xception/xception_deepfake.pth` | HongguLiu/Deepfake-Detection GitHub releases | ~80 MB |
| `models/freqnet/cnndetect_resnet50.pth` | PeterWang4158/CNNDetect GitHub releases | ~100 MB |

**CLI flags:**

```bash
python downloads/download_new_weights.py --help

--model-dir   Root directory (default: models/)
--skip-clip   Skip the 3.5 GB CLIP backbone
--force       Re-download even if file exists
--check-only  Print file status without downloading
```

If the UnivFD probe URL returns 404, the script offers to generate a zero-weight probe (for pipeline testing only — scores will be 0.5).

---

## Part 8 — Configuration & Meta Files

*(Unchanged from v1. See original allfiles.md Part 7 for `pyproject.toml`, `.env.example`, `.gitignore`, `NOTES.md`, `day18_test_results.txt`, `diagnostics_day14.py`.)*

**`requirements.txt` — v2 addition required:**

```
timm>=0.9.0
huggingface-hub>=0.20.0
```

---

## Part 9 — Processing Breakdown (v2 Photo Detection)

*(Steps 1–6, CPU phase, and early stopping are unchanged from v1. GPU phase updated below.)*

### Step 7 — Tool 6: UnivFD (Replaces SigLIP Adapter)

**File:** `core/tools/univfd_tool.py`
**Phase:** GPU | **Weight:** 0.15 | **Trust Tier:** 2

1. Loads CLIP-ViT-L/14 (FP16, frozen) from `models/clip-vit-large-patch14/`.
2. Loads linear probe (FP32) from `models/univfd/probe.pth`.
3. For each tracked face: runs `CLIPImageProcessor` → 224×224 CLIP-normalised tensor → FP16 backbone → FP32 cast → L2 normalise → probe → sigmoid.
4. TTA: original + H-flip crops; max-pool scores.
5. Returns worst-face score. Writes `siglip_score` shim into `details`.
6. Unloads: CPU offload → synchronize → del → gc → cache clear.

**Agent gate:** If `score > 0.70`, agent skips `run_sbi` (fully-synthetic images have no blend boundary).

### Step 8 — Tool 7: XceptionNet (NEW in v2)

**File:** `core/tools/xception_tool.py`
**Phase:** GPU | **Weight:** 0.10 | **Trust Tier:** 2

1. `timm.create_model('xception', num_classes=2)` loaded with FaceForensics++ weights.
2. Remaps HongguLiu checkpoint keys (`last_linear.*` → `fc.*`).
3. For each face: resize to 299×299 (Lanczos4) → ImageNet normalise → FP16 forward → softmax[:, 1].
4. TTA: original + H-flip; max-pool.
5. Returns worst-face score.
6. Unloads completely.

### Step 9 — Tool 8: SBI (unchanged)

*(See original allfiles.md Step 8 for full documentation.)*

### Step 10 — Tool 9: FreqNet (v2 — CNNDetect + FADHook)

**File:** `core/tools/freqnet_tool.py`
**Phase:** GPU | **Weight:** 0.09 | **Trust Tier:** 1

1. Loads `_CNNDetect` (ResNet-50 + Linear(2048,1)) from `models/freqnet/cnndetect_resnet50.pth`.
2. Expands face crop by 10% (edge-replicate padding, no black artefacts).
3. **Neural stream:** ImageNet normalise → FP16 ResNet-50 → sigmoid → `neural_score`.
4. **Statistical stream:** `DCTPreprocessor._dct_conv` hook → `FADHook.analyze()` → band energy excess → `fad_score`.
5. `combined = 0.70 × neural + 0.30 × fad`.
6. Returns worst-face combined score + band interpretation.
7. Unloads completely.

---

## Edge Case Matrix — Photo Detection (v2 update)

| Scenario | Handled? | How |
|---|---|---|
| No face detected | ✅ | `has_face=False` → INCONCLUSIVE |
| Fully synthetic (Midjourney/DALL-E/FLUX) | ✅ | UnivFD + FreqNet + Geometry all activate |
| Face swap (FaceSwap/DFL) | ✅ | XceptionNet + SBI detect boundary |
| Reenactment (Face2Face/NeuralTextures) | ✅ | XceptionNet (new in v2) |
| C2PA-signed authentic image | ✅ | C2PA short-circuits pipeline |
| PNG deepfake (no JPEG) | ⚠️ | DCT blind but UnivFD + FreqNet still active |
| All GPU tools abstain | ✅ | CPU tools still produce ensemble score |
| VRAM OOM during GPU tool | ✅ | `run_with_vram_cleanup()` retries once |
| CNNDetect weights missing | ⚠️ | FreqNet falls back to FADHook-only (fad_score only) |
| UnivFD probe missing | ⚠️ | Tool uses random-init probe (score near 0.5, confidence 0.05) |
| SBI weights missing | ❌ | Random-init head — scores unreliable (same as v1) |
| XceptionNet weights missing | ⚠️ | Falls back to ImageNet backbone + zero head (score ~0.5) |
| Agent SBI-skip gate not updated | ⚠️ | SBI always runs (no crash — just missing optimisation) |

---

## Training & Dataset Notes (v2)

With v2, **no training is required** for the core pipeline to function. All GPU tools use pre-trained public checkpoints.

If you wish to fine-tune or calibrate individual tools, the dataset recommendations from the original allfiles.md Section 4 remain valid, with one update:

- **SigLIP adapter training** (GenImage dataset) — **no longer applicable.** UnivFD's probe is already trained. If you want to improve generative-AI detection beyond UnivFD, consider training a larger head on top of CLIP features, but this is optional.
- **FreqNet training** (Synthbuster dataset) — the CNNDetect backbone is already pretrained. If you want to fine-tune on Synthbuster, you can adapt the `freqnet_dataset.py` augmentation script, but the out-of-the-box checkpoint already generalises well across GAN families.
- **SBI training / fine-tuning** (ForgeryNet + CocoGlide) — still recommended once official SBI weights are confirmed to load. Use the `train_sbi.py` script with the exact crop pipeline matching inference (1.15× / 1.25× BORDER_CONSTANT).
