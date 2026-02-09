
🛡️ Aegis-X: The Offline Deepfake Forensic Engine
Next-Gen Artifact Detection using AIMv2, Intrinsic Dimensionality & rPPG (Edge-Optimized)
📖 Executive Summary
Aegis-X is a military-grade, fully offline digital forensic tool designed to detect hyper-realistic AI-generated content (Flux.1, Midjourney v6, Sora).
Unlike traditional detectors that rely on black-box CNNs (like ResNet) to classify images based on visual patterns, Aegis-X employs a Hybrid Neuro-Symbolic Architecture. It fuses Generative Entropy Analysis (using Apple's AIMv2) with Hard Forensic Physics (Corneal Reflections, Wavelet Analysis) and Biological Signal Processing (rPPG Heartbeat Detection).
Critically, this system is engineered for Data Sovereignty. It runs entirely on consumer hardware (4GB VRAM) using advanced GGUF Split-Computing, ensuring sensitive forensic data never leaves the device.
⚡ The Core Innovation: Why Aegis-X?
Standard Deepfake detectors fail because they look for "artifacts" that Generative AI models are rapidly learning to hide. Aegis-X does not look for artifacts; it looks for Physical and Statistical Impossibilities.
1. The "Constraint is the Feature" (4GB VRAM Optimization)
Most SOTA Vision-Language Models (VLMs) require 24GB+ A100 GPUs. Aegis-X democratizes forensic AI by implementing a custom Split-Computing Pipeline:
 * Layers 1-20 (GPU): Acceleration for high-dimensional tensor operations.
 * Layers 21-40 (System RAM): Offloaded processing for memory-heavy weights.
 * Result: We run MiniCPM-V 2.6 (8 Billion Parameters) on a standard RTX 3050 / GTX 1650 Mobile.
2. Multi-Modal Verification (The "Judge & Jury" Approach)
We do not rely on a single probability score. We use a "Tri-Verification" system:
 * Physics Engine: Calculates geometric consistency of light reflections in the eyes.
 * Signal Engine: Analyzes high-frequency "checkerboard" artifacts using Wavelets.
 * Geometry Engine: Measures the Intrinsic Dimensionality of the data manifold.
🏗️ System Architecture
Aegis-X operates in two distinct modes depending on the input media.
🖼️ Mode A: Image Forensics (The "Deep Scan")
 * Corneal Specular Highlight Consistency:
   * Method: Extracts corneal reflection vectors using dlib landmarks.
   * Logic: In real physics, light vectors from both eyes must be parallel. Diffusion models often generate them independently.
   * Threshold: >5^\circ deviation = FAKE.
 * Wavelet Packet Decomposition (WPD):
   * Method: Haar Wavelet Transform on the grayscale spectrum.
   * Logic: Detects invisible frequency anomalies in the "High-High" (HH) sub-bands caused by upsamplers.
 * AIMv2 Entropy Analysis:
   * Method: Apple Image Model (AIMv2) Autoregressive backbone.
   * Logic: Measures pixel-level probability. High "surprise" (entropy) in texture regions indicates generative noise.
🎥 Mode B: Video Forensics (The "Bio-Liveness" Engine)
 * rPPG (Remote Photoplethysmography):
   * Method: Eulerian Video Magnification on the Green Channel of facial skin.
   * Logic: Real humans have a pulse (60-100 BPM). Generative Video (Sora/Runway) has zero biological signal.
   * Verdict: No Heartbeat Peak in FFT = FAKE.
📊 Technical Comparison
| Feature | Standard CNN Detector | Aegis-X (Ours) |
|---|---|---|
| Detection Method | Visual Pattern Matching | Physics + Signal + Entropy |
| Explainability | Black Box ("99% Fake") | Reasoned Verdict ("Fake: Inconsistent Lighting") |
| Gen-AI Robustness | Low (Fails on new models) | High (Physics laws don't change) |
| Video Analysis | Frame-by-Frame (Slow) | rPPG Biological Check (Fast) |
| Hardware Req | Cloud API / High VRAM | 4GB Consumer GPU (Edge) |
🛠️ Installation (100% Offline)
Prerequisites
 * Hardware: NVIDIA GPU (4GB+ VRAM) with CUDA 11.x/12.x.
 * OS: Linux (Ubuntu 22.04) or Windows 11 (WSL2 recommended).
 * Python: 3.10+.
1. Clone & Setup
git clone https://github.com/yourusername/aegis-x.git
cd aegis-x

# Create isolated environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python dlib scipy scikit-dimension timm

2. Install GPU-Accelerated Llama (Crucial)
To enable split-computing, llama-cpp-python must be compiled with CUDA support.
# Linux
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python

# Windows (PowerShell)
$env:CMAKE_ARGS = "-DLLAMA_CUBLAS=on"
pip install llama-cpp-python

3. Download Models (Manual Offline Setup)
Place these in the /models directory:
 * The Brain: MiniCPM-V-2.6-Q4_K_M.gguf (Download from HuggingFace).
 * The Eye: aimv2-large-patch14-224.pth (Auto-handled by timm or download manually).
🚀 Usage
Run the Forensic Dashboard
streamlit run app.py

Programmatic Usage (Python)
from aegis_core import ForensicEngine

# Initialize with Split-Computing (20 layers on GPU)
engine = ForensicEngine(model_path="models/MiniCPM.gguf", n_gpu_layers=20)

# Analyze an Image
report = engine.analyze_image("suspect_image.jpg")
print(report)
# Output:
# {
#   "verdict": "FAKE",
#   "confidence": 0.98,
#   "reasons": [
#       "Corneal Reflection Mismatch (Left vs Right > 12 deg)",
#       "Low Intrinsic Dimensionality (LID: 14.2)"
#   ]
# }

🔮 Future Roadmap & Research
 * Phase 1 (Current): Static Image Physics & Entropy Analysis.
 * Phase 2 (In Progress): Real-time rPPG video heartbeat detection.
 * Phase 3: Integration of 3D Morphable Models (3DMM) to detect geometry inconsistencies in head pose.
📜 License & Acknowledgements
This project is licensed under the MIT License.
Acknowledgements:
 * OpenBMB for MiniCPM-V.
 * Apple Machine Learning Research for AIMv2.
 * The llama.cpp Team for enabling LLMs on consumer hardware.
"Deepfakes simulate reality. Aegis-X validates it."
