"""
Aegis-X Model Downloader
------------------------
Downloads all required forensic models to local models/ directory.
Run once to enable offline inference.

Usage:
    python download/download_models.py
"""

import os
import sys
import torch
import torchvision.models as models
from pathlib import Path

# Configuration
ROOT_DIR = Path(__file__).parent.parent  # Root directory (parent of download/)
MODELS_DIR = ROOT_DIR / "models"
SBI_DIR = MODELS_DIR / "sbi"

# Model URLs (PyTorch Hub / torchvision default weights)
MODEL_CONFIGS = {
    "efficientnet_b4": {
        "filename": "efficientnet_b4.pth",
        "description": "EfficientNet-B4 Backbone for SBI Detector",
        "load_fn": lambda: models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    },
    # Add more models here as needed for other tools
    # "resnet50": {
    #     "filename": "resnet50.pth",
    #     "description": "ResNet-50 for XYZ Tool",
    #     "load_fn": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # }
}

def ensure_directory(path: Path):
    """Creates directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory ensured: {path}")

def download_model(model_name: str, config: dict) -> bool:
    """
    Downloads a single model to the models/ directory.
    Returns True if successful, False otherwise.
    """
    filename = config["filename"]
    description = config["description"]
    load_fn = config["load_fn"]
    
    target_path = SBI_DIR / filename
    
    # Check if already downloaded
    if target_path.exists():
        print(f"⏭️  Skipping {filename} (already exists)")
        print(f"   Location: {target_path}")
        return True
    
    print(f"⬇️  Downloading: {description}")
    print(f"   Target: {target_path}")
    
    try:
        # Load model with pretrained weights (this triggers download to torch cache)
        model = load_fn()
        
        # Extract state dict
        state_dict = model.state_dict()
        
        # Save to our models/ directory
        torch.save(state_dict, target_path)
        
        # Verify file was created
        if target_path.exists():
            file_size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"✅ Successfully downloaded: {filename} ({file_size_mb:.2f} MB)")
            return True
        else:
            print(f"❌ File not created: {target_path}")
            return False
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    print("🚀 Aegis-X Model Downloader")
    print("=" * 50)
    
    # Ensure directories exist
    ensure_directory(MODELS_DIR)
    ensure_directory(SBI_DIR)
    
    # Track results
    results = {}
    
    # Download each model
    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n📦 Processing: {model_name}")
        print("-" * 50)
        results[model_name] = download_model(model_name, config)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Download Summary")
    print("=" * 50)
    
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    for model_name, success in results.items():
        status = "✅ OK" if success else "❌ FAILED"
        print(f"  {model_name}: {status}")
    
    print(f"\nTotal: {successful}/{total} models downloaded")
    
    if successful == total:
        print("\n🎉 All models ready for offline inference!")
        print(f"   Models location: {MODELS_DIR.absolute()}")
        sys.exit(0)
    else:
        print("\n⚠️  Some models failed to download. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()