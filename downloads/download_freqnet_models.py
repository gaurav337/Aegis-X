"""
Aegis-X FreqNet Model Downloader
---------------------------------
Downloads ResNet-50 backbone weights for FreqNet tool.
Stores locally in models/freqnet/ for offline use.

Usage:
    python downloads/download_freqnet_models.py
"""

import os
import torch
import torchvision.models as models
from pathlib import Path

# Configuration
ROOT_DIR = Path(__file__).parent.parent
MODELS_DIR = ROOT_DIR / "models"
FREQNET_DIR = MODELS_DIR / "freqnet"

def ensure_directory(path: Path):
    """Creates directory if it doesn't exist"""
    path.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directory ensured: {path}")

def download_resnet50() -> bool:
    """Downloads ResNet-50 ImageNet weights"""
    filename = "resnet50_imagenet.pth"
    target_path = FREQNET_DIR / filename
    
    if target_path.exists():
        print(f"⏭️  Skipping {filename} (already exists)")
        print(f"   Location: {target_path}")
        return True
    
    print(f"⬇️  Downloading: ResNet-50 ImageNet pretrained weights")
    print(f"   Target: {target_path}")
    
    try:
        # Load model with pretrained weights (downloads to torch cache)
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Extract state dict
        state_dict = model.state_dict()
        
        # Save to our models/ directory
        torch.save(state_dict, target_path)
        
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
    print("🚀 Aegis-X FreqNet Model Downloader")
    print("=" * 50)
    
    ensure_directory(MODELS_DIR)
    ensure_directory(FREQNET_DIR)
    
    print(f"\n📦 Processing: ResNet-50 backbone")
    print("-" * 50)
    success = download_resnet50()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 FreqNet models ready for offline inference!")
        print(f"   Models location: {FREQNET_DIR.absolute()}")
    else:
        print("⚠️  Download failed. Check errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
