import os
from transformers import SiglipModel, SiglipConfig

def download_siglip_offline():
    model_id = "google/siglip-base-patch16-224"
    save_dir = "./models/siglip-base-patch16-224"
    
    print(f"📁 Creating directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"⬇️  Downloading {model_id} from Hugging Face...")
    
    # 1. Download and save the configuration
    config = SiglipConfig.from_pretrained(model_id)
    config.save_pretrained(save_dir)
    
    # 2. Download and save the PyTorch model weights
    # We download the standard weights. We will cast to FP16 dynamically during inference.
    model = SiglipModel.from_pretrained(model_id)
    model.save_pretrained(save_dir)
    
    print("✅ Download complete! SigLIP is now fully offline.")
    print(f"Files saved in: {os.path.abspath(save_dir)}")

if __name__ == "__main__":
    download_siglip_offline()