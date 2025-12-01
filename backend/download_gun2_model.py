#!/usr/bin/env python3
"""
Download second gun detection model from Hugging Face
"""
from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path

# Configuration
MODEL_REPO = "Subh775/Firearm_Detection_Yolov8n"
MODEL_FILENAME = "weights/best.pt"
OUTPUT_FILENAME = "gun2.pt"

print(f"Downloading model from {MODEL_REPO}...")
print(f"This may take a few minutes depending on your connection...")

try:
    # Download model from Hugging Face Hub
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME
    )
    
    print(f"✓ Model downloaded to: {model_path}")
    
    # Copy to current directory as gun2.pt
    output_path = Path(__file__).parent / OUTPUT_FILENAME
    shutil.copy(model_path, output_path)
    
    print(f"✓ Model saved as: {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    print("\n✅ SUCCESS! Model ready to use.")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    exit(1)
