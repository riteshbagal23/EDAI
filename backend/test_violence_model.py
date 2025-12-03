from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# Define paths
MODEL_PATH = os.path.abspath("best (2).pt")
IMG_PATH = os.path.abspath("uploads/strelka-street-fight.jpg")

print(f"Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print(f"Loading image from: {IMG_PATH}")
if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image not found at {IMG_PATH}")

# Run inference
print("Running inference...")
results = model.predict(source=IMG_PATH, imgsz=224, save=True, project="runs/predict", name="exp")
res = results[0]

# Extract probabilities and names robustly
probs = None
top1_idx = None
top1_conf = None

if hasattr(res, "probs"):
    # Ultralytics Probs object
    p = res.probs
    # top1 index and confidence
    if hasattr(p, "top1"):
        top1_idx = int(p.top1)
    if hasattr(p, "top1conf"):
        top1_conf = float(p.top1conf)
    
    # convert probs to numpy if needed
    try:
        probs = p.cpu().numpy()
    except Exception:
        try:
            probs = p.numpy()
        except Exception:
            probs = None
else:
    print("WARNING: res.probs is missing. Checking other attributes...")
    # Fallback logic if needed, but for classification models probs should be there.

# Get class names
names = res.names if hasattr(res, "names") else None
if names is None:
    raise RuntimeError("Class names not found on result object.")

# If top1_idx not available (edge case), compute from probs array
if top1_idx is None and probs is not None:
    if hasattr(probs, "data"):
         arr = probs.data.cpu().numpy().ravel()
    else:
         arr = np.array(probs).ravel()
    top1_idx = int(arr.argmax())
    top1_conf = float(arr[top1_idx])

if top1_idx is not None:
    top1_label = names[top1_idx]
    print("\n=== PREDICTION ===")
    print("Top-1 class:", top1_label)
    print("Confidence :", round(top1_conf, 4))
else:
    print("\n=== PREDICTION FAILED ===")
    print("Could not determine top1 class.")

print(f"Annotated image saved to: {res.save_dir}")
