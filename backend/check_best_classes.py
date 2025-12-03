from ultralytics import YOLO
import os

model_path = "best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"Model Classes: {model.names}")
else:
    print(f"Model not found at {model_path}")
