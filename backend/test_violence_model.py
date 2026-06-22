import cv2
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIOLENCE_MODEL_PATH = "best (2).pt"

def test_violence_model():
    try:
        logger.info(f"Loading violence model from {VIOLENCE_MODEL_PATH}")
        model = YOLO(VIOLENCE_MODEL_PATH)
        logger.info(f"Model loaded. Classes: {model.names}")
        
        # Create a dummy image (black)
        import numpy as np
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        logger.info("Running prediction on dummy image...")
        results = model.predict(dummy_img, imgsz=224, verbose=True)
        
        if results:
            res = results[0]
            logger.info(f"Result type: {type(res)}")
            
            if hasattr(res, "probs"):
                p = res.probs
                logger.info(f"Probs object: {p}")
                
                if hasattr(p, "top1"):
                    logger.info(f"Top1 Index: {p.top1}")
                    logger.info(f"Top1 Conf: {p.top1conf}")
                    
                    if hasattr(res, "names"):
                        logger.info(f"Top1 Label: {res.names[int(p.top1)]}")
                else:
                    logger.warning("Probs object has no top1 attribute")
            else:
                logger.warning("Result has no probs attribute (Is this a detection model instead of classification?)")
                
                # Check for boxes if it's a detection model
                if hasattr(res, "boxes"):
                    logger.info(f"Boxes: {res.boxes}")
                    
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_violence_model()
