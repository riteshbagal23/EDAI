"""
Model Manager for lazy loading and caching YOLO models.

This module provides a centralized way to manage model loading,
reducing memory usage and startup time through lazy loading.
"""

import logging
from pathlib import Path
from typing import Dict, Optional
from ultralytics import YOLO
import gc

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Singleton class to manage YOLO model loading and caching.
    
    Features:
    - Lazy loading: Models loaded only when first requested
    - Caching: Loaded models kept in memory for reuse
    - Memory management: Ability to unload unused models
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._models: Dict[str, YOLO] = {}
        self._model_paths: Dict[str, Path] = {}
        self._load_counts: Dict[str, int] = {}
        self._initialized = True
        
        # Define model paths
        root_dir = Path(__file__).parent.parent
        self._model_paths = {
            'best': root_dir / 'best.pt',
            'best1': root_dir / 'best (1).pt',
            'best2_violence': root_dir / 'best (2).pt',
            'best8': root_dir / 'best (8).pt',
            'best9_topview': root_dir / 'best (9).pt',
            'best10': root_dir / 'best (10).pt',
            'people': root_dir / 'yolov8n.pt',
            'thermal': root_dir / 'thermal.pt',
            'thermal_human': root_dir / 'thermalhuman.pt',
            'drone': root_dir / 'drone.pt',
        }
        
        logger.info("✅ ModelManager initialized with lazy loading enabled")
    
    def get_model(self, model_name: str) -> Optional[YOLO]:
        """
        Get a model by name, loading it if not already cached.
        
        Args:
            model_name: Name of the model (e.g., 'best', 'best1', 'people')
        
        Returns:
            YOLO model instance or None if model file doesn't exist
        """
        if model_name not in self._model_paths:
            logger.error(f"❌ Unknown model name: {model_name}")
            return None
        
        # Return cached model if available
        if model_name in self._models:
            self._load_counts[model_name] = self._load_counts.get(model_name, 0) + 1
            logger.debug(f"📦 Using cached model: {model_name} (used {self._load_counts[model_name]} times)")
            return self._models[model_name]
        
        # Load model if not cached
        model_path = self._model_paths[model_name]
        if not model_path.exists():
            logger.warning(f"⚠️ Model file not found: {model_path}")
            return None
        
        try:
            logger.info(f"🔄 Loading model: {model_name} from {model_path}")
            model = YOLO(str(model_path))
            self._models[model_name] = model
            self._load_counts[model_name] = 1
            logger.info(f"✅ Model loaded successfully: {model_name}. Classes: {model.names}")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {e}")
            return None
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory to free resources.
        
        Args:
            model_name: Name of the model to unload
        
        Returns:
            True if model was unloaded, False otherwise
        """
        if model_name in self._models:
            logger.info(f"🗑️ Unloading model: {model_name}")
            del self._models[model_name]
            if model_name in self._load_counts:
                del self._load_counts[model_name]
            gc.collect()
            return True
        return False
    
    def unload_all(self):
        """Unload all models from memory."""
        logger.info("🗑️ Unloading all models...")
        self._models.clear()
        self._load_counts.clear()
        gc.collect()
        logger.info("✅ All models unloaded")
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded model names."""
        return list(self._models.keys())
    
    def get_model_stats(self) -> Dict[str, int]:
        """Get usage statistics for loaded models."""
        return self._load_counts.copy()
    
    def preload_models(self, model_names: list):
        """
        Preload specific models at startup.
        
        Args:
            model_names: List of model names to preload
        """
        logger.info(f"🔄 Preloading models: {model_names}")
        for model_name in model_names:
            self.get_model(model_name)


# Global singleton instance
model_manager = ModelManager()


# Convenience functions for backward compatibility
def get_model(model_name: str) -> Optional[YOLO]:
    """Get a model by name."""
    return model_manager.get_model(model_name)


def unload_model(model_name: str) -> bool:
    """Unload a model from memory."""
    return model_manager.unload_model(model_name)
