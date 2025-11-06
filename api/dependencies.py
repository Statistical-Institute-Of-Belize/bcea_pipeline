from typing import Dict, Tuple, Any
from fastapi import HTTPException, status

from src.predict import load_model_and_tokenizer, get_confidence_grade
from src.utils import load_bcea_reference as load_reference_map
from .config import settings

# Keep a global reference to model data
_model_data = None


def get_model_and_tokenizer() -> Tuple[Any, Any, Dict, Dict]:
    """
    Load the BCEA classification model, tokenizer and mappings.
    Uses caching to avoid reloading the model on each request.
    Uses the original pipeline's load_model_and_tokenizer function
    for consistency.
    
    Returns:
        tuple: (model, tokenizer, id_to_label, bcea_descriptions)
    """
    global _model_data
    
    # If model is already loaded, return from cache
    if _model_data is not None:
        return _model_data
    
    try:
        # Use the same model loading function as the main pipeline
        model, tokenizer, id_to_label = load_model_and_tokenizer(settings.MODEL_PATH)
        
        # Load the reference data
        bcea_descriptions = load_reference_map(settings.REFERENCE_FILE)
        
        # Move model to device once at startup
        from src.model import get_device
        device = get_device({'model': {'device': 'auto'}})
        model.to(device)
        print(f"Model moved to device: {device}")
        
        # Cache the model, tokenizer and mappings
        _model_data = (model, tokenizer, id_to_label, bcea_descriptions)
        
        return _model_data
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )
