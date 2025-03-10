import os
import sys
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from fastapi import HTTPException, status

# Configure paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import from the original predict.py to maintain consistency
from src.predict import load_model_and_tokenizer, get_confidence_grade
from .config import settings

# Keep a global reference to model data
_model_data = None


def load_bcea_reference(reference_file: str) -> Dict[str, str]:
    """
    Load BCEA reference data with code descriptions
    
    Args:
        reference_file (str): Path to reference file
        
    Returns:
        dict: Mapping of BCEA codes to their descriptions
    """
    if not os.path.exists(reference_file):
        print(f"Warning: BCEA reference file not found: {reference_file}")
        return {}
        
    try:
        # Load reference data
        ref_df = pd.read_csv(reference_file)
        
        # Ensure required columns exist
        if 'bcea_code' not in ref_df.columns or 'short_description' not in ref_df.columns:
            print("BCEA reference file missing required columns")
            return {}
        
        # Convert codes to strings for consistent lookup
        ref_df['bcea_code'] = ref_df['bcea_code'].astype(str)
        
        # Create mapping dictionary
        code_to_desc = dict(zip(ref_df['bcea_code'], ref_df['short_description']))
        print(f"Loaded {len(code_to_desc)} BCEA code descriptions")
            
        return code_to_desc
        
    except Exception as e:
        print(f"Error loading BCEA reference file: {e}")
        return {}


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
        bcea_descriptions = load_bcea_reference(settings.REFERENCE_FILE)
        
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