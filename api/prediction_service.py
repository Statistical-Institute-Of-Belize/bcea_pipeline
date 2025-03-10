import torch
import sys
import os
from typing import Dict, List, Tuple, Any, Optional

# Configure paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Import from the original predict.py to ensure consistency
from src.predict import get_confidence_grade
from src.model import get_device

from .models import (
    BusinessInput, 
    BCEAPrediction, 
    AlternativePrediction
)
from .config import settings

# Cache for predictions to improve performance
_prediction_cache = {}

def predict_single_business(
    business: BusinessInput,
    model: Any, 
    tokenizer: Any, 
    label_map: Dict[str, str],
    bcea_descriptions: Dict[str, str],
) -> BCEAPrediction:
    """
    Predict BCEA code for a single business.
    
    Args:
        business: BusinessInput object with business name and description
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        label_map: Dictionary mapping from class IDs to BCEA codes
        bcea_descriptions: Dictionary mapping from BCEA codes to descriptions
    
    Returns:
        BCEAPrediction object
    """
    # Create cache key
    cache_key = f"{business.bus_name}|||{business.description}"
    
    # Check cache first
    if cache_key in _prediction_cache:
        return _prediction_cache[cache_key]
    
    # Combine business name and description
    separator = ' | '
    text = f"{business.bus_name}{separator}{business.description}" if business.bus_name and business.bus_name.strip() else business.description
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=settings.MAX_SEQ_LENGTH
    )
    
    # Move inputs to the correct device
    device = model.device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process output
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)[0]
    
    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probabilities, k=3, dim=0)
    
    # Convert to lists
    top_indices = [str(idx.item()) for idx in top_indices]
    top_probs = top_probs.cpu().tolist()
    
    # Get primary prediction
    primary_pred_id = top_indices[0]
    primary_confidence = top_probs[0]
    
    # Get BCEA code and description
    primary_code = label_map.get(primary_pred_id, "0000.0")
    primary_description = bcea_descriptions.get(primary_code, f"No description available for {primary_code}")
    
    # Get confidence grade
    confidence_grade = get_confidence_grade(primary_confidence)
    
    # Create alternatives list
    alternatives = []
    
    # Add first alternative if available
    if len(top_indices) > 1:
        alt1_id = top_indices[1]
        alt1_confidence = top_probs[1]
        alt1_code = label_map.get(alt1_id, "0000.0")
        alt1_description = bcea_descriptions.get(alt1_code, f"No description available for {alt1_code}")
        
        alternatives.append(
            AlternativePrediction(
                code=alt1_code,
                description=alt1_description,
                confidence=alt1_confidence
            )
        )
    
    # Add second alternative if available
    if len(top_indices) > 2:
        alt2_id = top_indices[2]
        alt2_confidence = top_probs[2]
        alt2_code = label_map.get(alt2_id, "0000.0")
        alt2_description = bcea_descriptions.get(alt2_code, f"No description available for {alt2_code}")
        
        alternatives.append(
            AlternativePrediction(
                code=alt2_code,
                description=alt2_description,
                confidence=alt2_confidence
            )
        )
    
    # Create prediction object
    prediction = BCEAPrediction(
        bus_name=business.bus_name,
        description=business.description,
        predicted_code=primary_code,
        industry_description=primary_description,
        confidence=primary_confidence,
        confidence_grade=confidence_grade,
        alternatives=alternatives
    )
    
    # Cache the prediction
    _prediction_cache[cache_key] = prediction
    
    return prediction


def predict_batch_businesses(
    businesses: List[BusinessInput],
    model: Any, 
    tokenizer: Any, 
    label_map: Dict[str, str],
    bcea_descriptions: Dict[str, str],
) -> List[BCEAPrediction]:
    """
    Predict BCEA codes for a batch of businesses.
    
    Args:
        businesses: List of BusinessInput objects
        model: Pre-loaded model
        tokenizer: Pre-loaded tokenizer
        label_map: Dictionary mapping from class IDs to BCEA codes
        bcea_descriptions: Dictionary mapping from BCEA codes to descriptions
    
    Returns:
        List of BCEAPrediction objects
    """
    # For small batches, process individually for better caching
    if len(businesses) <= 4:
        return [
            predict_single_business(
                business=business,
                model=model,
                tokenizer=tokenizer,
                label_map=label_map,
                bcea_descriptions=bcea_descriptions
            )
            for business in businesses
        ]
    
    # For larger batches, combine texts and process all at once
    texts = []
    separator = ' | '
    
    for business in businesses:
        text = f"{business.bus_name}{separator}{business.description}" if business.bus_name and business.bus_name.strip() else business.description
        texts.append(text)
    
    # Tokenize batch
    batch_inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=settings.MAX_SEQ_LENGTH
    )
    
    # Move to device
    device = model.device
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**batch_inputs)
    
    # Process output
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    
    # Get top 3 predictions for each item
    top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
    
    # Convert to CPU numpy
    top_indices = top_indices.cpu().numpy()
    top_probs = top_probs.cpu().numpy()
    
    # Create prediction objects
    predictions = []
    
    for i, (business, indices, probs) in enumerate(zip(businesses, top_indices, top_probs)):
        # Process main prediction
        primary_pred_id = str(indices[0])
        primary_confidence = probs[0]
        
        # Get BCEA code and description
        primary_code = label_map.get(primary_pred_id, "0000.0")
        primary_description = bcea_descriptions.get(primary_code, f"No description available for {primary_code}")
        
        # Get confidence grade
        confidence_grade = get_confidence_grade(primary_confidence)
        
        # Create alternatives list
        alternatives = []
        
        # Add first alternative
        if len(indices) > 1:
            alt1_id = str(indices[1])
            alt1_confidence = probs[1]
            alt1_code = label_map.get(alt1_id, "0000.0")
            alt1_description = bcea_descriptions.get(alt1_code, f"No description available for {alt1_code}")
            
            alternatives.append(
                AlternativePrediction(
                    code=alt1_code,
                    description=alt1_description,
                    confidence=alt1_confidence
                )
            )
        
        # Add second alternative
        if len(indices) > 2:
            alt2_id = str(indices[2])
            alt2_confidence = probs[2]
            alt2_code = label_map.get(alt2_id, "0000.0")
            alt2_description = bcea_descriptions.get(alt2_code, f"No description available for {alt2_code}")
            
            alternatives.append(
                AlternativePrediction(
                    code=alt2_code,
                    description=alt2_description,
                    confidence=alt2_confidence
                )
            )
        
        # Create prediction object
        prediction = BCEAPrediction(
            bus_name=business.bus_name,
            description=business.description,
            predicted_code=primary_code,
            industry_description=primary_description,
            confidence=primary_confidence,
            confidence_grade=confidence_grade,
            alternatives=alternatives
        )
        
        # Cache the prediction
        cache_key = f"{business.bus_name}|||{business.description}"
        _prediction_cache[cache_key] = prediction
        
        predictions.append(prediction)
    
    return predictions