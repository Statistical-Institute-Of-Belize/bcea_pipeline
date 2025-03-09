import os
import json
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import yaml
import sys
from pathlib import Path

# Add the parent directory to system path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config, check_memory_availability
from src.model import BCEADataset, get_device

def load_model_and_tokenizer(model_dir):
    """
    Load model and tokenizer from directory
    
    Args:
        model_dir (str): Directory containing model and tokenizer
        
    Returns:
        tuple: (model, tokenizer, id_to_label)
    """
    # Check if model directory exists
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    try:
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        logging.info(f"Model loaded from {model_dir}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logging.info("Tokenizer loaded")
        
        # Load ID to label mapping
        mapping_path = os.path.join(model_dir, 'id_to_label.json')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
            
        with open(mapping_path, 'r') as f:
            id_to_label = json.load(f)
        
        # Ensure all keys are strings for consistency
        id_to_label = {str(k): v for k, v in id_to_label.items()}
        
        logging.info(f"Loaded label mapping with {len(id_to_label)} labels")
        
        return model, tokenizer, id_to_label
        
    except Exception as e:
        logging.error(f"Error loading model and tokenizer: {e}")
        raise

def prepare_prediction_dataset(texts, tokenizer, max_seq_length, business_names=None, config=None):
    """
    Prepare dataset for prediction
    
    Args:
        texts (list): List of text descriptions
        tokenizer: Hugging Face tokenizer
        max_seq_length (int): Maximum sequence length
        business_names (list, optional): List of business names
        config (dict, optional): Configuration dictionary
        
    Returns:
        BCEADataset: Dataset for prediction
    """
    # If business names are provided, combine them with descriptions
    if business_names and len(business_names) == len(texts):
        separator = ' | '  # Default separator
        use_business_name = True  # Default value
        
        # Get values from config if available
        if config:
            separator = config.get('data', {}).get('text_separator', ' | ')
            use_business_name = config.get('data', {}).get('use_business_name', True)
        
        if use_business_name:
            combined_texts = [
                f"{bus_name}{separator}{desc}" if bus_name and bus_name.strip() else desc
                for bus_name, desc in zip(business_names, texts)
            ]
            logging.info(f"Combined business names with descriptions for prediction using separator '{separator}'")
            return BCEADataset(combined_texts, None, tokenizer, max_seq_length)
        else:
            logging.info("Business name use is disabled in config, using only descriptions")
            return BCEADataset(texts, None, tokenizer, max_seq_length)
    
    # Otherwise, just use the descriptions
    return BCEADataset(texts, None, tokenizer, max_seq_length)

def predict_all(model, dataset, device, batch_size=16, top_k=3):
    """
    Make predictions for a dataset
    
    Args:
        model: Hugging Face model
        dataset: BCEADataset for prediction
        device: PyTorch device
        batch_size (int): Batch size for prediction
        top_k (int): Number of top predictions to return
        
    Returns:
        tuple: (top_predictions, confidence_scores) - Lists of top-k predictions and their confidence scores
    """
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists for predictions and confidence scores
    all_top_predictions = []
    all_confidence_scores = []
    
    # Make predictions
    with torch.no_grad():
        for batch in dataloader:
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Get top-k predictions and confidence scores
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                
                # Get top-k predictions and their scores
                top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
                
                # Convert to lists
                batch_top_indices = top_indices.cpu().numpy()
                batch_top_probs = top_probs.cpu().numpy()
                
                # Add to overall lists
                for indices, probs in zip(batch_top_indices, batch_top_probs):
                    all_top_predictions.append([str(idx) for idx in indices])
                    all_confidence_scores.append(probs.tolist())
                
            except Exception as e:
                # Handle errors in prediction
                logging.warning(f"Error in prediction batch: {e}")
                # Add default predictions with low confidence
                for _ in range(len(batch['input_ids'])):
                    all_top_predictions.append(['0'] * top_k)
                    all_confidence_scores.append([0.1] * top_k)
    
    return all_top_predictions, all_confidence_scores

def flag_unknown_codes(predictions_df, review_dir):
    """
    Flag and save predictions for unknown BCEA codes
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with predictions
        review_dir (str): Directory to save unknown codes
        
    Returns:
        int: Count of unknown codes
    """
    # Create review directory if it doesn't exist
    os.makedirs(review_dir, exist_ok=True)
    
    # Flag unknown codes (not in our original label mapping)
    unknown_mask = predictions_df['bcea_code'] == 'Unknown'
    unknown_df = predictions_df[unknown_mask].copy()
    
    # Save unknown codes to CSV
    timestamp = datetime.now().strftime("%Y%m%d")
    unknown_path = os.path.join(review_dir, f'unknown_{timestamp}.csv')
    
    # If there are unknown predictions, save them
    unknown_count = len(unknown_df)
    if unknown_count > 0:
        unknown_df.to_csv(unknown_path, index=False)
        logging.warning(f"Flagged {unknown_count} unknown codes")
    else:
        logging.info("No unknown codes found")
    
    return unknown_count

def get_confidence_grade(confidence):
    """
    Convert confidence score to a grade
    
    Args:
        confidence (float): Confidence score between 0 and 1
        
    Returns:
        str: Confidence grade (very_low, low, medium, high, very_high)
    """
    if confidence < 0.5:
        return "very_low"
    elif confidence < 0.7:
        return "low"
    elif confidence < 0.8:
        return "medium"
    elif confidence < 0.9:
        return "high"
    else:
        return "very_high"

def load_bcea_reference():
    """
    Load BCEA reference data with code descriptions
    
    Returns:
        dict: Mapping of BCEA codes to their descriptions
    """
    ref_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                           'data/reference/bcea_ref.csv')
    
    if not os.path.exists(ref_path):
        logging.warning(f"BCEA reference file not found: {ref_path}")
        return {}
        
    try:
        # Load reference data
        ref_df = pd.read_csv(ref_path)
        
        # Ensure required columns exist
        if 'bcea_code' not in ref_df.columns or 'short_description' not in ref_df.columns:
            logging.warning("BCEA reference file missing required columns")
            return {}
            
        # Print first few rows for debugging
        logging.info("BCEA reference data sample:")
        for i, row in ref_df.head(3).iterrows():
            logging.info(f"  Code: '{row['bcea_code']}', Description: '{row['short_description']}'")
        
        # Convert codes to strings for consistent lookup
        ref_df['bcea_code'] = ref_df['bcea_code'].astype(str)
        
        # Create mapping dictionary
        code_to_desc = dict(zip(ref_df['bcea_code'], ref_df['short_description']))
        logging.info(f"Loaded {len(code_to_desc)} BCEA code descriptions")
        
        # Print a few sample mappings
        sample_keys = list(code_to_desc.keys())[:3]
        for key in sample_keys:
            logging.info(f"  Sample mapping: '{key}' -> '{code_to_desc[key]}'")
            
        return code_to_desc
        
    except Exception as e:
        logging.warning(f"Error loading BCEA reference file: {e}")
        return {}

def predict_batch(texts, model, tokenizer, id_to_label, config, original_df=None):
    """
    Predict BCEA codes for a batch of texts
    
    Args:
        texts (list): List of text descriptions
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        id_to_label (dict): ID to label mapping
        config (dict): Configuration dictionary
        original_df (pd.DataFrame, optional): Original DataFrame to preserve all columns
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Check memory availability
    check_memory_availability(
        required_gb=config.get('model', {}).get('memory', {}).get('min_required_gb'),
        percentage=config.get('model', {}).get('memory', {}).get('max_usage_percentage', 0.8)
    )
    
    # Load BCEA reference data for descriptions
    bcea_descriptions = load_bcea_reference()
    
    # Extract business names if available from the original dataframe
    business_names = None
    if original_df is not None and 'bus_name' in original_df.columns:
        business_names = original_df['bus_name'].tolist()
        logging.info(f"Using business name data for {len(business_names)} records")
    
    # Prepare dataset
    max_seq_length = config['model']['max_seq_length']
    dataset = prepare_prediction_dataset(texts, tokenizer, max_seq_length, 
                                        business_names=business_names, 
                                        config=config)
    
    # Get the appropriate device
    device = get_device(config)
    model.to(device)
    
    # Get batch size from config
    batch_size = config['model']['batch_size']
    
    # Make predictions with top 3 results
    top_predictions, confidence_scores = predict_all(model, dataset, device, batch_size, top_k=3)
    
    # Map predictions to BCEA codes with confidence scores
    bcea_codes = []
    bcea_descriptions_list = []
    confidences = []
    confidence_grades = []
    alt_bcea_codes_1 = []
    alt_descriptions_1 = []
    alt_confidences_1 = []
    alt_bcea_codes_2 = []
    alt_descriptions_2 = []
    alt_confidences_2 = []
    unknown_count = 0
    
    for preds, scores in zip(top_predictions, confidence_scores):
        # Process main prediction (first in the list)
        pred_str = preds[0]
        confidence = scores[0]
        
        if pred_str in id_to_label:
            code = id_to_label[pred_str]
            bcea_codes.append(code)
            # Add description if available (with debugging info)
            description = bcea_descriptions.get(code, f"No description available for {code}")
            # Log some samples for debugging
            if len(bcea_codes) <= 3:
                logging.info(f"Looking up description for code '{code}', found: '{description}'")
                if code not in bcea_descriptions:
                    logging.info(f"Available keys sample: {list(bcea_descriptions.keys())[:5]}")
            bcea_descriptions_list.append(description)
        else:
            unknown_count += 1
            bcea_codes.append('Unknown')
            bcea_descriptions_list.append("Unknown code")
            logging.warning(f"Unknown label ID: {pred_str}")
        
        confidences.append(confidence)
        confidence_grades.append(get_confidence_grade(confidence))
        
        # Process first alternative prediction
        if len(preds) > 1:
            alt_pred_str = preds[1]
            alt_confidence = scores[1]
            if alt_pred_str in id_to_label:
                code = id_to_label[alt_pred_str]
                alt_bcea_codes_1.append(code)
                # Add description if available
                description = bcea_descriptions.get(code, f"No description available for {code}")
                # Log first few for debugging
                if len(alt_bcea_codes_1) <= 1:
                    logging.info(f"Alt1: Looking up description for code '{code}', found: '{description}'")
                alt_descriptions_1.append(description)
            else:
                alt_bcea_codes_1.append('Unknown')
                alt_descriptions_1.append("Unknown code")
            alt_confidences_1.append(alt_confidence)
        else:
            alt_bcea_codes_1.append('Unknown')
            alt_descriptions_1.append("Unknown code")
            alt_confidences_1.append(0.0)
        
        # Process second alternative prediction
        if len(preds) > 2:
            alt_pred_str = preds[2]
            alt_confidence = scores[2]
            if alt_pred_str in id_to_label:
                code = id_to_label[alt_pred_str]
                alt_bcea_codes_2.append(code)
                # Add description if available
                description = bcea_descriptions.get(code, f"No description available for {code}")
                # Log first few for debugging
                if len(alt_bcea_codes_2) <= 1:
                    logging.info(f"Alt2: Looking up description for code '{code}', found: '{description}'")
                alt_descriptions_2.append(description)
            else:
                alt_bcea_codes_2.append('Unknown')
                alt_descriptions_2.append("Unknown code")
            alt_confidences_2.append(alt_confidence)
        else:
            alt_bcea_codes_2.append('Unknown')
            alt_descriptions_2.append("Unknown code")
            alt_confidences_2.append(0.0)
    
    if unknown_count > 0:
        logging.warning(f"Found {unknown_count} predictions without matching BCEA codes")
    
    # Create predictions DataFrame
    # Start with original columns if provided
    if original_df is not None:
        predictions_df = original_df.copy()
    else:
        predictions_df = pd.DataFrame({'description': texts})
    
    # Add prediction columns
    predictions_df['bcea_code'] = bcea_codes
    predictions_df['industry_description'] = bcea_descriptions_list
    predictions_df['confidence'] = confidences
    predictions_df['confidence_grade'] = confidence_grades
    
    # Add alternative predictions
    predictions_df['alt_bcea_code_1'] = alt_bcea_codes_1
    predictions_df['alt_industry_description_1'] = alt_descriptions_1
    predictions_df['alt_confidence_1'] = alt_confidences_1
    
    predictions_df['alt_bcea_code_2'] = alt_bcea_codes_2
    predictions_df['alt_industry_description_2'] = alt_descriptions_2
    predictions_df['alt_confidence_2'] = alt_confidences_2
    
    # Generate timestamp for output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save predictions with timestamp
    output_dir = config['data']['processed_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename based on input filename if available
    input_file = config.get('current_input_file', 'predictions')
    base_filename = os.path.basename(input_file)
    filename_no_ext = os.path.splitext(base_filename)[0]
    
    predictions_path = os.path.join(output_dir, f'{filename_no_ext}_predictions_{timestamp}.csv')
    predictions_df.to_csv(predictions_path, index=False)
    
    logging.info(f"Predicted {len(predictions_df)} codes")
    logging.info(f"Predictions saved to: {predictions_path}")
    print(f"✓ Predictions saved to: {predictions_path}")
    
    # Flag unknown codes
    unknown_count = flag_unknown_codes(predictions_df, config['data']['review_dir'])
    
    if unknown_count > 0:
        logging.warning(f"Flagged {unknown_count} unknown BCEA codes for review")
    
    return predictions_df

def predict_from_file(input_file, config):
    """
    Predict BCEA codes from input file
    
    Args:
        input_file (str): Path to input file
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Store input file path in config for later use
    config['current_input_file'] = input_file
    
    # Load input file
    df = pd.read_csv(input_file)
    
    # Check for required column
    if 'description' not in df.columns:
        error_msg = f"Required column 'description' missing from {input_file}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Clean input text fields
    if 'description' in df.columns:
        df['description'] = df['description'].fillna('').str.lower().str.strip()
    
    # Clean and process business name if available
    if 'bus_name' in df.columns:
        df['bus_name'] = df['bus_name'].fillna('').str.lower().str.strip()
        logging.info("Business name column found and will be used for prediction")
    else:
        logging.info("No business name column found. Using only descriptions for prediction.")
    
    # Get texts
    texts = df['description'].tolist()
    
    # Load model, tokenizer, and label mappings
    model, tokenizer, id_to_label = load_model_and_tokenizer(config['output']['best_model_dir'])
    
    # Make predictions, preserving all original columns
    return predict_batch(texts, model, tokenizer, id_to_label, config, original_df=df)

def run_prediction(input_file=None):
    """
    Run prediction pipeline
    
    Args:
        input_file (str, optional): Path to input file. If None, use test.csv
    """
    # Load config
    config = load_config("config.yaml")
    
    # If no input file is provided, use test.csv
    if input_file is None:
        input_file = os.path.join(config['data']['processed_dir'], 'test.csv')
    
    # Log input file
    logging.info(f"Predicting on {input_file}")
    print(f"✓ Starting prediction on {input_file}")
    
    # Predict from file
    predictions_df = predict_from_file(input_file, config)
    
    # Print summary
    print(f"✓ Successfully predicted {len(predictions_df)} BCEA codes")
    print(f"✓ Added confidence scores and alternative predictions")
    
    logging.info("Prediction completed successfully")
    print(f"✓ Prediction completed successfully")
    
    return predictions_df

if __name__ == "__main__":
    run_prediction()