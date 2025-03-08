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

def prepare_prediction_dataset(texts, tokenizer, max_seq_length):
    """
    Prepare dataset for prediction
    
    Args:
        texts (list): List of text descriptions
        tokenizer: Hugging Face tokenizer
        max_seq_length (int): Maximum sequence length
        
    Returns:
        BCEADataset: Dataset for prediction
    """
    return BCEADataset(texts, None, tokenizer, max_seq_length)

def predict_all(model, dataset, device, batch_size=16):
    """
    Make predictions for a dataset
    
    Args:
        model: Hugging Face model
        dataset: BCEADataset for prediction
        device: PyTorch device
        batch_size (int): Batch size for prediction
        
    Returns:
        list: Predicted label IDs
    """
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 0
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize predictions list
    all_predictions = []
    
    # Make predictions
    with torch.no_grad():
        for batch in dataloader:
            try:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward pass
                outputs = model(**batch)
                
                # Get predictions
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1).cpu().numpy()
                
                # Add predictions to list
                all_predictions.extend(predictions.tolist())
            except Exception as e:
                # Handle errors in prediction
                logging.warning(f"Error in prediction batch: {e}")
                # Add default predictions
                all_predictions.extend([0] * len(batch['input_ids']))
    
    # Convert predictions to strings for consistent mapping
    all_predictions = [str(pred) for pred in all_predictions]
    
    return all_predictions

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

def predict_batch(texts, model, tokenizer, id_to_label, config):
    """
    Predict BCEA codes for a batch of texts
    
    Args:
        texts (list): List of text descriptions
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        id_to_label (dict): ID to label mapping
        config (dict): Configuration dictionary
        
    Returns:
        pd.DataFrame: DataFrame with predictions
    """
    # Check memory availability
    check_memory_availability(
        required_gb=config.get('model', {}).get('memory', {}).get('min_required_gb'),
        percentage=config.get('model', {}).get('memory', {}).get('max_usage_percentage', 0.8)
    )
    
    # Prepare dataset
    max_seq_length = config['model']['max_seq_length']
    dataset = prepare_prediction_dataset(texts, tokenizer, max_seq_length)
    
    # Get the appropriate device
    device = get_device(config)
    model.to(device)
    
    # Get batch size from config
    batch_size = config['model']['batch_size']
    
    # Make predictions
    predictions = predict_all(model, dataset, device, batch_size)
    
    # Map predictions to BCEA codes
    bcea_codes = []
    unknown_count = 0
    
    for pred in predictions:
        # Ensure prediction is a string for consistency
        pred_str = str(pred)
        
        if pred_str in id_to_label:
            bcea_codes.append(id_to_label[pred_str])
        else:
            unknown_count += 1
            bcea_codes.append('Unknown')
            logging.warning(f"Unknown label ID: {pred_str}")
    
    if unknown_count > 0:
        logging.warning(f"Found {unknown_count} predictions without matching BCEA codes")
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'description': texts,
        'bcea_code': bcea_codes
    })
    
    # Save predictions
    predictions_path = os.path.join(config['data']['processed_dir'], 'predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    logging.info(f"Predicted {len(predictions_df)} codes")
    
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
    # Load input file
    df = pd.read_csv(input_file)
    
    # Check for required column
    if 'description' not in df.columns:
        error_msg = f"Required column 'description' missing from {input_file}"
        logging.error(error_msg)
        raise ValueError(error_msg)
    
    # Get texts
    texts = df['description'].tolist()
    
    # Load model, tokenizer, and label mappings
    model, tokenizer, id_to_label = load_model_and_tokenizer(config['output']['best_model_dir'])
    
    # Make predictions
    return predict_batch(texts, model, tokenizer, id_to_label, config)

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
    
    # Predict from file
    predict_from_file(input_file, config)
    
    logging.info("Prediction completed")

if __name__ == "__main__":
    run_prediction()