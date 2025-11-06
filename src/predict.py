import os
import json
import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime

from .utils import load_config, check_memory_availability, load_bcea_reference
from .model import BCEADataset, get_device

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
    
    code_column = 'predicted_code' if 'predicted_code' in predictions_df.columns else 'bcea_code'
    unknown_mask = predictions_df[code_column] == 'Unknown'
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

def _resolve_reference_path(config):
    """Return the absolute path to the BCEA reference CSV."""
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_cfg = config.get('data', {}) if config else {}

    reference_file = data_cfg.get('reference_file')
    if reference_file:
        return reference_file if os.path.isabs(reference_file) else os.path.join(project_root, reference_file)

    reference_dir = data_cfg.get('reference_dir')
    if reference_dir:
        base_dir = reference_dir if os.path.isabs(reference_dir) else os.path.join(project_root, reference_dir)
        return os.path.join(base_dir, 'bcea_ref.csv')

    return os.path.join(project_root, 'data', 'reference', 'bcea_ref.csv')


def _map_prediction_rows(top_predictions, confidence_scores, id_to_label, bcea_descriptions):
    """Convert raw classifier outputs into structured prediction rows."""
    rows = []
    unknown_primary = 0

    for indices, scores in zip(top_predictions, confidence_scores):
        mapped = []
        for class_id, score in zip(indices, scores):
            code = id_to_label.get(class_id)
            if code:
                description = bcea_descriptions.get(code, f"No description available for {code}")
            else:
                code = 'Unknown'
                description = 'Unknown code'
            mapped.append({'code': code, 'description': description, 'confidence': float(score)})

        while len(mapped) < 3:
            mapped.append({'code': 'Unknown', 'description': 'Unknown code', 'confidence': 0.0})

        if mapped[0]['code'] == 'Unknown':
            unknown_primary += 1

        rows.append(mapped[:3])

    return rows, unknown_primary




def predict_batch(texts, model, tokenizer, id_to_label, config, original_df=None, explain=False):
    """Predict BCEA codes for a batch of texts."""
    check_memory_availability(
        required_gb=config.get('model', {}).get('memory', {}).get('min_required_gb'),
        percentage=config.get('model', {}).get('memory', {}).get('max_usage_percentage', 0.8)
    )

    reference_path = _resolve_reference_path(config)
    bcea_descriptions = load_bcea_reference(reference_path)
    if not bcea_descriptions:
        logging.warning("BCEA reference file missing or empty at %s", reference_path)

    if explain:
        logging.warning("Explanation generation is not yet supported for the BCEA pipeline; continuing without it")

    business_names = None
    if original_df is not None and 'bus_name' in original_df.columns:
        business_names = original_df['bus_name'].tolist()
        logging.info("Using business name data for %s records", len(business_names))

    max_seq_length = config['model']['max_seq_length']
    dataset = prepare_prediction_dataset(
        texts,
        tokenizer,
        max_seq_length,
        business_names=business_names,
        config=config
    )

    device = get_device(config)
    model.to(device)

    training_cfg = config.get('training', {}) if config else {}
    batch_size = training_cfg.get('batch_size', config.get('model', {}).get('batch_size', 32))
    top_predictions, confidence_scores = predict_all(
        model, dataset, device, batch_size, top_k=3
    )

    mapped_rows, unknown_count = _map_prediction_rows(
        top_predictions, confidence_scores, id_to_label, bcea_descriptions
    )

    if mapped_rows:
        sample = mapped_rows[0][0]
        logging.info("Sample prediction mapping: %s -> %s", sample['code'], sample['description'])

    primary_codes = [row[0]['code'] for row in mapped_rows]
    primary_descriptions = [row[0]['description'] for row in mapped_rows]
    confidences = [row[0]['confidence'] for row in mapped_rows]
    confidence_grades = [get_confidence_grade(score) for score in confidences]

    alt_codes_1 = [row[1]['code'] for row in mapped_rows]
    alt_desc_1 = [row[1]['description'] for row in mapped_rows]
    alt_conf_1 = [row[1]['confidence'] for row in mapped_rows]

    alt_codes_2 = [row[2]['code'] for row in mapped_rows]
    alt_desc_2 = [row[2]['description'] for row in mapped_rows]
    alt_conf_2 = [row[2]['confidence'] for row in mapped_rows]

    prediction_cfg = config.get('prediction', {}) if config else {}
    confidence_threshold = prediction_cfg.get('confidence_threshold')
    if confidence_threshold is None:
        confidence_threshold = config.get('model', {}).get('confidence_threshold', 0.0)

    fallback_flags = []
    final_codes = []
    for code, confidence in zip(primary_codes, confidences):
        fallback = bool(confidence_threshold) and confidence < confidence_threshold
        fallback_flags.append(fallback)
        final_codes.append(code)

    if unknown_count > 0:
        logging.warning("Found %s predictions without matching BCEA codes", unknown_count)

    if original_df is not None:
        predictions_df = original_df.copy()
    else:
        predictions_df = pd.DataFrame({'description': texts})

    predictions_df['predicted_code'] = final_codes
    predictions_df['predicted_description'] = primary_descriptions
    predictions_df['confidence'] = confidences
    predictions_df['confidence_grade'] = confidence_grades
    predictions_df['is_fallback'] = fallback_flags
    predictions_df['alternative_1'] = alt_codes_1
    predictions_df['alternative_1_description'] = alt_desc_1
    predictions_df['alternative_1_confidence'] = alt_conf_1
    predictions_df['alternative_2'] = alt_codes_2
    predictions_df['alternative_2_description'] = alt_desc_2
    predictions_df['alternative_2_confidence'] = alt_conf_2

    # Backwards-compatible columns
    predictions_df['bcea_code'] = predictions_df['predicted_code']
    predictions_df['industry_description'] = predictions_df['predicted_description']
    predictions_df['alt_bcea_code_1'] = predictions_df['alternative_1']
    predictions_df['alt_industry_description_1'] = predictions_df['alternative_1_description']
    predictions_df['alt_confidence_1'] = predictions_df['alternative_1_confidence']
    predictions_df['alt_bcea_code_2'] = predictions_df['alternative_2']
    predictions_df['alt_industry_description_2'] = predictions_df['alternative_2_description']
    predictions_df['alt_confidence_2'] = predictions_df['alternative_2_confidence']

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config['data']['processed_dir']
    os.makedirs(output_dir, exist_ok=True)

    input_file = config.get('current_input_file', 'predictions')
    base_filename = os.path.basename(input_file)
    filename_no_ext = os.path.splitext(base_filename)[0]

    predictions_path = os.path.join(output_dir, f'{filename_no_ext}_predictions_{timestamp}.csv')
    predictions_df.to_csv(predictions_path, index=False)

    logging.info("Predicted %s codes", len(predictions_df))
    logging.info("Predictions saved to: %s", predictions_path)
    print(f"✓ Predictions saved to: {predictions_path}")

    unknown_count = flag_unknown_codes(predictions_df, config['data']['review_dir'])

    if unknown_count > 0:
        logging.warning("Flagged %s unknown BCEA codes for review", unknown_count)

    return predictions_df




def predict_from_file(input_file, config, explain=False):
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
    return predict_batch(texts, model, tokenizer, id_to_label, config, original_df=df, explain=explain)

def run_prediction(input_file=None, config=None, explain=False):
    """
    Run prediction pipeline
    
    Args:
        input_file (str, optional): Path to input file. If None, use test.csv
    """
    # Load config
    if config is None:
        config = load_config("config.yaml")
    
    # If no input file is provided, use test.csv
    if input_file is None:
        input_file = os.path.join(config['data']['processed_dir'], 'test.csv')
    
    # Log input file
    logging.info(f"Predicting on {input_file}")
    print(f"✓ Starting prediction on {input_file}")
    
    # Predict from file
    predictions_df = predict_from_file(input_file, config, explain=explain)
    
    # Print summary
    print(f"✓ Successfully predicted {len(predictions_df)} BCEA codes")
    print(f"✓ Added confidence scores and alternative predictions")
    
    logging.info("Prediction completed successfully")
    print(f"✓ Prediction completed successfully")
    
    return predictions_df

if __name__ == "__main__":
    run_prediction()
