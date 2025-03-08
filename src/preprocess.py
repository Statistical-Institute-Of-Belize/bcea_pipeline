import os
import re
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys
from pathlib import Path

# Add the parent directory to system path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config, check_memory_availability

def clean_text(texts):
    """
    Clean text by converting to lowercase and removing special characters
    
    Args:
        texts (pd.Series): Series of text descriptions
        
    Returns:
        pd.Series: Cleaned texts
    """
    # Handle NaN or None values
    texts = texts.fillna('')
    
    # Convert to lowercase
    texts = texts.str.lower()
    
    # Remove special characters but keep spaces and word boundaries
    texts = texts.apply(lambda x: re.sub(r'[^\w\s]', ' ', x) if isinstance(x, str) else '')
    
    # Replace multiple spaces with a single space
    texts = texts.apply(lambda x: re.sub(r'\s+', ' ', x) if isinstance(x, str) else '')
    
    # Strip leading and trailing spaces
    texts = texts.str.strip()
    
    # If any empty strings remain after cleaning, replace with placeholder
    texts = texts.replace('', 'unknown description')
    
    return texts

def validate_bcea_codes(df, output_dir):
    """
    Ensure BCEA codes are in the format ####.# using regex
    
    Args:
        df (pd.DataFrame): DataFrame with 'bcea_code' column
        output_dir (str): Directory to save invalid codes
        
    Returns:
        pd.DataFrame: DataFrame with valid BCEA codes
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Handle missing or NaN BCEA codes
    df['bcea_code'] = df['bcea_code'].fillna('invalid')
    
    # Convert all codes to strings
    df['bcea_code'] = df['bcea_code'].astype(str)
    
    # Use regex to validate BCEA codes (####.#)
    bcea_pattern = r'^\d{4}\.\d$'
    
    # Try to fix common issues:
    # 1. Codes without decimal point (e.g., "12345" -> "1234.5")
    # 2. Codes with extra zeros (e.g., "0001.2" -> "0001.2")
    # 3. Extra whitespace
    def fix_bcea_code(code):
        code = code.strip()
        
        # If the code is already in the correct format, return it
        if re.match(bcea_pattern, code):
            return code
        
        # If code has 5 digits without decimal, insert decimal point
        if re.match(r'^\d{5}$', code):
            return code[:4] + '.' + code[4:]
        
        # If code has 4 digits, try to add .0
        if re.match(r'^\d{4}$', code):
            return code + '.0'
            
        # Otherwise, mark as invalid
        return 'invalid'
    
    # Apply fixes
    df['original_bcea_code'] = df['bcea_code']  # Keep original for reference
    df['bcea_code'] = df['bcea_code'].apply(fix_bcea_code)
    
    # Identify valid and invalid codes after fixing attempts
    valid_mask = df['bcea_code'].str.match(bcea_pattern)
    valid_df = df[valid_mask].copy()
    invalid_df = df[~valid_mask].copy()
    
    # Log the counts
    logging.info(f"Found {len(valid_df)} valid BCEA codes and {len(invalid_df)} invalid codes after fixing attempts")
    
    # If fixes were applied, log the count
    fix_count = sum(df['original_bcea_code'] != df['bcea_code']) - len(invalid_df)
    if fix_count > 0:
        logging.info(f"Fixed {fix_count} BCEA codes to match the ####.# format")
    
    # Remove the temporary column from the valid dataframe
    valid_df = valid_df.drop(columns=['original_bcea_code'])
    
    # Save invalid codes to CSV for review
    if len(invalid_df) > 0:
        invalid_path = os.path.join(output_dir, 'invalid_codes.csv')
        invalid_df.to_csv(invalid_path, index=False)
        logging.warning(f"Invalid BCEA codes saved to {invalid_path}")
    
    return valid_df

def create_label_mappings(df, mappings_dir):
    """
    Create label mappings for BCEA codes
    
    Args:
        df (pd.DataFrame): DataFrame with 'bcea_code' column
        mappings_dir (str): Directory to save mappings
        
    Returns:
        tuple: (id_to_label dict, label_to_id dict)
    """
    # Create mappings directory if it doesn't exist
    os.makedirs(mappings_dir, exist_ok=True)
    
    # Extract unique BCEA codes
    unique_codes = sorted(df['bcea_code'].unique())
    
    # Create mappings with consistent string keys and values
    id_to_label = {str(i): code for i, code in enumerate(unique_codes)}
    label_to_id = {code: str(i) for i, code in enumerate(unique_codes)}
    
    # Create mappings dict for export
    mappings = {
        'id_to_label': id_to_label,
        'label_to_id': label_to_id
    }
    
    # Save mappings to JSON
    mappings_path = os.path.join(mappings_dir, 'bcea_mappings.json')
    with open(mappings_path, 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Create human-readable CSV version
    mappings_csv = pd.DataFrame({
        'id': list(id_to_label.keys()),
        'bcea_code': list(id_to_label.values())
    })
    mappings_csv_path = os.path.join(mappings_dir, 'bcea_mappings.csv')
    mappings_csv.to_csv(mappings_csv_path, index=False)
    
    # Log creation
    logging.info(f"Created mappings for {len(unique_codes)} unique BCEA codes")
    
    # Log sample entries
    sample_size = min(5, len(unique_codes))
    sample_entries = {str(i): id_to_label[str(i)] for i in range(sample_size)}
    logging.info(f"Sample mappings: {sample_entries}")
    
    return id_to_label, label_to_id

def split_and_save_data(df, processed_dir):
    """
    Split data into train, validation, and test sets
    
    Args:
        df (pd.DataFrame): DataFrame with 'description' and 'label_id' columns
        processed_dir (str): Directory to save processed data
    """
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Count occurrences of each label
    label_counts = df['label_id'].value_counts()
    
    # Filter out labels with too few samples (need at least 6 for train/val/test)
    # This ensures at least 4 samples in training and 1 each in validation and test
    min_samples_per_class = 6
    
    # Identify rare and common labels
    rare_labels = label_counts[label_counts < min_samples_per_class].index.tolist()
    common_labels = label_counts[label_counts >= min_samples_per_class].index.tolist()
    
    logging.info(f"Found {len(common_labels)} labels with {min_samples_per_class}+ samples (suitable for stratification)")
    
    if rare_labels:
        logging.warning(f"Found {len(rare_labels)} labels with fewer than {min_samples_per_class} samples.")
        
        # Option: Skip very rare labels completely
        extremely_rare = label_counts[label_counts < 3].index.tolist()
        if extremely_rare:
            logging.warning(f"Skipping {len(extremely_rare)} labels with fewer than 3 samples total")
            df = df[~df['label_id'].isin(extremely_rare)]
            rare_labels = [label for label in rare_labels if label not in extremely_rare]
        
        # Process labels with enough samples for stratification
        if common_labels:
            # Keep only labels with enough samples for stratification
            common_labels_mask = df['label_id'].isin(common_labels)
            common_df = df[common_labels_mask]
            
            # Split common labels with stratification
            train_common, temp_common = train_test_split(
                common_df, test_size=0.3, stratify=common_df['label_id'], random_state=42
            )
            val_common, test_common = train_test_split(
                temp_common, test_size=0.5, stratify=temp_common['label_id'], random_state=42
            )
            
            # Process rare labels (3-5 samples)
            semi_rare_labels = [label for label in rare_labels if label_counts[label] >= 3]
            if semi_rare_labels:
                semi_rare_mask = df['label_id'].isin(semi_rare_labels)
                semi_rare_df = df[semi_rare_mask]
                
                # For rare labels, use a fixed split: 2 samples to train, 1 to validation, 0-2 to test
                train_rare = pd.DataFrame()
                val_rare = pd.DataFrame()
                test_rare = pd.DataFrame()
                
                for label in semi_rare_labels:
                    label_df = df[df['label_id'] == label]
                    
                    # First two samples always go to training
                    label_train = label_df.iloc[:2]
                    
                    # Third sample goes to validation
                    label_val = pd.DataFrame()
                    if len(label_df) >= 3:
                        label_val = label_df.iloc[2:3]
                    
                    # Any remaining samples go to test
                    label_test = pd.DataFrame()
                    if len(label_df) > 3:
                        label_test = label_df.iloc[3:]
                    
                    # Add to respective splits
                    train_rare = pd.concat([train_rare, label_train])
                    val_rare = pd.concat([val_rare, label_val])
                    test_rare = pd.concat([test_rare, label_test])
                
                # Combine common and rare splits
                train_df = pd.concat([train_common, train_rare])
                val_df = pd.concat([val_common, val_rare])
                test_df = pd.concat([test_common, test_rare])
            else:
                # No rare labels to process
                train_df = train_common
                val_df = val_common
                test_df = test_common
        else:
            # No common labels, just use a simple random split
            logging.warning("No labels have enough samples for stratification. Using simple random split.")
            train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    else:
        # All labels have enough samples for stratification
        train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label_id'], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_id'], random_state=42)
    
    # Save splits to CSV
    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Log split sizes and label distribution
    logging.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    # Log label distribution in each split
    train_labels = train_df['label_id'].nunique()
    val_labels = val_df['label_id'].nunique()
    test_labels = test_df['label_id'].nunique()
    total_labels = df['label_id'].nunique()
    
    logging.info(f"Label distribution: {train_labels}/{total_labels} in train, "
                f"{val_labels}/{total_labels} in validation, "
                f"{test_labels}/{total_labels} in test")

def preprocess_data(input_csv, config):
    """
    Preprocess raw CSV data
    
    Args:
        input_csv (str): Path to input CSV file
        config (dict): Configuration dictionary
    """
    # Check memory availability
    check_memory_availability(
        required_gb=config.get('model', {}).get('memory', {}).get('min_required_gb'),
        percentage=config.get('model', {}).get('memory', {}).get('max_usage_percentage', 0.8)
    )
    
    # Load CSV file
    try:
        df = pd.read_csv(input_csv)
        logging.info(f"Loaded {len(df)} records from {input_csv}")
    except Exception as e:
        logging.error(f"Error loading {input_csv}: {e}")
        raise
    
    # Check for required columns
    required_columns = ['description', 'bcea_code']
    for col in required_columns:
        if col not in df.columns:
            error_msg = f"Required column '{col}' missing from {input_csv}"
            logging.error(error_msg)
            raise ValueError(error_msg)
    
    # Apply max_samples limit if specified
    max_samples = config.get('data', {}).get('max_samples')
    if max_samples is not None and max_samples < len(df):
        # Take a stratified sample to preserve label distribution
        if 'bcea_code' in df.columns:
            # Stratified sampling by BCEA code
            df = df.groupby('bcea_code', group_keys=False).apply(
                lambda x: x.sample(min(len(x), max(1, int(max_samples * len(x) / len(df)))))
            )
            # If we get more samples than requested, take a random subset
            if len(df) > max_samples:
                df = df.sample(max_samples, random_state=42)
        else:
            # Simple random sampling if BCEA code not available
            df = df.sample(max_samples, random_state=42)
        
        logging.info(f"Using subset of {len(df)} samples as specified by max_samples={max_samples}")
    
    # Clean text descriptions
    logging.info("Cleaning text descriptions")
    df['description'] = clean_text(df['description'])
    
    # Validate BCEA codes
    logging.info("Validating BCEA codes")
    df = validate_bcea_codes(df, config['data']['processed_dir'])
    
    # Create label mappings
    logging.info("Creating label mappings")
    id_to_label, label_to_id = create_label_mappings(df, config['data']['mappings_dir'])
    
    # Replace BCEA codes with label IDs
    df['label_id'] = df['bcea_code'].map(label_to_id)
    
    # Split and save data
    logging.info("Splitting and saving data")
    split_and_save_data(df, config['data']['processed_dir'])
    
    logging.info("Preprocessing completed")

def run_preprocessing(config=None):
    """
    Run the preprocessing pipeline
    
    Args:
        config (dict, optional): Configuration dictionary. If None, load from config.yaml
    """
    # Load config if not provided
    if config is None:
        config = load_config("config.yaml")
    
    # Ensure processed directory exists
    os.makedirs(config['data']['processed_dir'], exist_ok=True)
    
    # Preprocess data
    input_csv = os.path.join(config['data']['raw_dir'], 'historical_bcea.csv')
    
    # Check if input file exists
    if not os.path.exists(input_csv):
        logging.error(f"Input file not found: {input_csv}")
        raise FileNotFoundError(f"Input file not found: {input_csv}")
        
    preprocess_data(input_csv, config)

if __name__ == "__main__":
    run_preprocessing()