import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
import sys
from pathlib import Path

# Add the parent directory to system path to import utils
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import load_config
from src.model import BCEADataset

def load_test_data(config):
    """
    Load test data and label mappings
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (test_df, id_to_label)
    """
    # Load test data
    test_path = os.path.join(config['data']['processed_dir'], 'test.csv')
    test_df = pd.read_csv(test_path)
    
    # Load mappings
    mappings_path = os.path.join(config['data']['mappings_dir'], 'bcea_mappings.json')
    with open(mappings_path, 'r') as f:
        mappings = json.load(f)
    
    id_to_label = {int(k): v for k, v in mappings['id_to_label'].items()}
    
    return test_df, id_to_label

def prepare_dataset(test_df, max_seq_length, model_dir, config=None):
    """
    Prepare test dataset
    
    Args:
        test_df (pd.DataFrame): Test data
        max_seq_length (int): Maximum sequence length
        model_dir (str): Model directory
        config (dict): Configuration options (optional)
        
    Returns:
        tuple: (test_dataset, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Determine which text field to use based on config and available columns
    use_business_name = config.get('data', {}).get('use_business_name', True) if config else True
    
    # Check if combined_text is already available
    if 'combined_text' in test_df.columns:
        input_field = 'combined_text'
        logging.info(f"Using existing 'combined_text' field for evaluation")
    # If we should use business name but need to create combined_text
    elif use_business_name and 'bus_name' in test_df.columns:
        separator = config.get('data', {}).get('text_separator', ' | ') if config else ' | '
        logging.info(f"Creating combined text field from business name and description with separator '{separator}'")
        test_df['combined_text'] = test_df.apply(
            lambda row: f"{row['bus_name']}{separator}{row['description']}" if row['bus_name'] else row['description'],
            axis=1
        )
        input_field = 'combined_text'
    # Default to description only
    else:
        input_field = 'description'
        logging.info(f"Using only 'description' field for evaluation")
    
    # Create dataset
    test_dataset = BCEADataset(
        test_df[input_field].tolist(),
        test_df['label_id'].tolist(),
        tokenizer,
        max_seq_length
    )
    
    return test_dataset, tokenizer

def compute_metrics(model, test_dataset, id_to_label, device):
    """
    Compute evaluation metrics
    
    Args:
        model: Model
        test_dataset: Test dataset
        id_to_label (dict): ID to label mapping
        device: PyTorch device
        
    Returns:
        dict: Metrics and predictions
    """
    # Import utils for progress tracking
    from src.utils import create_rich_progress, print_status
    
    # Create DataLoader
    batch_size = 16
    dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists for predictions and labels
    all_preds = []
    all_labels = []
    
    # Create progress bar for prediction
    total_batches = len(dataloader)
    print_status(f"Starting evaluation on {len(test_dataset)} samples", "info")
    
    # Create progress bar
    with create_rich_progress() as progress:
        pred_task = progress.add_task(
            f"[cyan]Evaluating model...", 
            total=total_batches
        )
        
        # Make predictions
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Get labels
                labels = batch['labels'].cpu().numpy()
                all_labels.extend(labels)
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                
                # Forward pass
                outputs = model(**batch)
                
                # Get predictions
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                
                # Update progress
                progress.update(pred_task, advance=1, description=f"[cyan]Evaluating: {batch_idx+1}/{total_batches} batches")
                
        # Mark prediction as complete
        progress.update(pred_task, description="[green]Prediction complete!")
    
    print_status(f"Completed predictions on {len(all_preds)} samples", "success")
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate both macro and weighted F1 scores
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    # Keep the original f1 variable for backward compatibility
    f1 = macro_f1
    
    # Compute top-3 accuracy
    print_status("Computing top-3 accuracy...", "info")
    top3_correct = 0
    
    with create_rich_progress() as progress:
        top3_task = progress.add_task(
            f"[cyan]Computing top-3 accuracy...", 
            total=len(dataloader)
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                # Get labels
                labels = batch['labels'].cpu().numpy()
                
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                
                # Forward pass
                outputs = model(**batch)
                
                # Get top-3 predictions
                logits = outputs.logits
                k_value = min(3, logits.shape[1])  # Ensure k is not larger than number of classes
                top3_preds = torch.topk(logits, k=k_value, dim=1).indices.cpu().numpy()
                
                # Check if true label is in top-3
                for i, label in enumerate(labels):
                    if label in top3_preds[i]:
                        top3_correct += 1
                
                # Update progress
                progress.update(top3_task, advance=1, description=f"[cyan]Top-3 accuracy: {batch_idx+1}/{len(dataloader)} batches")
            
        # Mark as complete
        progress.update(top3_task, description="[green]Top-3 accuracy computed!")
    
    top3_accuracy = top3_correct / len(test_dataset)
    print_status(f"Top-3 accuracy: {top3_accuracy:.3f}", "success")
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create classification report with zero_division=0 to handle missing classes
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    
    # Map IDs to BCEA codes for better readability
    label_map = {}
    for i, label in enumerate(all_labels):
        # Handle potential key errors with safe lookups
        try:
            label_int = int(label) if isinstance(label, (np.integer, int)) else label
            pred_int = int(all_preds[i]) if isinstance(all_preds[i], (np.integer, int)) else all_preds[i]
            
            bcea_code = id_to_label.get(str(label_int), str(label_int))
            pred_code = id_to_label.get(str(pred_int), str(pred_int))
            
            label_map[i] = {
                'true': bcea_code,
                'pred': pred_code,
                'correct': bcea_code == pred_code
            }
        except Exception as e:
            logging.warning(f"Error mapping label {label} or prediction {all_preds[i]}: {e}")
            label_map[i] = {
                'true': str(label),
                'pred': str(all_preds[i]),
                'correct': label == all_preds[i]
            }
    
    return {
        'accuracy': accuracy,
        'f1': f1,  # This is macro_f1 for backward compatibility
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'top3_accuracy': top3_accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'label_map': label_map
    }

def perform_error_analysis(metrics, test_df, run_dir=None, output_dir=None, config=None):
    """
    Perform error analysis
    
    Args:
        metrics (dict): Metrics from compute_metrics
        test_df (pd.DataFrame): Test data
        run_dir (str): Existing run directory to use (prioritized if provided)
        output_dir (str): Output base directory (used only if run_dir not provided)
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (error_dir, metrics_dict) - Path to error analysis directory and metrics summary
    """
    # Import utils for progress tracking
    from src.utils import print_status, print_section_header
    
    # Load config if not provided
    if config is None:
        config = load_config("config.yaml")
        
    # Display error analysis section header
    print_section_header("Error Analysis")
    print_status("Starting detailed error analysis...", "info")
    # Determine run directory
    if run_dir is None:
        if output_dir is None:
            # Use default output directory
            config = load_config("config.yaml")
            output_dir = config['output']['model_dir']
            
        # Create new run directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, 'runs', f'run_{timestamp}')
    
    # Create error analysis directory
    error_dir = os.path.join(run_dir, 'error_analysis')
    os.makedirs(error_dir, exist_ok=True)
    logging.info(f"Saving evaluation results to {error_dir}")
    
    # Get predictions and labels
    preds = metrics['predictions']
    labels = metrics['labels']
    label_map = metrics['label_map']
    
    # Create misclassifications DataFrame
    misclassified_indices = np.where(preds != labels)[0]
    
    if len(misclassified_indices) > 0:
        # Get descriptions and labels for misclassified examples
        misclassified_df = pd.DataFrame()
        
        # Include appropriate text field(s)
        misclassified_df['description'] = test_df.iloc[misclassified_indices]['description'].reset_index(drop=True)
        if 'bus_name' in test_df.columns:
            misclassified_df['bus_name'] = test_df.iloc[misclassified_indices]['bus_name'].reset_index(drop=True)
        if 'combined_text' in test_df.columns:
            misclassified_df['combined_text'] = test_df.iloc[misclassified_indices]['combined_text'].reset_index(drop=True)
        
        misclassified_df['true_bcea_code'] = [label_map[i]['true'] for i in misclassified_indices]
        misclassified_df['pred_bcea_code'] = [label_map[i]['pred'] for i in misclassified_indices]
        
        # Load BCEA reference data for descriptions if available
        try:
            bcea_ref_path = os.path.join(config.get('data', {}).get('reference_dir', 'data/reference'), 'bcea_ref.csv')
            if os.path.exists(bcea_ref_path):
                bcea_ref = pd.read_csv(bcea_ref_path)
                
                # Convert all BCEA codes to strings for consistent matching
                bcea_ref['bcea_code_str'] = bcea_ref['bcea_code'].astype(str)
                misclassified_df['true_bcea_code_str'] = misclassified_df['true_bcea_code'].astype(str)
                misclassified_df['pred_bcea_code_str'] = misclassified_df['pred_bcea_code'].astype(str)
                
                # Create mappings from string BCEA codes to descriptions
                bcea_desc_map = dict(zip(bcea_ref['bcea_code_str'], bcea_ref['short_description']))
                
                # Add true and predicted descriptions using string keys
                misclassified_df['true_description'] = misclassified_df['true_bcea_code_str'].map(
                    lambda x: bcea_desc_map.get(x, "No description available")
                )
                misclassified_df['pred_description'] = misclassified_df['pred_bcea_code_str'].map(
                    lambda x: bcea_desc_map.get(x, "No description available")
                )
                
                # Remove the temporary string columns
                misclassified_df = misclassified_df.drop(columns=['true_bcea_code_str', 'pred_bcea_code_str'])
                
                # Reorder and filter columns as specified
                if 'combined_text' in misclassified_df.columns:
                    misclassified_df = misclassified_df.drop(columns=['combined_text'])
                
                # Define the desired column order
                column_order = [
                    'bus_name',
                    'description',
                    'pred_bcea_code',
                    'pred_description',
                    'true_bcea_code',
                    'true_description'
                ]
                
                # Make sure all columns exist (handle the case where bus_name might be missing)
                for col in column_order:
                    if col not in misclassified_df.columns and col != 'bus_name':
                        misclassified_df[col] = ""
                
                # For bus_name specifically, add if missing
                if 'bus_name' not in misclassified_df.columns:
                    misclassified_df['bus_name'] = ""
                
                # Select and reorder columns
                misclassified_df = misclassified_df[
                    [col for col in column_order if col in misclassified_df.columns]
                ]
                
                logging.info(f"Added BCEA descriptions from reference file: {bcea_ref_path}")
        except Exception as e:
            logging.warning(f"Could not add BCEA descriptions: {e}")
        
        # Save all misclassifications
        all_misclassified_path = os.path.join(error_dir, 'all_misclassifications.csv')
        misclassified_df.to_csv(all_misclassified_path, index=False)
        logging.info(f"All {len(misclassified_indices)} misclassifications saved to {all_misclassified_path}")
        
        # Save top 10 misclassifications
        top_10_path = os.path.join(error_dir, 'top_10_misclassifications.csv')
        misclassified_df.head(10).to_csv(top_10_path, index=False)
        logging.info(f"Top 10 misclassifications saved to {top_10_path}")
    
    # Get top-k labels
    try:
        # If id_to_label mapping is available in metrics
        id_to_label = metrics.get('id_to_label', {})
        # Get label names (limited to first 50 if there are too many)
        label_names = [id_to_label.get(str(i), str(i)) for i in range(min(50, len(id_to_label)))]
        use_labels = len(label_names) <= 50  # Only use labels if there aren't too many
    except:
        use_labels = False
        label_names = None
    
    # Save confusion matrix (with labels if available and not too many)
    plt.figure(figsize=(14, 12))
    cm = metrics['confusion_matrix']
    # If there are too many labels, show a square submatrix of the most common classes
    if use_labels:
        sns.heatmap(cm, cmap='Blues', annot=True, fmt='d', 
                   xticklabels=label_names, yticklabels=label_names)
    else:
        # Otherwise just show the raw matrix without labels
        sns.heatmap(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    cm_path = os.path.join(error_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    logging.info(f"Confusion matrix saved to {cm_path}")
    
    # Save accuracy by class
    try:
        # Create dataframe for per-class accuracy
        class_report = metrics['classification_report']
        class_df = pd.DataFrame()
        
        # Extract relevant metrics
        for class_id, stats in class_report.items():
            if class_id not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(stats, dict):
                bcea_code = id_to_label.get(class_id, class_id)
                # Create a row and use pandas concat instead of deprecated append
                row_df = pd.DataFrame([{
                    'bcea_code': bcea_code,
                    'precision': stats['precision'],
                    'recall': stats['recall'],
                    'f1-score': stats['f1-score'],
                    'support': stats['support']
                }])
                class_df = pd.concat([class_df, row_df], ignore_index=True)
        
        # Sort by support (frequency) and save
        if not class_df.empty:
            class_df = class_df.sort_values('support', ascending=False)
            class_stats_path = os.path.join(error_dir, 'class_performance.csv')
            class_df.to_csv(class_stats_path, index=False)
            logging.info(f"Per-class performance metrics saved to {class_stats_path}")
    except Exception as e:
        logging.warning(f"Could not generate per-class performance metrics: {e}")
    
    # Save metrics to JSON
    metrics_dict = {
        'accuracy': metrics['accuracy'],
        'f1': metrics['f1'],
        'macro_f1': metrics['macro_f1'],
        'weighted_f1': metrics['weighted_f1'],
        'top3_accuracy': metrics['top3_accuracy'],
        'misclassification_rate': len(misclassified_indices) / len(labels) if len(labels) > 0 else 0
    }
    metrics_path = os.path.join(error_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    logging.info(f"Metrics saved to {metrics_path}")
    print_status(f"Metrics saved to JSON file", "success")
    
    # Save classification report
    report_path = os.path.join(error_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(metrics['classification_report'], f, indent=2)
    logging.info(f"Classification report saved to {report_path}")
    print_status(f"Classification report generated", "success")
    
    # Final success message for error analysis
    print_status(f"Error analysis completed successfully", "success")
    
    # Create a summary plot
    plt.figure(figsize=(10, 6))
    metrics_to_plot = {
        'Accuracy': metrics_dict['accuracy'],
        'Macro F1': metrics_dict['macro_f1'],
        'Weighted F1': metrics_dict['weighted_f1'],
        'Top-3 Accuracy': metrics_dict['top3_accuracy']
    }
    bars = plt.bar(metrics_to_plot.keys(), metrics_to_plot.values())
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.ylim(0, 1.1)
    plt.title('Model Performance Metrics')
    plt.tight_layout()
    summary_path = os.path.join(error_dir, 'performance_summary.png')
    plt.savefig(summary_path)
    plt.close()
    
    return error_dir, metrics_dict

def evaluate_model_with_run_dir(config, model=None, tokenizer=None, id_to_label=None, run_dir=None):
    """
    Evaluate model performance and save results to specific run directory
    
    Args:
        config (dict): Configuration dictionary
        model: Optional pre-loaded model (will load from disk if not provided)
        tokenizer: Optional pre-loaded tokenizer (will load from disk if not provided)
        id_to_label: Optional pre-loaded ID to label mapping (will load from disk if not provided)
        run_dir: Optional run directory (will create a new one if not provided)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    try:
        # Load test data and label mappings if needed
        if id_to_label is None:
            test_df, id_to_label = load_test_data(config)
        else:
            test_df, _ = load_test_data(config)
        logging.info(f"Loaded {len(test_df)} test records")
        
        # Ensure id_to_label keys are strings
        id_to_label = {str(k): v for k, v in id_to_label.items()}
        
        # Load model if not provided
        if model is None:
            model_dir = config['output']['best_model_dir']
            model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            logging.info(f"Loaded model from {model_dir}")
        
        # Prepare dataset and tokenizer if needed
        if tokenizer is None:
            model_dir = config['output']['best_model_dir']
            test_dataset, tokenizer = prepare_dataset(
                test_df,
                config['model']['max_seq_length'],
                model_dir,
                config
            )
        else:
            # Create dataset with provided tokenizer - reuse prepare_dataset to handle text field selection
            test_dataset, _ = prepare_dataset(
                test_df,
                config['model']['max_seq_length'],
                model_dir,
                config
            )
        
        # Set device
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        model.to(device)
        
        # Compute metrics
        logging.info("Computing evaluation metrics")
        metrics = compute_metrics(model, test_dataset, id_to_label, device)
        
        # Log metrics
        logging.info(f"Accuracy: {metrics['accuracy']:.3f}, "
                    f"Macro F1: {metrics['macro_f1']:.3f}, "
                    f"Weighted F1: {metrics['weighted_f1']:.3f}, "
                    f"Top-3 Accuracy: {metrics['top3_accuracy']:.3f}")
        
        # Perform error analysis
        logging.info("Performing error analysis")
        error_dir, eval_metrics = perform_error_analysis(metrics, test_df, run_dir=run_dir, config=config)
        
        # Generate HTML report
        logging.info("Generating HTML report")
        report_path = generate_html_report(eval_metrics, error_dir, run_dir, id_to_label, config)
        logging.info(f"HTML report generated at {report_path}")
        
        logging.info("Evaluation analysis saved")
        
        return eval_metrics
        
    except Exception as e:
        logging.error(f"Evaluation failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        
        # Return basic metrics dictionary even if evaluation fails
        return {
            "error": str(e),
            "status": "failed",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def generate_html_report(metrics, error_dir, run_dir, id_to_label=None, config=None):
    """
    Generate HTML evaluation report
    
    Args:
        metrics (dict): Evaluation metrics
        error_dir (str): Directory with error analysis files
        run_dir (str): Run directory
        id_to_label (dict): ID to label mapping
        config (dict): Configuration dictionary
        
    Returns:
        str: Path to saved HTML report
    """
    import base64
    import io
    from datetime import datetime
    
    # Import utils for progress tracking
    from src.utils import print_status, print_section_header
    
    # Show section header
    print_section_header("HTML Report Generation")
    print_status("Creating interactive HTML evaluation report...", "info")
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get model name
    model_name = config.get('model', {}).get('name', 'BCEA Model') if config else 'BCEA Model'
    
    # Prepare CSS styles
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .metrics-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            width: 100%;
        }
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .metric {
            text-align: center;
            padding: 15px;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            margin: 10px 0;
            flex: 1;
            min-width: 200px;
            margin-right: 10px;
        }
        .metric h3 {
            margin-top: 0;
            font-size: 16px;
            color: #7f8c8d;
        }
        .metric .value {
            font-size: 36px;
            font-weight: bold;
            color: #2980b9;
        }
        .image-card {
            background-color: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            margin-bottom: 20px;
            width: 100%;
        }
        .confusion-matrix {
            text-align: center;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 14px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        tr:hover {
            background-color: #e9ecef;
        }
        .footer {
            margin-top: 30px;
            text-align: center;
            font-size: 12px;
            color: #7f8c8d;
            border-top: 1px solid #ecf0f1;
            padding-top: 20px;
        }
        .summary-chart {
            text-align: center;
            margin: 20px 0;
        }
        .model-info {
            font-size: 14px;
            margin-bottom: 20px;
        }
        .model-info p {
            margin: 5px 0;
        }
    </style>
    """
    
    # Start building HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BCEA Model Evaluation Report</title>
        {css}
    </head>
    <body>
        <h1>BCEA Model Evaluation Report</h1>
        
        <div class="model-info">
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Model:</strong> {model_name}</p>
            <p><strong>Run Directory:</strong> {run_dir}</p>
        </div>
        
        <div class="container">
            <div class="metrics-card">
                <h2>Performance Metrics</h2>
                <div class="metric-container">
                    <div class="metric">
                        <h3>Accuracy</h3>
                        <div class="value">{metrics.get('accuracy', 0):.3f}</div>
                    </div>
                    <div class="metric">
                        <h3>Macro F1</h3>
                        <div class="value">{metrics.get('macro_f1', 0):.3f}</div>
                    </div>
                    <div class="metric">
                        <h3>Weighted F1</h3>
                        <div class="value">{metrics.get('weighted_f1', 0):.3f}</div>
                    </div>
                    <div class="metric">
                        <h3>Top-3 Accuracy</h3>
                        <div class="value">{metrics.get('top3_accuracy', 0):.3f}</div>
                    </div>
                    <div class="metric">
                        <h3>Misclassification Rate</h3>
                        <div class="value">{metrics.get('misclassification_rate', 0):.3f}</div>
                    </div>
                </div>
                
                <div class="summary-chart">
    """
    
    # Add summary chart image if available
    summary_chart_path = os.path.join(error_dir, 'performance_summary.png')
    if os.path.exists(summary_chart_path):
        # Convert image to base64 for embedding
        with open(summary_chart_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        html_content += f"""
                    <img src="data:image/png;base64,{encoded_image}" alt="Performance Summary" style="max-width:600px;">
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="image-card">
                <h2>Confusion Matrix</h2>
                <div class="confusion-matrix">
    """
    
    # Add confusion matrix image if available
    cm_path = os.path.join(error_dir, 'confusion_matrix.png')
    if os.path.exists(cm_path):
        # Convert image to base64 for embedding
        with open(cm_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        html_content += f"""
                    <img src="data:image/png;base64,{encoded_image}" alt="Confusion Matrix" style="max-width:800px;">
        """
    
    html_content += """
                </div>
            </div>
            
            <div class="metrics-card">
                <h2>Top Misclassifications</h2>
    """
    
    # Add top misclassifications table if available
    misclass_path = os.path.join(error_dir, 'top_10_misclassifications.csv')
    if os.path.exists(misclass_path):
        misclass_df = pd.read_csv(misclass_path)
        
        # Check if combined_text is available
        has_combined_text = 'combined_text' in misclass_df.columns
        has_bus_name = 'bus_name' in misclass_df.columns
        has_descriptions = 'true_description' in misclass_df.columns
        
        # Create table header with consistent column order matching the CSV
        header = """
                <table>
                    <thead>
                        <tr>
        """
        
        # Add business name header if available
        if has_bus_name:
            header += """
                            <th>Business Name</th>
            """
            
        header += """
                            <th>Description</th>
        """
        
        # No longer showing combined text
        
        header += """
                            <th>Predicted BCEA Code</th>
        """
        
        # Add predicted description header if available
        if has_descriptions:
            header += """
                            <th>Predicted BCEA Description</th>
            """
        
        header += """
                            <th>True BCEA Code</th>
        """
        
        # Add true description header if available
        if has_descriptions:
            header += """
                            <th>True BCEA Description</th>
            """
        
        header += """
                        </tr>
                    </thead>
                    <tbody>
        """
        
        html_content += header
        
        # Add rows for misclassifications (up to 10)
        for _, row in misclass_df.head(10).iterrows():
            # Start table row
            row_content = """
                        <tr>
            """
            
            # Add business name if available
            if has_bus_name:
                bus_name = row['bus_name'] if not pd.isna(row['bus_name']) else ""
                # Limit business name length
                if len(bus_name) > 50:
                    bus_name = bus_name[:47] + "..."
                row_content += f"""
                            <td>{bus_name}</td>
                """
            
            # Add description
            description = row['description']
            # Limit description length
            if len(description) > 100:
                description = description[:97] + "..."
            row_content += f"""
                            <td>{description}</td>
            """
            
            # No longer showing combined_text
            
            # Add the predicted code
            row_content += f"""
                            <td>{row['pred_bcea_code']}</td>
            """
            
            # Add predicted description if available
            if has_descriptions:
                pred_desc = row['pred_description']
                # Limit description length
                if len(pred_desc) > 70:
                    pred_desc = pred_desc[:67] + "..."
                row_content += f"""
                            <td>{pred_desc}</td>
                """
            
            # Add the true code
            row_content += f"""
                            <td>{row['true_bcea_code']}</td>
            """
            
            # Add true description if available
            if has_descriptions:
                true_desc = row['true_description']
                # Limit description length
                if len(true_desc) > 70:
                    true_desc = true_desc[:67] + "..."
                row_content += f"""
                            <td>{true_desc}</td>
                """
            
            row_content += """
                        </tr>
            """
            
            html_content += row_content
            
        html_content += """
                    </tbody>
                </table>
        """
    else:
        html_content += "<p>No misclassification data available.</p>"
    
    # Add class performance if available
    html_content += """
            </div>
            
            <div class="metrics-card">
                <h2>Class Performance (Top 10 by Support)</h2>
    """
    
    class_perf_path = os.path.join(error_dir, 'class_performance.csv')
    if os.path.exists(class_perf_path):
        try:
            class_df = pd.read_csv(class_perf_path)
            # Sort by support and take top 10
            class_df = class_df.sort_values('support', ascending=False).head(10)
            
            html_content += """
                <table>
                    <thead>
                        <tr>
                            <th>BCEA Code</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1 Score</th>
                            <th>Support</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for _, row in class_df.iterrows():
                html_content += f"""
                        <tr>
                            <td>{row['bcea_code']}</td>
                            <td>{row['precision']:.3f}</td>
                            <td>{row['recall']:.3f}</td>
                            <td>{row['f1-score']:.3f}</td>
                            <td>{int(row['support'])}</td>
                        </tr>
                """
                
            html_content += """
                    </tbody>
                </table>
            """
        except Exception as e:
            logging.warning(f"Could not load class performance data: {e}")
            html_content += "<p>Error loading class performance data.</p>"
    else:
        html_content += "<p>No class performance data available.</p>"
    
    # Finish HTML
    html_content += """
            </div>
        </div>
        
        <div class="footer">
            <p>Generated by BCEA Pipeline Evaluation System</p>
            <p>© 2025 BCEA Classification Project</p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML to file
    report_path = os.path.join(run_dir, 'evaluation_report.html')
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    logging.info(f"HTML evaluation report saved to {report_path}")
    print_status(f"HTML report saved to {report_path}", "success")
    
    return report_path

def evaluate_model(config):
    """
    Evaluate model performance using standard approach
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (metrics, report_path) - Evaluation metrics and path to HTML report
    """
    # Create a run directory with timestamp that will be used consistently
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config['output']['model_dir'], 'runs', f'run_{timestamp}')
    
    # Run the evaluation with the specific run directory
    metrics = evaluate_model_with_run_dir(config, run_dir=run_dir)
    
    # The error_dir is inside the run_dir
    error_dir = os.path.join(run_dir, 'error_analysis')
    
    # Load id_to_label for report
    _, id_to_label = load_test_data(config)
    
    # Generate HTML report
    report_path = generate_html_report(metrics, error_dir, run_dir, id_to_label, config)
    
    return metrics, report_path

def run_evaluation():
    """Run evaluation pipeline"""
    # Import utils for progress tracking and display
    from src.utils import print_status, print_section_header, format_metrics_summary
    
    # Start with a section header
    print_section_header("BCEA Model Evaluation")
    
    # Load config
    config = load_config("config.yaml")
    print_status("Configuration loaded successfully", "success")
    
    # Evaluate model
    print_status("Starting model evaluation process", "info")
    metrics, report_path = evaluate_model(config)
    
    # Create a report output
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Display results section
    print_section_header("Evaluation Results")
    
    # Format for display
    print_status(f"Accuracy: {metrics['accuracy']:.4f}", "info")
    print_status(f"Macro F1 Score: {metrics['macro_f1']:.4f}", "info")
    print_status(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}", "info")
    print_status(f"Top-3 Accuracy: {metrics['top3_accuracy']:.4f}", "info")
    print_status(f"HTML Report: {report_path}", "info")
    
    # Final success message
    print_status("Evaluation completed successfully", "success")
    
    logging.info(f"Evaluation completed successfully. HTML report saved to {report_path}")
    
    return metrics, report_path

if __name__ == "__main__":
    run_evaluation()