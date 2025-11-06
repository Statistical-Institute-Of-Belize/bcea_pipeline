import argparse
import copy
import os
import logging
import yaml
import sys
import json
from datetime import datetime

import pandas as pd
from src.utils import load_config, setup_logging, print_status, print_section_header, format_metrics_summary
from src.preprocess import run_preprocessing
from src.model import run_training
from src.predict import run_prediction
from src.evaluate import run_evaluation

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='BCEA Classification Pipeline')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip preprocessing and training steps')

    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Skip evaluation stage entirely')

    parser.add_argument('--force-update-best', action='store_true',
                        help='Always sync the best model directory after training/fine-tuning')

    parser.add_argument('--enable-optimizations', action='store_true',
                        help='Enable optional training optimizations (mixed precision, gradient tricks)')

    parser.add_argument('--fine-tune', action='store_true',
                        help='Fine-tune the latest model using correction CSVs')

    parser.add_argument('--corrections-dir', type=str, default='data/corrections/',
                        help='Directory containing correction CSV files (default: data/corrections/)')

    parser.add_argument('--input', type=str,
                        help='Path to input CSV file for prediction')

    parser.add_argument('--explain', action='store_true',
                        help='Generate per-record explanations during prediction when supported')

    parser.add_argument('--evaluate', action='store_true',
                        help='Force an evaluation run even if training was skipped')

    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use for preprocessing (legacy option)')

    parser.add_argument('--subset-size', type=int, default=None,
                        help='Train on a random subset of this many samples when the dataset is large')
    
    return parser.parse_args()

def load_and_update_config(args):
    """
    Load and update configuration based on arguments
    
    Args:
        args (argparse.Namespace): Parsed arguments
        
    Returns:
        dict: Updated configuration
    """
    # Load config from file
    config = load_config(args.config)

    data_cfg = config.setdefault('data', {})
    training_cfg = config.setdefault('training', {})
    prediction_cfg = config.setdefault('prediction', {})

    if args.max_samples is not None:
        data_cfg['max_samples'] = args.max_samples

    if args.subset_size is not None:
        training_cfg['subset_size'] = args.subset_size
        training_cfg['subset_force'] = True

    if args.enable_optimizations:
        training_cfg['enable_optimizations'] = True
        training_cfg.setdefault('mixed_precision', True)
        training_cfg.setdefault('gradient_accumulation_steps', max(2, training_cfg.get('gradient_accumulation_steps', 1)))

    if args.force_update_best:
        training_cfg['force_update_best'] = True

    if args.fine_tune:
        data_cfg['corrections_dir'] = args.corrections_dir

    if args.explain:
        prediction_cfg['explain'] = True
    else:
        prediction_cfg.setdefault('explain', False)

    data_cfg.setdefault('corrections_dir', args.corrections_dir)

    return config

def run_preprocessing_and_training(config, auto_evaluate=True):
    """
    Run preprocessing and training steps with optional automatic evaluation
    
    Args:
        config (dict): Configuration dictionary
        auto_evaluate (bool): Whether to automatically run evaluation after training
        
    Returns:
        tuple: (model, run_dir) if successful, (None, None) otherwise
    """
    # Run preprocessing
    logging.info("=== Starting Preprocessing ===")
    print_status("Preprocessing data", "success")
    try:
        # Show max samples if configured
        if 'max_samples' in config.get('data', {}):
            max_samples = config['data']['max_samples']
            print_status(f"Using maximum of {max_samples} samples for training", "success")

        if config.get('training', {}).get('enable_optimizations'):
            logging.info("Training optimizations enabled via CLI/config")
            print_status("Training optimizations enabled", "success")
            
        run_preprocessing(config)
        logging.info("Preprocessing completed successfully")
        print_status("Preprocessing completed", "success")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        print_status(f"Preprocessing failed: {e}", "error")
        sys.exit(1)

    # Run training with optional evaluation
    logging.info("=== Starting Training ===")
    print_status("Training model", "success")
    try:
        result = run_training(config=config, auto_evaluate=auto_evaluate)
        if result:
            model, run_dir = result
            logging.info(f"Training completed successfully. Run data saved to {run_dir}")
            print_status(f"Training completed in {os.path.basename(run_dir)}", "success")
            return model, run_dir
        else:
            logging.error("Training returned no results")
            print_status("Training returned no results", "error")
            return None, None
    except Exception as e:
        logging.error(f"Training failed: {e}")
        print_status(f"Training failed: {e}", "error")
        sys.exit(1)


def run_fine_tuning(config):
    """Fine-tune the current model using correction CSVs if available."""

    data_cfg = config.get('data', {})
    corrections_dir = data_cfg.get('corrections_dir')

    if not corrections_dir:
        logging.info("No corrections directory configured; skipping fine-tuning")
        return

    if not os.path.exists(corrections_dir):
        logging.warning(f"Corrections directory not found: {corrections_dir}")
        return

    correction_files = [f for f in os.listdir(corrections_dir) if f.endswith('.csv')]
    if not correction_files:
        logging.warning(f"No correction CSVs detected in {corrections_dir}; skipping fine-tuning")
        return

    mappings_dir = data_cfg.get('mappings_dir')
    if not mappings_dir:
        raise ValueError("Mappings directory not configured; cannot map correction labels")

    mappings_path = os.path.join(mappings_dir, 'bcea_mappings.json')
    if not os.path.exists(mappings_path):
        raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

    with open(mappings_path, 'r') as mapping_file:
        mappings_data = json.load(mapping_file)
    label_to_id = {code: str(idx) for code, idx in mappings_data.get('label_to_id', {}).items()}

    correction_dfs = []
    for filename in correction_files:
        file_path = os.path.join(corrections_dir, filename)
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            logging.warning(f"Could not read correction file {filename}: {exc}")
            continue

        src_description_col = None
        for candidate in ('description', 'text', 'combined_text'):
            if candidate in df.columns:
                src_description_col = candidate
                break

        code_col = None
        for candidate in ('bcea_code', 'predicted_code', 'corrected_bcea_code'):
            if candidate in df.columns:
                code_col = candidate
                break

        if not src_description_col or not code_col:
            logging.warning(f"Skipping {filename}: missing description or code column")
            continue

        corrections_subset = df[[src_description_col, code_col]].dropna()
        corrections_subset = corrections_subset.rename(columns={
            src_description_col: 'combined_text',
            code_col: 'bcea_code'
        })
        corrections_subset['label_id'] = corrections_subset['bcea_code'].map(label_to_id)
        corrections_subset = corrections_subset.dropna(subset=['label_id'])
        correction_dfs.append(corrections_subset)

    if not correction_dfs:
        logging.warning("No usable corrections found; skipping fine-tuning")
        return

    corrections_df = pd.concat(correction_dfs, ignore_index=True)

    processed_dir = data_cfg.get('processed_dir')
    if not processed_dir:
        raise ValueError("Processed data directory not configured; cannot fine-tune")

    train_path = os.path.join(processed_dir, 'train.csv')
    val_path = os.path.join(processed_dir, 'val.csv')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError("Processed train/val splits not found. Run preprocessing first.")

    train_df = pd.read_csv(train_path)
    if 'combined_text' not in train_df.columns:
        if 'description' in train_df.columns:
            train_df['combined_text'] = train_df['description']
        else:
            raise ValueError("Training data missing 'combined_text' or 'description' columns.")

    fine_tune_dir = os.path.join(processed_dir, 'fine_tune')
    os.makedirs(fine_tune_dir, exist_ok=True)

    if 'label_id' not in train_df.columns:
        if 'bcea_code' in train_df.columns:
            train_df['label_id'] = train_df['bcea_code'].map(label_to_id)
        else:
            raise ValueError("Training data missing 'label_id'; cannot merge corrections")

    combined_df = pd.concat([train_df, corrections_df], ignore_index=True)
    combined_df.to_csv(os.path.join(fine_tune_dir, 'train.csv'), index=False)

    val_df = pd.read_csv(val_path)
    val_df.to_csv(os.path.join(fine_tune_dir, 'val.csv'), index=False)

    fine_tune_config = copy.deepcopy(config)
    fine_tune_config.setdefault('data', {})['processed_dir'] = fine_tune_dir

    training_cfg = fine_tune_config.setdefault('training', {})
    training_cfg.setdefault('epochs', max(1, training_cfg.get('fine_tune_epochs', 2)))
    training_cfg.setdefault('learning_rate', min(1e-5, training_cfg.get('learning_rate', 3.5e-5)))
    training_cfg['is_fine_tune'] = True

    logging.info(f"Fine-tuning with {len(corrections_df)} correction records")
    run_training(config=fine_tune_config, auto_evaluate=False)

def main():
    """Main pipeline function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load and update config
    config = load_and_update_config(args)
    
    # Get start time and log path from setup_logging
    start_time, log_file = setup_logging(
        config['logging']['dir'], 
        config['logging']['level'], 
        config['logging']['style']
    )
    
    model = None
    run_dir = None

    skip_evaluation = args.skip_evaluation
    force_evaluation = args.evaluate

    stages = []
    if not args.skip_training:
        stages.append("Preprocessing and Training")
    if args.fine_tune:
        stages.append("Fine-tuning")
    if args.input:
        stages.append("Prediction")
    if (not skip_evaluation) or force_evaluation:
        stages.append("Evaluation")

    if stages:
        print_status(f"Pipeline will run these stages: {', '.join(stages)}", "success")
    else:
        print_status("No stages selected – exiting", "warning")
        return

    current_stage = 0
    total_stages = len(stages)

    if not args.skip_training:
        current_stage += 1
        print_status(f"Stage {current_stage}/{total_stages}: Preprocessing and Training", "success")

        auto_evaluate = not skip_evaluation and not force_evaluation
        model, run_dir = run_preprocessing_and_training(config, auto_evaluate=auto_evaluate)

        print_status("Preprocessing and training completed successfully ✓", "success")

    if args.fine_tune:
        current_stage += 1
        print_status(f"Stage {current_stage}/{total_stages}: Fine-tuning", "success")
        try:
            run_fine_tuning(config)
            print_status("Fine-tuning completed successfully ✓", "success")
        except Exception as exc:
            logging.error(f"Fine-tuning failed: {exc}")
            print_status(f"Fine-tuning failed: {exc}", "error")
            sys.exit(1)

    if args.input:
        current_stage += 1
        print_status(f"Stage {current_stage}/{total_stages}: Prediction", "success")
        logging.info(f"=== Starting Prediction on {args.input} ===")
        try:
            prediction_explain = args.explain or config.get('prediction', {}).get('explain', False)
            predictions_df = run_prediction(args.input, config=config, explain=prediction_explain)
            logging.info("Prediction completed successfully")
            print_status("Prediction completed successfully ✓", "success")

            print_section_header("Prediction Summary")
            print(f"  • Input file: {args.input}")
            print(f"  • Records processed: {len(predictions_df)}")

            if 'confidence_grade' in predictions_df.columns:
                grade_counts = predictions_df['confidence_grade'].value_counts().to_dict()
                print("\nConfidence grade distribution:")
                for grade in ['very_high', 'high', 'medium', 'low', 'very_low']:
                    count = grade_counts.get(grade, 0)
                    percentage = (count / len(predictions_df)) * 100 if len(predictions_df) else 0
                    print(f"  • {grade.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

            print("\nThe prediction output includes:")
            print("  • Original data from input file")
            print("  • Primary prediction (predicted_code) with industry description")
            print("  • Confidence score, grade, and fallback flag")
            print("  • Two alternative predictions with descriptions and confidence scores")

            print("\nPrediction output saved to a timestamped CSV file in the 'data/processed/' directory.")

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            print_status(f"Prediction failed: {e}", "error")
            sys.exit(1)

    if (not skip_evaluation) or force_evaluation:
        current_stage += 1
        print_status(f"Stage {current_stage}/{total_stages}: Evaluation", "success")
        logging.info("=== Starting Evaluation ===")
        try:
            metrics, report_path = run_evaluation(config=config)

            if run_dir:
                logging.info(f"Also saving evaluation results to the training run directory: {run_dir}")
                try:
                    with open(os.path.join(run_dir, 'explicit_eval_metrics.json'), 'w') as f:
                        json.dump(metrics, f, indent=2)

                    html_link_path = os.path.join(run_dir, 'evaluation_report_link.html')
                    if os.path.exists(report_path):
                        try:
                            if os.path.exists(html_link_path):
                                os.remove(html_link_path)
                            os.symlink(os.path.relpath(report_path, os.path.dirname(html_link_path)), html_link_path)
                            logging.info("Created symlink to HTML report in training run directory")
                        except Exception as e:
                            logging.warning(f"Could not create symlink to HTML report: {e}")
                            import shutil
                            if os.path.exists(report_path):
                                shutil.copy2(report_path, os.path.join(run_dir, 'evaluation_report.html'))
                                logging.info("Copied HTML report to training run directory")
                            else:
                                logging.warning(f"Could not find evaluation report at: {report_path}")

                except Exception as e:
                    logging.warning(f"Could not save evaluation metrics to run directory: {e}")
                    print_status(f"Could not save evaluation metrics to run directory: {e}", "warning")

            print_status("Evaluation completed successfully ✓", "success")
            logging.info("Evaluation completed successfully")
        except Exception as e:
            logging.error(f"Evaluation failed: {e}")
            print_status(f"Evaluation failed: {e}", "error")
            sys.exit(1)
    
    # If we have a run_dir, log it for reference
    if run_dir:
        logging.info(f"Run data saved to {run_dir}")
    
    # Calculate total runtime
    end_time = datetime.now()
    total_runtime = end_time - start_time
    hours, remainder = divmod(total_runtime.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    runtime_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    
    # Print completion summary
    print_section_header("Pipeline Completed Successfully")
    
    # Print metrics if available
    if 'metrics' in locals() and metrics:
        print(format_metrics_summary(metrics))
        print()
    
    # Print time information
    print("Time Information:")
    print(f"  • Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  • Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  • Total time: {runtime_str}")
    print()
    
    # Print log file location
    print(f"Full logs saved to: {log_file}")
    
    logging.info("=== Pipeline Completed Successfully ===")

if __name__ == "__main__":
    main()
