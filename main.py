import argparse
import os
import logging
import yaml
import sys
from pathlib import Path

# Add the src directory to system path
sys.path.append(str(Path(__file__).parent))
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
    
    parser.add_argument('--input', type=str,
                        help='Path to input CSV file for prediction')
    
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation on test data')
    
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use for training (default: use all available data)')
    
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
    
    # Update config with command line arguments
    if args.max_samples is not None:
        # Add max_samples to data section if it doesn't exist
        if 'data' not in config:
            config['data'] = {}
        config['data']['max_samples'] = args.max_samples
    
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
        result = run_training(auto_evaluate=auto_evaluate)
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
    
    # Track the model and run_dir for potential later use
    model = None
    run_dir = None
    
    # Determine pipeline stages
    stages = []
    if not args.skip_training:
        stages.append("Preprocessing and Training")
    if args.input:
        stages.append("Prediction")
    if args.evaluate or (not args.skip_training and not args.evaluate):  # Auto-evaluate happens by default
        stages.append("Evaluation")
    
    # Print pipeline stages
    print_status(f"Pipeline will run these stages: {', '.join(stages)}", "success")
    
    # Track current stage
    current_stage = 0
    total_stages = len(stages)
    
    # Run preprocessing and training (with automatic evaluation unless explicitly requested)
    if not args.skip_training:
        current_stage += 1
        print_status(f"Stage {current_stage}/{total_stages}: Preprocessing and Training", "success")
        
        # Don't automatically evaluate if --evaluate flag is explicitly set
        auto_evaluate = not args.evaluate
        model, run_dir = run_preprocessing_and_training(config, auto_evaluate=auto_evaluate)
        
        print_status("Preprocessing and training completed successfully ✓", "success")
    
    # Run prediction
    if args.input:
        current_stage += 1
        print_status(f"Stage {current_stage}/{total_stages}: Prediction", "success")
        logging.info(f"=== Starting Prediction on {args.input} ===")
        try:
            run_prediction(args.input)
            logging.info("Prediction completed successfully")
            print_status("Prediction completed successfully ✓", "success")
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            print_status(f"Prediction failed: {e}", "error")
            sys.exit(1)
    
    # Run evaluation if explicitly requested
    # Note: If we already ran training with auto_evaluate=True, this will create a second evaluation
    if args.evaluate:
        current_stage += 1
        print_status(f"Stage {current_stage}/{total_stages}: Evaluation", "success")
        logging.info("=== Starting Explicit Evaluation ===")
        try:
            # If we have a model and run_dir from training, we could pass them here
            # but instead we'll do a fresh evaluation from disk for consistency
            metrics, report_path = run_evaluation()
            
            # If we have a run_dir from training, also save the evaluation results there
            if run_dir:
                logging.info(f"Also saving evaluation results to the training run directory: {run_dir}")
                try:
                    # Save metrics
                    with open(os.path.join(run_dir, 'explicit_eval_metrics.json'), 'w') as f:
                        json.dump(metrics, f, indent=2)
                        
                    # Create a symlink to the HTML report
                    html_link_path = os.path.join(run_dir, 'evaluation_report_link.html')
                    if os.path.exists(report_path):
                        try:
                            if os.path.exists(html_link_path):
                                os.remove(html_link_path)
                            os.symlink(os.path.relpath(report_path, os.path.dirname(html_link_path)), html_link_path)
                            logging.info(f"Created symlink to HTML report in training run directory")
                        except Exception as e:
                            logging.warning(f"Could not create symlink to HTML report: {e}")
                            # Try to copy the file instead
                            import shutil
                            shutil.copy2(report_path, os.path.join(run_dir, 'evaluation_report.html'))
                            logging.info(f"Copied HTML report to training run directory")
                            
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