import os
import yaml
from pydantic_settings import BaseSettings
from typing import Dict, Any


class APISettings(BaseSettings):
    """API settings for the BCEA pipeline."""
    APP_NAME: str = "BCEA Classification API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API for predicting BCEA (Business Classification with Economic Activity) codes from business names and descriptions"
    
    # Default model settings - using absolute paths to ensure consistency
    MODEL_PATH: str = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models/best_model/"))
    REFERENCE_FILE: str = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/reference/bcea_ref.csv"))
    MAX_SEQ_LENGTH: int = 160
    
    # Override from config file
    def load_from_config(self, config_path: str = None) -> None:
        """Load settings from the main config.yaml file."""
        if config_path is None:
            # Use absolute path to the config file
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.yaml"
            )
            
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Using default settings.")
            return
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update settings from config
        if 'output' in config and 'best_model_dir' in config['output']:
            model_path = config['output']['best_model_dir']
            # Make sure path is absolute
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    model_path
                ))
            self.MODEL_PATH = model_path
        
        if 'model' in config and 'max_seq_length' in config['model']:
            self.MAX_SEQ_LENGTH = config['model']['max_seq_length']
        
        if 'data' in config and 'reference_dir' in config['data']:
            ref_dir = config['data']['reference_dir']
            ref_file = os.path.join(ref_dir, 'bcea_ref.csv')
            # Use the absolute path from config if provided
            if os.path.isabs(ref_file):
                self.REFERENCE_FILE = ref_file
            else:
                # Otherwise, make relative path absolute
                self.REFERENCE_FILE = os.path.abspath(os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                    ref_file
                ))

# Create settings instance
settings = APISettings()

# Load from config file at module import time
# Use absolute path from the project root
settings.load_from_config()