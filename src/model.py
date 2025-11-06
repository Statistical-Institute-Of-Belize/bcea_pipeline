import os
import json
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
import yaml
import platform

from .utils import (
    load_config,
    create_rich_progress,
    check_memory_availability,
    cleanup_memory,
    print_status,
)

class BCEADataset(Dataset):
    """Dataset class for BCEA classification with memory optimization"""
    
    def __init__(self, text_inputs, labels, tokenizer, max_seq_length, lazy_loading=True):
        """
        Initialize dataset with memory optimization
        
        Args:
            text_inputs (list): List of text inputs (combined business name and description)
            labels (list): List of label IDs (as strings)
            tokenizer: Hugging Face tokenizer
            max_seq_length (int): Maximum sequence length for tokenization
            lazy_loading (bool): Whether to tokenize on-the-fly (True) or precompute (False)
        """
        self.text_inputs = text_inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.lazy_loading = lazy_loading
        
        # Pre-tokenize if not using lazy loading
        self.encodings = None
        if not lazy_loading:
            self._tokenize_all_texts()
    
    def _tokenize_all_texts(self):
        """Tokenize all texts at once (only used when lazy_loading=False)"""
        try:
            # Replace empty strings with placeholder
            clean_inputs = [
                t if isinstance(t, str) and t.strip() else "unknown description" 
                for t in self.text_inputs
            ]
            
            # Batch tokenize all at once
            self.encodings = self.tokenizer(
                clean_inputs,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
            
            # Clear text input list to save memory
            self.text_inputs = None
            
        except Exception as e:
            logging.error(f"Error during batch tokenization: {e}")
            # Fall back to lazy loading if batch tokenization fails
            self.lazy_loading = True
    
    def __len__(self):
        if self.lazy_loading:
            return len(self.text_inputs)
        else:
            return len(self.encodings['input_ids'])
    
    def __getitem__(self, idx):
        try:
            if self.lazy_loading:
                # Get the text input (combined business name and description)
                text_input = self.text_inputs[idx]
                
                # Handle empty strings
                if not isinstance(text_input, str) or not text_input.strip():
                    text_input = "unknown description"
                
                # Tokenize text with error handling
                encoding = self.tokenizer(
                    text_input,
                    padding='max_length',
                    truncation=True,
                    max_length=self.max_seq_length,
                    return_tensors='pt'
                )
                
                # Convert to dict and remove batch dimension from tensors
                item = {
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                }
            else:
                # Use pre-tokenized data
                item = {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx]
                }
            
            # Add label if available
            if self.labels is not None:
                # Convert string label to int safely
                label_value = int(self.labels[idx]) if self.labels[idx] is not None else 0
                item['labels'] = torch.tensor(label_value, dtype=torch.long)
            
            return item
            
        except Exception as e:
            # Log the error and return a default item
            logging.warning(f"Error processing item at index {idx}: {e}")
            
            # Create default item with appropriate shapes
            input_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
            attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
            
            # Set the first token to be the <PAD> token
            input_ids[0] = self.tokenizer.pad_token_id or 0
            
            item = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            if self.labels is not None:
                item['labels'] = torch.tensor(0, dtype=torch.long)
                
            return item


def _select_training_strategy(device, mixed_precision):
    """Return training mode: cuda_amp, mps_amp, or standard."""
    if mixed_precision and device.type == 'cuda':
        return 'cuda_amp'
    if mixed_precision and device.type == 'mps':
        return 'mps_amp'
    return 'standard'


def _move_batch_to_device(batch, device):
    """Move every tensor in the batch to the target device."""
    return {key: value.to(device) for key, value in batch.items()}


def _should_step(batch_idx, gradient_accumulation_steps):
    """Return True when it's time to update weights."""
    return (batch_idx + 1) % gradient_accumulation_steps == 0


def _maybe_cleanup_memory(device, cleanup_frequency, batch_idx):
    """Free memory following the legacy per-batch cleanup rules."""
    if cleanup_frequency <= 0 or not _should_step(batch_idx, cleanup_frequency):
        return

    if device.type == 'mps':
        cleanup_memory(device)
        import gc
        gc.collect()
        if torch.backends.mps.is_available() and hasattr(torch._C, "_mps_empty_cache"):
            try:
                torch._C._mps_empty_cache()
            except Exception:
                pass
    else:
        cleanup_memory(device)


def _train_one_epoch(
    *,
    model,
    optimizer,
    scheduler,
    train_loader,
    device,
    strategy,
    scaler,
    gradient_accumulation_steps,
    log_batch_frequency,
    memory_cleanup_freq,
    epoch_index,
    total_epochs,
):
    """Run a full training epoch and return the average loss."""

    def cuda_amp_step(batch_idx, batch):
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss / gradient_accumulation_steps
        scaler.scale(loss).backward()

        if _should_step(batch_idx, gradient_accumulation_steps):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        return loss.item() * gradient_accumulation_steps

    def mps_amp_step(batch_idx, batch):
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float16)
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        torch.set_default_dtype(orig_dtype)

        scale = 128.0
        (loss * scale).backward()

        if _should_step(batch_idx, gradient_accumulation_steps):
            skip_step = False
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.div_(scale)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        skip_step = True
            if skip_step:
                logging.warning("Skipping optimizer step due to NaN/inf gradients")
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            optimizer.zero_grad()

        return loss.item() * gradient_accumulation_steps

    def standard_step(batch_idx, batch):
        outputs = model(**batch)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()

        if _should_step(batch_idx, gradient_accumulation_steps):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        return loss.item() * gradient_accumulation_steps

    step_fn = {
        'cuda_amp': cuda_amp_step,
        'mps_amp': mps_amp_step,
        'standard': standard_step,
    }[strategy]

    train_loss = 0.0
    progress = create_rich_progress()
    optimizer.zero_grad()

    with progress:
        task_id = progress.add_task(
            f"[green]Epoch {epoch_index + 1}/{total_epochs}",
            total=len(train_loader)
        )

        for batch_idx, batch in enumerate(train_loader):
            batch = _move_batch_to_device(batch, device)

            loss_value = step_fn(batch_idx, batch)
            train_loss += loss_value

            display_loss = loss_value
            progress.update(
                task_id,
                advance=1,
                description=(
                    f"[green]Epoch {epoch_index + 1}/{total_epochs}"
                    f" - Loss: {display_loss:.3f}"
                )
            )

            if log_batch_frequency and (batch_idx + 1) % log_batch_frequency == 0:
                logging.info(
                    "Epoch %s/%s - Batch %s/%s - Loss: %.3f",
                    epoch_index + 1,
                    total_epochs,
                    batch_idx + 1,
                    len(train_loader),
                    display_loss,
                )
                print(
                    "{'loss': %.4f, 'learning_rate': %s, 'epoch': %.2f}" % (
                        display_loss,
                        scheduler.get_last_lr()[0],
                        epoch_index + (batch_idx + 1) / len(train_loader),
                    )
                )

            if memory_cleanup_freq:
                _maybe_cleanup_memory(device, memory_cleanup_freq, batch_idx)

    return train_loss / len(train_loader)


def _run_validation(model, val_loader, val_dataset, device, batch_size):
    """Evaluate the model on the validation set and return metrics."""
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []

    cleanup_memory(device)

    validation_batch_size = max(1, batch_size // 2)
    if validation_batch_size != batch_size:
        loader = DataLoader(
            val_dataset,
            batch_size=validation_batch_size,
            num_workers=0,
            pin_memory=False,
        )
    else:
        loader = val_loader

    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            val_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels)

    avg_val_loss = val_loss / len(loader)
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_macro_f1 = f1_score(val_labels, val_preds, average='macro')
    val_weighted_f1 = f1_score(val_labels, val_preds, average='weighted')
    val_f1 = f1_score(val_labels, val_preds, average='micro')

    model.train()

    return avg_val_loss, val_accuracy, val_macro_f1, val_weighted_f1, val_f1


def _get_training_param(config, key, default=None):
    training_cfg = config.get('training', {})
    if key in training_cfg and training_cfg[key] is not None:
        return training_cfg[key]
    return config.get('model', {}).get(key, default)

def load_data_and_mappings(processed_dir, mappings_dir):
    """
    Load training data and label mappings
    
    Args:
        processed_dir (str): Directory containing processed data
        mappings_dir (str): Directory containing label mappings
        
    Returns:
        tuple: (train_df, val_df, id_to_label dict, label_to_id dict)
    """
    try:
        # Load train and validation data
        train_path = os.path.join(processed_dir, 'train.csv')
        val_path = os.path.join(processed_dir, 'val.csv')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found: {val_path}")
            
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        
        # Load label mappings
        mappings_path = os.path.join(mappings_dir, 'bcea_mappings.json')
        
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Mappings file not found: {mappings_path}")
            
        with open(mappings_path, 'r') as f:
            mappings = json.load(f)
        
        # Ensure consistent mapping format (all keys and values as strings)
        id_to_label = {str(k): v for k, v in mappings['id_to_label'].items()}
        label_to_id = {k: str(v) for k, v in mappings['label_to_id'].items()}
        
        logging.info(f"Loaded {len(id_to_label)} label mappings")
        logging.info(f"Sample mapping: {list(id_to_label.items())[:3]}")
        
        return train_df, val_df, id_to_label, label_to_id
        
    except Exception as e:
        logging.error(f"Error loading data and mappings: {e}")
        raise

def prepare_datasets(train_df, val_df, max_seq_length, model_name, config=None):
    """
    Prepare train and validation datasets with memory optimizations
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        max_seq_length (int): Maximum sequence length
        model_name (str): Model name for tokenizer
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_dataset, val_dataset, tokenizer)
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine dataset loading strategy based on config
    dataset_opts = {}
    if config:
        if 'training' in config and 'dataset_opts' in config['training']:
            dataset_opts = config['training']['dataset_opts']
        elif 'model' in config and 'dataset_opts' in config['model']:
            dataset_opts = config['model']['dataset_opts']
    lazy_loading = dataset_opts.get('lazy_loading', True)
        
    logging.info(f"Dataset loading strategy: {'lazy loading' if lazy_loading else 'precompute tokenization'}")
    
    # Use combined_text field if available, otherwise fallback to description only
    input_field = 'combined_text' if 'combined_text' in train_df.columns else 'description'
    logging.info(f"Using '{input_field}' as input for model training")
    
    # Create training dataset
    train_dataset = BCEADataset(
        train_df[input_field].tolist(),
        train_df['label_id'].tolist(),
        tokenizer,
        max_seq_length,
        lazy_loading=lazy_loading
    )
    
    # Create validation dataset - always use precomputed for validation (typically smaller)
    val_dataset = BCEADataset(
        val_df[input_field].tolist(),
        val_df['label_id'].tolist(),
        tokenizer,
        max_seq_length,
        lazy_loading=False  # Precompute for validation set
    )
    
    # Log dataset sizes
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset, tokenizer

def initialize_model(model_name, num_labels):
    """
    Initialize model for sequence classification
    
    Args:
        model_name (str): Model name from Hugging Face
        num_labels (int): Number of labels for classification
        
    Returns:
        model: Hugging Face model
    """
    # Initialize model with classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="single_label_classification"
    )
    
    return model

def get_device(config):
    """
    Get the appropriate device based on config and availability
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        device: PyTorch device
    """
    device_preference = config['model'].get('device', 'auto')
    
    if device_preference == 'auto':
        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            if torch.backends.mps.is_available():
                logging.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
                # Configure MPS memory management
                from .utils import configure_mps_memory
                configure_mps_memory()
                return torch.device('mps')
        
        # Check for CUDA
        if torch.cuda.is_available():
            logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda')
        
        # Default to CPU
        logging.info("Using CPU for training")
        return torch.device('cpu')
    
    elif device_preference == 'mps':
        if torch.backends.mps.is_available():
            logging.info("Using MPS (Metal Performance Shaders) for Apple Silicon")
            # Configure MPS memory management
            from .utils import configure_mps_memory
            configure_mps_memory()
            return torch.device('mps')
        else:
            logging.warning("MPS requested but not available. Falling back to CPU.")
            return torch.device('cpu')
    
    elif device_preference == 'cuda':
        if torch.cuda.is_available():
            logging.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            return torch.device('cuda')
        else:
            logging.warning("CUDA requested but not available. Falling back to CPU.")
            return torch.device('cpu')
    
    else:
        logging.info("Using CPU for training (explicitly requested)")
        return torch.device('cpu')

def train_model_with_params(model, train_dataset, val_dataset, config):
    """Train model with specified parameters and return the best checkpoint."""

    memory_cfg = config.get('training', {}).get('memory') or config.get('model', {}).get('memory', {})
    check_memory_availability(
        required_gb=memory_cfg.get('min_required_gb'),
        percentage=memory_cfg.get('max_usage_percentage', 0.8)
    )

    batch_size = _get_training_param(config, 'batch_size', 32)
    epochs = _get_training_param(config, 'epochs', 3)
    learning_rate = _get_training_param(config, 'learning_rate', 5e-5)
    gradient_accumulation_steps = _get_training_param(config, 'gradient_accumulation_steps', 1)
    mixed_precision = _get_training_param(config, 'mixed_precision', config.get('model', {}).get('mixed_precision', False))
    early_stopping_patience = _get_training_param(config, 'early_stopping_patience', 0)
    log_batch_frequency = config['logging'].get('log_batch_frequency', 10)

    device = get_device(config)

    dataloader_opts = config.get('training', {}).get('dataloader_opts', config.get('model', {}).get('dataloader_opts', {}))
    num_workers = dataloader_opts.get('num_workers', 1)
    pin_memory = dataloader_opts.get('pin_memory', False)
    prefetch_factor = dataloader_opts.get('prefetch_factor', 2)

    print_status(f"Training dataset size: {len(train_dataset)} examples", "success")
    logging.info("DataLoader configuration: workers=%s, pin_memory=%s", num_workers, pin_memory)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    print_status(f"Using {device} device for training", "success")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs // max(1, gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=max(1, total_steps)
    )

    best_val_loss = float('inf')
    best_model = None
    patience_counter = 0

    strategy = _select_training_strategy(device, mixed_precision)
    scaler = torch.cuda.amp.GradScaler() if strategy == 'cuda_amp' else None

    if mixed_precision and strategy == 'cuda_amp':
        logging.info("Using CUDA mixed precision training with GradScaler")
        print_status("Using mixed precision training (CUDA)", "success")
    elif mixed_precision and strategy == 'mps_amp':
        logging.info("Using MPS mixed precision training with bfloat16")
        print_status("Using mixed precision training (MPS)", "success")
    elif mixed_precision:
        logging.info("Mixed precision requested but not supported on this device")

    if early_stopping_patience > 0:
        print_status(f"Early stopping enabled with patience {early_stopping_patience}", "success")

    print_status("Training Configuration:", "success")
    print_status(f"  • Dataset size: {len(train_dataset):,} examples", "success", indent=1)
    print_status(f"  • Batch size: {batch_size}", "success", indent=1)
    print_status(f"  • Gradient accumulation steps: {gradient_accumulation_steps}", "success", indent=1)
    print_status(f"  • Epochs: {epochs}", "success", indent=1)
    print_status(f"  • Learning rate: {learning_rate}", "success", indent=1)
    print_status(f"  • Total training steps: ~{total_steps}", "success", indent=1)
    print_status(f"  • Model size: {sum(p.numel() for p in model.parameters()):,} parameters", "success", indent=1)

    memory_cleanup_freq = memory_cfg.get('cleanup_frequency', 100)

    from datetime import datetime
    print_status("Starting model training", "success")
    start_time = datetime.now()

    for epoch in range(epochs):
        avg_train_loss = _train_one_epoch(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            device=device,
            strategy=strategy,
            scaler=scaler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_batch_frequency=log_batch_frequency,
            memory_cleanup_freq=memory_cleanup_freq,
            epoch_index=epoch,
            total_epochs=epochs,
        )

        (
            avg_val_loss,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1,
            val_f1,
        ) = _run_validation(model, val_loader, val_dataset, device, batch_size)

        logging.info(
            "Epoch %s/%s - Train Loss: %.3f, Val Loss: %.3f, Val Accuracy: %.3f, Val Macro F1: %.3f, Val Weighted F1: %.3f",
            epoch + 1,
            epochs,
            avg_train_loss,
            avg_val_loss,
            val_accuracy,
            val_macro_f1,
            val_weighted_f1,
        )

        print(
            "{'eval_loss': %.4f, 'eval_accuracy': %.4f, 'eval_macro_f1': %.4f, 'eval_weighted_f1': %.4f, 'eval_f1': %.4f, 'epoch': %d}" % (
                avg_val_loss,
                val_accuracy,
                val_macro_f1,
                val_weighted_f1,
                val_f1,
                epoch + 1,
            )
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            logging.info("New best model saved with validation loss: %.3f", best_val_loss)
            print_status(f"New best model saved (val_loss: {best_val_loss:.3f})", "success")
            patience_counter = 0
        else:
            patience_counter += 1
            logging.info(
                "Validation loss did not improve. Patience: %s/%s",
                patience_counter,
                early_stopping_patience,
            )
            if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                logging.info("Early stopping triggered after %s epochs", epoch + 1)
                print_status(f"Early stopping triggered after {epoch+1} epochs", "warning")
                break

        cleanup_memory(device)
        logging.info("Completed epoch %s/%s", epoch + 1, epochs)

        if hasattr(train_loader, 'dataset') and hasattr(train_loader.dataset, 'encodings'):
            if not train_loader.dataset.lazy_loading and epoch < epochs - 1:
                logging.info("Reloading training dataset encodings to free memory")
                train_loader.dataset._tokenize_all_texts()

    if best_model is not None:
        model.load_state_dict(best_model)
        logging.info("Loaded best model from training")

    end_time = datetime.now()
    training_time = end_time - start_time
    hours, remainder = divmod(training_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    training_time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    print_status(f"Training completed in {training_time_str}", "success")

    return model, best_val_loss

def save_best_model(model, tokenizer, best_model_dir, id_to_label):
    """
    Save trained model, tokenizer, and label mappings
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        best_model_dir (str): Directory to save model
        id_to_label (dict): ID to label mapping
    """
    # Create directory if it doesn't exist
    os.makedirs(best_model_dir, exist_ok=True)
    
    # Save model
    model.save_pretrained(best_model_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(best_model_dir)
    
    # Save ID to label mapping
    with open(os.path.join(best_model_dir, 'id_to_label.json'), 'w') as f:
        json.dump(id_to_label, f, indent=2)
    
    logging.info(f"Model saved to {best_model_dir}")

def train_model(config):
    """
    Train model with configuration
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (model, tokenizer, id_to_label, train_metrics)
    """
    # Create run directory with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(config['output']['model_dir'], 'runs', f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Save config to run directory
    run_config_path = os.path.join(run_dir, 'config.yaml')
    with open(run_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.info(f"Saved run configuration to {run_config_path}")
    
    # Load data and mappings
    train_df, val_df, id_to_label, label_to_id = load_data_and_mappings(
        config['data']['processed_dir'],
        config['data']['mappings_dir']
    )
    logging.info(f"Loaded {len(train_df)} training and {len(val_df)} validation records")

    training_cfg = config.get('training', {})
    subset_size = training_cfg.get('subset_size')
    subset_threshold = training_cfg.get('subset_threshold')
    subset_force = training_cfg.get('subset_force', False)
    original_train_size = len(train_df)
    if subset_size:
        apply_subset = subset_force or subset_threshold is None or original_train_size > subset_threshold
        if apply_subset:
            sample_size = min(subset_size, original_train_size)
            seed = training_cfg.get('subset_seed', 42)
            train_df = train_df.sample(sample_size, random_state=seed)
            logging.info(
                "Using a random training subset of %s examples (from %s)",
                sample_size,
                original_train_size
            )

    if subset_size and subset_threshold:
        logging.info(
            "Subset threshold set to %s; final training size %s",
            subset_threshold,
            len(train_df)
        )

    # Prepare datasets with memory optimization options
    train_dataset, val_dataset, tokenizer = prepare_datasets(
        train_df,
        val_df,
        config['model']['max_seq_length'],
        config['model']['name'],
        config=config  # Pass full config for memory optimizations
    )
    logging.info(f"Prepared datasets with max sequence length {config['model']['max_seq_length']}")
    
    # Free up memory
    import gc
    train_df = val_df = None
    gc.collect()
    
    # Initialize model
    num_labels = len(id_to_label)
    model = initialize_model(config['model']['name'], num_labels)
    logging.info(f"Initialized model {config['model']['name']} with {num_labels} labels")
    
    # Train model
    logging.info("Starting training")
    model, best_val_loss = train_model_with_params(
        model,
        train_dataset,
        val_dataset,
        config
    )
    
    # Save best model
    best_model_dir = config['output']['best_model_dir']
    save_best_model(
        model,
        tokenizer,
        best_model_dir,
        id_to_label
    )
    
    # Save run metadata
    train_metrics = {
        'best_val_loss': best_val_loss,
        'model_name': config['model']['name'],
        'timestamp': timestamp,
        'num_labels': num_labels
    }
    
    # Save training metrics to run directory
    metrics_path = os.path.join(run_dir, 'train_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(train_metrics, f, indent=2)
    logging.info(f"Saved training metrics to {metrics_path}")
    
    # Create symlink to best model in run directory
    run_model_dir = os.path.join(run_dir, 'model')
    try:
        # Create relative symlink to best model
        os.symlink(
            os.path.relpath(best_model_dir, os.path.dirname(run_model_dir)),
            run_model_dir
        )
        logging.info(f"Created symlink to best model in run directory")
    except Exception as e:
        logging.warning(f"Could not create symlink to best model: {e}")
        # Copy files instead as fallback
        import shutil
        if not os.path.exists(run_model_dir):
            os.makedirs(run_model_dir)
        for item in os.listdir(best_model_dir):
            src = os.path.join(best_model_dir, item)
            dst = os.path.join(run_model_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        logging.info(f"Copied best model files to run directory")
    
    # Return model and related objects
    return model, tokenizer, id_to_label, train_metrics, run_dir

def run_training(config=None, auto_evaluate=True):
    """
    Run model training with optional automatic evaluation
    
    Args:
        auto_evaluate (bool): Whether to automatically run evaluation after training
        
    Returns:
        tuple: (model, run_dir) if successful, None otherwise
    """
    # Load config
    if config is None:
        config = load_config("config.yaml")
    
    # Train model
    try:
        model, tokenizer, id_to_label, train_metrics, run_dir = train_model(config)
        logging.info("Training completed successfully")
        
        # Automatically run evaluation if requested
        if auto_evaluate:
            try:
                # Import here to avoid circular imports
                from .evaluate import evaluate_model_with_run_dir, generate_html_report
                
                logging.info("Starting automatic evaluation after training")
                eval_metrics = evaluate_model_with_run_dir(config, model, tokenizer, id_to_label, run_dir)
                
                # Add evaluation metrics to training metrics
                combined_metrics = {**train_metrics, **eval_metrics}
                
                # Save combined metrics
                metrics_path = os.path.join(run_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(combined_metrics, f, indent=2)
                logging.info(f"Saved combined metrics to {metrics_path}")
                
                # Generate HTML report
                error_dir = os.path.join(run_dir, 'error_analysis')
                report_path = generate_html_report(eval_metrics, error_dir, run_dir, id_to_label, config)
                logging.info(f"Generated HTML evaluation report at {report_path}")
                
                # Print report location for user
                print(f"\n=== Model Evaluation Report ===")
                print(f"Report saved to: {report_path}")
                print(f"Accuracy: {eval_metrics.get('accuracy', 0):.3f}")
                print(f"F1 Score: {eval_metrics.get('f1', 0):.3f}")
                print(f"Top-3 Accuracy: {eval_metrics.get('top3_accuracy', 0):.3f}")
                print(f"===============================\n")
                
            except Exception as e:
                logging.error(f"Automatic evaluation failed: {e}")
                import traceback
                logging.error(traceback.format_exc())
                
        return model, run_dir
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return None

if __name__ == "__main__":
    run_training()
