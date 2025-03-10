import os
import logging
import yaml
from datetime import datetime
from rich.logging import RichHandler
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
import psutil

def print_status(message, status="info", indent=0):
    """
    Print a status message with color and icon
    
    Args:
        message (str): Message to print
        status (str): Status type (info, success, warning, error)
        indent (int): Indentation level
    """
    indent_str = "  " * indent
    
    if status == "success":
        icon = "✓"
        color = "\033[92m"  # Green
    elif status == "warning":
        icon = "⚠️"
        color = "\033[93m"  # Yellow
    elif status == "error":
        icon = "✗"
        color = "\033[91m"  # Red
    else:
        icon = "•"
        color = "\033[0m"   # Default
        
    reset = "\033[0m"
    print(f"{indent_str}{color}{icon} {message}{reset}")

def print_section_header(title, width=80):
    """
    Print a section header with border
    
    Args:
        title (str): Title of section
        width (int): Width of the header
    """
    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width + "\n")

def format_metrics_summary(metrics):
    """
    Format metrics for display
    
    Args:
        metrics (dict): Metrics dictionary
        
    Returns:
        str: Formatted metrics string
    """
    lines = []
    lines.append("Key Metrics:")
    lines.append(f"  • Accuracy: {metrics.get('accuracy', 0):.4f}")
    
    # Use macro F1 and weighted F1 if available, otherwise fall back to just F1
    if 'macro_f1' in metrics and 'weighted_f1' in metrics:
        lines.append(f"  • Macro F1 Score: {metrics.get('macro_f1', 0):.4f}")
        lines.append(f"  • Weighted F1 Score: {metrics.get('weighted_f1', 0):.4f}")
    else:
        lines.append(f"  • F1 Score: {metrics.get('f1', 0):.4f}")
        
    lines.append(f"  • Top-3 Accuracy: {metrics.get('top3_accuracy', 0):.4f}")
    
    if 'misclassification_rate' in metrics:
        lines.append(f"  • Misclassification Rate: {metrics.get('misclassification_rate', 0):.4f}")
    
    return "\n".join(lines)

def load_config(config_path="config.yaml"):
    """
    Load and return the YAML config file as a dictionary
    
    Args:
        config_path (str): Path to the config file
        
    Returns:
        dict: Configuration dictionary
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Only log if logging is already configured
        if logging.getLogger().handlers:
            logging.info("Config loaded")
            print_status("Config loaded", "success")
            
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")
    except Exception as e:
        raise IOError(f"Error loading config file: {e}")

def setup_logging(log_dir="logs/", log_level="INFO", style="rich"):
    """
    Set up logging with rich formatting
    
    Args:
        log_dir (str): Directory to save log files
        log_level (str): Logging level (INFO, DEBUG, WARNING, ERROR)
        style (str): Logging style, 'rich' for colored output
        
    Returns:
        tuple: (timestamp, log_file) - Start timestamp and log file path
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with detailed timestamp
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"bcea_pipeline_{timestamp}.log")
    
    # Validate and set up level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    log_level = log_level.upper()
    if log_level not in valid_levels:
        print_status(f"Invalid log level '{log_level}'. Defaulting to INFO.", "warning")
        log_level = 'INFO'
    
    level = getattr(logging, log_level)
    
    # Configure logger
    if style.lower() == "rich":
        # Rich logging setup for console
        console_handler = RichHandler(rich_tracebacks=True)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        console_handler.setLevel(level)
        
        # File handler for log file
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", 
                                      datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(file_format)
        file_handler.setLevel(level)
        
        # Root logger setup
        logging.basicConfig(
            level=level,
            handlers=[console_handler, file_handler],
            format="%(message)s",
            force=True,  # Force reconfiguration if already configured
        )
    else:
        # Standard logging
        logging.basicConfig(
            level=level,
            format="[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True,  # Force reconfiguration if already configured
        )
    
    # Print welcome header
    print_section_header("BCEA Classification Pipeline")
    
    logging.info(f"=== Starting BCEA Pipeline ===")
    logging.info(f"Log file created at {log_file}")
    print_status(f"Log file created at {log_file}", "success")
    
    # Log system information
    log_system_info()
    
    return start_time, log_file

def create_rich_progress():
    """
    Create a rich progress bar for model training
    
    Returns:
        Progress: Rich progress bar object
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

def get_console():
    """
    Get a rich console for styled output
    
    Returns:
        Console: Rich console object
    """
    return Console()

def log_system_info():
    """
    Log system information including memory, CPU, and platform details
    """
    import platform
    import torch
    
    # System information
    logging.info("=== System Information ===")
    
    # CPU information
    cpu_count = os.cpu_count()
    logging.info(f"CPU: {cpu_count} cores")
    
    # Memory information
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024 ** 3)
    memory_available_gb = memory.available / (1024 ** 3)
    logging.info(f"Memory: {memory_available_gb:.2f}GB available / {memory_total_gb:.2f}GB total")
    print_status(f"System memory: {memory_total_gb:.1f} GB total, {memory_available_gb:.1f} GB available", "success")
    
    # Determine device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print_status(f"Using device: {device} ({device_name})", "success")
    elif platform.system() == "Darwin" and platform.machine() == "arm64" and torch.backends.mps.is_available():
        device = "mps"
        print_status(f"Using device: {device} (Apple Silicon)", "success")
    else:
        print_status(f"Using device: {device}", "success")
    
    # PyTorch information
    logging.info(f"PyTorch version: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        logging.info("Apple Silicon detected (M-series)")
        logging.info(f"MPS available: {torch.backends.mps.is_available()}")
    
    logging.info("=== End System Information ===")
    
    return device

def check_memory_availability(required_gb=None, percentage=0.8):
    """
    Check if sufficient memory is available
    
    Args:
        required_gb (float): Required memory in GB
        percentage (float): Maximum percentage of memory to use
        
    Returns:
        bool: True if sufficient memory is available
    """
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    total_gb = memory.total / (1024 ** 3)
    
    if required_gb is not None:
        if available_gb < required_gb:
            logging.warning(f"Insufficient memory: {available_gb:.2f}GB available, {required_gb:.2f}GB required")
            return False
    else:
        # Check if we're using too much memory already
        max_usable_gb = total_gb * percentage
        if available_gb < (total_gb * (1 - percentage)):
            logging.warning(f"High memory usage: {available_gb:.2f}GB available, recommended to have at least {max_usable_gb:.2f}GB")
            return False
    
    logging.info(f"Memory check passed: {available_gb:.2f}GB available")
    return True

class MPSGradScaler:
    """
    Gradient scaler for mixed precision training on MPS (Apple Silicon).
    Emulates functionality of torch.cuda.amp.GradScaler for MPS devices.
    """
    
    def __init__(self, enabled=True, init_scale=128.0, growth_factor=2.0, 
                 backoff_factor=0.5, growth_interval=2000):
        """
        Initialize MPSGradScaler.
        
        Args:
            enabled (bool): Whether the scaler is enabled
            init_scale (float): Initial scale factor
            growth_factor (float): Factor by which to increase scale after successful steps
            backoff_factor (float): Factor by which to decrease scale after NaN/inf
            growth_interval (int): Number of consecutive successful steps before increasing scale
        """
        self.enabled = enabled
        self.scale = init_scale if enabled else 1.0
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.successful_steps = 0
        self._found_inf = False
        
        logging.info(f"MPSGradScaler initialized: enabled={enabled}, init_scale={init_scale}")
        
    def scale_loss(self, loss):
        """Scale the loss value"""
        return loss * self.scale if self.enabled else loss
        
    def unscale_(self, optimizer):
        """Unscale gradients in the optimizer"""
        if not self.enabled:
            return
            
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.div_(self.scale)
                    
    def step(self, optimizer):
        """Step the optimizer with NaN/inf checking"""
        if not self.enabled:
            optimizer.step()
            return
            
        # Check for NaN/inf before stepping
        self._found_inf = False
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        self._found_inf = True
                        break
            if self._found_inf:
                break
                
        # Adjust scale based on gradient status
        if self._found_inf:
            # Reduce scale and skip optimizer step
            self.scale *= self.backoff_factor
            self.successful_steps = 0
            logging.warning(f"Found NaN/inf gradients, reducing scale to {self.scale}")
            return
        else:
            # Step optimizer normally
            optimizer.step()
            self.successful_steps += 1
            
            # Increase scale if enough successful steps
            if self.successful_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self.successful_steps = 0
                
    def update(self):
        """Update scaler state - no-op for this implementation"""
        pass

class MPSAutocast:
    """
    Context manager for mixed precision on MPS (Apple Silicon).
    Emulates torch.cuda.amp.autocast for MPS devices.
    """
    
    def __init__(self, enabled=True, dtype=None):
        """
        Initialize MPSAutocast.
        
        Args:
            enabled (bool): Whether autocast is enabled
            dtype (torch.dtype): Data type to use for autocast
        """
        import torch
        self.enabled = enabled
        self.dtype = dtype or torch.float16
        self.prev_dtype = None
        # Keep a reference to torch for enter/exit methods
        self.torch = torch

    def __enter__(self):
        if not self.enabled:
            return
        # Use the stored torch reference
        # Save previous default dtype
        self.prev_dtype = self.torch.get_default_dtype()
        
        # Set new default dtype
        self.torch.set_default_dtype(self.dtype)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return
        # Use the stored torch reference
        # Restore previous default dtype
        if self.prev_dtype is not None:
            self.torch.set_default_dtype(self.prev_dtype)

def configure_mps_memory():
    """
    Configure MPS (Metal Performance Shaders) memory for Apple Silicon.
    """
    import torch
    if not torch.backends.mps.is_available():
        return
        
    try:
        # Force MPS to release memory when not in use
        # Note: These are internal APIs and may change in future PyTorch versions
        if hasattr(torch._C, "_mps_set_cache_memory_fraction"):
            torch._C._mps_set_cache_memory_fraction(0.7)  # Use at most 70% of GPU memory
        if hasattr(torch._C, "_mps_set_flush_on_empty_cache"):
            torch._C._mps_set_flush_on_empty_cache(True)  # Flush memory when empty_cache() is called
        logging.info("MPS memory configured for Apple Silicon")
    except Exception as e:
        logging.warning(f"Could not configure MPS memory: {e}")

def cleanup_memory(device=None):
    """
    Force garbage collection and free up memory
    
    Args:
        device: PyTorch device (if provided, will clear CUDA cache for that device)
    
    Returns:
        float: Available memory in GB after cleanup
    """
    import gc
    import torch
    
    # Force garbage collection
    gc.collect()
    
    # Free CUDA cache if available
    if device and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Free MPS cache if available (Apple Silicon)
    if (device and device.type == 'mps' or 
        (not device and torch.backends.mps.is_available())):
        # Try to use MPS-specific memory management if available
        try:
            if hasattr(torch._C, "_mps_empty_cache"):
                torch._C._mps_empty_cache()
            
            # Create and delete a small tensor to force memory cleanup
            dummy = torch.zeros(1, device="mps")
            del dummy
            gc.collect()
        except Exception as e:
            logging.debug(f"Error cleaning MPS memory: {e}")
    
    # Get memory stats after cleanup
    memory = psutil.virtual_memory()
    available_gb = memory.available / (1024 ** 3)
    
    # Log memory status
    logging.info(f"Memory after cleanup: {available_gb:.2f}GB available")
    
    return available_gb