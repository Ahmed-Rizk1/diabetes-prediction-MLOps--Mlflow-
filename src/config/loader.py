import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the config file. Defaults to config/config.yaml relative to project root.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if config_path is None:
        # Assuming this file is in src/config/loader.py
        # Project root is ../../
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    return config

# Singleton-like access if needed
# config = load_config()
