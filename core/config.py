import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

def env_var_constructor(loader, node):
    """Constructor for handling !env tags in YAML."""
    value = loader.construct_scalar(node)
    return os.getenv(value, value)

# Register the !env tag constructor
yaml.SafeLoader.add_constructor('!env', env_var_constructor)

class ConfigurationManager:
    """Manages configuration for different implementations and data lakes."""
    
    def __init__(self, config_path: str):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_implementation_config(self) -> Dict[str, Any]:
        """Get configuration for the selected implementation."""
        impl_name = self.config.get('implementation', {}).get('name')
        if not impl_name:
            raise ValueError("Implementation name not specified in config")
        
        impl_config = self.config.get('implementation', {}).get('config', {})
        return {
            'name': impl_name,
            'config': impl_config
        }
    
    def get_data_lake_config(self) -> Dict[str, Any]:
        """Get configuration for the selected data lake."""
        data_lake = self.config.get('data_lake', {}).get('name')
        if not data_lake:
            raise ValueError("Data lake not specified in config")
        
        data_config = self.config.get('data_lake', {}).get('config', {})
        return {
            'name': data_lake,
            'config': data_config
        } 