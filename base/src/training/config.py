import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class that loads from YAML file."""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            self._set_attributes(config_dict)
        else:
            # Use defaults if no config file
            self._set_defaults()
    
    def _set_attributes(self, config_dict: Dict[str, Any]):
        """Set attributes from config dictionary."""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # Create nested namespace
                setattr(self, key, type('Config', (), value)())
            else:
                setattr(self, key, value)
    
    def _set_defaults(self):
        """Set default configuration values."""
        # Model hyperparameters
        self.model = type('Config', (), {
            'audio_embedding_dim': 512,
            'text_embedding_dim': 512,
            'temperature': 0.07,
            'learnable_temperature': True,
            'text_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'audio_model_type': 'cnn',  # 'cnn', 'transformer'
            'dropout': 0.1
        })()
        
        # Training parameters
        self.training = type('Config', (), {
            'batch_size': 32,
            'learning_rate': 1e-4,
            'num_epochs': 50,
            'weight_decay': 0.01,
            'warmup_steps': 1000,
            'gradient_clip': 1.0,
            'mixed_precision': True
        })()
        
        # Paths
        self.paths = type('Config', (), {
            'audio_model_path': 'models/audio_encoder.pth',
            'text_model_path': 'models/text_encoder.pth',
            'joint_model_path': 'models/joint_embedding.pth',
            'train_data_path': 'data/train_data.csv',
            'val_data_path': 'data/val_data.csv',
            'checkpoint_dir': 'checkpoints'
        })()
        
        # Logging
        self.logging = type('Config', (), {
            'log_interval': 10,
            'save_model_interval': 5,
            'use_wandb': False,
            'wandb_project': 'audio-text-retrieval'
        })()
    
    def display(self):
        """Display configuration settings."""
        print("=" * 50)
        print("Configuration Settings")
        print("=" * 50)
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr = getattr(self, attr_name)
                print(f"\n{attr_name.upper()}:")
                if hasattr(attr, '__dict__'):
                    for k, v in attr.__dict__.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {attr}")
        print("=" * 50)
    
    def save(self, path: str = None):
        """Save configuration to YAML file."""
        if path is None:
            path = self.config_path
        
        config_dict = {}
        for attr_name in dir(self):
            if not attr_name.startswith('_') and not callable(getattr(self, attr_name)):
                attr = getattr(self, attr_name)
                if hasattr(attr, '__dict__'):
                    config_dict[attr_name] = attr.__dict__
                elif attr_name != 'config_path':
                    config_dict[attr_name] = attr
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)