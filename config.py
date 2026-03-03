from dataclasses import dataclass, field
from typing import Dict, List, Optional
import argparse
import json
import os
from loguru import logger


@dataclass
class GGSMConfig:
    """Configuration"""
    
    # ========== LLM Configuration ==========
    llm_model_name: str = './Qwen2.5-0.5B-Instruct'
    hidden_size: int = 896
    
    # ========== Training Configuration ==========
    batch_size: int = 64
    real_batch: int = 256
    num_epochs: int = 1
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # ========== Gradient Accumulation ==========
    gradient_accumulation_steps: int = 4
    
    # ========== Learning Rate Schedule ==========
    warmup_ratio: float = 0.1
    total_training_steps: int = 2000
    warmup_steps: int = field(default=0, init=False)
    
    # ========== SOFT Causal Mask ==========
    ema_decay: float = 0.0
    
    # ========== Contrastive Learning ==========
    logit_scale: float = 0.02
    margin: float = 0.1
    
    # ========== Data Configuration ==========
    data_path: str = 'alpaca_data_cleaned.json'
    max_length: int = 512
    
    # ========== Save Configuration ==========
    save_steps_num: int = 100
    output_dir: str = './checkpoints'
    
    # ========== Gradient Clipping ==========
    clip_grad_flag: bool = True
    max_norm: float = 1.0
    
    # ========== LoRA Configuration ==========
    use_lora: bool = True
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_bias: str = "none"  # "none", "all", "lora_only"
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    
    def __post_init__(self):
        """Post-initialization validation and processing"""
        # Calculate warmup_steps from ratio
        self.warmup_steps = int(self.warmup_ratio * self.total_training_steps)
        
        # Calculate batch_size
        self.batch_size = self.real_batch // (self.gradient_accumulation_steps * 1)
        
        # Validate configuration
        self._validate()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("✅ Config initialized successfully")
    
    def _validate(self):
        """Validate configuration values"""
        errors = []
        
        # Validate numerical parameters
        if self.batch_size <= 0:
            errors.append("batch_size must be > 0")
        if self.lr <= 0:
            errors.append("lr must be > 0")
        if self.num_epochs <= 0:
            errors.append("num_epochs must be > 0")
        if self.warmup_steps < 0:
            errors.append("warmup_steps must be >= 0")
        if self.total_training_steps <= self.warmup_steps:
            errors.append("total_training_steps must be > warmup_steps")
        if self.max_length <= 0:
            errors.append("max_length must be > 0")
        if self.ema_decay < 0 or self.ema_decay > 1:
            errors.append("ema_decay must be in [0, 1]")
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            errors.append("warmup_ratio must be in [0, 1]")
        
        # LoRA validation
        if self.use_lora:
            if self.lora_r <= 0:
                errors.append("lora_r must be > 0")
            if self.lora_alpha <= 0:
                errors.append("lora_alpha must be > 0")
            if self.lora_dropout < 0 or self.lora_dropout > 1:
                errors.append("lora_dropout must be in [0, 1]")
            if self.lora_bias not in ["none", "all", "lora_only"]:
                errors.append("lora_bias must be 'none', 'all', or 'lora_only'")
            if not self.lora_target_modules:
                errors.append("lora_target_modules cannot be empty")
        
        # Validate file paths
        if not os.path.exists(self.llm_model_name):
            errors.append(f"LLM model path does not exist: {self.llm_model_name}")
        if not os.path.exists(self.data_path):
            errors.append(f"Data file does not exist: {self.data_path}")
        
        if errors:
            logger.error("❌ Config validation failed:")
            for error in errors:
                logger.error(f"  - {error}")
            raise ValueError("\n".join(errors))
        
        logger.info("✅ Config validation passed")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'llm_model_name': self.llm_model_name,
            'hidden_size': self.hidden_size,
            'batch_size': self.batch_size,
            'real_batch': self.real_batch,
            'num_epochs': self.num_epochs,
            'lr': self.lr,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'warmup_ratio': self.warmup_ratio,
            'warmup_steps': self.warmup_steps,
            'total_training_steps': self.total_training_steps,
            'ema_decay': self.ema_decay,
            'logit_scale': self.logit_scale,
            'margin': self.margin,
            'data_path': self.data_path,
            'max_length': self.max_length,
            'save_steps_num': self.save_steps_num,
            'output_dir': self.output_dir,
            'clip_grad_flag': self.clip_grad_flag,
            'max_norm': self.max_norm,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'lora_bias': self.lora_bias,
            'lora_target_modules': self.lora_target_modules,
        }
    
    def save_to_file(self, path: str = None):
        """Save configuration to JSON file"""
        path = path or os.path.join(self.output_dir, 'config.json')
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Config saved to {path}")
    
    @classmethod
    def from_file(cls, path: str) -> 'GGSMConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        logger.info(f"📂 Loaded config from: {path}")
        return cls(**config_dict)
    
    def __str__(self) -> str:
        """Print configuration"""
        config_str = "\n" + "=" * 80 + "\n"
        config_str += "📋 CONFIGURATION\n"
        config_str += "=" * 80 + "\n"
        
        # Display in sections
        sections = {
            'LLM Configuration': [
                'llm_model_name',
                'hidden_size',
            ],
            'Training Configuration': [
                'batch_size',
                'real_batch',
                'num_epochs',
                'lr',
                'beta1',
                'beta2',
                'eps',
            ],
            'Gradient Accumulation': [
                'gradient_accumulation_steps',
            ],
            'Learning Rate Schedule': [
                'warmup_ratio',
                'warmup_steps',
                'total_training_steps',
            ],
            'SOFT Causal Mask': [
                'ema_decay',
            ],
            'Contrastive Learning': [
                'logit_scale',
                'margin',
            ],
            'Data Configuration': [
                'data_path',
                'max_length',
            ],
            'Save Configuration': [
                'save_steps_num',
                'output_dir',
            ],
            'Gradient Clipping': [
                'clip_grad_flag',
                'max_norm',
            ],
            'LoRA Configuration': [
                'use_lora',
                'lora_r',
                'lora_alpha',
                'lora_dropout',
                'lora_bias',
                'lora_target_modules',
            ],
        }
        
        config_dict = self.to_dict()
        
        for section_name, keys in sections.items():
            config_str += f"\n{section_name}:\n"
            for key in keys:
                if key in config_dict:
                    value = config_dict[key]
                    if isinstance(value, list):
                        value = ", ".join(value)
                    config_str += f"  {key:<40} : {value}\n"
        
        # Add calculated info
        config_str += "\n" + "=" * 80 + "\n"
        config_str += "📊 Calculated Metrics:\n"
        config_str += f"  {'LoRA Parameters (approx)':<40} : {self._estimate_lora_params()}\n"
        config_str += f"  {'Steps per Epoch':<40} : {self._steps_per_epoch()}\n"
        config_str += "=" * 80 + "\n"
        
        return config_str
    
    def _estimate_lora_params(self) -> str:
        """Estimate number of LoRA parameters"""
        if not self.use_lora:
            return "N/A (LoRA disabled)"
        # Rough estimation: 2 * r * hidden_size * num_layers
        # For 0.5B model: ~24 layers
        estimated = 2 * self.lora_r * self.hidden_size * 24
        return f"~{estimated / 1e6:.2f}M parameters"
    
    def _steps_per_epoch(self) -> int:
        """Estimate steps per epoch"""
        # This is a rough estimate, actual value depends on dataset size
        return self.total_training_steps // self.num_epochs


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train DeepFind Model via GGSM with LoRA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training with LoRA
  python train.py --data_path data.json --output_dir ./checkpoints
  
  # Custom parameters
  python train.py --data_path data.json --num_epochs 10 --lr 5e-5 --use_lora True
  
  # Load from config file
  python train.py --config_file ./checkpoints/config.json
  
  # Override config file parameters
  python train.py --config_file ./checkpoints/config.json --lr 1e-4 --lora_r 16
  
  # Disable LoRA
  python train.py --data_path data.json --use_lora False
        """
    )
    
    # LLM Configuration
    parser.add_argument(
        '--llm_model_name',
        type=str,
        default='./Qwen2.5-0.5B-Instruct',
        help='Path to LLM model'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=896,
        help='Hidden size of the model'
    )
    
    # Training Configuration
    parser.add_argument(
        '--real_batch',
        type=int,
        default=256,
        help='Real batch size'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.9,
        help='AdamW beta1'
    )
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.999,
        help='AdamW beta2'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-8,
        help='AdamW epsilon'
    )
    
    # Gradient Accumulation
    parser.add_argument(
        '--gradient_accumulation_steps',
        type=int,
        default=4,
        help='Gradient accumulation steps'
    )
    
    # Learning Rate Schedule
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        default=0.1,
        help='Warmup ratio (0.1 = 10% of total steps)'
    )
    parser.add_argument(
        '--total_training_steps',
        type=int,
        default=2000,
        help='Total training steps'
    )
    
    # SOFT Causal Mask
    parser.add_argument(
        '--ema_decay',
        type=float,
        default=0.99,
        help='EMA decay coefficient'
    )
    
    # Contrastive Learning
    parser.add_argument(
        '--logit_scale',
        type=float,
        default=0.07,
        help='Temperature parameter for contrastive learning'
    )
    parser.add_argument(
        '--margin',
        type=float,
        default=0.1,
        help='Margin for contrastive learning'
    )
    
    # Data and Save
    parser.add_argument(
        '--data_path',
        type=str,
        default='alpaca_data_cleaned.json',
        help='Path to training data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./checkpoints',
        help='Output directory'
    )
    parser.add_argument(
        '--save_steps_num',
        type=int,
        default=100,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    
    # Gradient Clipping
    parser.add_argument(
        '--clip_grad_flag',
        type=bool,
        default=True,
        help='Whether to use gradient clipping'
    )
    parser.add_argument(
        '--max_norm',
        type=float,
        default=1.0,
        help='Max norm for gradient clipping'
    )
    
    # LoRA Configuration
    parser.add_argument(
        '--use_lora',
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help='Whether to use LoRA (True/False)'
    )
    parser.add_argument(
        '--lora_r',
        type=int,
        default=8,
        help='LoRA rank'
    )
    parser.add_argument(
        '--lora_alpha',
        type=int,
        default=16,
        help='LoRA alpha (usually 2x of lora_r)'
    )
    parser.add_argument(
        '--lora_dropout',
        type=float,
        default=0.05,
        help='LoRA dropout rate'
    )
    parser.add_argument(
        '--lora_bias',
        type=str,
        default='none',
        choices=['none', 'all', 'lora_only'],
        help='LoRA bias type'
    )
    parser.add_argument(
        '--lora_target_modules',
        type=str,
        default='q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj',
        help='Comma-separated list of target modules for LoRA'
    )
    
    # Config File
    parser.add_argument(
        '--config_file',
        type=str,
        default=None,
        help='Load configuration from JSON file'
    )
    parser.add_argument(
        '--save_config',
        type=str,
        default=None,
        help='Save configuration to JSON file'
    )
    
    return parser.parse_args()


def get_config(args=None) -> GGSMConfig:
    """Get configuration object"""
    
    if args is None:
        args = parse_args()
    
    # Load from config file if specified
    if args.config_file and os.path.exists(args.config_file):
        logger.info(f"📂 Loading config from: {args.config_file}")
        config = GGSMConfig.from_file(args.config_file)
        
        # Allow command line arguments to override config file
        defaults = vars(parse_args([]))
        for key, value in vars(args).items():
            if key in ['config_file', 'save_config']:
                continue
            # Check if user provided non-default value
            if value != defaults.get(key):
                # Special handling for lora_target_modules
                if key == 'lora_target_modules':
                    value = [m.strip() for m in value.split(',')]
                setattr(config, key, value)
                logger.info(f"  ✏️  Override {key} = {value}")
    else:
        # Convert comma-separated string to list for lora_target_modules
        lora_target_modules = [
            m.strip() for m in args.lora_target_modules.split(',')
        ] if isinstance(args.lora_target_modules, str) else args.lora_target_modules
        
        # Create config from command line arguments
        config = GGSMConfig(
            llm_model_name=args.llm_model_name,
            hidden_size=args.hidden_size,
            real_batch=args.real_batch,
            num_epochs=args.num_epochs,
            lr=args.lr,
            beta1=args.beta1,
            beta2=args.beta2,
            eps=args.eps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            total_training_steps=args.total_training_steps,
            ema_decay=args.ema_decay,
            logit_scale=args.logit_scale,
            margin=args.margin,
            data_path=args.data_path,
            output_dir=args.output_dir,
            save_steps_num=args.save_steps_num,
            max_length=args.max_length,
            clip_grad_flag=args.clip_grad_flag,
            max_norm=args.max_norm,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_bias=args.lora_bias,
            lora_target_modules=lora_target_modules,
        )
    
    # Print configuration
    logger.info(config)
    
    # Save configuration
    if args.save_config:
        config.save_to_file(args.save_config)
    else:
        config.save_to_file()
    
    return config
    