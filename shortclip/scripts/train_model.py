#!/usr/bin/env python3
"""
Script to train the FusionModel using pre-computed Moment objects.
"""

import argparse
import yaml
import os
import sys
import logging

from shortclip.training.dataset import HighlightDataset
from shortclip.training.trainer import Trainer


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    
    return config


def find_data_file(data_dir: str, filename: str) -> str:
    """
    Find data file in directory (supports .json and .pkl extensions).
    
    Args:
        data_dir: Directory path or file path
        filename: Base filename (e.g., "train")
        
    Returns:
        Path to data file
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    # If data_dir is a file, return it directly
    if os.path.isfile(data_dir):
        return data_dir
    
    # Try .json first, then .pkl
    for ext in ['.json', '.pkl', '.pickle']:
        file_path = os.path.join(data_dir, f"{filename}{ext}")
        if os.path.exists(file_path):
            return file_path
    
    raise FileNotFoundError(
        f"Data file not found: {filename}.json or {filename}.pkl in {data_dir}"
    )


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Train FusionModel using pre-computed Moment objects"
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Directory containing training data (train.json/pkl) or path to training data file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config.yaml file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        required=True,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (defaults to config value)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints (default: checkpoints)'
    )
    
    parser.add_argument(
        '--val_data',
        type=str,
        default=None,
        help='Optional validation data file or directory (looks for val.json/pkl)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Find training data file
    try:
        train_data_path = find_data_file(args.data_dir, "train")
        logger.info(f"Training data: {train_data_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load training dataset
    try:
        train_dataset = HighlightDataset(train_data_path)
        logger.info(f"Loaded {len(train_dataset)} training samples")
    except Exception as e:
        print(f"Error loading training dataset: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Load validation dataset (optional)
    val_dataset = None
    if args.val_data:
        try:
            val_data_path = find_data_file(args.val_data, "val")
            logger.info(f"Validation data: {val_data_path}")
            val_dataset = HighlightDataset(val_data_path)
            logger.info(f"Loaded {len(val_dataset)} validation samples")
        except FileNotFoundError as e:
            logger.warning(f"Validation data not found: {e}, continuing without validation")
        except Exception as e:
            logger.warning(f"Error loading validation dataset: {e}, continuing without validation")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    try:
        trainer = Trainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir=args.output_dir
        )
        logger.info("Trainer initialized successfully")
    except Exception as e:
        print(f"Error initializing trainer: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Train model
    try:
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=1e-3  # Default learning rate
        )
        logger.info(f"Training completed. Best checkpoint saved in {args.output_dir}")
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
