#!/usr/bin/env python3
"""
Script to process videos and generate highlight clips using the shortClip pipeline.
"""

import argparse
import yaml
import os
import sys

from shortclip.pipeline.multimodal_pipeline import MultimodalPipeline


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


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Process videos and generate highlight clips using shortClip"
    )
    
    parser.add_argument(
        '--videos',
        nargs='+',
        required=True,
        help='List of video file paths to process'
    )
    
    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Optional text query for filtering highlights'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config.yaml file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for the generated highlight video'
    )
    
    args = parser.parse_args()
    
    # Validate video files exist
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"Error: Video file not found: {video_path}", file=sys.stderr)
            sys.exit(1)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Instantiate pipeline
    try:
        pipeline = MultimodalPipeline(config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run pipeline
    try:
        output_path = pipeline.process(
            video_paths=args.videos,
            output_path=args.output,
            user_query=args.query
        )
        
        # Print final output path
        print(output_path)
        
    except Exception as e:
        print(f"Error processing videos: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
