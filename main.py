#!/usr/bin/env python3
"""
Main entry point for the AutoClip project.
Wrapper around shortclip/scripts/process_video.py
"""
import sys
import os

# Add the project root to the python path to ensure imports work
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from shortclip.scripts.process_video import main

if __name__ == "__main__":
    main()
