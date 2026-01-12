"""
Checkpoint manager for pipeline stage recovery and resumption.

Allows saving intermediate embeddings and pipeline state to disk,
enabling resumption from checkpoints if the pipeline is interrupted.
"""

import os
import pickle
import logging
from typing import Any, Dict, Optional, List
from pathlib import Path


class CheckpointManager:
    """Manages saving and loading pipeline checkpoints for recovery."""
    
    def __init__(self, checkpoint_dir: str = ".checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_checkpoint(
        self, 
        stage_name: str, 
        data: Any, 
        run_id: str = "default"
    ) -> Path:
        """
        Save a checkpoint for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage (e.g., 'visual_embeddings')
            data: Data to save (embeddings, contexts, etc.)
            run_id: Unique identifier for this pipeline run
            
        Returns:
            Path to the saved checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / f"{run_id}_{stage_name}.pkl"
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint {checkpoint_path}: {e}")
            raise
    
    def load_checkpoint(
        self, 
        stage_name: str, 
        run_id: str = "default"
    ) -> Optional[Any]:
        """
        Load a checkpoint for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage to load
            run_id: Unique identifier for the pipeline run
            
        Returns:
            Loaded data, or None if checkpoint doesn't exist
        """
        checkpoint_path = self.checkpoint_dir / f"{run_id}_{stage_name}.pkl"
        
        if not checkpoint_path.exists():
            self.logger.debug(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def checkpoint_exists(
        self, 
        stage_name: str, 
        run_id: str = "default"
    ) -> bool:
        """Check if a checkpoint exists for a stage."""
        checkpoint_path = self.checkpoint_dir / f"{run_id}_{stage_name}.pkl"
        return checkpoint_path.exists()
    
    def cleanup_checkpoints(self, run_id: str = "default") -> None:
        """
        Clean up all checkpoints for a run (call after successful completion).
        
        Args:
            run_id: Unique identifier for the pipeline run to clean up
        """
        try:
            for checkpoint_path in self.checkpoint_dir.glob(f"{run_id}_*.pkl"):
                checkpoint_path.unlink()
            self.logger.info(f"Cleaned up checkpoints for run: {run_id}")
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")
    
    def get_last_completed_stage(
        self, 
        stages: List[str],
        run_id: str = "default"
    ) -> Optional[int]:
        """
        Determine the last completed stage index.
        
        Args:
            stages: List of stage names in order
            run_id: Unique identifier for the pipeline run
            
        Returns:
            Index of last completed stage, or None if no checkpoints exist
        """
        for i in range(len(stages) - 1, -1, -1):
            if self.checkpoint_exists(stages[i], run_id):
                return i
        return None
