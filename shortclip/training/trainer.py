from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os

from shortclip.models.fusion_model import FusionModel
from shortclip.training.dataset import HighlightDataset


class Trainer:
    """Trains FusionModel using binary cross-entropy loss and Adam optimizer."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: HighlightDataset,
        val_dataset: Optional[HighlightDataset] = None,
        checkpoint_dir: str = "checkpoints"
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary containing model and processing parameters
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            checkpoint_dir: Directory to save checkpoints
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Setup device from config
        device_name = config.get("processing", {}).get("device", "cpu")
        if device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.logger.info(f"Using device: {self.device}")
        else:
            self.device = torch.device("cpu")
            if device_name == "cuda":
                self.logger.warning("CUDA requested but not available, using CPU")
            else:
                self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = FusionModel(config)
        self.model.to(self.device)
        self.model.unfreeze()  # Ensure model is trainable
        
        # Loss and optimizer
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
        # Training state
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.logger.info(
            f"Trainer initialized: model parameters={self.model.get_num_parameters()}, "
            f"train samples={len(train_dataset)}, "
            f"val samples={len(val_dataset) if val_dataset else 0}"
        )
    
    def train(
        self,
        epochs: int,
        batch_size: Optional[int] = None,
        learning_rate: float = 1e-3,
        val_freq: int = 1
    ) -> None:
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size (defaults to config value)
            learning_rate: Learning rate for Adam optimizer
            val_freq: Frequency of validation (every N epochs)
        """
        batch_size = batch_size or self.config.get("processing", {}).get("batch_size", 16)
        
        # Update learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Avoid multiprocessing issues
        )
        
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
        
        self.logger.info(f"Starting training for {epochs} epochs, batch_size={batch_size}, lr={learning_rate}")
        
        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.logger.info(f"Epoch {self.current_epoch}/{epochs} - Train Loss: {train_loss:.4f}")
            
            # Validation phase
            if val_loader and (epoch + 1) % val_freq == 0:
                val_loss = self._validate(val_loader)
                self.logger.info(f"Epoch {self.current_epoch}/{epochs} - Val Loss: {val_loss:.4f}")
                
                # Save best checkpoint
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best_model.pt")
                    self.logger.info(f"Saved best checkpoint (val_loss={val_loss:.4f})")
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            visual, audio, text, labels = batch
            
            # Move to device
            visual = visual.to(self.device)
            audio = audio.to(self.device)
            text = text.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Shape: (batch_size, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                visual_embedding=visual,
                audio_embedding=audio,
                text_embedding=text
            )
            
            # Compute loss (outputs already have sigmoid applied from FusionModel)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                visual, audio, text, labels = batch
                
                # Move to device
                visual = visual.to(self.device)
                audio = audio.to(self.device)
                text = text.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)  # Shape: (batch_size, 1)
                
                # Forward pass
                outputs = self.model(
                    visual_embedding=visual,
                    audio_embedding=audio,
                    text_embedding=text
                )
                
                # Compute loss (outputs already have sigmoid applied from FusionModel)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        torch.save(checkpoint, checkpoint_path)
        self.logger.debug(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
