from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class FusionModel(nn.Module):
    """
    Lightweight fusion model that combines visual, audio, and text embeddings.
    This is the ONLY trainable model in the pipeline.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the fusion model with architecture from config.
        
        Args:
            config: Configuration dictionary containing fusion model parameters
        """
        super().__init__()
        
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get model dimensions from config
        fusion_config = config.get("models", {}).get("fusion", {})
        self.input_dim = fusion_config.get("input_dim", 2560)
        hidden_dims = fusion_config.get("hidden_dims", [1024, 512])
        self.output_dim = fusion_config.get("output_dim", 1)
        
        # Verify input dimension matches expected concatenation
        visual_dim = config.get("models", {}).get("vision", {}).get("embedding_dim", 512)
        audio_dim = config.get("models", {}).get("audio", {}).get("embedding_dim", 1280)
        text_dim = config.get("models", {}).get("text", {}).get("embedding_dim", 768)
        expected_input_dim = visual_dim + audio_dim + text_dim
        
        if self.input_dim != expected_input_dim:
            self.logger.warning(
                f"Input dimension mismatch: config says {self.input_dim}, "
                f"but expected {expected_input_dim} (visual={visual_dim} + "
                f"audio={audio_dim} + text={text_dim}). Using config value."
            )
        
        # Build MLP layers
        layers = []
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=0.1))  # Small dropout for regularization
            prev_dim = hidden_dim
        
        # Output layer (no activation here, sigmoid applied in forward)
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(
            f"FusionModel initialized: input_dim={self.input_dim}, "
            f"hidden_dims={hidden_dims}, output_dim={self.output_dim}"
        )
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self,
        visual_embedding: Optional[torch.Tensor] = None,
        audio_embedding: Optional[torch.Tensor] = None,
        text_embedding: Optional[torch.Tensor] = None,
        combined_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the fusion model.
        
        Can accept either individual embeddings or a pre-combined feature vector.
        If combined_features is provided, it will be used directly.
        Otherwise, visual, audio, and text embeddings will be concatenated.
        
        Args:
            visual_embedding: Visual embedding tensor (batch_size, 512) or None
            audio_embedding: Audio embedding tensor (batch_size, 1280) or None
            text_embedding: Text embedding tensor (batch_size, 768) or None
            combined_features: Pre-combined feature tensor (batch_size, 2560) or None
            
        Returns:
            Relevance score tensor (batch_size, 1) with values in [0, 1] via sigmoid
        """
        # Use pre-combined features if provided
        if combined_features is not None:
            features = combined_features
        else:
            # Concatenate individual embeddings
            embedding_parts = []
            
            if visual_embedding is not None:
                embedding_parts.append(visual_embedding)
            else:
                # Use zeros if missing
                batch_size = self._get_batch_size(visual_embedding, audio_embedding, text_embedding)
                if batch_size > 0:
                    device = self._get_device(visual_embedding, audio_embedding, text_embedding)
                    embedding_parts.append(torch.zeros(batch_size, 512, device=device))
            
            if audio_embedding is not None:
                embedding_parts.append(audio_embedding)
            else:
                batch_size = self._get_batch_size(visual_embedding, audio_embedding, text_embedding)
                if batch_size > 0:
                    device = self._get_device(visual_embedding, audio_embedding, text_embedding)
                    embedding_parts.append(torch.zeros(batch_size, 1280, device=device))
            
            if text_embedding is not None:
                embedding_parts.append(text_embedding)
            else:
                batch_size = self._get_batch_size(visual_embedding, audio_embedding, text_embedding)
                if batch_size > 0:
                    device = self._get_device(visual_embedding, audio_embedding, text_embedding)
                    embedding_parts.append(torch.zeros(batch_size, 768, device=device))
            
            if not embedding_parts:
                raise ValueError("At least one embedding or combined_features must be provided")
            
            features = torch.cat(embedding_parts, dim=-1)
        
        # Verify feature dimension
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.input_dim}, "
                f"got {features.shape[-1]}"
            )
        
        # Forward through MLP
        logits = self.mlp(features)
        
        # Apply sigmoid to get relevance score in [0, 1]
        relevance_score = torch.sigmoid(logits)
        
        return relevance_score
    
    def _get_batch_size(
        self,
        visual_embedding: Optional[torch.Tensor],
        audio_embedding: Optional[torch.Tensor],
        text_embedding: Optional[torch.Tensor]
    ) -> int:
        """Get batch size from any available embedding."""
        for emb in [visual_embedding, audio_embedding, text_embedding]:
            if emb is not None:
                return emb.shape[0]
        return 0
    
    def _get_device(
        self,
        visual_embedding: Optional[torch.Tensor],
        audio_embedding: Optional[torch.Tensor],
        text_embedding: Optional[torch.Tensor]
    ) -> torch.device:
        """Get device from any available embedding."""
        for emb in [visual_embedding, audio_embedding, text_embedding]:
            if emb is not None:
                return emb.device
        # Default to CPU if no embeddings provided
        return torch.device("cpu")
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self):
        """Freeze all model parameters (not typically used, but available)."""
        for param in self.parameters():
            param.requires_grad = False
        self.logger.info("FusionModel parameters frozen")
    
    def unfreeze(self):
        """Unfreeze all model parameters (ensure model is trainable)."""
        for param in self.parameters():
            param.requires_grad = True
        self.logger.info("FusionModel parameters unfrozen (trainable)")
