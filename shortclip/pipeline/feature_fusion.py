from typing import List, Dict, Any, Optional
import torch
import numpy as np
import logging
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d

from .moment import Moment
from ..models.fusion_model import FusionModel


class FeatureFusion:
    """
    Fuses multimodal features using trained FusionModel and computes relevance scores.
    Applies temporal smoothing to scores within each video.
    """
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """
        Initialize the feature fusion processor.
        
        Args:
            config: Configuration dictionary containing model parameters
            model_path: Path to trained FusionModel checkpoint (optional, can be loaded later)
        """
        self.config = config
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
        
        self.model = None
        self.model_path = model_path
        self.batch_size = config.get("processing", {}).get("batch_size", 16)
        
        # Temporal smoothing parameters
        self.smoothing_sigma = config.get("processing", {}).get("temporal_smoothing_sigma", 1.0)
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """
        Load trained FusionModel from checkpoint.
        
        Args:
            model_path: Path to model checkpoint file
        """
        try:
            self.logger.info(f"Loading FusionModel from {model_path}")
            
            # Initialize model from config
            self.model = FusionModel(self.config)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                elif "state_dict" in checkpoint:
                    self.model.load_state_dict(checkpoint["state_dict"])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model_path = model_path
            self.logger.info(f"FusionModel loaded successfully. Parameters: {self.model.get_num_parameters()}")
            
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def compute_scores(self, moments: List[Moment]) -> List[Moment]:
        """
        Compute relevance scores for Moments using FusionModel and apply temporal smoothing.
        
        Args:
            moments: List of Moment objects with embeddings
            
        Returns:
            List of Moments with updated scores
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not moments:
            return moments
        
        self.logger.info(f"Computing scores for {len(moments)} moments")
        
        # Normalize embeddings
        normalized_moments = self._normalize_embeddings(moments)
        
        # Group moments by video_id for temporal smoothing
        moments_by_video = defaultdict(list)
        for moment in normalized_moments:
            moments_by_video[moment.video_id].append(moment)
        
        # Compute scores for all moments in batches
        all_scores = self._compute_model_scores(normalized_moments)
        
        # Assign scores to moments
        for moment, score in zip(normalized_moments, all_scores):
            moment.score = float(score)
        
        # Apply temporal smoothing per video
        smoothed_moments = []
        for video_id, video_moments in moments_by_video.items():
            smoothed_video_moments = self._apply_temporal_smoothing(video_moments)
            smoothed_moments.extend(smoothed_video_moments)
        
        # Sort back to original order (preserve order across videos)
        # Create mapping from (video_id, start_time) to index
        original_order = {id(m): i for i, m in enumerate(normalized_moments)}
        smoothed_moments.sort(key=lambda m: original_order.get(id(m), float('inf')))
        
        self.logger.info(
            f"Computed scores: mean={np.mean(all_scores):.4f}, "
            f"min={np.min(all_scores):.4f}, max={np.max(all_scores):.4f}"
        )
        
        return smoothed_moments
    
    def _normalize_embeddings(self, moments: List[Moment]) -> List[Moment]:
        """
        Normalize embeddings in Moments using L2 normalization.
        Creates new Moment objects with normalized embeddings.
        
        Args:
            moments: List of Moments with embeddings
            
        Returns:
            List of Moments with normalized embeddings
        """
        normalized_moments = []
        
        for moment in moments:
            # Create copies of embeddings and normalize
            visual_emb = self._normalize_embedding(moment.visual_embedding)
            audio_emb = self._normalize_embedding(moment.audio_embedding)
            text_emb = self._normalize_embedding(moment.text_embedding)
            
            # Create new moment with normalized embeddings
            normalized_moment = Moment(
                video_id=moment.video_id,
                video_path=moment.video_path,
                start_time=moment.start_time,
                end_time=moment.end_time,
                visual_embedding=visual_emb,
                audio_embedding=audio_emb,
                text_embedding=text_emb,
                label=moment.label,
                score=moment.score  # Preserve existing score if any
            )
            
            normalized_moments.append(normalized_moment)
        
        return normalized_moments
    
    def _normalize_embedding(self, embedding: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Normalize a single embedding using L2 normalization.
        
        Args:
            embedding: Embedding array or None
            
        Returns:
            Normalized embedding or None
        """
        if embedding is None or embedding.size == 0:
            return None
        
        # Ensure it's a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Flatten and normalize
        embedding = embedding.flatten().astype(np.float32)
        norm = np.linalg.norm(embedding)
        
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _compute_model_scores(self, moments: List[Moment]) -> np.ndarray:
        """
        Compute relevance scores using FusionModel in batches.
        
        Args:
            moments: List of Moments with normalized embeddings
            
        Returns:
            Array of relevance scores
        """
        all_scores = []
        
        # Process in batches
        for i in range(0, len(moments), self.batch_size):
            batch_moments = moments[i:i + self.batch_size]
            
            # Prepare batch tensors
            visual_embs = []
            audio_embs = []
            text_embs = []
            
            for moment in batch_moments:
                visual_embs.append(
                    torch.from_numpy(moment.visual_embedding) 
                    if moment.visual_embedding is not None 
                    else torch.zeros(512, dtype=torch.float32)
                )
                audio_embs.append(
                    torch.from_numpy(moment.audio_embedding) 
                    if moment.audio_embedding is not None 
                    else torch.zeros(1280, dtype=torch.float32)
                )
                text_embs.append(
                    torch.from_numpy(moment.text_embedding) 
                    if moment.text_embedding is not None 
                    else torch.zeros(768, dtype=torch.float32)
                )
            
            # Stack into batch tensors
            visual_batch = torch.stack(visual_embs).to(self.device)
            audio_batch = torch.stack(audio_embs).to(self.device)
            text_batch = torch.stack(text_embs).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                scores = self.model(
                    visual_embedding=visual_batch,
                    audio_embedding=audio_batch,
                    text_embedding=text_batch
                )
            
            # Extract scores
            batch_scores = scores.cpu().numpy().flatten()
            all_scores.extend(batch_scores)
        
        return np.array(all_scores, dtype=np.float32)
    
    def _apply_temporal_smoothing(
        self, 
        video_moments: List[Moment]
    ) -> List[Moment]:
        """
        Apply temporal smoothing to scores within a video.
        Uses Gaussian smoothing on scores ordered by time.
        
        Args:
            video_moments: List of Moments from the same video, should be sorted by time
            
        Returns:
            List of Moments with smoothed scores
        """
        if len(video_moments) <= 1:
            return video_moments
        
        # Sort by start_time to ensure temporal order
        sorted_moments = sorted(video_moments, key=lambda m: m.start_time)
        
        # Extract scores
        scores = np.array([m.score for m in sorted_moments], dtype=np.float32)
        
        # Apply Gaussian smoothing (1D filter)
        if len(scores) > 2 and self.smoothing_sigma > 0:
            smoothed_scores = gaussian_filter1d(scores, sigma=self.smoothing_sigma)
            # Ensure scores stay in valid range [0, 1]
            smoothed_scores = np.clip(smoothed_scores, 0.0, 1.0)
        else:
            smoothed_scores = scores
        
        # Update scores in moments
        smoothed_moments = []
        for moment, smoothed_score in zip(sorted_moments, smoothed_scores):
            # Create new moment with smoothed score
            smoothed_moment = Moment(
                video_id=moment.video_id,
                video_path=moment.video_path,
                start_time=moment.start_time,
                end_time=moment.end_time,
                visual_embedding=moment.visual_embedding,
                audio_embedding=moment.audio_embedding,
                text_embedding=moment.text_embedding,
                label=moment.label,
                score=float(smoothed_score)
            )
            smoothed_moments.append(smoothed_moment)
        
        return smoothed_moments
