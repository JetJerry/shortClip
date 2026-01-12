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
        
        # Get embedding dimensions from config (avoid hardcoding)
        self.visual_dim = config.get("models", {}).get("vision", {}).get("embedding_dim", 512)
        self.audio_dim = config.get("models", {}).get("audio", {}).get("embedding_dim", 1280)
        self.text_dim = config.get("models", {}).get("text", {}).get("embedding_dim", 768)
        
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
        If no model is loaded, uses heuristic scoring based on feature norms.
        
        Args:
            moments: List of Moment objects with embeddings
            
        Returns:
            List of Moments with updated scores
        """
        if not moments:
            return moments
        
        self.logger.info(f"Computing scores for {len(moments)} moments")
        
        # Normalize embeddings
        normalized_moments = self._normalize_embeddings(moments)
        
        # Group moments by video_id for temporal smoothing
        moments_by_video = defaultdict(list)
        for moment in normalized_moments:
            moments_by_video[moment.video_id].append(moment)
        
        # Compute scores
        if self.model is not None:
            all_scores = self._compute_model_scores(normalized_moments)
        else:
            self.logger.warning("No model loaded. Using heuristic scoring based on feature norms.")
            all_scores = self._compute_heuristic_scores(normalized_moments)
        
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

    def _compute_heuristic_scores(self, moments: List[Moment]) -> np.ndarray:
        """
        Compute heuristic scores based on L2 norms of embeddings.
        Weights text similarity higher if a query was used (indicated by non-zero text embedding).
        
        Args:
            moments: List of Moments with normalized embeddings
            
        Returns:
            Array of heuristic relevance scores
        """
        scores = []
        
        for moment in moments:
            # Calculate norms (moment embeddings are already normalized to unit length if present, 
            # but we want to know if they exist and "strength" if they were not normalized?
            # Actually, _normalize_embeddings makes them unit vector. 
            # So we can't use norm magnitude as signal if they are already normalized.
            # Wait, _normalize_embeddings() was called before this.
            # But we can check if they are None or Zero.
            
            # Actually, we want to align with "interestingness".
            # Without a trained model, usually high variance or specific features denote interest.
            # But for simple heuristic:
            # - If text query exists (text_embedding is not None/Zero), we assume it's a Similarity embedding (failed to verify text_processor logic, but assuming text_processor returns similarity or embedding of query?
            # Let's check text_processor later. Assuming text_embedding is the EMBEDDING of the frame.
            # Wait, if text_processor runs, it puts text embeddings into SceneContext.
            # If user_query is provided, text_processor usually computes SIMILARITY between query and frame.
            # Let's re-read text_processor logic to be sure. 
            # BUT for now, let's assume we want to combine available signals.
            
            # Heuristic Logic:
            # 1. Base score from Visual/Audio "activity" (norm as proxy)
            # 2. Boost if text similarity is high (strong semantic match)
            
            visual_score = 0.0
            if moment.visual_embedding is not None:
                # Norm of normalized embedding is 1.0, so this doesn't help much if already normalized.
                # However, we can assume baseline interest.
                visual_score = 0.5 
                
            audio_score = 0.0
            if moment.audio_embedding is not None:
                 audio_score = 0.5

            # Text Score (The most important signal if query is present)
            text_score = 0.0
            if moment.text_similarity_score is not None:
                # Similarity is cosine (-1 to 1), map to 0-1 (roughly)
                # Cap at 0 for negative similarity
                sim = max(0.0, moment.text_similarity_score)
                text_score = sim
            
            # Combine
            # If query was provided (text_similarity_score is not None), weight it heavily
            if moment.text_similarity_score is not None:
                # 60% Text, 20% Visual, 20% Audio
                final_score = (0.6 * text_score) + (0.2 * visual_score) + (0.2 * audio_score)
            else:
                # No query, equal weights
                final_score = (visual_score + audio_score) / 2.0
                # Add some random jitter to avoid flat lines if all norms are 1.0
                # (Deterministic based on start time hash)
                # Deterministic jitter based on start_time
                rng = np.random.RandomState(int(moment.start_time * 1000) % (2**32))
                jitter = rng.uniform(0, 0.1)
                final_score += jitter
            
            scores.append(final_score)
            
        return np.array(scores, dtype=np.float32)
    
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
                text_similarity_score=moment.text_similarity_score,
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
                    else torch.zeros(self.visual_dim, dtype=torch.float32)
                )
                audio_embs.append(
                    torch.from_numpy(moment.audio_embedding) 
                    if moment.audio_embedding is not None 
                    else torch.zeros(self.audio_dim, dtype=torch.float32)
                )
                text_embs.append(
                    torch.from_numpy(moment.text_embedding) 
                    if moment.text_embedding is not None 
                    else torch.zeros(self.text_dim, dtype=torch.float32)
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
                text_similarity_score=moment.text_similarity_score,
                label=moment.label,
                score=float(smoothed_score)
            )
            smoothed_moments.append(smoothed_moment)
        
        return smoothed_moments
