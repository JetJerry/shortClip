from typing import List, Optional, Dict, Any
import numpy as np
from dataclasses import dataclass, asdict

from .audio_processor import WindowAudioEmbedding
from .text_processor import WindowTextEmbedding


@dataclass
class SceneContext:
    """Represents a complete scene context for a video window."""
    window_start_time: float
    window_end_time: float
    video_id: str
    video_path: str
    
    # Embeddings
    visual_embedding: Optional[np.ndarray]  # CLIP visual embedding (512-dim)
    audio_embedding: Optional[np.ndarray]  # Whisper audio embedding (1280-dim)
    text_embedding: Optional[np.ndarray]  # Sentence-BERT text embedding (768-dim)
    
    # Metadata
    transcript: Optional[str]  # Combined transcript from audio/text processing
    has_speech: bool  # Whether speech was detected
    has_text: bool  # Whether text content exists
    text_similarity_score: Optional[float]  # Similarity to user query if provided
    
    # Combined feature vector (for fusion model)
    # Concatenated: visual (512) + audio (1280) + text (768) = 2560
    combined_features: Optional[np.ndarray]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert SceneContext to dictionary (for serialization)."""
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


class SceneContextBuilder:
    """Builds scene-level context by combining visual, audio, and text embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the scene context builder.
        
        Args:
            config: Configuration dictionary (for consistency with other processors)
        """
        self.config = config
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Expected embedding dimensions from config
        self.visual_dim = config.get("models", {}).get("vision", {}).get("embedding_dim", 512)
        self.audio_dim = config.get("models", {}).get("audio", {}).get("embedding_dim", 1280)
        self.text_dim = config.get("models", {}).get("text", {}).get("embedding_dim", 768)
        self.combined_dim = self.visual_dim + self.audio_dim + self.text_dim
        
        self.logger.info(
            f"SceneContextBuilder initialized: "
            f"visual={self.visual_dim}, audio={self.audio_dim}, text={self.text_dim}, "
            f"combined={self.combined_dim}"
        )
    
    def build_contexts(
        self,
        video_segments: List[tuple],  # List of (video_id, video_path, start_time, end_time)
        visual_embeddings: List[Optional[np.ndarray]],
        audio_embeddings: List[WindowAudioEmbedding],
        text_embeddings: List[WindowTextEmbedding]
    ) -> List[SceneContext]:
        """
        Combine visual, audio, and text embeddings into structured SceneContext objects.
        
        Args:
            video_segments: List of video segment tuples (video_id, video_path, start_time, end_time)
            visual_embeddings: List of visual embeddings (CLIP, 512-dim) or None
            audio_embeddings: List of WindowAudioEmbedding objects
            text_embeddings: List of WindowTextEmbedding objects
            
        Returns:
            List of SceneContext objects, one per window
        """
        if len(video_segments) != len(visual_embeddings) or \
           len(video_segments) != len(audio_embeddings) or \
           len(video_segments) != len(text_embeddings):
            self.logger.warning(
                f"Mismatched input lengths: segments={len(video_segments)}, "
                f"visual={len(visual_embeddings)}, audio={len(audio_embeddings)}, "
                f"text={len(text_embeddings)}"
            )
            # Use minimum length to avoid index errors
            n_windows = min(len(video_segments), len(visual_embeddings), 
                          len(audio_embeddings), len(text_embeddings))
        else:
            n_windows = len(video_segments)
        
        scene_contexts = []
        
        for i in range(n_windows):
            try:
                video_id, video_path, start_time, end_time = video_segments[i]
                visual_emb = visual_embeddings[i] if i < len(visual_embeddings) else None
                audio_emb = audio_embeddings[i] if i < len(audio_embeddings) else None
                text_emb = text_embeddings[i] if i < len(text_embeddings) else None
                
                # Extract embeddings with validation
                visual_embedding = self._validate_embedding(visual_emb, self.visual_dim, "visual")
                audio_embedding = None
                transcript = None
                has_speech = False
                
                if audio_emb:
                    audio_embedding = self._validate_embedding(
                        audio_emb.embedding, self.audio_dim, "audio"
                    )
                    transcript = audio_emb.transcript
                    has_speech = audio_emb.has_speech
                    # Verify window timing matches
                    if abs(audio_emb.window_start_time - start_time) > 0.01 or \
                       abs(audio_emb.window_end_time - end_time) > 0.01:
                        self.logger.warning(
                            f"Window timing mismatch for {video_id} at {start_time:.2f}s: "
                            f"audio window [{audio_emb.window_start_time:.2f}, {audio_emb.window_end_time:.2f}]"
                        )
                
                text_embedding = None
                has_text = False
                text_similarity_score = None
                
                if text_emb:
                    text_embedding = self._validate_embedding(
                        text_emb.embedding, self.text_dim, "text"
                    )
                    has_text = text_emb.has_text
                    text_similarity_score = text_emb.similarity_score
                    # Prefer transcript from text_emb if available (might have better processing)
                    if not transcript and text_emb.transcript:
                        transcript = text_emb.transcript
                    # Verify window timing matches
                    if abs(text_emb.window_start_time - start_time) > 0.01 or \
                       abs(text_emb.window_end_time - end_time) > 0.01:
                        self.logger.warning(
                            f"Window timing mismatch for {video_id} at {start_time:.2f}s: "
                            f"text window [{text_emb.window_start_time:.2f}, {text_emb.window_end_time:.2f}]"
                        )
                
                # Build combined feature vector by concatenating embeddings
                combined_features = self._combine_features(
                    visual_embedding, audio_embedding, text_embedding
                )
                
                # Create SceneContext
                scene_context = SceneContext(
                    window_start_time=start_time,
                    window_end_time=end_time,
                    video_id=video_id,
                    video_path=video_path,
                    visual_embedding=visual_embedding,
                    audio_embedding=audio_embedding,
                    text_embedding=text_embedding,
                    transcript=transcript,
                    has_speech=has_speech,
                    has_text=has_text,
                    text_similarity_score=text_similarity_score,
                    combined_features=combined_features
                )
                
                scene_contexts.append(scene_context)
                
            except Exception as e:
                self.logger.error(f"Error building context for window {i}: {e}")
                # Create empty context as fallback
                try:
                    video_id, video_path, start_time, end_time = video_segments[i]
                except:
                    video_id = "unknown"
                    video_path = "unknown"
                    start_time = 0.0
                    end_time = 0.0
                
                scene_contexts.append(
                    SceneContext(
                        window_start_time=start_time,
                        window_end_time=end_time,
                        video_id=video_id,
                        video_path=video_path,
                        visual_embedding=None,
                        audio_embedding=None,
                        text_embedding=None,
                        transcript=None,
                        has_speech=False,
                        has_text=False,
                        text_similarity_score=None,
                        combined_features=None
                    )
                )
        
        self.logger.info(
            f"Built {len(scene_contexts)} scene contexts: "
            f"{sum(1 for c in scene_contexts if c.visual_embedding is not None)} with visual, "
            f"{sum(1 for c in scene_contexts if c.audio_embedding is not None)} with audio, "
            f"{sum(1 for c in scene_contexts if c.text_embedding is not None)} with text"
        )
        
        return scene_contexts
    
    def _validate_embedding(
        self, 
        embedding: Optional[np.ndarray], 
        expected_dim: int, 
        modality: str
    ) -> Optional[np.ndarray]:
        """
        Validate and normalize embedding.
        
        Args:
            embedding: Embedding array or None
            expected_dim: Expected embedding dimension
            modality: Modality name for logging
            
        Returns:
            Validated embedding or None if invalid
        """
        if embedding is None:
            return None
        
        # Convert to numpy array if needed
        if not isinstance(embedding, np.ndarray):
            try:
                embedding = np.array(embedding, dtype=np.float32)
            except:
                self.logger.warning(f"Could not convert {modality} embedding to numpy array")
                return None
        
        # Check shape
        if embedding.ndim == 0 or embedding.size == 0:
            self.logger.warning(f"Empty {modality} embedding")
            return None
        
        # Flatten if needed
        embedding = embedding.flatten()
        
        # Check dimension
        if len(embedding) != expected_dim:
            self.logger.warning(
                f"{modality} embedding dimension mismatch: expected {expected_dim}, "
                f"got {len(embedding)}"
            )
            # Pad or truncate to expected dimension
            if len(embedding) < expected_dim:
                embedding = np.pad(
                    embedding, 
                    (0, expected_dim - len(embedding)), 
                    mode='constant'
                )
            else:
                embedding = embedding[:expected_dim]
        
        return embedding.astype(np.float32)
    
    def _combine_features(
        self,
        visual_embedding: Optional[np.ndarray],
        audio_embedding: Optional[np.ndarray],
        text_embedding: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """
        Combine visual, audio, and text embeddings into a single feature vector.
        Uses zero padding for missing embeddings.
        
        Args:
            visual_embedding: Visual embedding (512-dim) or None
            audio_embedding: Audio embedding (1280-dim) or None
            text_embedding: Text embedding (768-dim) or None
            
        Returns:
            Combined feature vector (2560-dim) or None if all embeddings are missing
        """
        parts = []
        
        # Add visual embedding (or zeros)
        if visual_embedding is not None:
            parts.append(visual_embedding)
        else:
            parts.append(np.zeros(self.visual_dim, dtype=np.float32))
        
        # Add audio embedding (or zeros)
        if audio_embedding is not None:
            parts.append(audio_embedding)
        else:
            parts.append(np.zeros(self.audio_dim, dtype=np.float32))
        
        # Add text embedding (or zeros)
        if text_embedding is not None:
            parts.append(text_embedding)
        else:
            parts.append(np.zeros(self.text_dim, dtype=np.float32))
        
        # Concatenate all parts
        combined = np.concatenate(parts, axis=0).astype(np.float32)
        
        # Verify final dimension
        if len(combined) != self.combined_dim:
            self.logger.warning(
                f"Combined feature dimension mismatch: expected {self.combined_dim}, "
                f"got {len(combined)}"
            )
            # Adjust if needed
            if len(combined) < self.combined_dim:
                combined = np.pad(combined, (0, self.combined_dim - len(combined)), mode='constant')
            else:
                combined = combined[:self.combined_dim]
        
        return combined
