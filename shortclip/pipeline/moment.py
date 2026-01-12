from typing import Optional, Dict, Any
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class Moment:
    """Represents a moment/clip in a video with multimodal embeddings."""
    
    video_id: str
    video_path: str
    start_time: float
    end_time: float
    visual_embedding: Optional[np.ndarray] = None
    audio_embedding: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    text_similarity_score: Optional[float] = None
    label: Optional[str] = None
    score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Moment to dictionary for serialization.
        Handles numpy arrays by converting them to lists.
        
        Returns:
            Dictionary representation of the Moment
        """
        result = asdict(self)
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Moment':
        """
        Create Moment instance from dictionary.
        Converts list representations back to numpy arrays where appropriate.
        
        Args:
            data: Dictionary representation of Moment
            
        Returns:
            Moment instance
        """
        # Convert embedding lists back to numpy arrays
        for embedding_key in ['visual_embedding', 'audio_embedding', 'text_embedding']:
            if embedding_key in data and data[embedding_key] is not None:
                if isinstance(data[embedding_key], list):
                    data[embedding_key] = np.array(data[embedding_key], dtype=np.float32)
        
        return cls(**data)
    
    def __post_init__(self):
        """Validate moment data after initialization."""
        # Validate time range
        if self.start_time < 0:
            raise ValueError(f"start_time must be non-negative, got {self.start_time}")
        if self.end_time <= self.start_time:
            raise ValueError(
                f"end_time must be greater than start_time, "
                f"got start_time={self.start_time}, end_time={self.end_time}"
            )
        
        # Validate embedding types
        for embedding_name, embedding in [
            ('visual_embedding', self.visual_embedding),
            ('audio_embedding', self.audio_embedding),
            ('text_embedding', self.text_embedding)
        ]:
            if embedding is not None and not isinstance(embedding, np.ndarray):
                # Try to convert to numpy array
                try:
                    setattr(self, embedding_name, np.array(embedding, dtype=np.float32))
                except Exception as e:
                    raise ValueError(
                        f"{embedding_name} must be a numpy array or convertible to one, "
                        f"got {type(embedding)}: {e}"
                    )
        
        # Ensure embeddings are float32
        for embedding_name in ['visual_embedding', 'audio_embedding', 'text_embedding']:
            embedding = getattr(self, embedding_name)
            if embedding is not None:
                setattr(self, embedding_name, embedding.astype(np.float32))
