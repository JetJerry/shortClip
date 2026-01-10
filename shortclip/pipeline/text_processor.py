from typing import Tuple, Dict, Any, List, Optional
import torch
import numpy as np
from dataclasses import dataclass

from .base import BaseProcessor
from .audio_processor import WindowAudioEmbedding


@dataclass
class WindowTextEmbedding:
    """Represents a text embedding aligned to a video window."""
    window_start_time: float
    window_end_time: float
    embedding: np.ndarray  # Sentence-BERT embedding for this window
    transcript: Optional[str]  # Original transcript text
    similarity_score: Optional[float]  # Cosine similarity to user query if provided
    has_text: bool  # Whether text content exists for this window


class TextProcessor(BaseProcessor):
    """Processes transcripts to extract semantic embeddings using Sentence-BERT."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the text processor.
        
        Args:
            config: Configuration dictionary containing model and processing parameters
        """
        super().__init__(config)
        self.model = None
        self.model_name = None
        self.embedding_dim = None
        self.batch_size = config.get("processing", {}).get("batch_size", 16)
        self.user_query_embedding = None  # Cached query embedding
    
    def setup(self) -> None:
        """
        Load Sentence-BERT model and set up processing.
        Model weights are frozen.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Get model name from config
            self.model_name = self.config.get("models", {}).get("text", {}).get("name", "sentence-transformers/all-mpnet-base-v2")
            self.embedding_dim = self.config.get("models", {}).get("text", {}).get("embedding_dim", 768)
            
            self.logger.info(f"Loading Sentence-BERT model: {self.model_name}")
            
            # Load Sentence-BERT model
            # SentenceTransformer handles device placement automatically
            self.model = SentenceTransformer(self.model_name, device=str(self.device))
            
            # Freeze all model weights
            for param in self.model.parameters():
                param.requires_grad = False
            
            self.logger.info(f"Sentence-BERT model loaded and frozen on {self.device}")
            
        except Exception as e:
            self._handle_error(e, "setup")
            raise
    
    def process(
        self, 
        audio_embeddings: List[WindowAudioEmbedding],
        user_query: Optional[str] = None
    ) -> List[WindowTextEmbedding]:
        """
        Encode transcripts from audio embeddings and compute similarity to optional query.
        Aligns text embeddings to video windows.
        
        Args:
            audio_embeddings: List of WindowAudioEmbedding objects with transcripts
            user_query: Optional user query string for similarity computation
            
        Returns:
            List of WindowTextEmbedding objects aligned to video windows
        """
        if self.model is None:
            raise RuntimeError("Processor not set up. Call setup() first.")
        
        try:
            # Encode user query if provided
            if user_query:
                self._encode_query(user_query)
            
            # Extract transcripts from audio embeddings
            transcripts = []
            window_info = []  # Store window timing info
            
            for audio_emb in audio_embeddings:
                transcript = audio_emb.transcript if audio_emb.has_speech and audio_emb.transcript else None
                transcripts.append(transcript)
                window_info.append({
                    "start": audio_emb.window_start_time,
                    "end": audio_emb.window_end_time,
                    "has_text": transcript is not None
                })
            
            # Encode transcripts in batches
            text_embeddings = self._encode_transcripts(transcripts)
            
            # Compute similarity scores if query provided
            similarity_scores = None
            if user_query and self.user_query_embedding is not None:
                similarity_scores = self._compute_similarities(text_embeddings, self.user_query_embedding)
            
            # Create WindowTextEmbedding objects aligned to windows
            window_text_embeddings = []
            for i, (emb, info) in enumerate(zip(text_embeddings, window_info)):
                similarity = similarity_scores[i] if similarity_scores is not None else None
                transcript = transcripts[i] if info["has_text"] else None
                
                window_text_embeddings.append(
                    WindowTextEmbedding(
                        window_start_time=info["start"],
                        window_end_time=info["end"],
                        embedding=emb,
                        transcript=transcript,
                        similarity_score=similarity,
                        has_text=info["has_text"]
                    )
                )
            
            self.logger.debug(
                f"Processed {len(window_text_embeddings)} windows: "
                f"{sum(1 for w in window_text_embeddings if w.has_text)} with text, "
                f"{sum(1 for w in window_text_embeddings if w.similarity_score is not None)} with similarity scores"
            )
            
            return window_text_embeddings
            
        except Exception as e:
            self._handle_error(e, "processing text embeddings")
            # Return empty embeddings on error
            return [
                WindowTextEmbedding(
                    window_start_time=emb.window_start_time,
                    window_end_time=emb.window_end_time,
                    embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                    transcript=None,
                    similarity_score=None,
                    has_text=False
                )
                for emb in audio_embeddings
            ]
    
    def _encode_query(self, query: str) -> None:
        """
        Encode user query and cache the embedding.
        
        Args:
            query: User query string
        """
        try:
            if not query or not query.strip():
                self.user_query_embedding = None
                return
            
            # Encode query
            query_embedding = self.model.encode(
                query.strip(),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            self.user_query_embedding = query_embedding.astype(np.float32)
            self.logger.debug(f"Encoded user query: '{query[:50]}...'")
            
        except Exception as e:
            self._handle_error(e, "encoding query")
            self.user_query_embedding = None
    
    def _encode_transcripts(self, transcripts: List[Optional[str]]) -> List[np.ndarray]:
        """
        Encode transcript texts in batches.
        Returns embeddings for all transcripts, using empty embeddings for None transcripts.
        
        Args:
            transcripts: List of transcript strings (or None for windows without speech)
            
        Returns:
            List of numpy arrays with embeddings (shape: embedding_dim,)
        """
        try:
            # Separate transcripts with text from those without
            text_transcripts = []
            text_indices = []
            
            for i, transcript in enumerate(transcripts):
                if transcript and transcript.strip():
                    text_transcripts.append(transcript.strip())
                    text_indices.append(i)
            
            # Encode all text transcripts in batches
            text_embeddings_dict = {}
            if text_transcripts:
                embeddings = self.model.encode(
                    text_transcripts,
                    batch_size=self.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                
                # Map embeddings back to indices
                for idx, emb in zip(text_indices, embeddings):
                    text_embeddings_dict[idx] = emb.astype(np.float32)
            
            # Create full list of embeddings (including empty ones)
            all_embeddings = []
            empty_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
            
            for i in range(len(transcripts)):
                if i in text_embeddings_dict:
                    all_embeddings.append(text_embeddings_dict[i])
                else:
                    # No text for this window - use zero embedding
                    all_embeddings.append(empty_embedding.copy())
            
            return all_embeddings
            
        except Exception as e:
            self._handle_error(e, "encoding transcripts")
            # Return empty embeddings on error
            return [
                np.zeros(self.embedding_dim, dtype=np.float32)
                for _ in transcripts
            ]
    
    def _compute_similarities(
        self, 
        text_embeddings: List[np.ndarray], 
        query_embedding: np.ndarray
    ) -> List[float]:
        """
        Compute cosine similarity between text embeddings and query embedding.
        
        Args:
            text_embeddings: List of text embeddings
            query_embedding: Query embedding (should be normalized)
            
        Returns:
            List of similarity scores (cosine similarity, range [-1, 1])
        """
        try:
            if query_embedding is None:
                return [None] * len(text_embeddings)
            
            # Stack embeddings for efficient computation
            embeddings_array = np.stack(text_embeddings)  # Shape: (n_windows, embedding_dim)
            
            # Compute cosine similarity (dot product since embeddings are normalized)
            similarities = np.dot(embeddings_array, query_embedding)  # Shape: (n_windows,)
            
            # Convert to list of floats
            similarity_scores = similarities.tolist()
            
            return similarity_scores
            
        except Exception as e:
            self._handle_error(e, "computing similarities")
            return [None] * len(text_embeddings)
    
    def teardown(self) -> None:
        """
        Cleanup resources: unload model and free memory.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            self.user_query_embedding = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Text processor resources cleaned up")
            
        except Exception as e:
            self._handle_error(e, "teardown")
