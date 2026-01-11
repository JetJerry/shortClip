from typing import Tuple, Dict, Any, List, Optional
import torch
import numpy as np
import os
from dataclasses import dataclass
from moviepy import VideoFileClip
import librosa

from .base import BaseProcessor


@dataclass
class TranscriptSegment:
    """Represents a transcript segment with timestamp and embedding."""
    start_time: float
    end_time: float
    text: str
    embedding: np.ndarray  # Whisper encoder embedding


@dataclass
class WindowAudioEmbedding:
    """Represents an audio embedding aligned to a video window."""
    window_start_time: float
    window_end_time: float
    embedding: np.ndarray  # Whisper encoder embedding for this window
    transcript: Optional[str]  # Transcription text if speech detected, None otherwise
    has_speech: bool  # Whether speech was detected in this window


class AudioProcessor(BaseProcessor):
    """Processes video segments to extract audio, transcribe, and get embeddings."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the audio processor.
        
        Args:
            config: Configuration dictionary containing model and processing parameters
        """
        super().__init__(config)
        self.model = None
        self.processor = None
        self.model_name = None
        self.embedding_dim = None
        self.sample_rate = 16000  # Whisper uses 16kHz sample rate
    
    def setup(self) -> None:
        """
        Load Whisper model and set up processing.
        Model weights are frozen.
        """
        try:
            from transformers import WhisperModel, WhisperProcessor
            
            # Get model name from config
            self.model_name = self.config.get("models", {}).get("audio", {}).get("name", "openai/whisper-base")
            self.embedding_dim = self.config.get("models", {}).get("audio", {}).get("embedding_dim", 1280)
            
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            
            # Load Whisper model and processor
            self.model = WhisperModel.from_pretrained(self.model_name)
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            
            # Freeze all model weights
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"Whisper model loaded and frozen on {self.device}")
            
        except Exception as e:
            self._handle_error(e, "setup")
            raise
    
    def process(self, segment: Tuple[str, str, float, float]) -> WindowAudioEmbedding:
        """
        Extract audio from video window, transcribe if speech present, and get embedding.
        Returns a single embedding aligned to the video window timestamps.
        Handles videos without speech gracefully by returning audio embedding anyway.
        
        Args:
            segment: Tuple of (video_id, video_path, start_time, end_time) representing a video window
            
        Returns:
            WindowAudioEmbedding object with window-aligned embedding and optional transcript
        """
        if self.model is None:
            raise RuntimeError("Processor not set up. Call setup() first.")
        
        video_id, video_path, start_time, end_time = segment
        
        # Default: return empty embedding if processing fails
        default_embedding = WindowAudioEmbedding(
            window_start_time=start_time,
            window_end_time=end_time,
            embedding=np.zeros(self.embedding_dim, dtype=np.float32),
            transcript=None,
            has_speech=False
        )
        
        try:
            # Check if video file exists
            if not os.path.exists(video_path):
                self.logger.warning(f"Video file not found: {video_path}")
                return default_embedding
            
            # Extract audio from video window
            audio_array = self._extract_audio(video_path, start_time, end_time)
            if audio_array is None:
                self.logger.debug(f"Failed to extract audio for {video_id} window [{start_time:.2f}s-{end_time:.2f}s]")
                return default_embedding
            
            # Even if audio array is empty or silent, we'll still try to get an embedding
            if len(audio_array) == 0:
                self.logger.debug(f"Empty audio array for {video_id} window [{start_time:.2f}s-{end_time:.2f}s]")
                # Create minimal silent audio for embedding extraction
                duration = end_time - start_time
                audio_array = np.zeros(max(1, int(duration * self.sample_rate)), dtype=np.float32)
            
            # Extract embedding for the entire window (always done, even without speech)
            window_embedding = self._extract_window_embedding(audio_array)
            
            if window_embedding is None:
                self.logger.warning(f"Failed to extract embedding for {video_id} window [{start_time:.2f}s-{end_time:.2f}s]")
                return default_embedding
            
            # Try to transcribe to detect speech and get transcript
            transcript = self._transcribe_audio(audio_array)
            has_speech = transcript is not None and len(transcript.strip()) > 0
            
            if has_speech:
                self.logger.debug(
                    f"Processed audio for {video_id} window [{start_time:.2f}s-{end_time:.2f}s]: "
                    f"speech detected, transcript: '{transcript[:50]}...'"
                )
            else:
                self.logger.debug(
                    f"Processed audio for {video_id} window [{start_time:.2f}s-{end_time:.2f}s]: "
                    f"no speech detected, returning audio embedding"
                )
            
            return WindowAudioEmbedding(
                window_start_time=start_time,
                window_end_time=end_time,
                embedding=window_embedding,
                transcript=transcript if has_speech else None,
                has_speech=has_speech
            )
            
        except Exception as e:
            self._handle_error(e, f"processing segment {segment}")
            return default_embedding
    
    def _extract_audio(self, video_path: str, start_time: float, end_time: float) -> Optional[np.ndarray]:
        """
        Extract audio from video segment and convert to the required format.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Audio array at 16kHz sample rate, or None if extraction failed
        """
        try:
            with VideoFileClip(video_path) as clip:
                # Validate time bounds
                if start_time < 0:
                    start_time = 0
                if end_time > clip.duration:
                    end_time = clip.duration
                if start_time >= end_time:
                    self.logger.warning(
                        f"Invalid time range [{start_time:.2f}, {end_time:.2f}] for {video_path}"
                    )
                    return None
                
                # Extract audio subclip
                try:
                    subclip = clip.subclip(start_time, end_time)
                    audio_clip = subclip.audio if hasattr(subclip, 'audio') else None
                except Exception as e:
                    self.logger.debug(f"Could not extract subclip from {video_path}: {e}")
                    audio_clip = None
                
                if audio_clip is None:
                    self.logger.debug(f"No audio track found for {video_path} window [{start_time:.2f}s-{end_time:.2f}s]")
                    # Return silent audio array instead of None to allow embedding extraction
                    duration = end_time - start_time
                    silent_audio = np.zeros(int(duration * self.sample_rate), dtype=np.float32)
                    return silent_audio
                
                # Write to temporary file and load with librosa
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    audio_clip.write_audiofile(
                        tmp_path, 
                        fps=self.sample_rate, 
                        verbose=False, 
                        logger=None
                    )
                    
                    # Load audio with librosa to get numpy array
                    audio_array, sr = librosa.load(tmp_path, sr=self.sample_rate, mono=True)
                    
                    return audio_array
                    
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    
        except Exception as e:
            self._handle_error(e, f"extracting audio from {video_path}")
            return None
    
    def _transcribe_audio(self, audio_array: np.ndarray) -> Optional[str]:
        """
        Transcribe audio and return text if speech is detected.
        Returns None if no speech is detected or transcription fails.
        
        Args:
            audio_array: Audio array at 16kHz
            
        Returns:
            Transcribed text string, or None if no speech detected/failed
        """
        try:
            # Check if audio has meaningful content (simple energy check)
            audio_energy = np.mean(np.abs(audio_array))
            if audio_energy < 1e-6:  # Very quiet or silence
                self.logger.debug("Audio energy too low, likely silence")
                return None
            
            # Process audio for Whisper
            inputs = self.processor(
                audio_array, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    max_length=448
                )
                
                # Decode transcription
                transcription = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
            
            # Filter out common non-speech outputs
            transcription = transcription.strip()
            
            # Check if transcription is meaningful (not just silence tokens or empty)
            if not transcription or transcription.lower() in ["", "thank you", "."]:
                return None
            
            # Check for very short transcriptions that might be noise
            if len(transcription.split()) < 2 and len(transcription) < 3:
                return None
            
            return transcription
            
        except Exception as e:
            # Fail gracefully - return None instead of raising
            self.logger.debug(f"Transcription failed or no speech detected: {e}")
            return None
    
    def _extract_window_embedding(self, audio_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract Whisper encoder embedding for the entire audio window.
        Always extracts embedding regardless of whether speech is present.
        
        Args:
            audio_array: Audio array at 16kHz for the window
            
        Returns:
            numpy array of shape (embedding_dim,) with the window embedding, or None if failed
        """
        try:
            # Ensure minimum audio length (Whisper needs at least some audio)
            min_length = self.sample_rate * 0.25  # At least 250ms
            if len(audio_array) < min_length:
                # Pad with zeros if too short
                audio_array = np.pad(
                    audio_array, 
                    (0, max(0, int(min_length) - len(audio_array))), 
                    mode='constant'
                )
            
            # Process audio through Whisper encoder
            inputs = self.processor(
                audio_array, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                # Get encoder outputs (embeddings)
                encoder_outputs = self.model.model.encoder(**inputs)
                # encoder_outputs.last_hidden_state shape: (batch, seq_len, hidden_dim)
                encoder_embeddings = encoder_outputs.last_hidden_state.squeeze(0).cpu().numpy()
                # Shape: (seq_len, hidden_dim)
            
            # Average pooling over sequence length to get single window embedding
            # This gives us one embedding per window, aligned to the window timestamps
            window_embedding = np.mean(encoder_embeddings, axis=0)
            
            # Normalize embedding
            norm = np.linalg.norm(window_embedding)
            if norm > 0:
                window_embedding = window_embedding / norm
            
            return window_embedding.astype(np.float32)
            
        except Exception as e:
            self._handle_error(e, "extracting window embedding")
            return None
    
    
    def teardown(self) -> None:
        """
        Cleanup resources: unload model and free memory.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Audio processor resources cleaned up")
            
        except Exception as e:
            self._handle_error(e, "teardown")
