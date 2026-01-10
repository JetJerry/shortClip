from typing import Tuple, Dict, Any, List, Optional
import torch
import numpy as np
import os
from dataclasses import dataclass
from moviepy.editor import VideoFileClip
import librosa

from .base import BaseProcessor


@dataclass
class TranscriptSegment:
    """Represents a transcript segment with timestamp and embedding."""
    start_time: float
    end_time: float
    text: str
    embedding: np.ndarray  # Whisper encoder embedding


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
    
    def process(self, segment: Tuple[str, str, float, float]) -> List[TranscriptSegment]:
        """
        Extract audio from video segment, transcribe, and get aligned embeddings.
        
        Args:
            segment: Tuple of (video_id, video_path, start_time, end_time)
            
        Returns:
            List of TranscriptSegment objects with timestamps, text, and embeddings
        """
        if self.model is None:
            raise RuntimeError("Processor not set up. Call setup() first.")
        
        try:
            video_id, video_path, start_time, end_time = segment
            
            # Check if video file exists
            if not os.path.exists(video_path):
                self.logger.warning(f"Video file not found: {video_path}")
                return []
            
            # Extract audio from video segment
            audio_array = self._extract_audio(video_path, start_time, end_time)
            if audio_array is None:
                return []
            
            # Run transcription and get segments with timestamps
            transcript_segments = self._transcribe_with_segments(audio_array, start_time)
            
            # Extract embeddings aligned with transcript segments
            transcript_segments_with_embeddings = self._extract_aligned_embeddings(
                audio_array, transcript_segments
            )
            
            self.logger.debug(
                f"Processed audio for {video_id} ({start_time:.2f}s-{end_time:.2f}s): "
                f"found {len(transcript_segments_with_embeddings)} transcript segments"
            )
            
            return transcript_segments_with_embeddings
            
        except Exception as e:
            self._handle_error(e, f"processing segment {segment}")
            return []
    
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
                audio_clip = clip.subclip(start_time, end_time).audio
                
                if audio_clip is None:
                    self.logger.warning(f"No audio track found in {video_path}")
                    return None
                
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
    
    def _transcribe_with_segments(
        self, 
        audio_array: np.ndarray, 
        offset_time: float
    ) -> List[Dict[str, Any]]:
        """
        Transcribe audio and get segments with timestamps.
        Uses Whisper's timestamp token capabilities.
        
        Args:
            audio_array: Audio array at 16kHz
            offset_time: Time offset to add to segment timestamps (start_time of video segment)
            
        Returns:
            List of dictionaries with 'start', 'end', 'text' keys
        """
        try:
            # Process audio for Whisper
            inputs = self.processor(
                audio_array, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Calculate audio duration for timestamp estimation
            duration = len(audio_array) / self.sample_rate
            
            # Generate transcription
            with torch.no_grad():
                # Generate with return_timestamps to enable timestamp tokens
                generated_ids = self.model.generate(
                    inputs["input_features"],
                    return_timestamps=True,
                    max_length=448
                )
                
                # Decode transcription
                transcription = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )[0]
            
            segments = []
            
            # Extract segments from transcription
            # For longer transcriptions, split into multiple segments
            # Timestamp extraction can be improved with word-level alignment if needed
            if transcription.strip():
                words = transcription.split()
                
                # If transcription is short, create single segment
                if len(words) <= 10:
                    segments.append({
                        "start": offset_time,
                        "end": offset_time + duration,
                        "text": transcription
                    })
                else:
                    # Split longer transcriptions into approximately equal segments
                    num_segments = min(3, max(1, len(words) // 10))
                    words_per_segment = len(words) // num_segments
                    segment_duration = duration / num_segments
                    
                    for i in range(num_segments):
                        start_idx = i * words_per_segment
                        end_idx = (i + 1) * words_per_segment if i < num_segments - 1 else len(words)
                        segment_text = " ".join(words[start_idx:end_idx])
                        
                        if segment_text.strip():
                            segments.append({
                                "start": offset_time + i * segment_duration,
                                "end": offset_time + (i + 1) * segment_duration if i < num_segments - 1 else offset_time + duration,
                                "text": segment_text.strip()
                            })
            
            return segments
            
        except Exception as e:
            self._handle_error(e, "transcribing audio")
            # Fallback: create single segment with full duration
            try:
                duration = len(audio_array) / self.sample_rate
                # Try to get at least the transcription text
                inputs = self.processor(
                    audio_array, 
                    sampling_rate=self.sample_rate, 
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    generated_ids = self.model.generate(inputs["input_features"])
                    transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if transcription.strip():
                    return [{
                        "start": offset_time,
                        "end": offset_time + duration,
                        "text": transcription
                    }]
            except:
                pass
            
            return []
    
    def _extract_aligned_embeddings(
        self, 
        audio_array: np.ndarray, 
        transcript_segments: List[Dict[str, Any]]
    ) -> List[TranscriptSegment]:
        """
        Extract Whisper encoder embeddings aligned with transcript segments.
        
        Args:
            audio_array: Full audio array
            transcript_segments: List of transcript segments with timestamps
            
        Returns:
            List of TranscriptSegment objects with embeddings
        """
        transcript_segments_with_embeddings = []
        
        try:
            # Process full audio through encoder once
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
            
            # Align embeddings to transcript segments
            total_duration = len(audio_array) / self.sample_rate
            seq_len = encoder_embeddings.shape[0]
            
            for segment in transcript_segments:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                
                # Map time to sequence position
                # Whisper uses mel-spectrogram frames, roughly 20ms per frame
                # Each encoder token corresponds to ~2 seconds of audio (approximate)
                start_ratio = (start_time % total_duration) / total_duration if total_duration > 0 else 0
                end_ratio = (end_time % total_duration) / total_duration if total_duration > 0 else 1
                
                start_idx = int(start_ratio * seq_len)
                end_idx = int(end_ratio * seq_len)
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx + 1, min(end_idx, seq_len))
                
                # Average embeddings over the segment
                segment_embeddings = encoder_embeddings[start_idx:end_idx]
                segment_embedding = np.mean(segment_embeddings, axis=0)
                
                transcript_segments_with_embeddings.append(
                    TranscriptSegment(
                        start_time=start_time,
                        end_time=end_time,
                        text=text,
                        embedding=segment_embedding
                    )
                )
            
        except Exception as e:
            self._handle_error(e, "extracting aligned embeddings")
            # Return segments without embeddings if extraction fails
            for segment in transcript_segments:
                # Create empty embedding as fallback
                empty_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                transcript_segments_with_embeddings.append(
                    TranscriptSegment(
                        start_time=segment["start"],
                        end_time=segment["end"],
                        text=segment["text"],
                        embedding=empty_embedding
                    )
                )
        
        return transcript_segments_with_embeddings
    
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
