from typing import Tuple, Dict, Any, List, Union, Optional
import torch
from PIL import Image
from moviepy import VideoFileClip
import numpy as np
import os

from .base import BaseProcessor


class VisualProcessor(BaseProcessor):
    """Processes video segments to extract visual embeddings using CLIP."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the visual processor.
        
        Args:
            config: Configuration dictionary containing model and processing parameters
        """
        super().__init__(config)
        self.model = None
        self.preprocess = None
        self.model_name = None
        self.embedding_dim = None
        self.batch_size = config.get("processing", {}).get("batch_size", 16)
    
    def setup(self) -> None:
        """
        Load CLIP model and set up preprocessing.
        Model weights are frozen.
        """
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            # Get model name from config
            self.model_name = self.config.get("models", {}).get("vision", {}).get("name", "openai/clip-vit-base-patch32")
            self.embedding_dim = self.config.get("models", {}).get("vision", {}).get("embedding_dim", 512)
            
            self.logger.info(f"Loading CLIP model: {self.model_name}")
            
            # Load CLIP model and processor
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.preprocess = CLIPProcessor.from_pretrained(self.model_name)
            
            # Freeze all model weights
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info(f"CLIP model loaded and frozen on {self.device}")
            self.logger.info(f"Batch size: {self.batch_size}")
            
        except Exception as e:
            self._handle_error(e, "setup")
            raise
    
    def process(
        self, 
        segments: Union[Tuple[str, str, float, float], List[Tuple[str, str, float, float]]]
    ) -> Union[np.ndarray, List[Optional[np.ndarray]]]:
        """
        Extract visual embeddings from video segment(s).
        Extracts one frame per window and returns 512-dim embeddings.
        Supports both single segment and batch processing.
        
        Args:
            segments: Either a single tuple (video_id, video_path, start_time, end_time)
                     or a list of such tuples
            
        Returns:
            For single segment: numpy array of shape (512,) containing the visual embedding
            For batch: list of numpy arrays (512,) or None for failed segments
        """
        if self.model is None:
            raise RuntimeError("Processor not set up. Call setup() first.")
        
        # Handle single segment
        if isinstance(segments, tuple):
            return self._process_single(segments)
        
        # Handle batch of segments
        return self._process_batch(segments)
    
    def _process_single(self, segment: Tuple[str, str, float, float]) -> Optional[np.ndarray]:
        """
        Process a single segment.
        
        Args:
            segment: Tuple of (video_id, video_path, start_time, end_time)
            
        Returns:
            numpy array of shape (512,) or None if processing failed
        """
        try:
            video_id, video_path, start_time, end_time = segment
            
            # Check if video file exists
            if not os.path.exists(video_path):
                self.logger.warning(f"Video file not found: {video_path}")
                return None
            
            # Extract one frame from the middle of the window
            frame_time = (start_time + end_time) / 2.0
            frame = self._extract_frame(video_path, frame_time)
            
            if frame is None:
                return None
            
            # Process frame through CLIP
            with torch.no_grad():
                inputs = self.preprocess(images=frame, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                
                # Normalize features (CLIP typically uses normalized features)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Extract embedding (should be 512-dim for clip-vit-base-patch32)
                embedding = image_features.cpu().numpy().flatten()
            
            self.logger.debug(
                f"Extracted embedding for {video_id} at {frame_time:.2f}s: "
                f"shape={embedding.shape}"
            )
            
            return embedding
            
        except Exception as e:
            self._handle_error(e, f"processing segment {segment}")
            return None
    
    def _process_batch(self, segments: List[Tuple[str, str, float, float]]) -> List[Optional[np.ndarray]]:
        """
        Process multiple segments in batches.
        
        Args:
            segments: List of tuples (video_id, video_path, start_time, end_time)
            
        Returns:
            List of numpy arrays (512,) or None for failed segments
        """
        results = []
        
        # Process in batches
        for i in range(0, len(segments), self.batch_size):
            batch_segments = segments[i:i + self.batch_size]
            batch_results = self._process_batch_chunk(batch_segments)
            results.extend(batch_results)
        
        successful = sum(1 for r in results if r is not None)
        self.logger.info(f"Processed {successful}/{len(segments)} segments successfully")
        
        return results
    
    def _process_batch_chunk(
        self, 
        segments: List[Tuple[str, str, float, float]]
    ) -> List[Optional[np.ndarray]]:
        """
        Process a single batch chunk of segments.
        
        Args:
            segments: List of segment tuples for this batch
            
        Returns:
            List of embeddings or None for failed segments
        """
        # Extract frames for all segments in batch
        frames = []
        valid_indices = []
        
        for idx, segment in enumerate(segments):
            video_id, video_path, start_time, end_time = segment
            
            try:
                # Check if video file exists
                if not os.path.exists(video_path):
                    self.logger.warning(f"Video file not found: {video_path}")
                    frames.append(None)
                    continue
                
                # Extract frame
                frame_time = (start_time + end_time) / 2.0
                frame = self._extract_frame(video_path, frame_time)
                
                if frame is not None:
                    frames.append(frame)
                    valid_indices.append(idx)
                else:
                    frames.append(None)
                    
            except Exception as e:
                self._handle_error(e, f"extracting frame for segment {segment}")
                frames.append(None)
        
        # Initialize results list with None
        results = [None] * len(segments)
        
        # Process valid frames in batch if any
        if valid_frames := [f for f in frames if f is not None]:
            try:
                with torch.no_grad():
                    # Process all valid frames at once
                    inputs = self.preprocess(images=valid_frames, return_tensors="pt").to(self.device)
                    image_features = self.model.get_image_features(**inputs)
                    
                    # Normalize features
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    
                    # Extract embeddings and assign to results
                    embeddings = image_features.cpu().numpy()
                    
                    for result_idx, embedding in zip(valid_indices, embeddings):
                        results[result_idx] = embedding.flatten()
                
            except Exception as e:
                self._handle_error(e, f"processing batch of {len(valid_frames)} frames")
                # Results already initialized with None for failures
        
        return results
    
    def _extract_frame(self, video_path: str, timestamp: float) -> Optional[Image.Image]:
        """
        Extract a single frame from video at the given timestamp.
        Handles missing or corrupt frames safely.
        
        Args:
            video_path: Path to the video file
            timestamp: Time in seconds to extract frame from
            
        Returns:
            PIL Image of the extracted frame, or None if extraction failed
        """
        try:
            # Check if file exists
            if not os.path.exists(video_path):
                self.logger.warning(f"Video file does not exist: {video_path}")
                return None
            
            # Try to open and extract frame
            with VideoFileClip(video_path) as clip:
                # Validate video duration
                if clip.duration <= 0:
                    self.logger.warning(f"Invalid video duration for {video_path}")
                    return None
                
                # Ensure timestamp is within video bounds
                timestamp = max(0, min(timestamp, clip.duration - 0.1))
                
                # Extract frame as numpy array
                try:
                    frame = clip.get_frame(timestamp)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to extract frame at {timestamp:.2f}s from {video_path}: {e}"
                    )
                    return None
                
                # Validate frame data
                if frame is None or frame.size == 0:
                    self.logger.warning(
                        f"Empty or invalid frame extracted from {video_path} at {timestamp:.2f}s"
                    )
                    return None
                
                # Check frame shape and values
                if len(frame.shape) != 3 or frame.shape[2] != 3:
                    self.logger.warning(
                        f"Unexpected frame shape {frame.shape} from {video_path} at {timestamp:.2f}s"
                    )
                    return None
                
                # Convert numpy array to PIL Image
                try:
                    # Ensure values are in valid range [0, 255] and correct dtype
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                    
                    frame_image = Image.fromarray(frame)
                    
                    # Validate image
                    if frame_image.size[0] == 0 or frame_image.size[1] == 0:
                        self.logger.warning(
                            f"Invalid image size {frame_image.size} from {video_path}"
                        )
                        return None
                    
                    return frame_image
                    
                except Exception as e:
                    self.logger.warning(
                        f"Failed to convert frame to PIL Image from {video_path}: {e}"
                    )
                    return None
                
        except FileNotFoundError:
            self.logger.warning(f"Video file not found: {video_path}")
            return None
        except Exception as e:
            self._handle_error(e, f"extracting frame at {timestamp:.2f}s from {video_path}")
            return None
    
    def teardown(self) -> None:
        """
        Cleanup resources: unload model and free memory.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.preprocess is not None:
                del self.preprocess
                self.preprocess = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info("Visual processor resources cleaned up")
            
        except Exception as e:
            self._handle_error(e, "teardown")
