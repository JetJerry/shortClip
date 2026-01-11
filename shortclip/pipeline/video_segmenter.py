import logging
from typing import List, Tuple, Dict, Any
from moviepy import VideoFileClip


class VideoSegmenter:
    """Segments videos into fixed time windows."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the video segmenter.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.window_size_sec = config.get("processing", {}).get("window_size_sec", 2.0)
        self.logger.info(f"Initialized with window size: {self.window_size_sec} seconds")
    
    def segment_videos(self, video_paths: List[str]) -> List[Tuple[str, str, float, float]]:
        """
        Segment multiple videos into fixed time windows.
        
        Args:
            video_paths: List of paths to video files
            
        Returns:
            List of tuples: (video_id, video_path, start_time, end_time)
        """
        segments = []
        
        for idx, video_path in enumerate(video_paths):
            video_id = f"video_{idx:04d}"
            
            try:
                # Get video duration
                with VideoFileClip(video_path) as clip:
                    duration = clip.duration
                
                # Generate segments
                video_segments = self._generate_segments(video_id, video_path, duration)
                segments.extend(video_segments)
                
                self.logger.info(
                    f"Segmented {video_path} (duration: {duration:.2f}s) "
                    f"into {len(video_segments)} segments"
                )
                
            except Exception as e:
                self.logger.error(f"Error segmenting video {video_path}: {e}")
                continue
        
        return segments
    
    def _generate_segments(
        self, 
        video_id: str, 
        video_path: str, 
        duration: float
    ) -> List[Tuple[str, str, float, float]]:
        """
        Generate fixed-size segments for a single video.
        
        Args:
            video_id: Unique identifier for the video
            video_path: Path to the video file
            duration: Total duration of the video in seconds
            
        Returns:
            List of tuples: (video_id, video_path, start_time, end_time)
        """
        segments = []
        start_time = 0.0
        
        while start_time < duration:
            end_time = min(start_time + self.window_size_sec, duration)
            segments.append((video_id, video_path, start_time, end_time))
            start_time = end_time
        
        return segments
