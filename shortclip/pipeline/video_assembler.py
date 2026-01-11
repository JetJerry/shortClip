from typing import List, Tuple, Dict, Any, Optional
import logging
import os
from moviepy import VideoFileClip, concatenate_videoclips


class VideoAssembler:
    """
    Assembles selected clips from multiple videos into a single short video.
    Extracts clips and concatenates them while preserving original audio.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the video assembler.
        
        Args:
            config: Configuration dictionary (for consistency with other processors)
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("VideoAssembler initialized")
    
    def assemble(
        self,
        clips: List[Tuple[str, float, float]],
        output_path: str,
        video_path_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Assemble clips from multiple videos into a single output video.
        
        Args:
            clips: List of tuples (video_path, start_time, end_time) or (video_id, start_time, end_time)
                  If video_id is used, video_path_map must be provided
            output_path: Path to save the output video
            video_path_map: Optional mapping from video_id to video_path
                          (required if clips contain video_id instead of video_path)
        
        Returns:
            Path to the output video file
        
        Raises:
            ValueError: If video_path_map is required but not provided
            RuntimeError: If no valid clips could be extracted
        """
        if not clips:
            raise ValueError("No clips provided for assembly")
        
        # Extract subclips
        subclips = []
        for clip_info in clips:
            if len(clip_info) == 3:
                identifier, start_time, end_time = clip_info
                
                # Determine if identifier is video_path or video_id
                if video_path_map and identifier in video_path_map:
                    video_path = video_path_map[identifier]
                elif os.path.exists(identifier):
                    video_path = identifier
                else:
                    if video_path_map is None:
                        raise ValueError(
                            f"Clip identifier '{identifier}' is not a valid file path "
                            "and video_path_map was not provided"
                        )
                    self.logger.warning(
                        f"Video ID '{identifier}' not found in video_path_map, skipping"
                    )
                    continue
                
                subclip = self._extract_clip(video_path, start_time, end_time)
                if subclip is not None:
                    subclips.append(subclip)
        
        if not subclips:
            raise RuntimeError("No valid clips could be extracted for assembly")
        
        self.logger.info(f"Extracted {len(subclips)} clips, concatenating...")
        
        # Concatenate clips
        try:
            final_clip = concatenate_videoclips(subclips, method="compose")
            
            # Write output video
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Close clips to free resources
            final_clip.close()
            for subclip in subclips:
                subclip.close()
            
            self.logger.info(f"Video assembled successfully: {output_path}")
            return output_path
            
        except Exception as e:
            # Clean up clips on error
            for subclip in subclips:
                try:
                    subclip.close()
                except:
                    pass
            self.logger.error(f"Error assembling video: {e}")
            raise
    
    def _extract_clip(
        self,
        video_path: str,
        start_time: float,
        end_time: float
    ) -> Optional[VideoFileClip]:
        """
        Extract a clip from a video file.
        
        Args:
            video_path: Path to the video file
            start_time: Start time in seconds
            end_time: End time in seconds
        
        Returns:
            VideoFileClip subclip, or None if extraction failed
        """
        try:
            # Validate video file exists
            if not os.path.exists(video_path):
                self.logger.warning(f"Video file not found: {video_path}")
                return None
            
            # Open video
            clip = VideoFileClip(video_path)
            
            # Validate time bounds
            if start_time < 0:
                start_time = 0
            if end_time > clip.duration:
                end_time = clip.duration
            if start_time >= end_time:
                self.logger.warning(
                    f"Invalid time range [{start_time:.2f}, {end_time:.2f}] for {video_path}"
                )
                clip.close()
                return None
            
            # Extract subclip (preserves audio by default)
            try:
                subclip = clip.subclip(start_time, end_time)
                self.logger.debug(
                    f"Extracted clip from {video_path}: [{start_time:.2f}s, {end_time:.2f}s] "
                    f"(duration: {end_time - start_time:.2f}s)"
                )
                # Note: We close the original clip but return the subclip
                # The subclip will be closed by the caller
                clip.close()
                return subclip
                
            except Exception as e:
                self.logger.warning(
                    f"Failed to extract subclip from {video_path} "
                    f"[{start_time:.2f}s, {end_time:.2f}s]: {e}"
                )
                clip.close()
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting clip from {video_path}: {e}")
            return None