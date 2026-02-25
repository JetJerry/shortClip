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
        """
        if not clips:
            raise ValueError("No clips provided for assembly")
        
        # Check for VFX flags
        features = self.config.get("features", {})
        use_zoom = features.get("use_zoom_motion", False)
        use_music = features.get("use_music_overlay", False)
        
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
                    # Apply V2 Effects: Zoom
                    if use_zoom:
                        try:
                            subclip = self.apply_zoom_effect(subclip)
                        except Exception as e:
                            self.logger.error(f"Failed to apply zoom effect: {e}")
                            
                    subclips.append(subclip)
        
        if not subclips:
            raise RuntimeError("No valid clips could be extracted for assembly")
        
        self.logger.info(f"Extracted {len(subclips)} clips, concatenating...")
        
        # Concatenate clips
        try:
            final_clip = concatenate_videoclips(subclips, method="compose")
            
            # Apply Music Overlay
            if use_music:
                try:
                    final_clip = self.apply_music_overlay(final_clip)
                except Exception as e:
                    self.logger.error(f"Failed to apply music overlay: {e}")

            # Write output video
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
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

    def apply_music_overlay(self, video_clip):
        """
        Selects a random music track and overlays it on the video.
        """
        import random
        from moviepy.audio.io.AudioFileClip import AudioFileClip
        from moviepy.audio.AudioClip import CompositeAudioClip
        
        music_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "music")
        # Hardcoded fallback relative path: d:/autoClip_04/model/shortclip/pipeline/../../../music
        # -> d:/autoClip_04/music
        
        if not os.path.exists(music_dir):
             self.logger.warning(f"Music directory not found at {music_dir}")
             return video_clip

        tracks = [f for f in os.listdir(music_dir) if f.endswith(('.mp3', '.wav'))]
        if not tracks:
             self.logger.warning("No music tracks found.")
             return video_clip
             
        track_name = random.choice(tracks)
        music_path = os.path.join(music_dir, track_name)
        self.logger.info(f"Adding background music: {track_name}")
        
        music = AudioFileClip(music_path)
        
        # Loop music if shorter than video
        if music.duration < video_clip.duration:
            from moviepy.audio.fx.all import audio_loop
            music = audio_loop(music, duration=video_clip.duration)
        else:
            music = music.subclipped(0, video_clip.duration)
            
        # Ducking: Set volume low (0.15) to not overpower speech
        music = music.with_volume_scaled(0.15)
        
        # Combine with original audio
        if video_clip.audio:
            final_audio = CompositeAudioClip([video_clip.audio, music])
        else:
            final_audio = music
            
        video_clip.audio = final_audio
        return video_clip

    def apply_zoom_effect(self, clip, zoom_ratio=0.04):
        """
        Applies a dynamic 'Ken Burns' style zoom-in effect.
        Args:
            clip: MoviePy VideoClip
            zoom_ratio: Total zoom percentage (0.04 means 4% zoom in)
        """
        w, h = clip.size
        
        # Define zoom function: t -> (new_w, new_h)
        def get_new_size(t):
            scale = 1 + (zoom_ratio * (t / clip.duration))
            return (int(w * scale), int(h * scale))

        # Apply resize using the zoom function
        zoomed_clip = clip.resize(get_new_size)
        
        # Center Crop using clip.crop
        # We want to keep the center of the zoomed frame
        # crop(x_center, y_center, width, height)
        # However, clip.crop usually takes static values. 
        # For dynamic cropping consistent with dynamic resize, we might need a custom filter if crop(t) isn't supported.
        # But standard MoviePy crop often doesn't support time-varying params unless using vfx.crop.
        
        # Alternative simpler 'Ken Burns':
        # 1. Resize the CLIP to be (W*1.04, H*1.04) statically? No that's static.
        
        # Let's try the safest path for dynamic zoom:
        # Use margin/crop.
        
        # As a fallback for stability if dynamic resize fails:
        # Just do a slight static zoom (1.02x) to verify it works first?
        # No, let's try to do it right.
        
        try:
             # Standard v1.0.3 way with vfx locally imported if it exists, or using clip.resize
             # If resize(lambda t...) works, then the clip size changes over time.
             # Then we just need to force it to center 
             return zoomed_clip.set_position('center').crop(x_center=zoomed_clip.w/2, y_center=zoomed_clip.h/2, width=w, height=h)
        except Exception:
             # Fallback: simple static zoom if dynamic fails
             self.logger.warning("Dynamic zoom failed, trying static zoom fallback.")
             return clip.resize(1.0 + zoom_ratio).crop(x_center=w/2 * (1.0+zoom_ratio), y_center=h/2 * (1.0+zoom_ratio), width=w, height=h)

        # End of valid zoom logic

    
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
                subclip = clip.subclipped(start_time, end_time)
                self.logger.debug(
                    f"Extracted clip from {video_path}: [{start_time:.2f}s, {end_time:.2f}s] "
                    f"(duration: {end_time - start_time:.2f}s)"
                )
                # Note: We close the original clip but return the subclip
                # The subclip will be closed by the caller
                # clip.close()  <-- Do NOT close here, it breaks the subclip
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