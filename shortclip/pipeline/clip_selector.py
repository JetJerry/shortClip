from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import logging
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from .moment import Moment


class ClipSelector:
    """
    Selects highlight clips from scored Moments across all videos.
    Applies temporal smoothing, peak detection, and non-overlap constraints.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the clip selector.
        
        Args:
            config: Configuration dictionary containing selection parameters
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Get selection parameters from config
        selection_config = config.get("selection", {})
        self.max_clips_per_video = selection_config.get("max_clips_per_video", 2)
        self.min_clip_sec = selection_config.get("min_clip_sec", 2)
        self.max_clip_sec = selection_config.get("max_clip_sec", 10)
        
        # Temporal smoothing parameter
        self.smoothing_sigma = config.get("processing", {}).get("temporal_smoothing_sigma", 1.0)
        
        self.logger.info(
            f"ClipSelector initialized: max_clips_per_video={self.max_clips_per_video}, "
            f"min_clip_sec={self.min_clip_sec}, max_clip_sec={self.max_clip_sec}"
        )
    
    def select_clips(self, moments: List[Moment]) -> List[Tuple[str, float, float]]:
        """
        Select highlight clips from scored moments.
        
        Args:
            moments: List of Moment objects with scores from all videos
            
        Returns:
            List of tuples (video_id, start_time, end_time) for selected clips
        """
        if not moments:
            self.logger.warning("No moments provided for clip selection")
            return []
        
        # Filter out moments without scores
        scored_moments = [m for m in moments if m.score is not None]
        if not scored_moments:
            self.logger.warning("No moments with scores found")
            return []
        
        # Group moments by video_id
        moments_by_video = defaultdict(list)
        for moment in scored_moments:
            moments_by_video[moment.video_id].append(moment)
        
        selected_clips = []
        
        # Process each video separately
        for video_id, video_moments in moments_by_video.items():
            video_clips = self._select_clips_for_video(video_moments)
            selected_clips.extend(video_clips)
            self.logger.debug(
                f"Selected {len(video_clips)} clips from {video_id} "
                f"(out of {len(video_moments)} moments)"
            )
        
        self.logger.info(f"Selected {len(selected_clips)} clips total from {len(moments_by_video)} videos")
        return selected_clips
    
    def _select_clips_for_video(self, moments: List[Moment]) -> List[Tuple[str, float, float]]:
        """
        Select clips for a single video.
        
        Args:
            moments: List of Moment objects from the same video
            
        Returns:
            List of tuples (video_id, start_time, end_time) for selected clips
        """
        if not moments:
            return []
        
        # Get video_id (all moments should have the same video_id)
        video_id = moments[0].video_id
        
        # Sort moments by start_time
        sorted_moments = sorted(moments, key=lambda m: m.start_time)
        
        # Apply temporal smoothing
        smoothed_moments = self._apply_temporal_smoothing(sorted_moments)
        
        # Extract scores and timestamps
        scores = np.array([m.score for m in smoothed_moments], dtype=np.float32)
        start_times = np.array([m.start_time for m in smoothed_moments], dtype=np.float32)
        end_times = np.array([m.end_time for m in smoothed_moments], dtype=np.float32)
        
        # Find peaks in scores
        peak_indices = self._find_peaks(scores)
        
        if len(peak_indices) == 0:
            self.logger.debug(f"No peaks found for {video_id}")
            return []
        
        # Select clips from peaks with constraints
        selected_clips = self._select_from_peaks(
            peak_indices,
            smoothed_moments,
            scores,
            video_id
        )
        
        return selected_clips
    
    def _apply_temporal_smoothing(self, moments: List[Moment]) -> List[Moment]:
        """
        Apply temporal smoothing to scores within a video.
        
        Args:
            moments: List of Moments from the same video, sorted by time
            
        Returns:
            List of Moments with smoothed scores
        """
        if len(moments) <= 1:
            return moments
        
        # Extract scores
        scores = np.array([m.score for m in moments], dtype=np.float32)
        
        # Apply Gaussian smoothing
        if len(scores) > 2 and self.smoothing_sigma > 0:
            smoothed_scores = gaussian_filter1d(scores, sigma=self.smoothing_sigma)
            smoothed_scores = np.clip(smoothed_scores, 0.0, 1.0)
        else:
            smoothed_scores = scores
        
        # Create new moments with smoothed scores
        smoothed_moments = []
        for moment, smoothed_score in zip(moments, smoothed_scores):
            smoothed_moment = Moment(
                video_id=moment.video_id,
                video_path=moment.video_path,
                start_time=moment.start_time,
                end_time=moment.end_time,
                visual_embedding=moment.visual_embedding,
                audio_embedding=moment.audio_embedding,
                text_embedding=moment.text_embedding,
                label=moment.label,
                score=float(smoothed_score)
            )
            smoothed_moments.append(smoothed_moment)
        
        return smoothed_moments
    
    def _find_peaks(self, scores: np.ndarray) -> np.ndarray:
        """
        Find peaks in the score sequence.
        
        Args:
            scores: Array of scores
            
        Returns:
            Array of peak indices
        """
        if len(scores) < 3:
            # Too few points for peak detection, return the highest scoring index
            if len(scores) > 0:
                return np.array([np.argmax(scores)])
            return np.array([])
        
        # Find peaks with minimum height and distance constraints
        # height: at least 0.1 above minimum score
        min_height = np.min(scores) + 0.1 * (np.max(scores) - np.min(scores))
        # distance: at least 2 points apart (to avoid too many peaks)
        min_distance = max(1, len(scores) // 20)
        
        peaks, properties = find_peaks(
            scores,
            height=min_height,
            distance=min_distance
        )
        
        return peaks
    
    def _select_from_peaks(
        self,
        peak_indices: np.ndarray,
        moments: List[Moment],
        scores: np.ndarray,
        video_id: str
    ) -> List[Tuple[str, float, float]]:
        """
        Select clips from peaks respecting non-overlap and max_clips_per_video constraints.
        
        Args:
            peak_indices: Indices of peaks in the moments list
            moments: List of Moments (sorted by time)
            scores: Array of scores
            video_id: Video identifier
            
        Returns:
            List of tuples (video_id, start_time, end_time) for selected clips
        """
        if len(peak_indices) == 0:
            return []
        
        # Get peak scores and sort by score (descending)
        peak_scores = scores[peak_indices]
        sorted_indices = peak_indices[np.argsort(peak_scores)[::-1]]  # Sort descending
        
        selected_clips = []
        used_times = []  # Track used time ranges to enforce non-overlap
        
        for peak_idx in sorted_indices:
            if len(selected_clips) >= self.max_clips_per_video:
                break
            
            moment = moments[peak_idx]
            start_time = moment.start_time
            end_time = moment.end_time
            
            # Check if this clip overlaps with already selected clips
            if self._overlaps_with_used(start_time, end_time, used_times):
                continue
            
            # Ensure clip duration is within bounds
            duration = end_time - start_time
            if duration < self.min_clip_sec:
                # Extend clip to meet minimum duration
                center = (start_time + end_time) / 2.0
                start_time = max(0, center - self.min_clip_sec / 2.0)
                end_time = start_time + self.min_clip_sec
            elif duration > self.max_clip_sec:
                # Trim clip to meet maximum duration
                center = (start_time + end_time) / 2.0
                start_time = max(0, center - self.max_clip_sec / 2.0)
                end_time = start_time + self.max_clip_sec
            
            # Final validation: ensure start_time >= 0 (already handled above, but double-check)
            start_time = max(0, start_time)
            
            # Skip if the adjusted clip now overlaps with used times
            if self._overlaps_with_used(start_time, end_time, used_times):
                continue
            
            # Add clip
            selected_clips.append((video_id, float(start_time), float(end_time)))
            used_times.append((start_time, end_time))
        
        # Sort selected clips by start_time
        selected_clips.sort(key=lambda x: x[1])
        
        return selected_clips
    
    def _overlaps_with_used(
        self,
        start_time: float,
        end_time: float,
        used_times: List[Tuple[float, float]]
    ) -> bool:
        """
        Check if a time range overlaps with any used time ranges.
        
        Args:
            start_time: Start time of the clip
            end_time: End time of the clip
            used_times: List of (start, end) tuples for already selected clips
            
        Returns:
            True if overlap exists, False otherwise
        """
        for used_start, used_end in used_times:
            # Check for overlap: not (end <= used_start or start >= used_end)
            if not (end_time <= used_start or start_time >= used_end):
                return True
        return False
