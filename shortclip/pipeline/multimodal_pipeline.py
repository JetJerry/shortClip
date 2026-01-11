from typing import List, Dict, Any, Optional, Tuple
import logging
import os
from collections import defaultdict
import numpy as np

from .video_segmenter import VideoSegmenter
from .visual_processor import VisualProcessor
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor
from .scene_context import SceneContextBuilder
from .feature_fusion import FeatureFusion
from .clip_selector import ClipSelector
from .video_assembler import VideoAssembler
from .moment import Moment


class MultimodalPipeline:
    """
    Orchestrates all pipeline stages to process multiple videos and generate a short highlight video.
    
    Stages:
    1. Segmentation - segment videos into fixed time windows
    2. Visual processing - extract CLIP embeddings
    3. Audio processing - extract Whisper embeddings and transcripts
    4. Text processing - extract Sentence-BERT embeddings (with optional query)
    5. Scene context - combine embeddings into SceneContext objects
    6. Feature fusion - score moments using FusionModel
    7. Clip selection - select top clips with constraints
    8. Video assembly - assemble selected clips into final video
    """
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """
        Initialize the multimodal pipeline.
        
        Args:
            config: Configuration dictionary containing all model and processing parameters
            model_path: Path to trained FusionModel checkpoint (required for scoring)
        """
        self.config = config
        self.model_path = model_path
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize all processors
        self.segmenter = VideoSegmenter(config)
        self.visual_processor = VisualProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.text_processor = TextProcessor(config)
        self.scene_builder = SceneContextBuilder(config)
        self.feature_fusion = FeatureFusion(config, model_path)
        self.clip_selector = ClipSelector(config)
        self.video_assembler = VideoAssembler(config)
        
        self.logger.info("MultimodalPipeline initialized")
    
    def process(
        self,
        video_paths: List[str],
        output_path: str,
        user_query: Optional[str] = None
    ) -> str:
        """
        Process multiple videos and generate a short highlight video.
        
        Args:
            video_paths: List of paths to input video files
            output_path: Path to save the output video
            user_query: Optional query string for text-based filtering
        
        Returns:
            Path to the output video file
        
        Raises:
            RuntimeError: If processing fails at any stage
        """
        if not video_paths:
            raise ValueError("No video paths provided")
        
        self.logger.info(f"Starting pipeline for {len(video_paths)} video(s)")
        if user_query:
            self.logger.info(f"User query: '{user_query}'")
        
        try:
            # Stage 1: Segmentation
            self.logger.info("Stage 1: Segmenting videos...")
            video_segments = self.segmenter.segment_videos(video_paths)
            if not video_segments:
                raise RuntimeError("No video segments generated")
            self.logger.info(f"Generated {len(video_segments)} segments")
            
            # Build video_path_map for later use
            video_path_map = {}
            for video_id, video_path, _, _ in video_segments:
                if video_id not in video_path_map:
                    video_path_map[video_id] = video_path
            
            # Stage 2: Visual processing
            self.logger.info("Stage 2: Processing visual features...")
            self.visual_processor.setup()
            try:
                visual_embeddings = self.visual_processor.process(video_segments)
                self.logger.info(f"Processed {len(visual_embeddings)} visual embeddings")
            finally:
                self.visual_processor.teardown()
            
            # Stage 3: Audio processing
            self.logger.info("Stage 3: Processing audio features...")
            self.audio_processor.setup()
            try:
                audio_embeddings = []
                for segment in video_segments:
                    audio_emb = self.audio_processor.process(segment)
                    audio_embeddings.append(audio_emb)
                self.logger.info(f"Processed {len(audio_embeddings)} audio embeddings")
            finally:
                self.audio_processor.teardown()
            
            # Stage 4: Text processing
            self.logger.info("Stage 4: Processing text features...")
            self.text_processor.setup()
            try:
                text_embeddings = self.text_processor.process(audio_embeddings, user_query=user_query)
                self.logger.info(f"Processed {len(text_embeddings)} text embeddings")
            finally:
                self.text_processor.teardown()
            
            # Stage 5: Scene context building
            self.logger.info("Stage 5: Building scene contexts...")
            scene_contexts = self.scene_builder.build_contexts(
                video_segments,
                visual_embeddings,
                audio_embeddings,
                text_embeddings
            )
            self.logger.info(f"Built {len(scene_contexts)} scene contexts")
            
            # Stage 6: Feature fusion (scoring)
            self.logger.info("Stage 6: Scoring moments with FusionModel...")
            # Convert SceneContext to Moment objects
            moments = self._scene_contexts_to_moments(scene_contexts)
            scored_moments = self.feature_fusion.compute_scores(moments)
            self.logger.info(f"Scored {len(scored_moments)} moments")
            
            # Stage 7: Clip selection
            self.logger.info("Stage 7: Selecting highlight clips...")
            selected_clips = self.clip_selector.select_clips(scored_moments)
            if not selected_clips:
                raise RuntimeError("No clips selected for assembly")
            self.logger.info(f"Selected {len(selected_clips)} clips")
            
            # Stage 8: Video assembly
            self.logger.info("Stage 8: Assembling final video...")
            final_output_path = self.video_assembler.assemble(
                selected_clips,
                output_path,
                video_path_map=video_path_map
            )
            self.logger.info(f"Pipeline completed successfully: {final_output_path}")
            
            return final_output_path
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise RuntimeError(f"Pipeline processing failed: {e}") from e
    
    def generate_explanations(
        self,
        selected_clips: List[Tuple[str, float, float]],
        scored_moments: List[Moment],
        scene_contexts: List
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for each selected clip.
        
        Args:
            selected_clips: List of tuples (video_id, start_time, end_time) for selected clips
            scored_moments: List of Moment objects with scores
            scene_contexts: List of SceneContext objects with transcripts
        
        Returns:
            List of explanation dictionaries, each containing:
            - video_id: Video identifier
            - start_time: Clip start time
            - end_time: Clip end time
            - relevance_score: Relevance score for this clip
            - strongest_modality: "visual", "audio", or "text"
            - transcript: Transcript snippet if available
        """
        explanations = []
        
        # Build lookup maps for fast access
        # Map (video_id, start_time, end_time) to Moment
        moment_map = {}
        for moment in scored_moments:
            key = (moment.video_id, moment.start_time, moment.end_time)
            moment_map[key] = moment
        
        # Map (video_id, window_start_time, window_end_time) to SceneContext
        context_map = {}
        for ctx in scene_contexts:
            key = (ctx.video_id, ctx.window_start_time, ctx.window_end_time)
            context_map[key] = ctx
        
        # Generate explanations for each selected clip
        for video_id, start_time, end_time in selected_clips:
            # Find matching moment (exact match or closest)
            moment = self._find_matching_moment(
                video_id, start_time, end_time, moment_map
            )
            
            # Find matching scene context
            scene_context = self._find_matching_context(
                video_id, start_time, end_time, context_map
            )
            
            # Get relevance score
            relevance_score = moment.score if moment and moment.score is not None else 0.0
            
            # Determine strongest contributing modality
            strongest_modality = self._determine_strongest_modality(moment, scene_context)
            
            # Get transcript snippet
            transcript = None
            if scene_context and scene_context.transcript:
                transcript = scene_context.transcript.strip()
            
            explanation = {
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "relevance_score": float(relevance_score),
                "strongest_modality": strongest_modality,
                "transcript": transcript
            }
            
            explanations.append(explanation)
        
        return explanations
    
    def _find_matching_moment(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        moment_map: Dict[Tuple[str, float, float], Moment]
    ) -> Optional[Moment]:
        """Find matching moment for a clip (exact or closest match)."""
        # Try exact match first
        key = (video_id, start_time, end_time)
        if key in moment_map:
            return moment_map[key]
        
        # Find closest match (same video_id, overlapping time)
        best_moment = None
        best_overlap = 0.0
        
        for (vid, st, et), moment in moment_map.items():
            if vid != video_id:
                continue
            
            # Calculate overlap
            overlap_start = max(start_time, st)
            overlap_end = min(end_time, et)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_moment = moment
        
        return best_moment
    
    def _find_matching_context(
        self,
        video_id: str,
        start_time: float,
        end_time: float,
        context_map: Dict[Tuple[str, float, float], Any]
    ) -> Optional[Any]:
        """Find matching scene context for a clip (exact or closest match)."""
        # Try exact match first
        key = (video_id, start_time, end_time)
        if key in context_map:
            return context_map[key]
        
        # Find closest match (same video_id, overlapping time)
        best_context = None
        best_overlap = 0.0
        
        for (vid, st, et), context in context_map.items():
            if vid != video_id:
                continue
            
            # Calculate overlap
            overlap_start = max(start_time, st)
            overlap_end = min(end_time, et)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_context = context
        
        return best_context
    
    def _determine_strongest_modality(
        self,
        moment: Optional[Moment],
        scene_context: Optional[Any]
    ) -> str:
        """
        Determine the strongest contributing modality.
        
        Args:
            moment: Moment object with embeddings
            scene_context: SceneContext object with metadata
        
        Returns:
            "visual", "audio", or "text"
        """
        # If we have scene context with text similarity score, prefer text if high
        if scene_context and scene_context.text_similarity_score is not None:
            if scene_context.text_similarity_score > 0.5:
                return "text"
        
        # Compare embedding magnitudes (L2 norm)
        if moment:
            visual_norm = 0.0
            audio_norm = 0.0
            text_norm = 0.0
            
            if moment.visual_embedding is not None:
                visual_norm = np.linalg.norm(moment.visual_embedding)
            
            if moment.audio_embedding is not None:
                audio_norm = np.linalg.norm(moment.audio_embedding)
            
            if moment.text_embedding is not None:
                text_norm = np.linalg.norm(moment.text_embedding)
            
            # Find modality with highest norm
            modalities = [
                ("visual", visual_norm),
                ("audio", audio_norm),
                ("text", text_norm)
            ]
            modalities.sort(key=lambda x: x[1], reverse=True)
            
            # Return the one with highest norm, or "visual" as default
            return modalities[0][0] if modalities[0][1] > 0 else "visual"
        
        # Default to visual if no moment available
        return "visual"
    
    def _scene_contexts_to_moments(self, scene_contexts: List) -> List[Moment]:
        """
        Convert SceneContext objects to Moment objects for feature fusion.
        
        Args:
            scene_contexts: List of SceneContext objects
        
        Returns:
            List of Moment objects
        """
        moments = []
        for ctx in scene_contexts:
            moment = Moment(
                video_id=ctx.video_id,
                video_path=ctx.video_path,
                start_time=ctx.window_start_time,
                end_time=ctx.window_end_time,
                visual_embedding=ctx.visual_embedding,
                audio_embedding=ctx.audio_embedding,
                text_embedding=ctx.text_embedding,
                label=None,
                score=None  # Will be set by feature fusion
            )
            moments.append(moment)
        
        return moments
