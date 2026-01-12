from typing import Tuple, Dict, Any, List, Union, Optional
import os

import torch
import numpy as np
from PIL import Image
from moviepy import VideoFileClip
from tqdm import tqdm

from .base import BaseProcessor


class VisualProcessor(BaseProcessor):
    """
    Processes video segments to extract visual embeddings using CLIP.
    Extracts one frame per segment and returns normalized CLIP embeddings.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.preprocess = None
        self.model_name = None
        self.embedding_dim = None
        self.batch_size = config.get("processing", {}).get("batch_size", 16)

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """
        Load CLIP model and processor.
        Model is frozen and moved to the configured device.
        """
        try:
            from transformers import CLIPProcessor, CLIPModel

            self.model_name = (
                self.config.get("models", {})
                .get("vision", {})
                .get("name", "openai/clip-vit-base-patch32")
            )
            self.embedding_dim = (
                self.config.get("models", {})
                .get("vision", {})
                .get("embedding_dim", 512)
            )

            self.logger.info(f"Loading CLIP model: {self.model_name}")

            dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = CLIPModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype
            ).to(self.device)

            self.preprocess = CLIPProcessor.from_pretrained(self.model_name)

            # Freeze weights
            for param in self.model.parameters():
                param.requires_grad = False

            torch.set_grad_enabled(False)
            self.model.eval()

            self.logger.info(f"CLIP model loaded and frozen on {self.device}")
            self.logger.info(f"Embedding dim: {self.embedding_dim}")
            self.logger.info(f"Batch size: {self.batch_size}")

        except Exception as e:
            self._handle_error(e, "setup")
            raise

    # ------------------------------------------------------------------
    # PUBLIC PROCESS API
    # ------------------------------------------------------------------
    def process(
        self,
        segments: Union[
            Tuple[str, str, float, float],
            List[Tuple[str, str, float, float]]
        ]
    ) -> Union[np.ndarray, List[Optional[np.ndarray]]]:
        """
        Process a single segment or a list of segments.
        """
        if self.model is None:
            raise RuntimeError("VisualProcessor not initialized. Call setup() first.")

        if isinstance(segments, tuple):
            return self._process_single(segments)

        return self._process_batch(segments)

    # ------------------------------------------------------------------
    # SINGLE SEGMENT
    # ------------------------------------------------------------------
    def _process_single(
        self,
        segment: Tuple[str, str, float, float]
    ) -> Optional[np.ndarray]:
        try:
            _, video_path, start_time, end_time = segment

            frame_time = (start_time + end_time) / 2.0
            frame = self._extract_frame(video_path, frame_time)

            if frame is None:
                return None

            with torch.no_grad():
                inputs = self.preprocess(
                    images=frame,
                    return_tensors="pt"
                ).to(self.device)

                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().flatten()

        except Exception as e:
            self._handle_error(e, f"processing segment {segment}")
            return None

    # ------------------------------------------------------------------
    # BATCH PROCESSING
    # ------------------------------------------------------------------
    def _process_batch(
        self,
        segments: List[Tuple[str, str, float, float]]
    ) -> List[Optional[np.ndarray]]:

        results: List[Optional[np.ndarray]] = []

        for i in tqdm(
            range(0, len(segments), self.batch_size),
            desc="Processing visual batches"
        ):
            batch = segments[i:i + self.batch_size]
            batch_results = self._process_batch_chunk(batch)
            results.extend(batch_results)

        success = sum(r is not None for r in results)
        self.logger.info(f"Visual embeddings: {success}/{len(segments)} successful")

        return results

    def _process_batch_chunk(
        self,
        segments: List[Tuple[str, str, float, float]]
    ) -> List[Optional[np.ndarray]]:

        frames: List[Optional[Image.Image]] = []
        valid_indices: List[int] = []

        for idx, (_, video_path, start_time, end_time) in enumerate(segments):
            frame_time = (start_time + end_time) / 2.0
            frame = self._extract_frame(video_path, frame_time)

            if frame is not None:
                frames.append(frame)
                valid_indices.append(idx)
            else:
                frames.append(None)

        results: List[Optional[np.ndarray]] = [None] * len(segments)

        valid_frames = [f for f in frames if f is not None]
        if not valid_frames:
            return results

        try:
            with torch.no_grad():
                inputs = self.preprocess(
                    images=valid_frames,
                    return_tensors="pt"
                ).to(self.device)

                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.cpu().numpy()

            for out_idx, emb in zip(valid_indices, features):
                results[out_idx] = emb.flatten()

        except Exception as e:
            self._handle_error(e, "processing visual batch")

        return results

    # ------------------------------------------------------------------
    # FRAME EXTRACTION
    # ------------------------------------------------------------------
    def _extract_frame(
        self,
        video_path: str,
        timestamp: float
    ) -> Optional[Image.Image]:

        if not os.path.exists(video_path):
            self.logger.warning(f"Video not found: {video_path}")
            return None

        try:
            with VideoFileClip(video_path) as clip:
                if clip.duration <= 0:
                    return None

                timestamp = max(0.0, min(timestamp, clip.duration - 0.1))
                frame = clip.get_frame(timestamp)

            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                return None

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

            return Image.fromarray(frame)

        except Exception as e:
            self.logger.warning(
                f"Frame extraction failed ({video_path} @ {timestamp:.2f}s): {e}"
            )
            return None

    # ------------------------------------------------------------------
    # TEARDOWN
    # ------------------------------------------------------------------
    def teardown(self) -> None:
        try:
            if self.model is not None:
                del self.model
                self.model = None

            if self.preprocess is not None:
                del self.preprocess
                self.preprocess = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info("VisualProcessor cleaned up")

        except Exception as e:
            self._handle_error(e, "teardown")
