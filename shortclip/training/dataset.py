from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import pickle
import os

from shortclip.pipeline.moment import Moment


class HighlightDataset(Dataset):
    """
    Dataset for training highlight selection model.
    Loads Moment objects and provides window-level binary labels.
    Assumes weak supervision (labels may be noisy).
    """
    
    def __init__(
        self,
        data_path: str,
        label_key: str = "label",
        positive_labels: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to file containing saved Moment objects (JSON or pickle)
            label_key: Key in Moment.label to use for binary classification
            positive_labels: List of label strings considered positive (default: ["highlight", "1", "positive"])
            score_threshold: If provided, use Moment.score >= threshold as positive label
                           (overrides label_key if both provided)
        """
        self.data_path = data_path
        self.label_key = label_key
        self.positive_labels = positive_labels or ["highlight", "1", "positive", "true"]
        self.score_threshold = score_threshold
        
        # Load moments
        self.moments = self._load_moments(data_path)
        
        # Filter out moments without embeddings
        self.moments = [m for m in self.moments if self._has_embeddings(m)]
        
        print(f"Loaded {len(self.moments)} moments from {data_path}")
    
    def _load_moments(self, data_path: str) -> List[Moment]:
        """Load Moment objects from file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        ext = os.path.splitext(data_path)[1].lower()
        
        if ext == ".json":
            with open(data_path, 'r') as f:
                data = json.load(f)
            # Handle both list of dicts and single dict
            if isinstance(data, list):
                return [Moment.from_dict(d) for d in data]
            else:
                return [Moment.from_dict(data)]
        
        elif ext in [".pkl", ".pickle"]:
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list):
                return data if all(isinstance(m, Moment) for m in data) else [Moment.from_dict(d) for d in data]
            elif isinstance(data, dict):
                return [Moment.from_dict(data)]
            else:
                return [data] if isinstance(data, Moment) else []
        
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use .json or .pkl")
    
    def _has_embeddings(self, moment: Moment) -> bool:
        """Check if moment has at least one embedding."""
        return (
            moment.visual_embedding is not None or
            moment.audio_embedding is not None or
            moment.text_embedding is not None
        )
    
    def _get_binary_label(self, moment: Moment) -> int:
        """Convert moment label to binary (0 or 1)."""
        # Use score threshold if provided
        if self.score_threshold is not None and moment.score is not None:
            return 1 if moment.score >= self.score_threshold else 0
        
        # Use label string
        if moment.label is None:
            return 0
        
        label_str = str(moment.label).lower().strip()
        return 1 if label_str in [l.lower() for l in self.positive_labels] else 0
    
    def __len__(self) -> int:
        return len(self.moments)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single data sample.
        
        Returns:
            Tuple of (visual_embedding, audio_embedding, text_embedding, label)
            All as torch.Tensor, embeddings padded with zeros if missing
        """
        moment = self.moments[idx]
        
        # Get embeddings, pad with zeros if missing
        visual = torch.from_numpy(moment.visual_embedding) if moment.visual_embedding is not None else torch.zeros(512, dtype=torch.float32)
        audio = torch.from_numpy(moment.audio_embedding) if moment.audio_embedding is not None else torch.zeros(1280, dtype=torch.float32)
        text = torch.from_numpy(moment.text_embedding) if moment.text_embedding is not None else torch.zeros(768, dtype=torch.float32)
        
        # Get binary label
        label = torch.tensor(self._get_binary_label(moment), dtype=torch.float32)
        
        return visual, audio, text, label
