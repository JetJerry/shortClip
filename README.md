# autoClip / shortClip — Automatic Video Highlighting ⚡️

**Short description:** autoClip (the `shortclip` module) extracts and selects short, high-quality highlight clips from long videos using a multimodal pipeline (visual CLIP features, Whisper audio + transcripts, and Sentence-BERT text embeddings) and a small trainable fusion model.

---

## Features 🔧
- Segment videos into fixed-size windows and extract one-frame visual embeddings (CLIP).
- Extract audio, transcribe with Whisper, and compute Whisper encoder embeddings.
- Compute semantic text embeddings using Sentence-BERT (for transcripts and optional user queries).
- Build scene contexts combining visual/audio/text features.
- Score moments via a lightweight FusionModel and select top clips with temporal smoothing and non-overlap constraints.
- Assemble selected clips into a single highlight video (preserves audio).
- CLI scripts for inference (`shortclip/scripts/process_video.py`) and training (`shortclip/scripts/train_model.py`).

---

## Quick Start — Installation ✅

**Prerequisites:**
- Python 3.8+ (conda recommended)
- ffmpeg binary available in PATH (or use the included `ffmpeg/` folder)
- (Optional) CUDA-enabled GPU for faster inference/training

Install dependencies (example):

```bash
conda create -n autoclip python=3.11 -y
conda activate autoclip
pip install -r requirements.txt
```

Key packages: `torch`, `transformers`, `openai-whisper`, `sentence-transformers`, `clip-anytorch`, `moviepy`, `librosa`.

Model weights are downloaded automatically by the libraries on first run:
- Vision: `openai/clip-vit-base-patch32`
- Audio: `openai/whisper-base`
- Text: `sentence-transformers/all-mpnet-base-v2`

---

## Configuration 🛠️
Default configuration is in `config.yaml`. Important options:

- `models` — names & embedding dims for vision, audio, text, fusion
- `processing` — `window_size_sec`, `batch_size`, `device`, `temporal_smoothing_sigma`
- `selection` — `max_clips_per_video`, `min_clip_sec`, `max_clip_sec`

Example excerpt:

```yaml
processing:
  window_size_sec: 2
  batch_size: 16
  device: "cuda"
selection:
  max_clips_per_video: 2
  min_clip_sec: 2
  max_clip_sec: 10
```

---

## Usage — Inference CLI 🏃

Process video(s) and generate a highlights video:

```bash
python -m shortclip.scripts.process_video \
  --videos path/to/video1.mp4 path/to/video2.mp4 \
  --output path/to/highlights.mp4 \
  --config config.yaml \
  --query "goal celebration"
```
- `--query` (optional): biases selection using text similarity via the `TextProcessor`.
- The script prints the final output path on success.

---

## Usage — Training FusionModel 🧠

Prepare training data as `train.json` or `train.pkl` containing a list of `Moment` objects (see `shortclip/pipeline/moment.py`).

Train example:

```bash
python -m shortclip.scripts.train_model \
  --data_dir path/to/data_dir_or_file \
  --epochs 10 \
  --config config.yaml \
  --output_dir checkpoints \
  --batch_size 32
```
- `HighlightDataset` expects Moments with embeddings and optional `label` or `score`.
- Best checkpoint is saved as `best_model.pt` in `--output_dir`.

---

## API Reference (high-level) 📚
- `shortclip.pipeline.MultimodalPipeline(config, model_path=None)` — orchestrates the pipeline.
  - `process(video_paths, output_path, user_query=None)` → returns output video path.
  - `generate_explanations(...)` → returns per-clip explanations (dominant modality, transcript, score).
- Processors: `VideoSegmenter`, `VisualProcessor`, `AudioProcessor`, `TextProcessor`, `SceneContextBuilder`, `FeatureFusion`, `ClipSelector`, `VideoAssembler`.
- Model: `shortclip.models.FusionModel` — MLP used for scoring.
- Training: `shortclip.training.Trainer` and `shortclip.training.HighlightDataset`.

---

## Output & Explanations 📄
- Final output: assembled MP4 with selected clips.
- Use `MultimodalPipeline.generate_explanations` to obtain human-readable summaries per selected clip (relevance score, strongest modality, transcript snippet).

---

## Testing & Examples 🧪
- Tests (currently minimal): `shortclip/tests/` (placeholder `test_structure.py`).
- Recommended: add small sample video(s) and `train.json` example for reproducible tests.

---

## Development & Contributing 🤝
- `setup.py` exists but is currently empty — add packaging metadata and `console_scripts` if publishing.
- Suggested workflow: feature branch → tests → PR → CI.
- CI: add GitHub Actions to run unit tests.

---

## Known limitations & Notes ⚠️
- GPU recommended for speed. Whisper+CLIP on CPU is slow.
- Processors are defensive and return defaults when extraction fails; for best results, use reasonably clean input videos.

---

## License & Acknowledgements 📜
- See `autoClip/LICENSE` and `autoClip/LICENSE_PYTHON.txt` for license terms.
- Uses public pretrained models (OpenAI CLIP/Whisper, SentenceTransformers) — please follow their licensing and usage terms.

---

## Next steps (suggestions) ✅
- Add example dataset (`data/sample.mp4`, `data/train.json`) and a short tutorial script.
- Add end-to-end integration tests and CI.
- Fill `setup.py` with packaging metadata and add an installable entry point.

---

If you want, I can add a small example video scaffold and a sample `train.json`, or add CI test scaffolding next — tell me which you'd prefer.