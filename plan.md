# KymoButler Python Reimplementation Plan

## Overview

Reimplement the KymoButler kymograph analysis tool from Wolfram Language (Mathematica) to Python. KymoButler automatically detects and tracks particles in kymographs (time-space diagrams from microscopy) using deep learning segmentation and algorithmic tracking.

## Architecture

```
kymobutler/
├── __init__.py                  # Package exports
├── preprocessing.py             # Image preprocessing (grayscale, normalization, negation detection)
├── networks.py                  # U-Net and Vision Module architectures (PyTorch)
├── segmentation.py              # Neural network inference + morphological post-processing
├── tracking.py                  # Seed detection, greedy tracking, vision-module-guided decisions
├── postprocessing.py            # Track cleaning, overlap resolution, derived quantities
├── visualization.py             # Colored overlays, labeled images
├── pipeline.py                  # UniKymoButler / BiKymoButler top-level pipelines
├── wavelet.py                   # (Optional) Wavelet-based segmentation alternative
└── cli.py                       # Command-line interface
tests/
├── __init__.py
├── test_preprocessing.py
├── test_segmentation.py
├── test_tracking.py
├── test_postprocessing.py
└── test_pipeline.py
pyproject.toml
```

## Dependencies

- **PyTorch** – neural network definitions, inference, GPU support
- **NumPy** – array operations
- **scikit-image** – morphological operations (thinning, pruning, labeling, hit-miss)
- **OpenCV** (`opencv-python`) – image I/O, resize, basic processing
- **NetworkX** – graph shortest-path for gap filling
- **matplotlib** – visualization and histograms
- **Pillow** – image I/O fallback
- **PyWavelets** – for optional wavelet-based segmentation
- **click** or **argparse** – CLI

## Step-by-Step Implementation

### Step 1: Project scaffolding & configuration
- Create `pyproject.toml` with dependencies (PyTorch, scikit-image, OpenCV, NetworkX, matplotlib, numpy)
- Create package directory `kymobutler/` with `__init__.py`
- Create `tests/` directory
- Copy test images into `tests/data/`

### Step 2: `preprocessing.py` – Image preprocessing
Port these functions from `KymoButler.wl`:
- **`load_image(path)`** – Load image, remove alpha channel, convert to grayscale float [0,1]
- **`is_negated(image)`** – Detect if background is white (mean > 0.5 → negated)
- **`normalize_lines(image)`** – Per-row normalization: divide each row by its mean, clip to [0,1]. Ported from `normlines[]`
- **`preprocess(image)`** – Full pipeline: grayscale → negate-if-needed → normalize → pad to multiple of 16

### Step 3: `networks.py` – Neural network architectures (PyTorch)
Port from `NeuralNetworkDefs.wl`:
- **`LeakyReLUBlock`** – Conv2d → BatchNorm → LeakyReLU(0.1)
- **`DSConvBlock`** – Depthwise-separable convolution block
- **`UNet`** – Encoder-decoder with skip connections for bidirectional segmentation (1 output channel)
- **`UNetUnidirectional`** – U-Net variant with 2 output channels (anterograde + retrograde)
- **`VisionModule`** – Triple-input U-Net: takes (image_tile, binary_mask, full_binary) → probability map
- Note: Weights will need to be retrained or converted from Wolfram format. For initial implementation, define architectures matching the Wolfram definitions exactly so converted weights can be loaded.

### Step 4: `segmentation.py` – Segmentation pipeline
Port the segmentation portion of `UniKymoButlerSegment[]` and `BiKymoButlerSegment[]`:
- **`segment_bidirectional(image, net, threshold, device)`** – Run U-Net → binarize → morphological cleanup
- **`segment_unidirectional(image, net, threshold, device)`** – Run U-Net → split antero/retro → binarize each → cleanup
- **`smooth_bin(binary_image, min_size)`** – Ported from `SmoothBin[]`: dilate → erode → thin → prune → remove small components → select by span
- **`smooth_bin_uni(binary_image, min_size)`** – Variant for unidirectional: same morphology + component filtering by aspect ratio

### Step 5: `tracking.py` – Particle tracking
Port from `MakeTrack[]`, `GetCand[]`, `GetNextCoord[]` etc.:
- **`find_seeds(skeleton)`** – Use hit-miss transform to find track endpoints (seed points). Port the 8 structural elements from the Wolfram code.
- **`make_track(seed, skeleton, image, vision_net, threshold, device)`** – Greedy tracking from seed:
  1. Start at seed pixel
  2. Find unvisited neighbors in skeleton
  3. If 1 neighbor → advance
  4. If 0 or >1 neighbors → use `get_candidate()` (vision module) to decide
  5. Track must progress in time (enforce temporal monotonicity)
  6. Fill gaps with shortest-path on skeleton graph
- **`get_candidate(position, skeleton, image, vision_net, threshold)`** – Extract 48×48 tile centered on current position, build 3-channel input (image tile, local binary, full binary), run vision module, pick highest-probability unvisited pixel
- **`track_all(skeleton, image, vision_net, threshold, device)`** – Find all seeds, run `make_track` for each, collect tracks
- **`catch_straddlers(tracks, skeleton, ...)`** – Re-extract seeds from remaining skeleton pixels, re-track. Port from `CatchStraddlers[]`.

### Step 6: `postprocessing.py` – Track cleaning & metrics
Port from `KymoButler.wl` (track cleaning) and `KymoButlerPProc.wl` (metrics):
- **`resolve_overlaps(tracks, confidences)`** – When tracks share pixels, keep the higher-confidence one
- **`remove_subsets(tracks)`** – Remove tracks that are subsets of others
- **`split_gaps(track, max_gap)`** – Split a track at time gaps > max_gap frames
- **`clean_tracks(tracks, min_frames, image_shape)`** – Full cleaning: clamp to bounds, split gaps, filter by duration, average coords per timepoint, round
- **`get_derived_quantities(tracks, pixel_size, frame_interval)`** – Compute per-track: velocity, direction (anterograde/retrograde/stationary), distance, duration, number of reversals

### Step 7: `visualization.py` – Output rendering
- **`draw_tracks(image, tracks)`** – Draw colored lines on image, each track a different color
- **`draw_overlay(image, tracks)`** – Blend colored tracks over original image
- **`draw_labeled(image, tracks)`** – Add numbered labels to each track
- **`plot_histograms(quantities)`** – Velocity, duration, displacement histograms

### Step 8: `pipeline.py` – Top-level API
- **`uni_kymobutler(image, binthresh=0.2, device="cpu", net=None, min_size=3, min_frames=3)`**
  - Preprocess → segment (unidirectional) → track → clean → visualize → return results
- **`bi_kymobutler(image, binthresh=0.2, vthresh=0.5, device="cpu", seg_net=None, vis_net=None, min_size=10, min_frames=10)`**
  - Preprocess → segment (bidirectional) → track with vision module → catch straddlers → resolve overlaps → clean → visualize → return results
- Return dataclass/named tuple: `KymoResult(original, tracks_image, overlay, labeled, tracks)`

### Step 9: `cli.py` – Command-line interface
- `kymobutler analyze <image> --mode uni|bi --threshold 0.2 --device cpu|cuda --output-dir ./results`
- Save: overlay image, track coordinates as CSV, derived quantities as CSV, histograms as PNG

### Step 10: `wavelet.py` – (Optional) Wavelet-based segmentation
Port from `WaveletPackage.wl`:
- Stationary Wavelet Transform using PyWavelets
- Combine specific wavelet subbands
- Binarize and feed into same tracking pipeline
- This is an alternative to neural-network segmentation

### Step 11: Tests
- **`test_preprocessing.py`** – Test grayscale conversion, negation detection, normalization
- **`test_segmentation.py`** – Test morphological operations (smooth_bin) on synthetic binary images
- **`test_tracking.py`** – Test seed finding, single-track extraction on synthetic skeletons
- **`test_postprocessing.py`** – Test overlap resolution, subset removal, gap splitting
- **`test_pipeline.py`** – Integration tests using test images (bitest.png, unitest.png, unitest2.png)

## Key Technical Decisions

1. **PyTorch over TensorFlow** – Better ecosystem for custom architectures, easier debugging, better GPU support on research machines
2. **scikit-image for morphology** – Has `skeletonize`, `thin`, `label`, `remove_small_objects` which map well to the Wolfram morphological ops
3. **NetworkX for pathfinding** – Lightweight, has `shortest_path` on pixel adjacency graphs matching the Wolfram `FindShortestPath` usage
4. **Coordinate convention** – The Wolfram code uses (row, col) = (time, space). We'll maintain this: axis-0 = time, axis-1 = space

## Neural Network Weight Handling

The pre-trained weights are in Wolfram format (`.wlnet`). Options:
1. **Convert weights** – Write a Mathematica script to export weights as NumPy arrays, then load into PyTorch models
2. **Retrain** – If training data is available, retrain the PyTorch models from scratch
3. **Placeholder** – For initial implementation, the architecture is defined but weights are loaded from `.pth` files that users must provide

For the initial reimplementation, we'll go with option 3 (placeholder) and provide a weight conversion utility later.

## Implementation Order

| Priority | Step | Deliverable |
|----------|------|-------------|
| 1 | Step 1 | Project scaffolding |
| 2 | Step 2 | `preprocessing.py` |
| 3 | Step 3 | `networks.py` |
| 4 | Step 4 | `segmentation.py` |
| 5 | Step 5 | `tracking.py` |
| 6 | Step 6 | `postprocessing.py` |
| 7 | Step 7 | `visualization.py` |
| 8 | Step 8 | `pipeline.py` |
| 9 | Step 9 | `cli.py` |
| 10 | Step 11 | Tests |
| 11 | Step 10 | `wavelet.py` (optional) |
