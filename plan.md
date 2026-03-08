# KymoButler Python Reimplementation – Detailed Plan

## Overview

Reimplement KymoButler (a Wolfram Language kymograph particle tracker) in Python. KymoButler uses U-Net segmentation + greedy tracking with a vision-module for decision-making to extract particle trajectories from kymographs (time-space microscopy images).

## Directory Structure

```
kymobutler/
├── __init__.py
├── preprocessing.py
├── networks.py
├── segmentation.py
├── tracking.py
├── postprocessing.py
├── visualization.py
├── pipeline.py
├── wavelet.py
├── benchmarking.py
├── cli.py
tests/
├── __init__.py
├── data/
│   ├── bitest.png      (copied from TestAndDeploy/)
│   ├── unitest.png     (copied from TestAndDeploy/)
│   └── unitest2.png    (copied from TestAndDeploy/)
├── test_preprocessing.py
├── test_segmentation.py
├── test_tracking.py
├── test_postprocessing.py
└── test_pipeline.py
pyproject.toml
```

## Coordinate Convention

**Critical**: The Wolfram code uses `{row, col}` where `row` = time (vertical, top-to-bottom after rotation), `col` = space. Wolfram `ImageDimensions` returns `{width, height}` = `{cols, rows}`. Pixel coordinates in the code are `{y, x}` where `y` indexes from top of image (time) and `x` is spatial position.

In Python (NumPy), `image[row, col]` where `row` = time (axis 0), `col` = space (axis 1). `image.shape = (height, width) = (n_time, n_space)`.

Track format: list of `[time, space]` pairs as integer coordinates (1-indexed in Wolfram → 0-indexed in Python).

---

## Step 1: `pyproject.toml` – Project Scaffolding

```toml
[project]
name = "kymobutler"
version = "2.0.0"
description = "AI-powered kymograph analysis for particle tracking"
license = {text = "GPL-3.0"}
requires-python = ">=3.9"
dependencies = [
    "torch>=2.0",
    "numpy>=1.24",
    "scikit-image>=0.21",
    "opencv-python>=4.8",
    "networkx>=3.0",
    "matplotlib>=3.7",
    "Pillow>=10.0",
    "click>=8.0",
]

[project.optional-dependencies]
wavelet = ["PyWavelets>=1.4"]
dev = ["pytest>=7.0", "pytest-cov"]

[project.scripts]
kymobutler = "kymobutler.cli:main"
```

Also create `kymobutler/__init__.py` exporting top-level API:
```python
from .pipeline import uni_kymobutler, bi_kymobutler, KymoResult
```

---

## Step 2: `preprocessing.py` – Image Preprocessing

### Functions to implement:

#### `load_image(path: str | Path) -> np.ndarray`
- Load with `PIL.Image.open(path)`
- Convert RGBA→RGB if alpha channel present (composite onto white)
- Convert to grayscale: `skimage.color.rgb2gray()` or use luminance formula
- Return as float64 array in [0, 1], shape `(H, W)`

#### `is_negated(image: np.ndarray) -> bool`
Port of `isNegated[]`:
```python
def is_negated(image):
    # Binarize at 0.5
    n1 = np.sum(image > 0.5)  # white pixel count
    n2 = np.sum((1.0 - image) > 0.5)  # count after negation
    return n1 >= n2  # True means background is white → need to negate
```

#### `normalize_lines(image: np.ndarray) -> np.ndarray`
Port of `normlines[]`:
```python
def normalize_lines(image):
    result = image.copy()
    for i in range(image.shape[0]):
        row_mean = np.mean(image[i])
        if row_mean > 0:
            result[i] = image[i] / row_mean
    # Clip and rescale to [0, 1] (ImageAdjust equivalent)
    result = np.clip(result, 0, None)
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    return result
```

#### `pad_to_multiple(image: np.ndarray, multiple: int = 16) -> np.ndarray`
- Resize image so both dimensions are multiples of 16
- Use: `round(dim / 16) * 16` for each dimension
- Resize with `skimage.transform.resize` or `cv2.resize`

#### `preprocess(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]`
Full pipeline returning `(preprocessed_image, raw_grayscale, is_neg)`:
1. `raw = image_adjust(rgb2gray(remove_alpha(image)))`
2. `neg = is_negated(raw)`
3. If negated: `processed = 1.0 - raw`
4. `processed = normalize_lines(processed)`
5. Return `(processed, raw, neg)`

---

## Step 3: `networks.py` – Neural Network Architectures (PyTorch)

### Building blocks:

#### `LeakyReLUBlock(nn.Module)`
Port of `basicBlock[]`: `Conv2d(in_ch, out_ch, kernel, padding=kernel//2) → BatchNorm2d → LeakyReLU(0.1)`

```python
class LeakyReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

#### `DSConvBlock(nn.Module)`
Port of `dsconvBlock[]`: Depthwise conv (groups=in_ch) → 1×1 conv → BatchNorm → LeakyReLU(0.1)

```python
class DSConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                    stride=stride, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.act(self.bn(self.pointwise(self.depthwise(x))))
```

### U-Net Architectures:

#### `UNet(nn.Module)` – Bidirectional segmentation
Port of `UNET` with n=64. Exact structure:
- **Encoder:**
  - `conv1`: LeakyReLUBlock(in_ch→64) → LeakyReLUBlock(64→64) → Dropout(0.1)
  - `pool1`: MaxPool2d(2,2)
  - `conv2`: LeakyReLUBlock(64→128) → LeakyReLUBlock(128→128) → Dropout(0.1)
  - `pool2`: MaxPool2d(2,2)
  - `conv3`: LeakyReLUBlock(128→256) → LeakyReLUBlock(256→256) → Dropout(0.1)
  - `pool3`: MaxPool2d(2,2)
  - `conv4`: LeakyReLUBlock(256→512) → LeakyReLUBlock(512→512) → Dropout(0.1)
  - `pool4`: MaxPool2d(2,2)
- **Bottleneck:**
  - `conv5`: LeakyReLUBlock(512→1024) → LeakyReLUBlock(1024→1024) → Dropout(0.2)
- **Decoder:**
  - `up1`: ConvTranspose2d(1024→512, kernel=2, stride=2)
  - `cat1`: Concatenate with conv4 output → 1024 channels
  - `uconv1`: LeakyReLUBlock(1024→512) → LeakyReLUBlock(512→512)
  - `up2`: ConvTranspose2d(512→256, kernel=2, stride=2)
  - `cat2`: Concatenate with conv3 output → 512 channels
  - `uconv2`: LeakyReLUBlock(512→256) → LeakyReLUBlock(256→256)
  - `up3`: ConvTranspose2d(256→128, kernel=2, stride=2)
  - `cat3`: Concatenate with conv2 output → 256 channels
  - `uconv3`: LeakyReLUBlock(256→128) → LeakyReLUBlock(128→128)
  - `up4`: ConvTranspose2d(128→64, kernel=2, stride=2)
  - `cat4`: Concatenate with conv1 output → 128 channels
  - `uconv4`: LeakyReLUBlock(128→64) → LeakyReLUBlock(64→64)
- **Head:**
  - `classifier`: Conv2d(64→2, kernel=1) → permute → Softmax(dim=-1)
  - Output shape: `(B, H, W, 2)` – probability of background vs foreground
  - Extract channel 1 (foreground probability) as the prediction

**Input**: `(B, 1, H, W)` grayscale image (or 3 channels for vision module input)
**Output**: `(B, H, W)` probability map

#### `UNetUnidirectional(nn.Module)`
Port of `UNETunidirectional[]`. Same encoder/decoder as UNet but with **two separate heads**:
- `ant_head`: Conv2d(64→2, 1) → permute → Softmax → extracts anterograde probability
- `ret_head`: Conv2d(64→2, 1) → permute → Softmax → extracts retrograde probability
- Returns dict: `{"ant": (B, H, W), "ret": (B, H, W)}`

#### `UNetDSW(nn.Module)` and `UNetDSWUnidirectional(nn.Module)`
Same as above but replace `LeakyReLUBlock` with `DSConvBlock` in encoder levels 2-5 and all decoder levels. Level 1 encoder still uses one `LeakyReLUBlock` followed by one `DSConvBlock`.

#### `VisionModule(nn.Module)`
Port of `VisionModule[]`. Takes 3 inputs, concatenates, runs through UNet:
```python
class VisionModule(nn.Module):
    def __init__(self, tile_size=48):
        super().__init__()
        self.drop_bin = nn.Dropout(0.05)
        self.drop_fullbin = nn.Dropout(0.5)
        self.scale_bin = 1 - 0.05  # manual scale factor
        self.scale_fullbin = 1 - 0.5
        # UNet with 3 input channels (img + bin + fullbin)
        self.unet = UNet(in_channels=3)

    def forward(self, img, bin_mask, full_bin):
        # img: (B, 1, 48, 48), bin_mask: (B, 1, 48, 48), full_bin: (B, 1, 48, 48)
        bin_dropped = self.drop_bin(bin_mask) * self.scale_bin
        fullbin_dropped = self.drop_fullbin(full_bin) * self.scale_fullbin
        x = torch.cat([img, bin_dropped, fullbin_dropped], dim=1)  # (B, 3, 48, 48)
        return self.unet(x)  # (B, 48, 48) probability map
```

### Weight loading:
```python
def load_weights(model, path):
    """Load .pth weights into model."""
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model
```

---

## Step 4: `segmentation.py` – Segmentation Pipeline

### `SmoothBin` – Morphological hit-miss cleanup for bidirectional

Port of `SmoothBin[]`. Uses hit-miss transform with 8 structuring element pairs.

**Fill kernels** (add pixels where these patterns match):
```python
FILL_KERNELS = [
    np.array([[0, 1, 1],
              [0,-1, 1],
              [0, 1, 1]]),     # right-side gap fill
    np.array([[1, 1, 1],
              [1,-1, 1],
              [0, 0, 0]]),     # top gap fill
    np.array([[1, 1, 0],
              [1,-1, 0],
              [1, 1, 0]]),     # left-side gap fill
    np.array([[0, 0, 0],
              [1,-1, 1],
              [1, 1, 1]]),     # bottom gap fill
]
```

**Remove kernels** (remove pixels where these match):
```python
REMOVE_KERNELS = [
    np.array([[ 0,-1,-1],
              [ 0, 1,-1],
              [ 0,-1,-1]]),    # isolated right protrusion
    np.array([[-1,-1,-1],
              [-1, 1,-1],
              [ 0, 0, 0]]),    # isolated top protrusion
    np.array([[-1,-1, 0],
              [-1, 1, 0],
              [-1,-1, 0]]),    # isolated left protrusion
    np.array([[ 0, 0, 0],
              [-1, 1,-1],
              [-1,-1,-1]]),    # isolated bottom protrusion
]
```

Implementation using `scipy.ndimage` or `skimage.morphology`:
```python
def smooth_bin(binary: np.ndarray) -> np.ndarray:
    """Apply hit-miss fill and removal. In HitMissTransform, 1=foreground, -1=background, 0=don't care."""
    result = binary.copy()
    for kernel in FILL_KERNELS:
        fg = (kernel == 1).astype(np.uint8)
        bg = (kernel == -1).astype(np.uint8)
        hits = ndimage.binary_hit_or_miss(result, fg, bg)
        result = result | hits
    for kernel in REMOVE_KERNELS:
        fg = (kernel == 1).astype(np.uint8)
        bg = (kernel == -1).astype(np.uint8)
        hits = ndimage.binary_hit_or_miss(result, fg, bg)
        result = result & ~hits
    return result
```

Apply twice: `smooth_bin(smooth_bin(binary))`

### `SmoothBinUni` – Unidirectional variant

Port of `SmoothBinUni[]`. Only fill kernels (no removal), applied 3 times with thinning between:
```python
FILL_KERNELS_UNI = [
    np.array([[0, 1, 0],
              [0,-1, 1],
              [0, 1, 0]]),
    np.array([[0, 1, 0],
              [1,-1, 1],
              [0, 0, 0]]),
    np.array([[0, 1, 0],
              [1,-1, 0],
              [0, 1, 0]]),
    np.array([[0, 0, 0],
              [1,-1, 1],
              [0, 1, 0]]),
]
```

Unidirectional morphology pipeline:
```python
def morphology_uni(binary, min_size, min_frames):
    # Thin → SmoothBinUni x3 → Thin → Prune(2) → SelectComponents
    thin = skimage.morphology.thin(binary)
    for _ in range(3):
        thin = smooth_bin_uni(thin)
    thin = skimage.morphology.thin(thin)
    pruned = prune(thin, iterations=2)  # remove 2-pixel branches
    # Select components by size >= min_size AND bounding_box height >= min_frames
    labeled = skimage.measure.label(pruned)
    props = skimage.measure.regionprops(labeled)
    mask = np.zeros_like(binary, dtype=bool)
    for p in props:
        if p.area >= min_size and (p.bbox[2] - p.bbox[0]) >= min_frames:
            mask |= (labeled == p.label)
    return mask
```

### `segment_bidirectional(image, net, threshold, device)`
Port of `BiKymoButlerSegment` + initial morphology from `BiKymoButlerTrack`:
1. Preprocess image → `preprocessed`
2. Resize to multiple of 16
3. Run through `net`: `pred = net(preprocessed_tensor)`  → probability map
4. Resize prediction back to original dimensions
5. Binarize at `threshold`
6. `smooth_bin(smooth_bin(binary))`
7. Thin → Prune(3) → SelectComponents(count >= min_size AND bbox_height >= min_frames)
8. Return skeleton image

### `segment_unidirectional(image, net, threshold, device)`
Port of `UniKymoButlerSegment` + `UniKymoButlerTrack` morphology:
1. Preprocess image
2. Resize to multiple of 16
3. Run net → `{"ant": ant_prob, "ret": ret_prob}`
4. Resize both back to original dimensions
5. For each (ant, ret):
   - Binarize at threshold
   - `morphology_uni(binary, min_size, min_frames)`
6. Return `(ant_skeleton, ret_skeleton)`

---

## Step 5: `tracking.py` – Particle Tracking

### Seed Detection

#### `chew_ends(binary) -> np.ndarray`
Port of `chewEnds[]`. Remove endpoint pixels iteratively:
```python
CHEW_KERNELS = [
    # Left endpoint:  fg=center+right, bg=all others except right col
    (np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),   # foreground
     np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]])),   # background
    # Right endpoint: fg=center+left, bg=all others except left col
    (np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
     np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]])),
]
```

#### `chew_all_ends(binary) -> np.ndarray`
Iteratively apply `chew_ends` until convergence (no change).

#### `find_seeds(skeleton) -> list[tuple[int, int]]`
Port of seed detection from `BiKymoButlerTrack`:
1. `chewed = chew_all_ends(skeleton)`
2. Hit-miss with kernel: `fg=[[0,0,0],[0,1,0],[0,0,0]]` where top row is background, bottom row is don't-care:
   ```python
   seed_kernel_fg = np.array([[0, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]])
   seed_kernel_bg = np.array([[1, 1, 1],
                               [1, 0, 1],
                               [0, 0, 0]])
   ```
   This finds pixels with no foreground neighbors above them (top-of-track seeds).
3. Get coordinates of seed pixels
4. Sort by time coordinate (ascending) to process from top to bottom
5. Return list of `(row, col)` tuples

### Core Tracking

#### `sort_coords(coords: list[tuple]) -> list[tuple]`
Port of `SortCoords[]`. Greedy nearest-neighbor chain building:
```python
def sort_coords(coords):
    """Build two chains (forward-biased and backward-biased) from coords,
    return the longer one."""
    if len(coords) <= 1:
        return coords
    start = coords[0]
    remaining = list(coords[1:])

    # Chain 1: prefer neighbor with smallest col (SortBy[foo, Last] → First)
    chain_l = [start]
    rem = list(remaining)
    while rem:
        neighbors = [p for p in rem if euclidean(p, chain_l[-1]) <= 1.5]
        if not neighbors:
            break
        pick = min(neighbors, key=lambda p: p[1])  # First@SortBy[foo,Last] → min by col
        chain_l.append(pick)
        rem.remove(pick)

    # Chain 2: prefer neighbor with largest col (Last@SortBy[foo, Last] → max)
    chain_r = [start]
    rem = list(remaining)
    while rem:
        neighbors = [p for p in rem if euclidean(p, chain_r[-1]) <= 1.5]
        if not neighbors:
            break
        pick = max(neighbors, key=lambda p: p[1])  # Last@SortBy
        chain_r.append(pick)
        rem.remove(pick)

    return chain_l if len(chain_l) >= len(chain_r) else chain_r
```

#### `find_short_path_image(binary, start, finish) -> list[tuple]`
Port of `FindShortPathImage[]`. Graph-based shortest path through connected binary image:
```python
def find_short_path_image(binary, start, finish):
    """Find shortest path from start to finish through foreground pixels
    using connected component graph."""
    # 1. Set start and finish pixels to 0 in binary
    # 2. Label connected components (renumber starting from 3)
    # 3. Set start=1, finish=2 in labeled image
    # 4. Build adjacency graph: components that are neighbors (8-connected)
    # 5. Find shortest path from component 1 to component 2
    # 6. Return centroid coordinates of each component along the path
    labeled = skimage.measure.label(binary_modified, connectivity=2)
    # ... build NetworkX graph from component neighbors
    # ... find shortest path
    # ... return pixel positions for each component in path
```

#### `get_tile(image, track, all_coords, tile_size=48) -> tuple`
Port of `GetTile[]`. Extract a tile around the last track position:
```python
def get_tile(image, track, all_coords, tile_size=48):
    """Extract tile_size×tile_size windows centered on last track position.
    Returns (image_tile, track_binary_tile, all_coords_binary_tile, window_bounds)."""
    h, w = image.shape
    center = track[-1]
    half = tile_size // 2

    # Window bounds with boundary clamping
    r_start = center[0] - half
    r_end = center[0] + half
    c_start = center[1] - half
    c_end = center[1] + half

    # Shift window if it goes out of bounds
    if r_start < 0:
        r_end -= r_start; r_start = 0
    if r_end > h:
        r_start -= (r_end - h); r_end = h
    if c_start < 0:
        c_end -= c_start; c_start = 0
    if c_end > w:
        c_start -= (c_end - w); c_end = w

    # Create tiles
    img_tile = image[r_start:r_end, c_start:c_end]

    # Binary mask of track within window
    track_bin = np.zeros((tile_size, tile_size))
    for r, c in track:
        rr, cc = r - r_start, c - c_start
        if 0 <= rr < tile_size and 0 <= cc < tile_size:
            track_bin[rr, cc] = 1

    # Binary mask of all skeleton coords within window
    full_bin = np.zeros((tile_size, tile_size))
    for r, c in all_coords:
        rr, cc = r - r_start, c - c_start
        if 0 <= rr < tile_size and 0 <= cc < tile_size:
            full_bin[rr, cc] = 1

    return img_tile, track_bin, full_bin, (r_start, r_end, c_start, c_end)
```

#### `get_cand_from_pmap(pmap, threshold) -> list[tuple]`
Port of `GetCandFromPmap[]`:
```python
def get_cand_from_pmap(pmap, threshold):
    """Extract candidate pixels from vision module probability map.
    Returns thinned coordinates of the largest connected component above threshold."""
    binary = pmap > threshold
    labeled = skimage.measure.label(binary)
    props = skimage.measure.regionprops(labeled)
    if not props:
        return [], 0.0

    # Find component with maximum area
    largest = max(props, key=lambda p: p.area)
    mask = (labeled == largest.label)

    # Compute confidence: mean probability in the mask
    confidence = np.sum(mask * pmap) / np.sum(mask)

    # Thin the mask to get skeleton coords
    thinned = skimage.morphology.thin(mask)
    coords = list(zip(*np.where(thinned)))
    return coords, confidence
```

#### `get_cand(image, track, all_coords, threshold, vision_net, device) -> list[tuple]`
Port of `GetCand[]`. The main vision-module decision function:
```python
def get_cand(image, track, all_coords, threshold, vision_net, device):
    """Use vision module to predict next track segment."""
    tile_size = 48
    pad = tile_size // 2 + 1

    # 1. Pad image with 0.1 border
    padded = np.pad(image, pad, constant_values=0.1)

    # 2. Shift all coordinates by pad amount
    track_shifted = [(r + pad, c + pad) for r, c in track]
    all_shifted = [(r + pad, c + pad) for r, c in all_coords]

    # 3. Find nearby skeleton coords (within tile_size * 1.5 of last track point)
    last = track_shifted[-1]
    nearby = sorted(
        [p for p in all_shifted if euclidean(p, last) <= tile_size * 1.5],
        key=lambda p: p[0]
    )
    if not nearby:
        return []

    # 4. Drop last point from track (it becomes the reference)
    track_for_tile = track_shifted[:-1]

    # 5. Get tile, track binary, full binary
    img_tile, track_bin, full_bin, window = get_tile(
        padded, track_for_tile, nearby, tile_size
    )

    # 6. Run vision module
    with torch.no_grad():
        img_t = torch.tensor(img_tile, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        bin_t = torch.tensor(track_bin, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        full_t = torch.tensor(full_bin, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        pmap = vision_net(img_t, bin_t, full_t).cpu().numpy()[0]

    # 7. Extract candidates from probability map
    cands, confidence = get_cand_from_pmap(pmap, threshold)

    # 8. Remove already-visited coordinates
    track_set = set(map(tuple, track_shifted))
    cands = [c for c in cands if tuple(c) not in track_set]

    # 9. Sort by distance to last track point (in tile coordinates)
    last_in_tile = (last[0] - window[0], last[1] - window[2])
    cands.sort(key=lambda c: euclidean(c, last_in_tile))

    # 10. Validate: need >2 candidates, mean time >= last time - 1
    if len(cands) <= 2:
        return []
    if np.mean([c[0] for c in cands]) - last_in_tile[0] < -1:
        return []

    # 11. Sort into connected chain
    cands = sort_coords(cands)

    # 12. Fill gaps with shortest-path if first candidate is > 1.5 pixels away
    if euclidean(last_in_tile, cands[0]) > 1.5:
        all_in_tile = np.zeros((tile_size, tile_size))
        for r, c in nearby:
            rr, cc = r - window[0], c - window[2]
            if 0 <= rr < tile_size and 0 <= cc < tile_size:
                all_in_tile[rr, cc] = 1
        gap_path = find_short_path_image(all_in_tile, last_in_tile, cands[0])
        cands = gap_path + cands

    # 13. Take at most 24 coordinates
    cands = cands[:24]

    # 14. Final sanity check: mean time must not go backwards
    if np.mean([c[0] for c in cands]) - last_in_tile[0] < -1:
        return []

    # 15. Convert back to original (unpadded) coordinates
    return [(r + window[0] - pad, c + window[2] - pad) for r, c in cands]
```

#### `go_back(track) -> list[tuple]`
Port of `GoBack[]`. Remove trailing backwards-in-time coordinates:
```python
def go_back(track):
    """Remove trailing coordinates that go backwards in time."""
    i = len(track) - 1
    while i > 0 and track[i][0] - track[i-1][0] <= 0:
        i -= 1
    return track[:i]
```

#### `get_next_coord(track, backwards_count, all_coords, image, threshold, vision_net, device) -> tuple`
Port of `GetNextCoord[]`:
```python
def get_next_coord(track, backwards_count, all_coords, image, threshold, vision_net, device):
    """Find next coordinate(s) to extend track."""
    last = track[-1]
    track_set = set(map(tuple, track))

    # Find simple 8-connected neighbors in skeleton that aren't already in track
    candidates = [p for p in all_coords
                  if euclidean(p, last) <= 1.5 and tuple(p) not in track_set]

    new_track = list(track)

    # If exactly 1 neighbor → use it; otherwise invoke vision module
    if len(candidates) != 1:
        if len(new_track) > 2:
            extension = get_cand(image, new_track, all_coords, threshold, vision_net, device)
        else:
            extension = []
    else:
        extension = candidates

    # Check temporal direction
    if extension:
        if extension[-1][0] - new_track[-1][0] > 0:
            backwards_count = 0  # moving forward → reset
        elif extension[-1][0] - new_track[-1][0] < 0:
            if backwards_count < 1:
                backwards_count += 1  # allow 1 step backwards
            else:
                new_track = go_back(track)  # revert backwards portion
                extension = []

    if extension:
        return new_track + extension, backwards_count
    else:
        return new_track + [(0, 0)], backwards_count  # sentinel for termination
```

#### `make_track(image, all_coords, threshold, seed, vision_net, device) -> list[tuple]`
Port of `MakeTrack[]`:
```python
def make_track(image, all_coords, threshold, seed, vision_net, device):
    """Build a complete track starting from seed pixel."""
    h, w = image.shape

    # Find initial neighbor (prefer one with larger time coordinate)
    neighbors = [p for p in all_coords
                 if euclidean(p, seed) <= 1.5 and p != seed]

    if not neighbors:
        return [seed]

    if len(neighbors) > 1:
        neighbors = [max(neighbors, key=lambda p: p[0])]

    track = [seed, neighbors[0]]
    backwards_count = 0

    # Iteratively extend track until sentinel (0,0) is produced
    while track[-1] != (0, 0):
        track, backwards_count = get_next_coord(
            track, backwards_count, all_coords, image, threshold, vision_net, device
        )

    # Remove sentinel and out-of-bounds coordinates
    track = [p for p in track[:-1]
             if 0 < p[0] <= h and 0 < p[1] <= w]

    return track
```

#### `track_all_bidirectional(skeleton, image, threshold, vision_net, device, min_size, min_frames) -> list`
Full bidirectional tracking pipeline:
```python
def track_all_bidirectional(skeleton, image, threshold, vision_net, device, min_size, min_frames):
    """Complete bidirectional tracking: seeds → tracks → catch straddlers → clean."""
    h, w = skeleton.shape

    # 1. Find seeds and all skeleton coordinates
    seeds = find_seeds(skeleton)
    all_coords = list(zip(*np.where(skeleton)))
    all_coords.sort(key=lambda p: p[0])

    # 2. Track from each seed
    tracks = [make_track(image, all_coords, threshold, s, vision_net, device)
              for s in seeds]

    # 3. Find remaining untracked skeleton pixels
    tracked_set = set()
    for t in tracks:
        tracked_set.update(map(tuple, t))
    remaining = skeleton.copy()
    for r, c in tracked_set:
        if 0 <= r < h and 0 <= c < w:
            remaining[r, c] = 0

    # 4. Iteratively catch straddlers (up to 500 iterations)
    for _ in range(500):
        remaining_labeled = skimage.measure.label(remaining)
        props = skimage.measure.regionprops(remaining_labeled)
        significant = [p for p in props
                       if p.area > 5 and (p.bbox[2] - p.bbox[0]) >= 3]
        if not significant or np.sum(remaining) <= 5:
            break
        new_seeds = find_seeds(remaining)
        if not new_seeds:
            break
        # Also add first coord from each unmatched component
        # (port of the mask-matching logic from CatchStraddlers[])
        new_tracks = [make_track(image, all_coords, threshold, s, vision_net, device)
                      for s in new_seeds]
        tracks.extend(new_tracks)
        # Update remaining
        tracked_set = set()
        for t in tracks:
            tracked_set.update(map(tuple, t))
        remaining = skeleton.copy()
        for r, c in tracked_set:
            if 0 <= r < h and 0 <= c < w:
                remaining[r, c] = 0

    return tracks
```

#### `track_all_unidirectional(ant_skeleton, ret_skeleton) -> tuple[list, list]`
Simpler than bidirectional – no vision module needed:
```python
def track_all_unidirectional(ant_skeleton, ret_skeleton):
    """Extract tracks from unidirectional skeletons via connected components."""
    ant_tracks = extract_component_tracks(ant_skeleton)
    ret_tracks = extract_component_tracks(ret_skeleton)
    return ant_tracks, ret_tracks

def extract_component_tracks(skeleton):
    """Label connected components, extract pixel coordinates per component,
    group by time, average per timepoint, round."""
    labeled = skimage.measure.label(skeleton, connectivity=2)
    tracks = []
    for region in skimage.measure.regionprops(labeled):
        coords = region.coords  # (N, 2) array of [row, col]
        # Group by row (time), average col (space) per timepoint
        track = {}
        for r, c in coords:
            track.setdefault(r, []).append(c)
        track = sorted([(t, round(np.mean(xs))) for t, xs in track.items()])
        tracks.append(track)
    return tracks
```

---

## Step 6: `postprocessing.py` – Track Cleaning & Metrics

### Track cleaning (port from BiKymoButlerTrack lines 458–507):

#### `clamp_to_bounds(tracks, height, width) -> list`
```python
def clamp_to_bounds(tracks, height, width):
    """Clamp coordinates to image bounds."""
    result = []
    for track in tracks:
        clamped = [(max(1, min(t, height)), max(1, min(s, width)))
                   for t, s in track]
        result.append(clamped)
    return result
```

#### `average_per_timepoint(track) -> list[tuple]`
```python
def average_per_timepoint(track):
    """Group coordinates by time, take mean position, round."""
    from collections import defaultdict
    groups = defaultdict(list)
    for t, s in track:
        groups[t].append(s)
    return sorted([(t, round(np.mean(positions))) for t, positions in groups.items()])
```

#### `remove_subsets(tracks, min_size) -> list`
Port of `checkifAnyTrkisSubset`:
```python
def remove_subsets(tracks, min_size):
    """Remove tracks that are approximate subsets of other tracks."""
    track_sets = [set(map(tuple, t)) for t in tracks]
    keep = [True] * len(tracks)
    for i in range(len(tracks)):
        for j in range(len(tracks)):
            if i == j:
                continue
            overlap = len(track_sets[i] & track_sets[j])
            if abs(overlap - len(tracks[j])) <= min_size and len(tracks[j]) < len(tracks[i]):
                keep[j] = False
    return [t for t, k in zip(tracks, keep) if k]
```

#### `resolve_overlaps(tracks, confidences) -> list`
Port of overlap resolution (lines 480-489):
```python
def resolve_overlaps(tracks, confidences):
    """When tracks overlap by >10 pixels, remove overlapping portions
    from lower-confidence tracks."""
    track_sets = [set(map(tuple, t)) for t in tracks]
    changed = True
    while changed:
        changed = False
        for i in range(len(tracks)):
            for j in range(i + 1, len(tracks)):
                overlap = track_sets[i] & track_sets[j]
                if len(overlap) > 10:
                    # Keep overlap in higher-confidence track
                    if confidences[i] >= confidences[j]:
                        tracks[j] = [p for p in tracks[j] if tuple(p) not in track_sets[i]]
                    else:
                        tracks[i] = [p for p in tracks[i] if tuple(p) not in track_sets[j]]
                    track_sets = [set(map(tuple, t)) for t in tracks]
                    changed = True
                    break
            if changed:
                break
    return tracks
```

#### `split_gaps(tracks, max_gap) -> list`
Port from line 503: `Split[#, #2[[1]] - #1[[1]] <= 2*minSz &]`
```python
def split_gaps(tracks, max_gap):
    """Split tracks where time gap exceeds max_gap."""
    result = []
    for track in tracks:
        if len(track) < 2:
            result.append(track)
            continue
        current = [track[0]]
        for i in range(1, len(track)):
            if track[i][0] - track[i-1][0] > max_gap:
                result.append(current)
                current = [track[i]]
            else:
                current.append(track[i])
        result.append(current)
    return result
```

#### `filter_by_duration(tracks, min_frames) -> list`
```python
def filter_by_duration(tracks, min_frames):
    return [t for t in tracks if len(t) >= 2 and t[-1][0] - t[0][0] >= min_frames]
```

#### `clean_tracks(tracks, confidences, min_size, min_frames, image_shape) -> list`
Full cleaning pipeline:
1. `clamp_to_bounds(tracks, *image_shape)`
2. `tracks = [average_per_timepoint(t) for t in tracks]`
3. `remove_subsets(tracks, min_size)` (first pass)
4. `resolve_overlaps(tracks, confidences)`
5. `remove_subsets(tracks, min_size)` (second pass)
6. `split_gaps(tracks, 2 * min_size)`
7. `filter_by_duration(tracks, min_frames)`

### Derived quantities (port of `KymoButlerPProc.wl`):

#### `get_derived_quantities(track, pixel_time=1.0, pixel_space=1.0) -> dict`
Port of `getDerivedQuantities[]`:
```python
def get_derived_quantities(track, pixel_time=1.0, pixel_space=1.0):
    """Compute track metrics."""
    if len(track) < 2:
        return None

    times = [p[0] for p in track]
    spaces = [p[1] for p in track]

    # Frame-to-frame velocity: mean of |Δspace/Δtime| per step
    diffs = [(times[i+1] - times[i], spaces[i+1] - spaces[i])
             for i in range(len(track) - 1)]
    velocities = [abs(ds / dt) if dt != 0 else 0 for dt, ds in diffs]
    v = np.mean(velocities) if velocities else 0

    # Direction: sign of (last_space - first_space)
    direction = np.sign(spaces[-1] - spaces[0])

    # Total distance: sum of |Δspace|
    dist = sum(abs(spaces[i+1] - spaces[i]) for i in range(len(spaces) - 1))

    # Duration in frames
    T = abs(times[-1] - times[0]) + 1

    return {
        "direction": int(direction),
        "velocity": round(v * pixel_space / pixel_time, 4),
        "distance": round(dist * pixel_space, 4),
        "duration": round(T * pixel_time, 4),
        "start_to_end_velocity": round(dist * pixel_space / T / pixel_time, 4),
    }
```

---

## Step 7: `visualization.py` – Output Rendering

#### `draw_tracks(image, tracks, seed=None) -> np.ndarray`
```python
def draw_tracks(image, tracks, seed=None):
    """Draw colored lines for each track on a black background.
    Returns RGB image same size as input."""
    h, w = image.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    rng = np.random.RandomState(seed)
    colors = [rng.rand(3) for _ in tracks]
    for track, color in zip(tracks, colors):
        for i in range(len(track) - 1):
            r0, c0 = track[i][0] - 1, track[i][1] - 1  # 1-indexed → 0-indexed
            r1, c1 = track[i+1][0] - 1, track[i+1][1] - 1
            rr, cc = skimage.draw.line(r0, c0, r1, c1)
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            canvas[rr[valid], cc[valid]] = color
    return canvas
```

#### `draw_overlay(image, tracks_image) -> np.ndarray`
```python
def draw_overlay(image, tracks_image):
    """Composite colored tracks over grayscale image."""
    gray_rgb = np.stack([image] * 3, axis=-1) if image.ndim == 2 else image
    mask = np.any(tracks_image > 0.01, axis=-1)
    result = gray_rgb.copy()
    result[mask] = tracks_image[mask]
    return result
```

#### `draw_labeled(overlay, tracks) -> np.ndarray`
```python
def draw_labeled(overlay, tracks):
    """Add numbered labels at track centroids using matplotlib."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(overlay.shape[1]/100, overlay.shape[0]/100), dpi=100)
    ax.imshow(overlay)
    for i, track in enumerate(tracks):
        centroid_r = np.mean([p[0] for p in track])
        centroid_c = np.mean([p[1] for p in track])
        ax.text(centroid_c, centroid_r, str(i + 1), color='white', fontsize=8, ha='center')
    ax.axis('off')
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    labeled = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    labeled = labeled.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return labeled
```

#### `plot_histograms(quantities, pixel_time, pixel_space) -> matplotlib.Figure`
Generate velocity, duration, and distance histograms.

---

## Step 8: `pipeline.py` – Top-Level API

```python
from dataclasses import dataclass

@dataclass
class KymoResult:
    original: np.ndarray           # Original grayscale image
    tracks_image: np.ndarray       # Colored track lines on black
    overlay: np.ndarray            # Tracks composited over original
    labeled: np.ndarray            # Overlay with numbered labels
    tracks: list[list[tuple]]      # List of tracks, each = list of (time, space)
    anterograde_tracks: list | None = None  # For unidirectional only
    retrograde_tracks: list | None = None   # For unidirectional only


def uni_kymobutler(
    image: np.ndarray | str,
    binthresh: float = 0.2,
    device: str = "cpu",
    net = None,
    min_size: int = 3,
    min_frames: int = 3,
    weights_path: str | None = None,
) -> KymoResult:
    """Full unidirectional kymograph analysis pipeline."""
    # 1. Load image if path
    # 2. Preprocess
    # 3. Segment (unidirectional) → ant_skeleton, ret_skeleton
    # 4. Extract tracks from each skeleton via connected components
    # 5. Clean: average per timepoint, filter by min_frames
    # 6. Visualize
    # 7. Return KymoResult


def bi_kymobutler(
    image: np.ndarray | str,
    binthresh: float = 0.2,
    vthresh: float = 0.5,
    device: str = "cpu",
    seg_net = None,
    vision_net = None,
    min_size: int = 10,
    min_frames: int = 10,
    seg_weights_path: str | None = None,
    vision_weights_path: str | None = None,
) -> KymoResult:
    """Full bidirectional kymograph analysis pipeline."""
    # 1. Load image if path
    # 2. Preprocess
    # 3. Segment (bidirectional) → skeleton
    # 4. Track all (with vision module, straddler catching)
    # 5. Clean tracks (overlaps, subsets, gaps, duration filter)
    # 6. Visualize
    # 7. Return KymoResult
```

---

## Step 9: `cli.py` – Command-Line Interface

```python
import click

@click.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--mode", type=click.Choice(["uni", "bi"]), required=True)
@click.option("--threshold", default=0.2, help="Binarization threshold")
@click.option("--vthreshold", default=0.5, help="Vision module threshold (bi only)")
@click.option("--device", default="cpu", type=click.Choice(["cpu", "cuda"]))
@click.option("--min-size", default=10, help="Minimum component pixel count")
@click.option("--min-frames", default=10, help="Minimum track duration in frames")
@click.option("--seg-weights", default=None, help="Path to segmentation network weights")
@click.option("--vis-weights", default=None, help="Path to vision module weights")
@click.option("--pixel-time", default=1.0, help="Pixel size in time (seconds)")
@click.option("--pixel-space", default=1.0, help="Pixel size in space (micrometers)")
@click.option("--output-dir", default="./results", help="Output directory")
def main(image_path, mode, threshold, vthreshold, device, min_size, min_frames,
         seg_weights, vis_weights, pixel_time, pixel_space, output_dir):
    """Analyze a kymograph image."""
    # 1. Load image
    # 2. Run uni_kymobutler or bi_kymobutler
    # 3. Save: overlay.png, tracks.csv (time,space per row, tracks separated by blank lines),
    #          quantities.csv (direction, velocity, duration, distance per track),
    #          histograms.png
```

---

## Step 10: `wavelet.py` – Wavelet-Based Segmentation (Optional)

Port of `AnalyseKymographBIwavelet[]`:
```python
import pywt

def wavelet_segment(image, binthresh=0.2, min_size=10, min_frames=10):
    """Segment kymograph using stationary wavelet transform."""
    # 1. Apply SWT with 2 levels
    coeffs = pywt.swt2(image, 'db1', level=2)

    # 2. Sum specific subbands: {0}, {1}, {2}, {0,0}, {0,2}, {0,1}
    # These correspond to detail coefficients at different levels/orientations
    # Level 1: cA1, (cH1, cV1, cD1) → {0}=cH1, {1}=cV1, {2}=cD1
    # Level 2: cA2, (cH2, cV2, cD2) → {0,0}=cH2, {0,1}=cV2, {0,2}=cD2
    combined = sum of selected subbands (absolute values)

    # 3. Binarize
    binary = combined > binthresh

    # 4. Morphological cleanup:
    #    Dilate(1) → Thin → Prune(5) → DeleteSmallComponents(5) → Thin
    #    → SelectComponents(count >= min_size AND bbox_height >= min_frames)

    return skeleton
```

---

## Step 11: `benchmarking.py` – Evaluation Metrics

Port of benchmarking functions:
```python
def benchmark_prediction(predicted_tracks, ground_truth_tracks):
    """Compute precision, recall, F1 for predicted vs ground truth tracks."""
    # For each GT track, find closest predicted track (by nearest-neighbor within 3.2px)
    # Recall = mean over GT tracks of (1 - |len(matched_pred) - len(gt)| / len(gt))
    # Precision = mean over pred tracks of (1 - |len(matched_gt) - len(pred)| / len(pred))
    # F1 = 2 * precision * recall / (precision + recall)
```

---

## Step 12: Tests

### `test_preprocessing.py`
- Test `load_image` with PNG, JPEG inputs
- Test `is_negated`: white-background image → True, dark-background → False
- Test `normalize_lines`: verify each row has mean ≈ 1 after normalization
- Test `pad_to_multiple`: dimensions are multiples of 16

### `test_segmentation.py`
- Test `smooth_bin` on synthetic binary with known gaps → verify gaps filled
- Test `smooth_bin` on synthetic binary with isolated protrusions → verify removed
- Test `smooth_bin_uni` produces valid thinned output
- Test full `morphology_uni` pipeline on simple synthetic image

### `test_tracking.py`
- Test `find_seeds` on a diagonal line → finds endpoints
- Test `sort_coords` on scattered points → produces connected chain
- Test `go_back` removes trailing backwards coordinates
- Test `make_track` on simple synthetic skeleton (no vision module)
- Test `extract_component_tracks` on multi-component labeled image

### `test_postprocessing.py`
- Test `average_per_timepoint` groups and rounds correctly
- Test `remove_subsets`: track A ⊂ track B → A removed
- Test `resolve_overlaps`: overlapping tracks resolved by confidence
- Test `split_gaps`: track with 20-frame gap → split into 2
- Test `filter_by_duration`: short tracks removed
- Test `get_derived_quantities`: known track → expected velocity/direction/distance

### `test_pipeline.py`
Integration tests using test images:
- `unitest.png`: expect ~7 retrograde tracks (uni mode, last output)
- `unitest2.png`: expect ~9 retrograde tracks
- `bitest.png`: expect ~13 tracks (bi mode)
- Post-processing: `pprocLocal` on bitest tracks → 15 data rows
- **Note**: These numbers match only with correct weights. Without weights, test that pipeline runs end-to-end without errors using random weights.

---

## Implementation Order

| Step | Module | Est. Complexity | Dependencies |
|------|--------|----------------|--------------|
| 1 | `pyproject.toml`, `__init__.py` | Low | None |
| 2 | `preprocessing.py` | Low | NumPy, skimage, PIL |
| 3 | `networks.py` | Medium | PyTorch |
| 4 | `segmentation.py` | Medium | preprocessing, networks, skimage |
| 5 | `tracking.py` | High | segmentation, networks, NetworkX |
| 6 | `postprocessing.py` | Medium | NumPy |
| 7 | `visualization.py` | Low | matplotlib, skimage |
| 8 | `pipeline.py` | Low | All above |
| 9 | `cli.py` | Low | pipeline, click |
| 10 | `wavelet.py` | Medium | PyWavelets, tracking |
| 11 | `benchmarking.py` | Low | NumPy |
| 12 | Tests | Medium | All above, pytest |

---

## Key Differences from Wolfram Implementation

1. **0-indexed coordinates** in Python vs 1-indexed in Wolfram. All boundary checks adjusted.
2. **Image axis order**: NumPy `(rows, cols)` vs Wolfram `ImageDimensions` returns `(width, height)`.
3. **Morphological thinning**: `skimage.morphology.thin()` replaces Wolfram `Thinning[]`.
4. **Pruning**: `skimage.morphology` doesn't have a direct `Pruning[]` equivalent. Implement as iterative endpoint removal for N iterations.
5. **SparseArray for coordinate marking**: replaced with direct NumPy array indexing.
6. **Softmax output convention**: Wolfram outputs `(H, W, 2)` after TransposeLayer; PyTorch outputs `(B, 2, H, W)` → need to permute and take channel 1.
