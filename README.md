![alt text](misc/logo.png "KymoButler")

## The AI that analyses your kymograph

KymoButler uses deep learning to automatically segment and track particles in kymographs (space-time images from live microscopy). It supports both bidirectional and unidirectional transport analysis.

### Installation

```bash
pip install -e .
```

For model weight conversion from Mathematica (optional):
```bash
pip install -e ".[convert]"
```

### Quick start

#### Python API

```python
from kymobutler.models.weights import load_default_models
from kymobutler.segmentation import segment_bidirectional
from kymobutler.tracking import track_bidirectional
from kymobutler.postprocessing import postprocess

models = load_default_models()
was_negated, raw, preprocessed, prediction = segment_bidirectional("kymograph.png", models["binet"])
tracks = track_bidirectional(prediction, preprocessed, was_negated, vision_net=models["decnet"])
stats = postprocess(tracks, pixel_time=0.5, pixel_space=0.1)
```

#### CLI

```bash
kymobutler analyze --mode bidirectional kymograph.png
kymobutler analyze --mode unidirectional kymograph.png
```

### Model weights

Model weights must be placed in `~/.kymobutler/models/`. To export from the original Mathematica models:

```bash
# Step 1: Export ONNX from Mathematica (requires Wolfram Desktop)
wolframscript -file scripts/export_to_onnx.wls

# Step 2: Convert ONNX to PyTorch state dicts (optional)
pip install -e ".[convert]"
python scripts/convert_weights.py
```

### Testing

```bash
pip install -e ".[dev]"
pytest
```

### Mathematica version

The original Mathematica implementation is still available. Open `KymoButler.nb` in Mathematica, run the first cell to download the neural networks and import the package, then use `BiKymoButler[]` or `UniKymoButler[]`.

### References

If you use KymoButler, please cite:

Jakobs, Franze & Bhatt (2019). KymoButler, a deep learning software for automated kymograph analysis. *eLife* 8:e42024.

### License

GPL-3.0
