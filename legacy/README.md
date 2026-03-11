# Legacy Mathematica Implementation

This directory contains the original Mathematica implementation of KymoButler.

The active implementation is the Python version in `src/kymobutler/`. See the root [README](../README.md) for usage instructions.

## Contents

- `packages/` - Mathematica package files (.wl)
- `scripts/` - Mathematica scripts for model export and debugging (.wls)
- `KymoButler.nb` - Original Mathematica notebook
- `KymoButlerTestandDeploy.wls` - Mathematica test and deployment script
- `eLife2019_Analysis_Code/` - Analysis code from the original eLife 2019 publication

## Generating models from Mathematica

If you need to re-export ONNX models from the Mathematica source:

```bash
wolframscript -file legacy/scripts/export_to_onnx.wls
```

This requires a Mathematica license and the trained neural network files.
