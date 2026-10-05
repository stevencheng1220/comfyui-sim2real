# comfyui-sim2real

ComfyUI custom nodes that turn simulator depth and instance segmentation into ControlNet
conditioning.

## Installation

Requires Python 3.12+. Clone into ComfyUI's `custom_nodes` directory, then install the package
into the Python environment that runs ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone <repository-url> comfyui-sim2real
cd comfyui-sim2real
pip install -e .
```

Restart ComfyUI. ComfyUI loads the repository's root `__init__.py`, which imports the nodes from
the installed `comfyui_sim2real` package; without `pip install -e .` that import fails and the
nodes do not register. The install pulls in no runtime dependencies: torch, numpy and Pillow come
from ComfyUI's own environment.

## Layout

```
__init__.py                 ComfyUI entry shim (NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS)
src/comfyui_sim2real/
    nodes.py                node implementations
    ade20k_palette.py       ADE20K 150-class palette and class names
tests/                      pytest suite
```

## Nodes

All nodes appear under the `sim2real` category.

### LoadDepthNPY

Loads metric depth maps from NumPy .npy files (as exported by Isaac Sim).

**Inputs:**
- `file_path` (STRING): Path to .npy depth file

**Output:**
- `depth` (IMAGE): Depth tensor [1, H, W, 1] in meters

### LoadSegmentationPNG

Loads 16-bit PNG instance segmentation maps. Rejects non-PNG files and PNGs that are not 16-bit
grayscale (mode `I;16`).

**Inputs:**
- `file_path` (STRING): Path to 16-bit PNG segmentation file

**Output:**
- `segmentation` (SEGMENTATION): Segmentation tensor [1, H, W] with int32 instance IDs

### InstanceSegToADE20K

Colors an instance segmentation map with the ADE20K palette for segmentation ControlNets.

**Inputs:**
- `segmentation` (SEGMENTATION): Instance segmentation from LoadSegmentationPNG
- `id_to_class` (STRING): JSON object mapping instance IDs to ADE20K class IDs (1-150),
  e.g. `{"1": 1, "2": 35, "3": 13}`

Instance ID 0 is always background (black) and cannot be mapped. Any instance ID present in the
map but missing from `id_to_class` raises an error.

**Output:**
- `segmentation_ade20k` (IMAGE): RGB tensor [1, H, W, 3] in [0, 1]

### SimDepthToControlNet

Converts metric depth maps (float32 meters) to ControlNet-compatible format.

**Inputs:**
- `depth` (IMAGE): Metric depth map from LoadDepthNPY
- `near` (FLOAT): Near clipping distance in meters (default: 0.1)
- `far` (FLOAT): Far clipping distance in meters (default: 10.0)

**Output:**
- `depth_controlnet` (IMAGE): Normalized depth for ControlNet (white=near, black=far)

## Workflow

```
LoadDepthNPY (depth.npy) → SimDepthToControlNet → Apply ControlNet (depth model)
LoadSegmentationPNG (seg.png) → InstanceSegToADE20K → Apply ControlNet (segmentation model)
```

## Development Setup

```bash
# Clone the repository
git clone <repository-url> comfyui-sim2real
cd comfyui-sim2real

# Create a Python 3.12+ virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies (torch, numpy, Pillow, pytest, ruff)
pip install -e ".[dev]"

# Lint and test
ruff check .
pytest
```

pytest puts `src/` on the import path (`pythonpath` in `pyproject.toml`), so the tests import the
working tree directly.

## Dependencies

The package declares no runtime dependencies. ComfyUI provides torch, numpy, and Pillow at
runtime; the `dev` extra installs them for local testing.
