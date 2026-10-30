# comfyui-sim2real

Adds ComfyUI nodes that turn simulator depth and instance segmentation into ControlNet conditioning, so a rendered scene can steer photoreal generation.

## Requirements

- A ComfyUI install running Python 3.12+, which supplies torch, NumPy, and Pillow.
- Simulator exports: metric depth as float32 `.npy` files in meters, and instance segmentation as
  16-bit grayscale PNGs.

## Setup

In the Python environment that runs ComfyUI:

<!-- host interpreter: comfyui -->
```sh
cd ComfyUI/custom_nodes
git clone https://github.com/stevencheng1220/comfyui-sim2real.git
cd comfyui-sim2real
pip install -e .
```

Restart ComfyUI. The four nodes appear under the `sim2real` category.

## Usage

In a ComfyUI graph:

```text
LoadDepthNPY (depth.npy)      → SimDepthToControlNet → Apply ControlNet (depth model)
LoadSegmentationPNG (seg.png) → InstanceSegToADE20K  → Apply ControlNet (segmentation model)
```

The nodes are plain Python classes, so the depth path also runs outside ComfyUI. With a frame at
`depth/frame_000001.npy`:

```sh
python - <<'EOF'
from comfyui_sim2real.service import LoadDepthNPY, SimDepthToControlNet
(depth,) = LoadDepthNPY().load("depth/frame_000001.npy")
(cond,) = SimDepthToControlNet().convert(depth, near=0.1, far=10.0)
print(tuple(depth.shape), "->", tuple(cond.shape), f"min={cond.min():.3f} max={cond.max():.3f}")
EOF
```

```text
(1, 480, 640, 1) -> (1, 480, 640, 3) min=0.000 max=0.960
```

## Development

```sh
git clone https://github.com/stevencheng1220/comfyui-sim2real.git
cd comfyui-sim2real
uv sync --inexact --locked --extra dev
uv run ruff check .
uv run ruff format --check .
uv run lint-imports
uv run mypy
uv run pytest
```

`uv.lock` takes torch's CPU-only build on Linux and PyPI's on macOS. CI runs these checks, a
dependency audit, and a secret scan on every push. Agent rules are in [`CLAUDE.md`](CLAUDE.md).

## Nodes

### LoadDepthNPY

Loads a metric depth map from a `.npy` file.

- Input `file_path` (STRING): path to the `.npy` file; non-float32 arrays are cast to float32.
- Output `depth` (IMAGE): tensor `[1, H, W, 1]` in meters.

### SimDepthToControlNet

Converts metric depth to ControlNet depth: clips to `[near, far]`, normalizes, and inverts so near
is white and far is black.

- Input `depth` (IMAGE): from `LoadDepthNPY`; a 3-channel input uses its first channel.
- Input `near` (FLOAT, default `0.1`) and `far` (FLOAT, default `10.0`): clipping distances in
  meters; `near` must be less than `far`.
- Output `depth_controlnet` (IMAGE): tensor `[B, H, W, 3]` in `[0, 1]`.

### LoadSegmentationPNG

Loads an instance segmentation map. Rejects anything but a 16-bit grayscale PNG (mode `I;16`),
since 8 bits cannot hold the instance ids.

- Input `file_path` (STRING): path to the PNG.
- Output `segmentation` (SEGMENTATION): tensor `[1, H, W]` of int32 instance ids.

### InstanceSegToADE20K

Colors an instance map with the ADE20K palette for segmentation ControlNets.

- Input `segmentation` (SEGMENTATION): from `LoadSegmentationPNG`.
- Input `id_to_class` (STRING): JSON object mapping instance ids to ADE20K class ids 1–150, for
  example `{"1": 1, "2": 35, "3": 13}`. Class names are in
  `src/comfyui_sim2real/constants.py`.
- Output `segmentation_ade20k` (IMAGE): RGB tensor `[1, H, W, 3]` in `[0, 1]`.

Instance id 0 is always background (black) and cannot be mapped. Any other id in the map that is
missing from `id_to_class` raises an error.
