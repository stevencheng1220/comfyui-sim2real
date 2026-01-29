# comfyui-sim2real

ComfyUI custom nodes for sim-to-real depth conditioning with ControlNet.

## Installation

1. Clone into ComfyUI custom_nodes directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone <repository-url> comfyui-sim2real
   ```

2. Restart ComfyUI

## Nodes

### LoadDepthNPY

Loads metric depth maps from NumPy .npy files (as exported by Isaac Sim).

**Inputs:**
- `file_path` (STRING): Path to .npy depth file

**Output:**
- `depth` (IMAGE): Depth tensor [1, H, W, 1] in meters

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
```

## No External Dependencies

This package is self-contained. It does not require any external ComfyUI extensions.
