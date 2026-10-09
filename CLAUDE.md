# CLAUDE.md

## Constraints

- Never declare torch, NumPy, or Pillow in `[project] dependencies`. ComfyUI's environment supplies
  them, and a declared torch can replace its CUDA build.
- Never rename a node class, its `NODE_CLASS_MAPPINGS` key, or its input and output names. Saved
  ComfyUI workflows reference nodes and sockets by those names.
- Never fall back silently on bad simulator data. An 8-bit segmentation PNG, an unmapped instance
  id, or `near >= far` raises a `ValueError` that names the problem.

## Project

Four ComfyUI custom nodes (Python 3.12, torch, NumPy, Pillow) that convert simulator depth and
instance segmentation into ControlNet conditioning. Installed editable into ComfyUI's environment;
no runtime dependencies of its own. Setup and node reference: `README.md`.

## Commands

```sh
python3.12 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # Linux only
.venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/lint-imports
.venv/bin/pytest
```

## Architecture

Flow: ComfyUI imports the root `__init__.py` by file path → it imports the installed
`comfyui_sim2real` package → `NODE_CLASS_MAPPINGS` registers the nodes → ComfyUI calls each node's
`FUNCTION` method with its `INPUT_TYPES`.

- `__init__.py` (repo root) — ComfyUI entry shim: node mappings and display names only.
- `src/comfyui_sim2real/` — node classes in `service.py`; `constants.py` is the ADE20K palette
  table and imports nothing.
- `tests/` — unit tests per node and an end-to-end depth test on synthetic frames in the simulator
  export layout (`depth/frame_NNNNNN.npy`).

## Conventions

- A new node is added in two places: its class in `service.py` and both mappings in the root
  `__init__.py`.
- Tensors follow ComfyUI's layout: IMAGE is `[B, H, W, C]` float32 in `[0, 1]` (depth stays in
  meters until `SimDepthToControlNet`); SEGMENTATION is this package's own `[1, H, W]` int32 type.
- Tests build their inputs with NumPy and Pillow in `tmp_path`; never commit simulator frames.

## Gotchas

- ComfyUI does not put the node directory on `sys.path` → without `pip install -e .` in ComfyUI's
  environment the root import fails and no node registers.
- Tests import from `src/` through `pythonpath` → they pass without an install, so a passing suite
  does not prove ComfyUI can load the package.
- `pip install -e '.[dev]'` on Linux pulls the multi-GB CUDA torch → install the CPU wheel first,
  as CI does.
- ControlNet depth uses the MiDaS convention (white = near) and expects 3 channels → keep the
  inversion and the RGB expand when changing `SimDepthToControlNet`.
- `id_to_class` keys are JSON strings and values must be integers 1–150 → class 0 is background
  and is assigned automatically to instance id 0.
