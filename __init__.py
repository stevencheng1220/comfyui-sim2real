"""ComfyUI entry shim for the sim-to-real conditioning nodes.

ComfyUI imports ``custom_nodes/<node dir>/__init__.py`` by file path and registers the
nodes listed in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS. It does not add the
node directory to sys.path, so the implementation in src/comfyui_sim2real must be
pip-installed into ComfyUI's Python environment (``pip install -e .``) for the import
below to resolve.
"""

from comfyui_sim2real import (
    InstanceSegToADE20K,
    LoadDepthNPY,
    LoadSegmentationPNG,
    SimDepthToControlNet,
)

NODE_CLASS_MAPPINGS = {
    "InstanceSegToADE20K": InstanceSegToADE20K,
    "LoadDepthNPY": LoadDepthNPY,
    "LoadSegmentationPNG": LoadSegmentationPNG,
    "SimDepthToControlNet": SimDepthToControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InstanceSegToADE20K": "Instance Seg to ADE20K",
    "LoadDepthNPY": "Load Depth NPY",
    "LoadSegmentationPNG": "Load Segmentation PNG",
    "SimDepthToControlNet": "Sim Depth to ControlNet",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
