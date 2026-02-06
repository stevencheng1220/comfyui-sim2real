"""ComfyUI custom nodes for sim-to-real conditioning.

This module is the entry point for ComfyUI's node loading.
The actual implementation is in the comfyui_sim2real subpackage.
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
