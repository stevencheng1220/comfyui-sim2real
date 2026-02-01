"""ComfyUI custom nodes for sim-to-real depth conditioning."""

from .nodes import LoadDepthNPY, SimDepthToControlNet

NODE_CLASS_MAPPINGS = {
    "LoadDepthNPY": LoadDepthNPY,
    "SimDepthToControlNet": SimDepthToControlNet,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadDepthNPY": "Load Depth NPY",
    "SimDepthToControlNet": "Sim Depth to ControlNet",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
