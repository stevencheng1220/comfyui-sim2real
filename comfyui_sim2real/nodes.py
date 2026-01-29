"""Custom nodes for sim-to-real depth conditioning.

Nodes:
    LoadDepthNPY: Load metric depth from NumPy files
    SimDepthToControlNet: Normalize depth for ControlNet conditioning
"""

import numpy as np
import torch


class LoadDepthNPY:
    """Load metric depth map from NumPy binary file.

    Input: Path to .npy file containing float32 depth in meters
    Output: ComfyUI IMAGE tensor [1, H, W, 1]
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to .npy depth file (float32 meters)",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "load"
    CATEGORY = "sim2real"
    DESCRIPTION = "Load metric depth map from NumPy .npy file"

    def load(self, file_path: str) -> tuple[torch.Tensor]:
        """Load depth from NPY file.

        Args:
            file_path: Path to .npy file containing float32 depth array.

        Returns:
            Tuple containing depth tensor [1, H, W, 1] in meters.
        """
        depth = np.load(file_path)

        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(-1)

        return (tensor,)


class SimDepthToControlNet:
    """Convert simulator metric depth to ControlNet-compatible format.

    Input: Metric depth map in meters (from LoadDepthNPY)
    Output: Normalized depth map for ControlNet (white=near, black=far)

    Algorithm:
        1. Clip depth values to [near, far] range
        2. Normalize to [0, 1]: (depth - near) / (far - near)
        3. Invert for MiDaS convention: 1.0 - normalized
        4. Expand single channel to RGB (ControlNet expects 3 channels)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE",),
                "near": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Near clipping distance in meters. Depths closer than this become white.",
                }),
                "far": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 1000.0,
                    "step": 0.1,
                    "tooltip": "Far clipping distance in meters. Depths beyond this become black.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_controlnet",)
    FUNCTION = "convert"
    CATEGORY = "sim2real"
    DESCRIPTION = "Converts metric depth (meters) to ControlNet depth format (white=near, black=far)"

    def convert(
        self,
        depth: torch.Tensor,
        near: float,
        far: float,
    ) -> tuple[torch.Tensor]:
        """Convert metric depth to ControlNet format.

        Args:
            depth: Input depth tensor [B, H, W, C] in meters.
                   C can be 1 (single channel) or 3 (RGB where R=G=B).
            near: Near clipping distance in meters.
            far: Far clipping distance in meters.

        Returns:
            Tuple containing normalized depth tensor [B, H, W, 3] in range [0, 1].
        """
        if near >= far:
            raise ValueError(f"near ({near}) must be less than far ({far})")

        # Handle both single-channel and RGB depth inputs
        if depth.shape[-1] == 3:
            # Take first channel if RGB (assume R=G=B for depth)
            depth = depth[..., 0:1]

        depth_clipped = torch.clamp(depth, min=near, max=far)
        depth_normalized = (depth_clipped - near) / (far - near)

        # Invert for MiDaS convention (white = near, black = far)
        depth_inverted = 1.0 - depth_normalized

        # Expand to RGB (ControlNet expects 3 channels)
        depth_rgb = depth_inverted.expand(-1, -1, -1, 3)

        return (depth_rgb,)
