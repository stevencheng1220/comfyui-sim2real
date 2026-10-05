"""Custom nodes for sim-to-real conditioning.

Nodes:
    LoadDepthNPY: Load metric depth from NumPy files
    LoadSegmentationPNG: Load 16-bit PNG instance segmentation maps
    InstanceSegToADE20K: Convert instance segmentation to ADE20K format
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


class LoadSegmentationPNG:
    """Load 16-bit PNG instance segmentation map.

    Input: Path to 16-bit grayscale PNG file containing instance IDs
    Output: SEGMENTATION tensor [1, H, W] with int32 instance IDs

    Instance ID Conventions:
        - ID 0: Background/unlabeled pixels
        - ID 1+: Unique object instances

    Note: This node validates that the input is a 16-bit PNG. Standard
    8-bit images will be rejected to prevent silent data loss.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Path to 16-bit PNG segmentation file (instance IDs)",
                    },
                ),
            }
        }

    RETURN_TYPES = ("SEGMENTATION",)
    RETURN_NAMES = ("segmentation",)
    FUNCTION = "load"
    CATEGORY = "sim2real"
    DESCRIPTION = "Load 16-bit PNG instance segmentation preserving instance IDs"

    def load(self, file_path: str) -> tuple[torch.Tensor]:
        """Load segmentation from 16-bit PNG file.

        Args:
            file_path: Path to 16-bit grayscale PNG file.

        Returns:
            Tuple containing segmentation tensor [1, H, W] as int32.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If file is not a valid 16-bit PNG.
        """
        from pathlib import Path

        from PIL import Image

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {file_path}")

        if path.suffix.lower() != ".png":
            raise ValueError(f"Expected PNG file, got: {path.suffix}")

        img = Image.open(path)

        if img.mode != "I;16":
            raise ValueError(
                f"Expected 16-bit grayscale PNG (mode I;16), got mode: {img.mode}. "
                f"8-bit images cannot preserve instance IDs."
            )

        # Convert to numpy array (PIL returns uint16 for I;16 mode)
        seg_array = np.array(img)

        # Convert to int32 to preserve full range and match downstream expectations
        seg_int32 = seg_array.astype(np.int32)

        # Create torch tensor with batch dimension [1, H, W]
        tensor = torch.from_numpy(seg_int32).unsqueeze(0)

        return (tensor,)


class InstanceSegToADE20K:
    """Convert instance segmentation to ADE20K semantic segmentation format.

    Input: Instance segmentation tensor from LoadSegmentationPNG
    Output: RGB image with ADE20K palette colors for ControlNet conditioning

    The id_to_class parameter maps instance IDs to ADE20K class IDs:
        - Keys: Instance IDs as strings (e.g., "1", "2", "3")
        - Values: ADE20K class IDs (1-150, see ade20k_palette.py for class names)

    Special handling:
        - Instance ID 0 always maps to ADE20K class 0 (background, black)
        - Unmapped instance IDs raise ValueError (strict validation)

    Example id_to_class JSON for climbing scene:
        {"1": 1, "2": 35, "3": 35, "4": 13}
        Maps: wall=1, holds=35 (rock/stone), person=13
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segmentation": ("SEGMENTATION",),
                "id_to_class": (
                    "STRING",
                    {
                        "default": "{}",
                        "multiline": True,
                        "tooltip": (
                            "JSON mapping of instance IDs to ADE20K class IDs. "
                            'Example: {"1": 1, "2": 35, "3": 13}'
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("segmentation_ade20k",)
    FUNCTION = "convert"
    CATEGORY = "sim2real"
    DESCRIPTION = "Convert instance segmentation to ADE20K color format for ControlNet"

    def convert(
        self,
        segmentation: torch.Tensor,
        id_to_class: str,
    ) -> tuple[torch.Tensor]:
        """Convert instance segmentation to ADE20K RGB format.

        Args:
            segmentation: Instance segmentation tensor [1, H, W] as int32.
            id_to_class: JSON string mapping instance IDs to ADE20K class IDs.

        Returns:
            Tuple containing RGB tensor [1, H, W, 3] as float32 in [0, 1].

        Raises:
            ValueError: If tensor invalid, JSON invalid, class IDs out of range, or unmapped IDs.
        """
        import json

        from .ade20k_palette import ADE20K_PALETTE, NUM_CLASSES

        # Validate input tensor shape and dtype
        if segmentation.dim() != 3:
            raise ValueError(
                f"Expected segmentation tensor with 3 dimensions [1, H, W], "
                f"got {segmentation.dim()}D tensor with shape {list(segmentation.shape)}"
            )
        if segmentation.shape[0] != 1:
            raise ValueError(
                f"Expected batch size 1, got {segmentation.shape[0]}"
            )
        if segmentation.dtype != torch.int32:
            raise ValueError(
                f"Expected int32 segmentation tensor, got {segmentation.dtype}"
            )

        # Parse JSON mapping
        try:
            mapping_raw = json.loads(id_to_class)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in id_to_class: {e}") from e

        if not isinstance(mapping_raw, dict):
            raise ValueError(
                f"id_to_class must be a JSON object, got {type(mapping_raw).__name__}"
            )

        # Convert string keys to int and validate class IDs
        mapping: dict[int, int] = {}
        for instance_id_str, class_id in mapping_raw.items():
            try:
                instance_id = int(instance_id_str)
            except ValueError:
                raise ValueError(
                    f"Instance ID must be an integer, got: {instance_id_str!r}"
                )

            if not isinstance(class_id, int):
                raise ValueError(
                    f"ADE20K class ID must be an integer, got {type(class_id).__name__} "
                    f"for instance ID {instance_id}"
                )

            if not 1 <= class_id <= NUM_CLASSES:
                raise ValueError(
                    f"ADE20K class ID must be in range [1, {NUM_CLASSES}], "
                    f"got {class_id} for instance ID {instance_id}. "
                    f"(Class 0 is reserved for background and assigned automatically)"
                )

            if instance_id == 0:
                raise ValueError(
                    "Instance ID 0 cannot be mapped - it is always background. "
                    "Remove '0' from id_to_class."
                )

            mapping[instance_id] = class_id

        # Instance ID 0 always maps to class 0 (background)
        mapping[0] = 0

        # Get unique instance IDs in the segmentation
        seg_np = segmentation.numpy()
        unique_ids = set(np.unique(seg_np).tolist())

        # Check for unmapped instance IDs (strict validation)
        unmapped_ids = unique_ids - set(mapping.keys())
        if unmapped_ids:
            unmapped_list = sorted(unmapped_ids)
            raise ValueError(
                f"Unmapped instance IDs found in segmentation: {unmapped_list}. "
                f"Add these to id_to_class or verify your segmentation data."
            )

        # Build lookup table for vectorized conversion
        # Find max instance ID to size the lookup table
        max_instance_id = max(unique_ids)
        lookup_table = np.zeros((max_instance_id + 1, 3), dtype=np.uint8)

        for instance_id, class_id in mapping.items():
            if instance_id <= max_instance_id:
                lookup_table[instance_id] = ADE20K_PALETTE[class_id]

        # Apply lookup table to segmentation
        # seg_np shape: [1, H, W] -> need to index with [H, W] values
        seg_2d = seg_np[0]  # [H, W]
        rgb_uint8 = lookup_table[seg_2d]  # [H, W, 3]

        # Convert to float32 [0, 1] and add batch dimension
        rgb_float = rgb_uint8.astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb_float).unsqueeze(0)  # [1, H, W, 3]

        return (tensor,)
