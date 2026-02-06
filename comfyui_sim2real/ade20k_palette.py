"""ADE20K semantic segmentation palette.

Provides the standard 150-class ADE20K color palette used by MMSegmentation
and compatible with ControlNet segmentation models.

Reference: https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/ade.py
"""

# fmt: off
# ADE20K palette: 150 RGB colors for classes 1-150
# Source: MMSegmentation mmseg/datasets/ade.py METAINFO["palette"]
_ADE20K_PALETTE_150 = [
    [120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
    [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
    [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
    [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
    [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
    [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
    [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
    [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
    [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
    [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
    [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
    [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
    [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
    [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
    [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
    [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
    [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
    [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
    [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
    [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
    [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
    [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
    [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
    [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
    [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
    [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
    [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
    [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
    [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
    [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
    [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
    [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
    [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
    [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
    [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
    [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
    [102, 255, 0], [92, 0, 255],
]
# fmt: on

# Class names for ADE20K (classes 1-150, from objectInfo150.csv)
# Included for documentation and debugging purposes
ADE20K_CLASS_NAMES = [
    "background",  # Class 0 (added for indexing convenience)
    "wall", "building", "sky", "floor", "tree",
    "ceiling", "road", "bed", "windowpane", "grass",
    "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair",
    "car", "water", "painting", "sofa", "shelf",
    "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock",
    "wardrobe", "lamp", "bathtub", "railing", "cushion",
    "base", "box", "column", "signboard", "chest",
    "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway",
    "case", "pool", "pillow", "screen", "stairway",
    "river", "bridge", "bookcase", "blind", "coffee",
    "toilet", "flower", "book", "hill", "bench",
    "countertop", "stove", "palm", "kitchen", "computer",
    "swivel", "boat", "bar", "arcade", "hovel",
    "bus", "towel", "light", "truck", "tower",
    "chandelier", "awning", "streetlight", "booth", "television",
    "airplane", "dirt", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet",
    "poster", "stage", "van", "ship", "fountain",
    "conveyer", "canopy", "washer", "plaything", "swimming",
    "stool", "barrel", "basket", "waterfall", "tent",
    "bag", "minibike", "cradle", "oven", "ball",
    "food", "step", "tank", "trade", "microwave",
    "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce",
    "vase", "traffic", "tray", "ashcan", "fan",
    "pier", "crt", "plate", "monitor", "bulletin",
    "shower", "radiator", "glass", "clock", "flag",
]

# Number of semantic classes (excluding background)
NUM_CLASSES = 150

# Full palette with class 0 (background) prepended: 151 entries total
# Index directly with ADE20K class ID (0-150)
ADE20K_PALETTE = [[0, 0, 0]] + _ADE20K_PALETTE_150


def get_color(class_id: int) -> tuple[int, int, int]:
    """Get RGB color for an ADE20K class ID.

    Args:
        class_id: ADE20K class ID (0-150). 0 is background.

    Returns:
        RGB tuple (r, g, b) with values in [0, 255].

    Raises:
        ValueError: If class_id is out of range [0, 150].
    """
    if not 0 <= class_id <= NUM_CLASSES:
        raise ValueError(
            f"ADE20K class ID must be in range [0, {NUM_CLASSES}], got {class_id}"
        )
    return tuple(ADE20K_PALETTE[class_id])


def get_class_name(class_id: int) -> str:
    """Get human-readable name for an ADE20K class ID.

    Args:
        class_id: ADE20K class ID (0-150). 0 is background.

    Returns:
        Class name string.

    Raises:
        ValueError: If class_id is out of range [0, 150].
    """
    if not 0 <= class_id <= NUM_CLASSES:
        raise ValueError(
            f"ADE20K class ID must be in range [0, {NUM_CLASSES}], got {class_id}"
        )
    return ADE20K_CLASS_NAMES[class_id]


__all__ = [
    "ADE20K_PALETTE",
    "ADE20K_CLASS_NAMES",
    "NUM_CLASSES",
    "get_color",
    "get_class_name",
]
