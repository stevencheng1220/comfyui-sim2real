"""Unit tests for comfyui-sim2real nodes."""

from pathlib import Path

import numpy as np
import pytest
import torch

from comfyui_sim2real.nodes import (
    InstanceSegToADE20K,
    LoadDepthNPY,
    LoadSegmentationPNG,
    SimDepthToControlNet,
)


class TestLoadDepthNPY:
    """Tests for LoadDepthNPY node."""

    @pytest.fixture
    def node(self):
        return LoadDepthNPY()

    def test_load_depth_file(self, node, tmp_path):
        """Test loading a valid NPY depth file."""
        depth_data = np.random.rand(1024, 1024).astype(np.float32) * 10
        file_path = tmp_path / "depth.npy"
        np.save(file_path, depth_data)

        result = node.load(str(file_path))
        output = result[0]

        assert output.shape == (1, 1024, 1024, 1)
        assert output.dtype == torch.float32

        np.testing.assert_array_almost_equal(
            output[0, :, :, 0].numpy(), depth_data
        )

    def test_load_converts_dtype(self, node, tmp_path):
        """Test that non-float32 arrays are converted."""
        depth_data = np.random.rand(64, 64).astype(np.float64)
        file_path = tmp_path / "depth.npy"
        np.save(file_path, depth_data)

        result = node.load(str(file_path))
        assert result[0].dtype == torch.float32


class TestSimDepthToControlNet:
    """Tests for SimDepthToControlNet node."""

    @pytest.fixture
    def node(self):
        return SimDepthToControlNet()

    def test_basic_normalization(self, node):
        """Test that depth values are correctly normalized and inverted."""
        # Create depth tensor: values from 1m to 10m
        depth = torch.tensor([[[[1.0], [5.5], [10.0]]]])  # [1, 1, 3, 1]

        result = node.convert(depth, near=1.0, far=10.0)
        output = result[0]

        # near (1m) should become white (1.0)
        assert torch.isclose(output[0, 0, 0, 0], torch.tensor(1.0), atol=1e-6)
        # mid (5.5m) should become gray (0.5)
        assert torch.isclose(output[0, 0, 1, 0], torch.tensor(0.5), atol=1e-6)
        # far (10m) should become black (0.0)
        assert torch.isclose(output[0, 0, 2, 0], torch.tensor(0.0), atol=1e-6)

    def test_clipping_near(self, node):
        """Test that values below near are clipped."""
        depth = torch.tensor([[[[0.1]]]])  # Below near
        result = node.convert(depth, near=1.0, far=10.0)
        # Should be clipped to near, which inverts to 1.0 (white)
        assert torch.isclose(result[0][0, 0, 0, 0], torch.tensor(1.0), atol=1e-6)

    def test_clipping_far(self, node):
        """Test that values above far are clipped."""
        depth = torch.tensor([[[[100.0]]]])  # Above far
        result = node.convert(depth, near=1.0, far=10.0)
        # Should be clipped to far, which inverts to 0.0 (black)
        assert torch.isclose(result[0][0, 0, 0, 0], torch.tensor(0.0), atol=1e-6)

    def test_rgb_output(self, node):
        """Test that output has 3 channels (RGB)."""
        depth = torch.ones(1, 64, 64, 1)
        result = node.convert(depth, near=0.5, far=20.0)
        assert result[0].shape == (1, 64, 64, 3)

    def test_rgb_input_handling(self, node):
        """Test that RGB input (R=G=B) is handled correctly."""
        depth_single = torch.full((1, 64, 64, 1), 5.0)
        depth_rgb = torch.full((1, 64, 64, 3), 5.0)

        result_single = node.convert(depth_single, near=0.5, far=20.0)
        result_rgb = node.convert(depth_rgb, near=0.5, far=20.0)

        assert torch.allclose(result_single[0], result_rgb[0])

    def test_invalid_near_far(self, node):
        """Test that near >= far raises error."""
        depth = torch.ones(1, 64, 64, 1)

        with pytest.raises(ValueError, match="near.*must be less than far"):
            node.convert(depth, near=10.0, far=5.0)

        with pytest.raises(ValueError, match="near.*must be less than far"):
            node.convert(depth, near=5.0, far=5.0)

    def test_batch_processing(self, node):
        """Test that batched inputs are handled correctly."""
        depth = torch.rand(4, 128, 128, 1) * 10 + 1  # [4, 128, 128, 1], range 1-11
        result = node.convert(depth, near=1.0, far=10.0)
        assert result[0].shape == (4, 128, 128, 3)

    def test_output_range(self, node):
        """Test that output values are in [0, 1] range."""
        depth = torch.rand(1, 256, 256, 1) * 100  # Random depths 0-100m
        result = node.convert(depth, near=0.5, far=20.0)
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0


class TestEndToEnd:
    """End-to-end tests combining both nodes."""

    def test_load_and_normalize(self, tmp_path):
        """Test loading NPY and normalizing in sequence."""
        # Create test depth file with known values
        depth_data = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32)
        file_path = tmp_path / "depth.npy"
        np.save(file_path, depth_data)

        # Load
        loader = LoadDepthNPY()
        depth_tensor = loader.load(str(file_path))[0]

        # Normalize
        normalizer = SimDepthToControlNet()
        result = normalizer.convert(depth_tensor, near=5.0, far=20.0)[0]

        # Verify output shape
        assert result.shape == (1, 2, 2, 3)

        # Verify normalization (near=5 -> 1.0, far=20 -> 0.0)
        assert torch.isclose(result[0, 0, 0, 0], torch.tensor(1.0), atol=1e-6)  # 5m -> white
        assert torch.isclose(result[0, 1, 1, 0], torch.tensor(0.0), atol=1e-6)  # 20m -> black


class TestLoadSegmentationPNG:
    """Tests for LoadSegmentationPNG node."""

    @pytest.fixture
    def node(self):
        return LoadSegmentationPNG()

    @pytest.fixture
    def sample_segmentation_file(self, tmp_path):
        """Create a valid 16-bit PNG segmentation file."""
        from PIL import Image

        # Create segmentation with known instance IDs
        seg_data = np.zeros((64, 64), dtype=np.uint16)
        seg_data[10:30, 10:30] = 1  # Instance 1
        seg_data[35:55, 35:55] = 2  # Instance 2
        seg_data[5:15, 40:60] = 65535  # Max uint16 value

        file_path = tmp_path / "segmentation.png"
        Image.fromarray(seg_data, mode="I;16").save(file_path)

        return file_path, seg_data

    def test_load_valid_16bit_png(self, node, sample_segmentation_file):
        """Test loading a valid 16-bit PNG returns correct tensor."""
        file_path, expected_data = sample_segmentation_file

        result = node.load(str(file_path))
        output = result[0]

        assert output.shape == (1, 64, 64)
        assert output.dtype == torch.int32
        np.testing.assert_array_equal(output[0].numpy(), expected_data.astype(np.int32))

    def test_preserves_instance_ids(self, node, sample_segmentation_file):
        """Test that instance IDs including max value are preserved."""
        file_path, _ = sample_segmentation_file

        result = node.load(str(file_path))
        output = result[0]

        unique_ids = torch.unique(output).tolist()
        assert 0 in unique_ids  # Background
        assert 1 in unique_ids  # Instance 1
        assert 2 in unique_ids  # Instance 2
        assert 65535 in unique_ids  # Max uint16

    def test_file_not_found_raises_error(self, node, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            node.load(str(tmp_path / "nonexistent.png"))

    def test_non_png_file_raises_error(self, node, tmp_path):
        """Test that non-PNG files raise ValueError."""
        file_path = tmp_path / "test.jpg"
        file_path.touch()

        with pytest.raises(ValueError, match="Expected PNG"):
            node.load(str(file_path))

    def test_8bit_png_raises_error(self, node, tmp_path):
        """Test that 8-bit PNG raises ValueError."""
        from PIL import Image

        # Create 8-bit PNG
        img_data = np.zeros((64, 64), dtype=np.uint8)
        file_path = tmp_path / "8bit.png"
        Image.fromarray(img_data, mode="L").save(file_path)

        with pytest.raises(ValueError, match="16-bit grayscale"):
            node.load(str(file_path))

    def test_roundtrip_matches_generator_export_format(self, node, tmp_path):
        """Test round-trip with generator's export format (exporter.py:208).

        This test verifies compatibility with isaac-simulator's FrameExporter,
        which exports segmentation using:
            Image.fromarray(seg_uint16, mode="I;16").save(path)
        """
        from PIL import Image

        # Simulate generator export: int32 -> clip to uint16 -> save as I;16
        # (matches exporter.py lines 206-208)
        original_int32 = np.array([[0, 1, 100], [1000, 10000, 65535]], dtype=np.int32)
        seg_uint16 = np.clip(original_int32, 0, 65535).astype(np.uint16)

        file_path = tmp_path / "generator_export.png"
        Image.fromarray(seg_uint16, mode="I;16").save(file_path)

        # Load through node
        result = node.load(str(file_path))
        output = result[0]

        # Verify exact round-trip preservation
        assert output.dtype == torch.int32
        np.testing.assert_array_equal(
            output[0].numpy(),
            original_int32  # Should match original values exactly
        )


class TestInstanceSegToADE20K:
    """Tests for InstanceSegToADE20K node."""

    @pytest.fixture
    def node(self):
        return InstanceSegToADE20K()

    @pytest.fixture
    def sample_segmentation(self):
        """Create a sample segmentation tensor with known instance IDs."""
        # [1, H, W] int32 tensor with instances 0 (bg), 1, 2, 3
        seg = np.zeros((4, 4), dtype=np.int32)
        seg[0, :] = 0  # Background (row 0)
        seg[1, :] = 1  # Instance 1 (row 1)
        seg[2, :] = 2  # Instance 2 (row 2)
        seg[3, :] = 3  # Instance 3 (row 3)
        return torch.from_numpy(seg).unsqueeze(0)  # [1, 4, 4]

    def test_valid_mapping(self, node, sample_segmentation):
        """Test that valid mapping produces correct ADE20K colors."""
        from comfyui_sim2real.ade20k_palette import ADE20K_PALETTE

        # Map: 1->wall(1), 2->rock(35), 3->person(13)
        id_to_class = '{"1": 1, "2": 35, "3": 13}'

        result = node.convert(sample_segmentation, id_to_class)
        output = result[0]

        assert output.shape == (1, 4, 4, 3)
        assert output.dtype == torch.float32
        assert output.min() >= 0.0
        assert output.max() <= 1.0

        # Verify colors (convert back to uint8 for comparison)
        rgb_uint8 = (output[0].numpy() * 255).astype(np.uint8)

        # Row 0: background -> class 0 -> [0, 0, 0]
        np.testing.assert_array_equal(rgb_uint8[0, 0], [0, 0, 0])

        # Row 1: instance 1 -> class 1 (wall) -> palette[1]
        np.testing.assert_array_equal(rgb_uint8[1, 0], ADE20K_PALETTE[1])

        # Row 2: instance 2 -> class 35 (rock) -> palette[35]
        np.testing.assert_array_equal(rgb_uint8[2, 0], ADE20K_PALETTE[35])

        # Row 3: instance 3 -> class 13 (person) -> palette[13]
        np.testing.assert_array_equal(rgb_uint8[3, 0], ADE20K_PALETTE[13])

    def test_unmapped_instance_raises_error(self, node, sample_segmentation):
        """Test that unmapped instance IDs raise ValueError."""
        # Only map instance 1, leaving 2 and 3 unmapped
        id_to_class = '{"1": 1}'

        with pytest.raises(ValueError, match="Unmapped instance IDs.*2.*3"):
            node.convert(sample_segmentation, id_to_class)

    def test_invalid_json_raises_error(self, node, sample_segmentation):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            node.convert(sample_segmentation, "not valid json")

    def test_class_id_out_of_range_raises_error(self, node, sample_segmentation):
        """Test that ADE20K class ID > 150 raises ValueError."""
        id_to_class = '{"1": 151, "2": 35, "3": 13}'

        with pytest.raises(ValueError, match="range.*1.*150"):
            node.convert(sample_segmentation, id_to_class)
