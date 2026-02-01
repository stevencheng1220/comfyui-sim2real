"""Integration tests with simulated Task 1.4 output."""

from pathlib import Path

import numpy as np
import pytest
import torch

from nodes import LoadDepthNPY, SimDepthToControlNet


class TestTask14Integration:
    """Integration tests simulating Task 1.4 output format."""

    @pytest.fixture
    def task_14_output_dir(self, tmp_path):
        """Create a directory structure mimicking Task 1.4 output."""
        # Simulate Task 1.4 nested directory structure
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        # Create realistic depth data (climbing scene simulation)
        # Depth ranges from 2m (close wall) to 8m (far objects)
        height, width = 1024, 1024
        depth_data = np.random.uniform(2.0, 8.0, (height, width)).astype(np.float32)

        # Add some structure to simulate a realistic scene
        # Closer region (climbing wall)
        depth_data[300:700, 400:600] = np.random.uniform(2.0, 3.5, (400, 200))

        # Save as .npy (Task 1.4 format)
        depth_path = depth_dir / "frame_000001.npy"
        np.save(depth_path, depth_data)

        return tmp_path, depth_path, depth_data

    def test_full_pipeline_with_task_14_output(self, task_14_output_dir):
        """Test complete pipeline: Load Task 1.4 .npy → Normalize for ControlNet."""
        tmp_path, depth_path, original_depth = task_14_output_dir

        # Step 1: Load depth using LoadDepthNPY (as exported by Task 1.4)
        loader = LoadDepthNPY()
        depth_tensor = loader.load(str(depth_path))[0]

        # Verify loaded data matches original
        assert depth_tensor.shape == (1, 1024, 1024, 1)
        assert depth_tensor.dtype == torch.float32
        np.testing.assert_array_almost_equal(
            depth_tensor[0, :, :, 0].numpy(),
            original_depth,
            decimal=5
        )

        # Step 2: Convert to ControlNet format
        # Use realistic near/far for climbing scene (matches design docs)
        normalizer = SimDepthToControlNet()
        controlnet_depth = normalizer.convert(
            depth_tensor,
            near=0.1,  # Default from TECH_DOC_1.5
            far=10.0   # Default from TECH_DOC_1.5
        )[0]

        # Verify output format
        assert controlnet_depth.shape == (1, 1024, 1024, 3)
        assert controlnet_depth.dtype == torch.float32

        # Verify value range [0, 1]
        assert controlnet_depth.min() >= 0.0
        assert controlnet_depth.max() <= 1.0

        # Verify RGB channels are identical (grayscale)
        assert torch.allclose(controlnet_depth[..., 0], controlnet_depth[..., 1])
        assert torch.allclose(controlnet_depth[..., 1], controlnet_depth[..., 2])

        # Verify inversion (closer objects should be brighter)
        # Climbing wall region (2-3.5m) should be brighter than far regions (5-8m)
        wall_region = controlnet_depth[0, 300:700, 400:600, 0]
        far_region = controlnet_depth[0, 0:200, 0:200, 0]

        assert wall_region.mean() > far_region.mean(), \
            "Closer objects (wall) should be brighter than far objects"

    def test_batch_export_simulation(self, tmp_path):
        """Test processing multiple frames as Task 1.4 would export them."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        # Simulate Task 1.4 exporting multiple frames
        num_frames = 5
        frame_ids = []

        for frame_id in range(1, num_frames + 1):
            depth_data = np.random.uniform(1.0, 10.0, (512, 512)).astype(np.float32)
            depth_path = depth_dir / f"frame_{frame_id:06d}.npy"
            np.save(depth_path, depth_data)
            frame_ids.append(str(depth_path))

        # Process all frames through pipeline
        loader = LoadDepthNPY()
        normalizer = SimDepthToControlNet()

        results = []
        for frame_path in frame_ids:
            # Load
            depth = loader.load(frame_path)[0]
            # Normalize
            controlnet = normalizer.convert(depth, near=1.0, far=10.0)[0]
            results.append(controlnet)

        # Verify all processed correctly
        assert len(results) == num_frames
        for result in results:
            assert result.shape == (1, 512, 512, 3)
            assert result.min() >= 0.0
            assert result.max() <= 1.0

    def test_realistic_depth_values(self, tmp_path):
        """Test with realistic depth values from Isaac Sim."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        # Simulate realistic Isaac Sim depth output
        # - Most values in 1-10m range (typical indoor/climbing scene)
        # - Some outliers at max depth (sky/background)
        # - Float32 precision
        depth_data = np.concatenate([
            np.random.uniform(1.5, 8.0, (512, 256)),    # Foreground
            np.random.uniform(8.0, 20.0, (512, 256)),   # Background
        ], axis=1).astype(np.float32)

        depth_path = depth_dir / "frame_000001.npy"
        np.save(depth_path, depth_data)

        # Process through pipeline
        loader = LoadDepthNPY()
        normalizer = SimDepthToControlNet()

        depth = loader.load(str(depth_path))[0]
        controlnet = normalizer.convert(depth, near=0.1, far=10.0)[0]

        # Verify normalization behavior
        # Foreground should be brighter (closer to white)
        foreground = controlnet[0, :, :256, 0]
        background = controlnet[0, :, 256:, 0]

        assert foreground.mean() > background.mean(), \
            "Foreground should be brighter than background after inversion"

        # Background beyond far clipping should all be black (0.0)
        # (original values 10-20m get clipped to 10m, normalized to 0.0, inverted to 1.0, then 1-1=0)
        far_pixels = controlnet[0, 0, 256:, 0]  # Background pixels
        # Many should be close to 0 (far clipping)
        assert (far_pixels < 0.1).sum() > 0, \
            "Some far pixels should be clipped to black"
