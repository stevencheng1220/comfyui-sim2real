"""Integration tests over depth frames in the isaac-simulator FrameExporter layout.

FrameExporter writes one float32-meters array per frame to ``depth/frame_NNNNNN.npy``.
"""


import numpy as np
import pytest
import torch

from comfyui_sim2real.nodes import LoadDepthNPY, SimDepthToControlNet


class TestExportedDepthPipeline:
    """Load exported .npy depth frames and normalize them for ControlNet."""

    @pytest.fixture
    def exported_depth_frame(self, tmp_path):
        """Write one depth frame where FrameExporter puts it: depth/frame_000001.npy."""
        # FrameExporter writes depth frames under <output_dir>/depth/
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        # Create realistic depth data (climbing scene simulation)
        # Depth ranges from 2m (close wall) to 8m (far objects)
        height, width = 1024, 1024
        depth_data = np.random.uniform(2.0, 8.0, (height, width)).astype(np.float32)

        # Add some structure to simulate a realistic scene
        # Closer region (climbing wall)
        depth_data[300:700, 400:600] = np.random.uniform(2.0, 3.5, (400, 200))

        # float32 meters, one .npy per frame
        depth_path = depth_dir / "frame_000001.npy"
        np.save(depth_path, depth_data)

        return tmp_path, depth_path, depth_data

    def test_full_pipeline_with_exported_frame(self, exported_depth_frame):
        """Test complete pipeline: load an exported .npy frame → normalize for ControlNet."""
        tmp_path, depth_path, original_depth = exported_depth_frame

        # Step 1: Load depth using LoadDepthNPY (FrameExporter .npy output)
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
        # Use the node's default near/far clipping planes
        normalizer = SimDepthToControlNet()
        controlnet_depth = normalizer.convert(
            depth_tensor,
            near=0.1,  # SimDepthToControlNet default
            far=10.0   # SimDepthToControlNet default
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
        """Test processing several frames named as FrameExporter writes them."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        # Write frame_000001.npy ... frame_000005.npy, FrameExporter's naming
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

        # Background spans 8-20m; values at or beyond far (10m) clip to 10m,
        # normalize to 1.0 and invert to 0.0 (black)
        far_pixels = controlnet[0, 0, 256:, 0]  # Background pixels
        # Many should be close to 0 (far clipping)
        assert (far_pixels < 0.1).sum() > 0, \
            "Some far pixels should be clipped to black"
