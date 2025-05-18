import torch
import comfy.utils
import comfy.model_management
import traceback
import numpy as np
from scipy.fftpack import idctn
from scipy.ndimage import gaussian_filter

try:
    from nodes import MAX_RESOLUTION
except ImportError:
    try:
        from comfy.comfy_types import MAX_RESOLUTION
    except ImportError:
        MAX_RESOLUTION = 16384  # Fallback

BLOCK_SIZE = 8

DEFAULT_Q_TABLE_NP = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
)


# Helper functions for DCT noise generation
def _generate_dct_block_with_spatial_dc(
    pre_quant_dc, q_table, ac_coeff_laplacian_scale, rng  # numpy.random.Generator
):
    dct_coeffs_dequantized = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=np.float32)
    quantized_dc = np.round(pre_quant_dc / q_table[0, 0])
    dct_coeffs_dequantized[0, 0] = quantized_dc * q_table[0, 0]

    for r in range(BLOCK_SIZE):
        for c in range(BLOCK_SIZE):
            if r == 0 and c == 0:
                continue
            pre_quant_ac = rng.laplace(loc=0, scale=ac_coeff_laplacian_scale)
            quantized_ac = np.round(pre_quant_ac / q_table[r, c])
            dct_coeffs_dequantized[r, c] = quantized_ac * q_table[r, c]
    return dct_coeffs_dequantized


def _idct_2d(dct_block):
    return idctn(dct_block, norm="ortho", type=2)


def _generate_single_channel_dct_noise(
    H_latent,
    W_latent,
    q_table,
    dc_map_base_range,
    dc_map_smooth_sigma,
    ac_scale,
    rng,  # numpy.random.Generator
):
    if H_latent % BLOCK_SIZE != 0 or W_latent % BLOCK_SIZE != 0:
        # This should not happen if latent_h/w are calculated correctly from pixel_h/w
        # Forcing adjustment if it does, though this might lead to unexpected sizes.
        # A better approach might be to require pixel_h/w to be multiples of 64.
        print(
            f"Warning: Latent dimensions H({H_latent}) or W({W_latent}) not multiples of BLOCK_SIZE ({BLOCK_SIZE}). Adjusting..."
        )
        H_latent = (H_latent // BLOCK_SIZE) * BLOCK_SIZE
        W_latent = (W_latent // BLOCK_SIZE) * BLOCK_SIZE
        if H_latent == 0 or W_latent == 0:
            raise ValueError(
                f"Adjusted latent dimensions are zero. Original H_latent={H_latent}, W_latent={W_latent}"
            )

    num_blocks_h = H_latent // BLOCK_SIZE
    num_blocks_w = W_latent // BLOCK_SIZE

    if num_blocks_h == 0 or num_blocks_w == 0:
        # This case can happen if H_latent or W_latent is less than BLOCK_SIZE (8)
        # e.g. original image width/height < 64
        # We'll create a small constant block in this scenario.
        print(
            f"Warning: num_blocks_h ({num_blocks_h}) or num_blocks_w ({num_blocks_w}) is zero. Generating minimal block. H_latent={H_latent}, W_latent={W_latent}"
        )
        # Fallback: generate a single block and tile/crop if necessary, or just return small noise
        # For now, let's make a small noise block if the target is smaller than BLOCK_SIZE
        # and hope it gets upscaled or handled by Comfy.
        # A more robust solution would be to pad to BLOCK_SIZE, generate, then crop.
        # Or, just return zeros for very small dimensions if they are problematic.

        # If the requested latent dim is smaller than block_size, create a small noise patch
        # and then we'll crop from it.
        target_h, target_w = H_latent, W_latent
        if H_latent < BLOCK_SIZE or W_latent < BLOCK_SIZE:
            H_gen = max(H_latent, BLOCK_SIZE)
            W_gen = max(W_latent, BLOCK_SIZE)
            num_blocks_h = H_gen // BLOCK_SIZE
            num_blocks_w = W_gen // BLOCK_SIZE
        else:  # Should not be reached if already handled above
            H_gen, W_gen = H_latent, W_latent

        initial_dc_map = rng.uniform(
            dc_map_base_range[0],
            dc_map_base_range[1],
            size=(num_blocks_h, num_blocks_w),
        )
        if dc_map_smooth_sigma > 0:
            smoothed_dc_map = gaussian_filter(initial_dc_map, sigma=dc_map_smooth_sigma)
        else:
            smoothed_dc_map = initial_dc_map

        spatial_image_full = np.zeros((H_gen, W_gen), dtype=np.float32)
        for r_idx in range(num_blocks_h):
            for c_idx in range(num_blocks_w):
                pre_quant_dc_for_block = smoothed_dc_map[r_idx, c_idx]
                dct_block = _generate_dct_block_with_spatial_dc(
                    pre_quant_dc_for_block, q_table, ac_scale, rng
                )
                spatial_block = _idct_2d(dct_block)
                row_start, row_end = r_idx * BLOCK_SIZE, (r_idx + 1) * BLOCK_SIZE
                col_start, col_end = c_idx * BLOCK_SIZE, (c_idx + 1) * BLOCK_SIZE
                spatial_image_full[row_start:row_end, col_start:col_end] = spatial_block

        return spatial_image_full[:H_latent, :W_latent]  # Crop if generated larger

    # Original logic for when H_latent/W_latent are >= BLOCK_SIZE
    initial_dc_map = rng.uniform(
        dc_map_base_range[0], dc_map_base_range[1], size=(num_blocks_h, num_blocks_w)
    )
    if dc_map_smooth_sigma > 0:
        smoothed_dc_map = gaussian_filter(initial_dc_map, sigma=dc_map_smooth_sigma)
    else:
        smoothed_dc_map = initial_dc_map

    spatial_image = np.zeros((H_latent, W_latent), dtype=np.float32)
    for r_idx in range(num_blocks_h):
        for c_idx in range(num_blocks_w):
            pre_quant_dc_for_block = smoothed_dc_map[r_idx, c_idx]
            dct_block = _generate_dct_block_with_spatial_dc(
                pre_quant_dc_for_block, q_table, ac_scale, rng
            )
            spatial_block = _idct_2d(dct_block)
            row_start, row_end = r_idx * BLOCK_SIZE, (r_idx + 1) * BLOCK_SIZE
            col_start, col_end = c_idx * BLOCK_SIZE, (c_idx + 1) * BLOCK_SIZE
            spatial_image[row_start:row_end, col_start:col_end] = spatial_block

    return spatial_image


class WolfDCTNoiseScriptableLatent:
    """
    Generates an initial latent tensor using DCT-based noise synthesis.
    This method aims to produce noise with JPEG-like compression artifacts.
    The characteristics of the noise can be controlled through various parameters.
    """

    def __init__(self):
        self.node_device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "device_selection": (["AUTO", "CPU", "GPU"], {"default": "AUTO"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Input sigmas. The first sigma is used for scaling."},
                ),
                "dc_map_base_min": (
                    "FLOAT",
                    {"default": -800.0, "min": -2000.0, "max": 2000.0, "step": 10.0},
                ),
                "dc_map_base_max": (
                    "FLOAT",
                    {"default": 800.0, "min": -2000.0, "max": 2000.0, "step": 10.0},
                ),
                "dc_map_smooth_sigma": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "ac_coeff_laplacian_scale": (
                    "FLOAT",
                    {"default": 30.0, "min": 0.1, "max": 200.0, "step": 0.1},
                ),
                "q_table_multiplier": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "normalization": (
                    [
                        "None",
                        "Mean0Std1_channel",
                        "Mean0Std1_tensor",
                        "ScaleToStd1_channel",
                        "ScaleToStd1_tensor",
                    ],
                    {"default": "Mean0Std1_channel"},
                ),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_dct_noise_latent"
    CATEGORY = "latent/noise"  # Or "Wolf Custom Nodes/Latent"

    def generate_dct_noise_latent(
        self,
        width,
        height,
        batch_size,
        device_selection,
        seed,
        sigmas,
        dc_map_base_min,
        dc_map_base_max,
        dc_map_smooth_sigma,
        ac_coeff_laplacian_scale,
        q_table_multiplier,
        normalization,
        model=None,
    ):
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() == 0:
            raise ValueError(
                "Sigmas input is required and cannot be empty for WolfDCTNoiseScriptableLatent."
            )
        target_sigma_val = sigmas[0].item()

        # Determine target device
        if device_selection == "CPU":
            final_tensor_device = torch.device("cpu")
        elif device_selection == "GPU":
            final_tensor_device = comfy.model_management.get_torch_device()
        else:  # AUTO
            if model is not None and hasattr(model, "device"):
                final_tensor_device = model.device
            elif (
                model is not None
                and hasattr(model, "model_devices")
                and model.model_devices
            ):
                final_tensor_device = model.model_devices[0]
            else:
                final_tensor_device = self.node_device

        print(
            f"{self.__class__.__name__}: Final tensor on {final_tensor_device}, Target sigma for scaling: {target_sigma_val:.4f}"
        )

        # Determine latent dimensions and channels
        latent_h = height // 8
        latent_w = width // 8

        if latent_h == 0 or latent_w == 0:
            raise ValueError(
                f"Calculated latent dimensions {latent_h}x{latent_w} are invalid. Original width={width}, height={height}. Ensure width and height are at least 8."
            )

        num_latent_channels = 4
        if model is not None:
            if (
                hasattr(model, "model_type")
                and model.model_type is not None
                and str(model.model_type).lower() == "flux"
            ):
                num_latent_channels = 16
            elif "Flux" in model.__class__.__name__ or (
                hasattr(model, "diffusion_model")
                and "Flux" in model.diffusion_model.__class__.__name__
            ):
                num_latent_channels = 16

        print(
            f"{self.__class__.__name__}: Generating {width}x{height} (batch: {batch_size}) -> latent {latent_w}x{latent_h}x{num_latent_channels}"
        )

        # Prepare Q-Table
        current_q_table = DEFAULT_Q_TABLE_NP * q_table_multiplier
        current_q_table[current_q_table < 1] = 1  # Ensure values are at least 1

        # Initialize RNG for numpy
        rng = np.random.default_rng(seed)

        batch_latents_np = []
        for i in range(batch_size):
            channel_latents_np = []
            # Add a unique seed component for each batch item if desired, or use the same seed for all.
            # For now, using the same rng object which progresses.
            # If truly independent noise per batch item with same seed is needed, re-seed rng here.
            # rng_item = np.random.default_rng(seed + i) # for per-item deterministic from global seed

            for _ in range(num_latent_channels):
                single_channel_noise = _generate_single_channel_dct_noise(
                    latent_h,
                    latent_w,
                    current_q_table,
                    (dc_map_base_min, dc_map_base_max),
                    dc_map_smooth_sigma,
                    ac_coeff_laplacian_scale,
                    rng,  # Use the progressing rng object
                )
                channel_latents_np.append(single_channel_noise)

            stacked_channels = np.stack(channel_latents_np, axis=0)  # Shape: (C, H, W)
            batch_latents_np.append(stacked_channels)

        final_np_array = np.stack(batch_latents_np, axis=0)  # Shape: (B, C, H, W)
        output_tensor = torch.from_numpy(final_np_array).to(dtype=torch.float32)

        # Normalization
        if normalization != "None":
            if output_tensor.numel() == 0:  # Skip normalization for empty tensor
                print(
                    f"{self.__class__.__name__}: Skipping normalization for empty tensor."
                )
            elif normalization == "Mean0Std1_channel":
                for c in range(output_tensor.shape[1]):
                    channel_data = output_tensor[:, c, :, :]
                    if channel_data.numel() > 0:  # avoid issues with 0-element tensors
                        mean = torch.mean(channel_data)
                        std = torch.std(channel_data)
                        if std > 1e-6:  # avoid division by zero
                            output_tensor[:, c, :, :] = (channel_data - mean) / std
                        else:  # if std is zero, just subtract mean (or set to 0 if mean is also 0)
                            output_tensor[:, c, :, :] = channel_data - mean
                    else:  # Should not happen with current checks but good for safety
                        print(
                            f"{self.__class__.__name__}: Warning: Channel {c} has 0 elements during normalization."
                        )

            elif normalization == "Mean0Std1_tensor":
                if output_tensor.numel() > 0:
                    mean = torch.mean(output_tensor)
                    std = torch.std(output_tensor)
                    if std > 1e-6:
                        output_tensor = (output_tensor - mean) / std
                    else:
                        output_tensor = output_tensor - mean
                else:
                    print(
                        f"{self.__class__.__name__}: Warning: Tensor has 0 elements during global normalization."
                    )

            elif normalization == "ScaleToStd1_channel":
                for c in range(output_tensor.shape[1]):
                    channel_data = output_tensor[:, c, :, :]
                    if channel_data.numel() > 0:
                        std = torch.std(channel_data)
                        if std > 1e-6:
                            output_tensor[:, c, :, :] = channel_data / std
                        # If std is zero, tensor is constant, division makes it nan/inf or keeps it 0 if it was 0.
                        # No change if std is zero might be safer.
                    else:
                        print(
                            f"{self.__class__.__name__}: Warning: Channel {c} has 0 elements during normalization."
                        )

            elif normalization == "ScaleToStd1_tensor":
                if output_tensor.numel() > 0:
                    std = torch.std(output_tensor)
                    if std > 1e-6:
                        output_tensor = output_tensor / std
                else:
                    print(
                        f"{self.__class__.__name__}: Warning: Tensor has 0 elements during global normalization."
                    )

        # Scale by sigmas[0] to make it a directly usable initial latent
        output_tensor *= target_sigma_val

        output_tensor = output_tensor.to(final_tensor_device)

        print(
            f"{self.__class__.__name__}: Final DCT noise latent on device: {output_tensor.device}, shape: {output_tensor.shape}, dtype: {output_tensor.dtype}, mean: {torch.mean(output_tensor).item():.4f}, std: {torch.std(output_tensor).item():.4f}"
        )

        # Sanity check shape (mostly for FLUX vs SDXL)
        expected_shape_tuple = (batch_size, num_latent_channels, latent_h, latent_w)
        if output_tensor.shape != expected_shape_tuple:
            print(
                f"Warning: Output tensor shape {output_tensor.shape} differs from expected {expected_shape_tuple}. This might be an issue."
            )

        return ({"samples": output_tensor},)


NODE_CLASS_MAPPINGS = {"WolfDCTNoiseScriptableLatent": WolfDCTNoiseScriptableLatent}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfDCTNoiseScriptableLatent": "DCT Noise Latent (Scriptable) (Wolf)"
}
