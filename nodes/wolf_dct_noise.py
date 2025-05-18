import torch
import comfy.model_management
import numpy as np
from scipy.fftpack import idctn
from scipy.ndimage import gaussian_filter

try:
    from nodes import MAX_RESOLUTION
except ImportError:
    try:
        from comfy.comfy_types import MAX_RESOLUTION
    except ImportError:
        MAX_RESOLUTION = 16384

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


def _generate_dct_block_with_spatial_dc(
    pre_quant_dc, q_table, ac_coeff_laplacian_scale, rng
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
    H_latent, W_latent, q_table, dc_map_base_range, dc_map_smooth_sigma, ac_scale, rng
):
    if H_latent == 0 or W_latent == 0:  # Added before any division
        # This can happen if input_samples has zero height or width for a channel.
        # Return an empty array of the correct type, or a small zero array.
        print(
            f"Warning: H_latent ({H_latent}) or W_latent ({W_latent}) is zero. Returning empty noise for this channel."
        )
        return np.zeros((H_latent, W_latent), dtype=np.float32)

    # Original check for block size divisibility and adjustment
    # This logic might be less critical if H_latent, W_latent are derived from an existing tensor,
    # but good to keep for robustness if direct small sizes are ever passed.
    if H_latent % BLOCK_SIZE != 0 or W_latent % BLOCK_SIZE != 0:
        print(
            f"Warning: Latent dimensions H({H_latent}) or W({W_latent}) not multiples of BLOCK_SIZE ({BLOCK_SIZE}). Adjusting internal generation logic for this channel."
        )
        # If we adjust, we still need to return H_latent x W_latent
        # The generation will happen on a slightly larger, block-aligned canvas then cropped.
        H_gen_canvas = ((H_latent + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
        W_gen_canvas = ((W_latent + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    else:
        H_gen_canvas = H_latent
        W_gen_canvas = W_latent

    num_blocks_h = H_gen_canvas // BLOCK_SIZE
    num_blocks_w = W_gen_canvas // BLOCK_SIZE

    if num_blocks_h == 0 or num_blocks_w == 0:
        # This should be caught by the H_latent == 0 or W_latent == 0 check earlier if they were inputs
        # but if H_gen_canvas became 0 after rounding (e.g. H_latent < BLOCK_SIZE), handle it.
        # This case means the target H_latent or W_latent is < BLOCK_SIZE.
        # We generate a single BLOCK_SIZE x BLOCK_SIZE tile and then crop from it.
        print(
            f"Warning: num_blocks_h ({num_blocks_h}) or num_blocks_w ({num_blocks_w}) is zero after canvas adjustment. Generating minimal block and cropping. Target: {H_latent}x{W_latent}"
        )
        H_gen_canvas = BLOCK_SIZE
        W_gen_canvas = BLOCK_SIZE
        num_blocks_h = 1
        num_blocks_w = 1

    initial_dc_map = rng.uniform(
        dc_map_base_range[0], dc_map_base_range[1], size=(num_blocks_h, num_blocks_w)
    )
    if dc_map_smooth_sigma > 0:
        smoothed_dc_map = gaussian_filter(initial_dc_map, sigma=dc_map_smooth_sigma)
    else:
        smoothed_dc_map = initial_dc_map

    # This generates on H_gen_canvas x W_gen_canvas
    spatial_image_canvas = np.zeros((H_gen_canvas, W_gen_canvas), dtype=np.float32)
    for r_idx in range(num_blocks_h):
        for c_idx in range(num_blocks_w):
            pre_quant_dc_for_block = smoothed_dc_map[r_idx, c_idx]
            dct_block = _generate_dct_block_with_spatial_dc(
                pre_quant_dc_for_block, q_table, ac_scale, rng
            )
            spatial_block = _idct_2d(dct_block)
            row_start, row_end = r_idx * BLOCK_SIZE, (r_idx + 1) * BLOCK_SIZE
            col_start, col_end = c_idx * BLOCK_SIZE, (c_idx + 1) * BLOCK_SIZE
            spatial_image_canvas[row_start:row_end, col_start:col_end] = spatial_block

    # Crop to original H_latent, W_latent if canvas was larger
    return spatial_image_canvas[:H_latent, :W_latent]


class ExecutableDCTNoise:
    def __init__(
        self,
        seed,
        dc_map_base_min,
        dc_map_base_max,
        dc_map_smooth_sigma,
        ac_coeff_laplacian_scale,
        q_table_multiplier,
        normalization_mode,
        gaussian_blend_factor,
        device_string,
        node_sigmas,
    ):
        self.seed = seed
        self.dc_map_base_min = dc_map_base_min
        self.dc_map_base_max = dc_map_base_max
        self.dc_map_smooth_sigma = dc_map_smooth_sigma
        self.ac_coeff_laplacian_scale = ac_coeff_laplacian_scale
        self.q_table_multiplier = q_table_multiplier
        self.normalization_mode = normalization_mode
        self.gaussian_blend_factor = gaussian_blend_factor
        self.device = torch.device(device_string)  # Store as torch.device
        self.node_sigmas = node_sigmas  # Sigmas provided to the main node

    def generate_noise(self, latent_image_dict):
        input_samples = latent_image_dict["samples"]
        sampler_sigmas = latent_image_dict.get("sigmas", self.node_sigmas)
        batch_size, num_latent_channels, latent_h, latent_w = input_samples.shape

        # --- Raw DCT Noise Generation (on CPU) ---
        current_q_table = DEFAULT_Q_TABLE_NP * self.q_table_multiplier
        current_q_table[current_q_table < 1] = 1
        rng_numpy = np.random.default_rng(self.seed)

        batch_latents_np = []
        for i in range(batch_size):
            channel_latents_np = []
            for _ in range(num_latent_channels):
                single_channel_noise_np = _generate_single_channel_dct_noise(
                    latent_h,
                    latent_w,
                    current_q_table,
                    (self.dc_map_base_min, self.dc_map_base_max),
                    self.dc_map_smooth_sigma,
                    self.ac_coeff_laplacian_scale,
                    rng_numpy,
                )
                channel_latents_np.append(single_channel_noise_np)
            stacked_channels_np = np.stack(channel_latents_np, axis=0)
            batch_latents_np.append(stacked_channels_np)

        dct_raw_torch_cpu = torch.from_numpy(np.stack(batch_latents_np, axis=0)).to(
            dtype=input_samples.dtype  # Ensure dtype matches early, remains on CPU
        )

        # --- Raw Gaussian Noise Generation (N(0,1) on CPU) ---
        # Use a CPU-based generator for consistent seed application before potential device moves
        gaussian_rng_cpu = torch.Generator(device="cpu").manual_seed(self.seed)
        gaussian_raw_torch_cpu = torch.randn(
            dct_raw_torch_cpu.shape,  # Match shape of DCT component
            generator=gaussian_rng_cpu,
            dtype=input_samples.dtype,
            device="cpu",
        )

        # --- Blend Raw Components (on CPU) ---
        # torch.lerp(input, end, weight) -> input if weight=0, end if weight=1
        # input = dct_raw_torch_cpu, end = gaussian_raw_torch_cpu
        blended_raw_torch_cpu = torch.lerp(
            dct_raw_torch_cpu, gaussian_raw_torch_cpu, self.gaussian_blend_factor
        )

        # --- Apply Normalization to the Blended Noise (on CPU) ---
        # The normalization functions will operate on blended_raw_torch_cpu
        # and the result will be assigned to processed_noise_cpu
        processed_noise_cpu = blended_raw_torch_cpu  # Start with the blended raw noise

        if self.normalization_mode != "None":
            if (
                processed_noise_cpu.numel() > 0
            ):  # Check if the tensor to normalize has elements
                if self.normalization_mode == "Mean0Std1_channel":
                    temp_normalized_channels = []
                    for c in range(processed_noise_cpu.shape[1]):
                        channel_data = processed_noise_cpu[:, c, :, :]
                        if channel_data.numel() > 0:
                            mean = torch.mean(channel_data)
                            std = torch.std(channel_data)
                            if std > 1e-6:
                                temp_normalized_channels.append(
                                    (channel_data - mean) / std
                                )
                            else:
                                temp_normalized_channels.append(channel_data - mean)
                        else:
                            temp_normalized_channels.append(
                                channel_data
                            )  # Append empty or unchanged if no elements
                    if temp_normalized_channels:  # Ensure list is not empty
                        processed_noise_cpu = torch.stack(
                            [
                                ch.to(processed_noise_cpu.device)
                                for ch in temp_normalized_channels
                            ],
                            dim=1,
                        )
                elif self.normalization_mode == "Mean0Std1_tensor":
                    mean = torch.mean(processed_noise_cpu)
                    std = torch.std(processed_noise_cpu)
                    if std > 1e-6:
                        processed_noise_cpu = (processed_noise_cpu - mean) / std
                    else:
                        processed_noise_cpu = processed_noise_cpu - mean
                elif self.normalization_mode == "ScaleToStd1_channel":
                    temp_scaled_channels = []
                    for c in range(processed_noise_cpu.shape[1]):
                        channel_data = processed_noise_cpu[:, c, :, :]
                        if channel_data.numel() > 0:
                            std = torch.std(channel_data)
                            if std > 1e-6:
                                temp_scaled_channels.append(channel_data / std)
                            else:
                                temp_scaled_channels.append(
                                    channel_data
                                )  # If std is zero, keep as is
                        else:
                            temp_scaled_channels.append(channel_data)
                    if temp_scaled_channels:
                        processed_noise_cpu = torch.stack(
                            [
                                ch.to(processed_noise_cpu.device)
                                for ch in temp_scaled_channels
                            ],
                            dim=1,
                        )
                elif self.normalization_mode == "ScaleToStd1_tensor":
                    std = torch.std(processed_noise_cpu)
                    if std > 1e-6:
                        processed_noise_cpu = processed_noise_cpu / std
            else:
                print(
                    f"{self.__class__.__name__}: Skipping normalization for empty tensor (blended raw)."
                )

        # Move the processed (normalized blend) noise to the sampler's input tensor device
        noise_on_sampler_device = processed_noise_cpu.to(input_samples.device)

        # Return the (normalized) blended noise, now on the sampler's input device.
        # The final move to self.device (node's target device) is important.
        return noise_on_sampler_device.to(
            self.device
        )  # Move to the node's target device


class WolfDCTNoise:
    def __init__(self):
        self.node_device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "sigmas": ("SIGMAS", {"tooltip": "Sigma schedule for noise scaling."}),
                "device_selection": (["AUTO", "CPU", "GPU"], {"default": "AUTO"}),
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
                "gaussian_blend_factor": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0.0 for 100% DCT noise, 1.0 for 100% Gaussian noise.",
                    },
                ),
            },
            "optional": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "create_dct_noise_generator"
    CATEGORY = "sampling/custom_sampling/noise"

    def create_dct_noise_generator(
        self,
        seed,
        sigmas,
        device_selection,
        dc_map_base_min,
        dc_map_base_max,
        dc_map_smooth_sigma,
        ac_coeff_laplacian_scale,
        q_table_multiplier,
        normalization,
        gaussian_blend_factor,
        model=None,
    ):
        target_device_str = ""
        if device_selection == "CPU":
            target_device_str = "cpu"
        elif device_selection == "GPU":
            if comfy.model_management.is_device_mps(self.node_device):
                target_device_str = str(comfy.model_management.get_torch_device())
            elif torch.cuda.is_available():
                target_device_str = str(comfy.model_management.get_torch_device())
            else:
                print(
                    f"{self.__class__.__name__} Warning: GPU selected but no CUDA available. Falling back to CPU."
                )
                target_device_str = "cpu"
        else:  # AUTO
            determined_model_device = None
            if model is not None:
                if hasattr(model, "device") and model.device is not None:
                    determined_model_device = str(model.device)
                elif (
                    hasattr(model, "model_devices")
                    and model.model_devices
                    and model.model_devices[0] is not None
                ):
                    determined_model_device = str(model.model_devices[0])

            if determined_model_device:
                target_device_str = determined_model_device
            elif comfy.model_management.is_device_mps(self.node_device):
                target_device_str = str(self.node_device)
            elif torch.cuda.is_available():
                target_device_str = str(comfy.model_management.get_torch_device())
            else:
                target_device_str = str(self.node_device)

        print(
            f"{self.__class__.__name__}: DCT Noise generator configured for device: '{target_device_str}', seed: {seed}, blend: {gaussian_blend_factor}"
        )

        dct_noise_generator_object = ExecutableDCTNoise(
            seed=seed,
            dc_map_base_min=dc_map_base_min,
            dc_map_base_max=dc_map_base_max,
            dc_map_smooth_sigma=dc_map_smooth_sigma,
            ac_coeff_laplacian_scale=ac_coeff_laplacian_scale,
            q_table_multiplier=q_table_multiplier,
            normalization_mode=normalization,
            gaussian_blend_factor=gaussian_blend_factor,
            device_string=target_device_str,
            node_sigmas=sigmas,  # Pass the sigmas from the node input
        )
        return (dct_noise_generator_object,)


NODE_CLASS_MAPPINGS = {"WolfDCTNoise": WolfDCTNoise}
NODE_DISPLAY_NAME_MAPPINGS = {"WolfDCTNoise": "DCT Noise (Wolf)"}
