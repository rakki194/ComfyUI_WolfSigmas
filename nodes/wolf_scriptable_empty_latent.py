import torch
import math
import comfy.utils
import comfy.model_management

# MAX_RESOLUTION is usually available in the global scope when nodes.py is loaded
# or directly via from nodes import MAX_RESOLUTION if this file is treated similarly.
# For custom nodes, it's safer to define it or handle its absence.
# We'll assume it's available or use a large fallback.
try:
    from nodes import MAX_RESOLUTION
except ImportError:
    try:
        from comfy.comfy_types import MAX_RESOLUTION
    except ImportError:
        MAX_RESOLUTION = 16384  # Fallback
import torch.nn.functional as F
import traceback  # Added for script error logging

# Perlin noise implementation adapted from
# https://github.com/Extraltodeus/noise_latent_perlinpinpin
# Patched to handle non-divisible shape/resolution via interpolation.


def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d(shape, res, fade_func=_fade, device="cpu", generator=None):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid_y = torch.linspace(0, res[0], shape[0], device=device)
    grid_x = torch.linspace(0, res[1], shape[1], device=device)
    grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=-1) % 1

    # Use the provided generator for torch.rand if available
    rand_opts = {"device": device}
    if generator:
        rand_opts["generator"] = generator
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1, **rand_opts)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    target_shape_hw = (shape[0], shape[1])

    def get_resized_gradient(slice_y, slice_x):
        grad_tile = gradients[slice_y[0] : slice_y[1], slice_x[0] : slice_x[1]]
        repeat_y = max(1, int(d[0])) if d[0] > 0 else 1
        repeat_x = max(1, int(d[1])) if d[1] > 0 else 1
        repeated_grad = grad_tile.repeat_interleave(repeat_y, 0).repeat_interleave(
            repeat_x, 1
        )
        repeated_grad_permuted = repeated_grad.permute(2, 0, 1).unsqueeze(0)
        resized_grad_permuted = F.interpolate(
            repeated_grad_permuted,
            size=target_shape_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized_grad_permuted.squeeze(0).permute(1, 2, 0)

    g00 = get_resized_gradient((0, res[0]), (0, res[1]))
    g10 = get_resized_gradient((1, res[0] + 1), (0, res[1]))
    g01 = get_resized_gradient((0, res[0]), (1, res[1] + 1))
    g11 = get_resized_gradient((1, res[0] + 1), (1, res[1] + 1))

    dot = lambda grad, shift_y, shift_x: (
        torch.stack((grid[..., 0] + shift_y, grid[..., 1] + shift_x), dim=-1) * grad
    ).sum(dim=-1)

    n00 = dot(g00, 0, 0)
    n10 = dot(g10, -1, 0)
    n01 = dot(g01, 0, -1)
    n11 = dot(g11, -1, -1)

    t = fade_func(grid)

    lerp_x1 = torch.lerp(n00, n10, t[..., 0])
    lerp_x2 = torch.lerp(n01, n11, t[..., 0])
    lerped = torch.lerp(lerp_x1, lerp_x2, t[..., 1])

    return math.sqrt(2) * lerped


def rand_perlin_2d_octaves(
    shape,
    res,
    octaves=1,
    persistence=0.5,
    frequency_factor=2.0,
    device="cpu",
    fade_func=_fade,
    generator=None,  # Pass generator through
):
    noise = torch.zeros(shape, device=device)
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        current_res_h = max(1, int(frequency * res[0]))
        current_res_w = max(1, int(frequency * res[1]))
        current_res = (current_res_h, current_res_w)
        noise += amplitude * rand_perlin_2d(
            shape,
            current_res,
            fade_func=fade_func,
            device=device,
            generator=generator,  # Pass generator
        )
        frequency *= frequency_factor
        amplitude *= persistence
    return noise


DEFAULT_SCRIPT = """# Script to generate latent samples
# Available variables:
# - width, height, batch_size, seed (integers)
# - sigmas (1D torch.Tensor)
# - model (comfy model object)
# - device (string, e.g., "cuda:0" or "cpu") - target device for tensor generation
# - perlin_params (dict: blend_factor, res_factor_h, res_factor_w, frequency_factor, octaves, persistence)
# - torch (module)
# - math (module)
# - F (torch.nn.functional)
# - rand_perlin_2d_octaves_fn(shape, res, octaves, persistence, frequency_factor, device, generator) (function)
# - rand_perlin_2d_fn(shape, res, fade_func, device, generator) (function)
# - _fade_fn (function, the default fade for perlin)
#
# The script MUST assign the final latent tensor (NCHW format, float32)
# to a variable named 'output_latent_samples'.

print(f"Scriptable Latent (Perlin): Generating {width}x{height} (batch: {batch_size}) with seed {seed} on {device}")
print(f"Scriptable Latent (Perlin): Input sigmas: {sigmas}")
print(f"Scriptable Latent (Perlin): Perlin params: {perlin_params}")

# Determine latent characteristics based on model type
model_is_flux = False
detection_reason = "Initial: Unknown"
# Define the target model_type name for FLUX, case-insensitive comparison will be used on .name
COMFY_MODEL_MANAGEMENT_FLUX_NAME_UPPER = "FLUX" 

if model is not None:
    actual_model_for_check = None
    # Prefer model.model if it exists (typical for ModelPatcher)
    if hasattr(model, 'model') and model.model is not None:
        actual_model_for_check = model.model
    else: # Otherwise, use the model object directly
        actual_model_for_check = model

    if actual_model_for_check is not None:
        # Primary check: model_type.name attribute
        if hasattr(actual_model_for_check, 'model_type') and \
           actual_model_for_check.model_type is not None and \
           hasattr(actual_model_for_check.model_type, 'name') and \
           actual_model_for_check.model_type.name.upper() == COMFY_MODEL_MANAGEMENT_FLUX_NAME_UPPER:
            model_is_flux = True
            detection_reason = f"actual_model.model_type.name == '{COMFY_MODEL_MANAGEMENT_FLUX_NAME_UPPER}' (model type: {type(actual_model_for_check).__name__})"
        # Secondary check: class name contains 'Flux' (more specific than just 'flux')
        elif 'Flux' in actual_model_for_check.__class__.__name__: # Check for 'Flux' (capitalized)
            model_is_flux = True
            detection_reason = f"actual_model.__class__.__name__ ('{actual_model_for_check.__class__.__name__}') contains 'Flux'"
    
    # Fallback to top-level model object if specific checks on actual_model_for_check failed or didn't identify FLUX
    # This handles cases where the top-level 'model' is the one with the type info, or 'actual_model_for_check' was None.
    if not model_is_flux:
        if hasattr(model, 'model_type') and \
           model.model_type is not None and \
           hasattr(model.model_type, 'name') and \
           model.model_type.name.upper() == COMFY_MODEL_MANAGEMENT_FLUX_NAME_UPPER:
            model_is_flux = True
            detection_reason = f"top-level_model.model_type.name == '{COMFY_MODEL_MANAGEMENT_FLUX_NAME_UPPER}' (model type: {type(model).__name__})"
        elif 'Flux' in model.__class__.__name__: # Check for 'Flux' (capitalized) in top-level model class name
            model_is_flux = True
            detection_reason = f"top-level_model.__class__.__name__ ('{model.__class__.__name__}') contains 'Flux'"
else:
    detection_reason = "Model object provided to script is None"


if model_is_flux:
    print(f"Scriptable Latent (Perlin) Script: Detected FLUX model type. Reason: {detection_reason}.")
    print(f"Scriptable Latent (Perlin) Script: Configuring for FLUX: Channels=16, Spatial Divisor=8, VAE Scale Factor=1.0")
    num_latent_channels = 16
    latent_height = height // 8 
    latent_width = width // 8
    vae_scaling_factor = 1.0
else:
    # Default to SDXL-like parameters if not FLUX or if model is None
    reason_for_default = detection_reason
    if model is None and detection_reason == "Initial: Unknown": # More specific reason if model was None from start
        reason_for_default = "Model object provided to script is None"
    print(f"Scriptable Latent (Perlin) Script: Detected non-FLUX (e.g., SDXL) model type. Reason: {reason_for_default}.")
    print(f"Scriptable Latent (Perlin) Script: Configuring for non-FLUX: Channels=4, Spatial Divisor=8, VAE Scale Factor (model intrinsic)=0.18215")
    num_latent_channels = 4
    latent_height = height // 8
    latent_width = width // 8
    vae_scaling_factor = 0.18215 # This is the model's intrinsic VAE scale, kept for info

if latent_height <= 0 or latent_width <= 0:
    raise ValueError(f"Calculated latent dimensions are too small or zero ({latent_height}x{latent_width}). Image H/W: {height}x{width}. Please check model type and image dimensions.")

shape_dim4 = (batch_size, num_latent_channels, latent_height, latent_width)
perlin_shape_dim2 = (latent_height, latent_width) # For Perlin noise generation per channel
print(f"Scriptable Latent (Perlin) Script: Target latent shape: {shape_dim4}")

if sigmas is None or len(sigmas) == 0:
    raise ValueError("Scriptable Latent (Perlin) Script: 'sigmas' input is required and cannot be empty.")
target_sigma = sigmas[0].item() # Use the first sigma for initial noise scaling

# For initial noise generation, the Karras sigmas (target_sigma) usually define the desired std dev directly.
# So, the noise adjustment factor should be 1.0, making effective_sigma = target_sigma.
# The vae_scaling_factor (0.18215 for SDXL, 1.0 for FLUX) is primarily for VAE encoding/decoding context.
effective_sigma = target_sigma 
print(f"Scriptable Latent (Perlin) Script: Model VAE Scale (intrinsic): {vae_scaling_factor:.5f}, Target Sigma (from input): {target_sigma:.4f}, Effective Sigma for noise (target_sigma directly): {effective_sigma:.4f}")

perlin_res_h = max(1, latent_height // perlin_params['res_factor_h'])
perlin_res_w = max(1, latent_width // perlin_params['res_factor_w'])
perlin_res_tuple = (perlin_res_h, perlin_res_w)

# Generate Gaussian noise (standard normal distribution)
gen_gaussian = torch.Generator(device=torch.device(device))
gen_gaussian.manual_seed(seed)
gaussian_noise_tensor = torch.randn(shape_dim4, generator=gen_gaussian, device=torch.device(device))

# Generate Perlin noise, re-seeding generator for each channel for variety if desired, or use a single seed for consistent patterns across channels
perlin_noise_tensor = torch.zeros_like(gaussian_noise_tensor) # Initialize with zeros
perlin_generator_master = torch.Generator(device=torch.device(device)) # Master generator for Perlin

for b_idx in range(batch_size):
    for c_idx in range(num_latent_channels): # Iterate up to num_latent_channels
        # Unique seed for each channel and batch item to ensure variety
        # If consistent Perlin pattern across channels is desired, use a fixed seed or (seed + b_idx)
        perlin_instance_seed = seed + b_idx * num_latent_channels + c_idx 
        perlin_generator_master.manual_seed(perlin_instance_seed)
        
        current_perlin_channel = rand_perlin_2d_octaves_fn(
            shape=perlin_shape_dim2, # (H, W) for a single channel
            res=perlin_res_tuple, 
            octaves=perlin_params['octaves'], 
            persistence=perlin_params['persistence'], 
            frequency_factor=perlin_params['frequency_factor'],
            device=device, 
            generator=perlin_generator_master
        )
        perlin_noise_tensor[b_idx, c_idx] = current_perlin_channel

# Apply Perlin contrast scale before normalization
perlin_contrast = perlin_params.get('perlin_contrast_scale', 1.0) # Get from params, default 1.0
if perlin_contrast != 1.0:
    perlin_noise_tensor = perlin_noise_tensor * perlin_contrast
    print(f"Script Debug: Applied perlin_contrast_scale: {perlin_contrast} to perlin_noise_tensor")

# Normalize both noises individually before blending to ensure consistent amplitude ranges
# This helps make the blend_factor more predictable in its effect.
gauss_mean = torch.mean(gaussian_noise_tensor, dim=(-2, -1), keepdim=True)
gauss_std = torch.std(gaussian_noise_tensor, dim=(-2, -1), keepdim=True)
# Avoid division by zero if std is very small (e.g., for very small latents or specific seeds)
normalized_gaussian = (gaussian_noise_tensor - gauss_mean) / (gauss_std + 1e-5)

perlin_mean = torch.mean(perlin_noise_tensor, dim=(-2, -1), keepdim=True)
perlin_std = torch.std(perlin_noise_tensor, dim=(-2, -1), keepdim=True)
normalized_perlin = (perlin_noise_tensor - perlin_mean) / (perlin_std + 1e-5)

print(f"Script Debug: Normalized Gaussian: mean={torch.mean(normalized_gaussian).item():.4f}, std={torch.std(normalized_gaussian).item():.4f}, sample[0,0,0,0]={'N/A' if normalized_gaussian.numel() == 0 else f'{normalized_gaussian[0,0,0,0].item():.4f}'}")
print(f"Script Debug: Normalized Perlin:   mean={torch.mean(normalized_perlin).item():.4f}, std={torch.std(normalized_perlin).item():.4f}, sample[0,0,0,0]={'N/A' if normalized_perlin.numel() == 0 else f'{normalized_perlin[0,0,0,0].item():.4f}'}")

# Blend the normalized noises
blend_factor = perlin_params['blend_factor']
print(f"Script Debug: Blend Factor before LERP: {blend_factor}")
blended_noise_tensor = torch.lerp(normalized_gaussian, normalized_perlin, blend_factor)
print(f"Script Debug: Blended Noise (post-LERP): mean={torch.mean(blended_noise_tensor).item():.4f}, std={torch.std(blended_noise_tensor).item():.4f}, sample[0,0,0,0]={'N/A' if blended_noise_tensor.numel() == 0 else f'{blended_noise_tensor[0,0,0,0].item():.4f}'}")

# Re-normalize the blended result to ensure it's standard normal N(0,1) before scaling by effective_sigma
# This ensures that the final latent has the desired std dev = effective_sigma
blended_mean = torch.mean(blended_noise_tensor, dim=(-2, -1), keepdim=True)
blended_std = torch.std(blended_noise_tensor, dim=(-2, -1), keepdim=True)
normalized_blended_noise = (blended_noise_tensor - blended_mean) / (blended_std + 1e-5)

# Scale the N(0,1) blended noise by the effective_sigma
final_output_tensor = normalized_blended_noise * effective_sigma
output_latent_samples = final_output_tensor

print(f"Script Debug: Final Output Tensor: mean={torch.mean(final_output_tensor).item():.4f}, std={torch.std(final_output_tensor).item():.4f}")
print(f"Scriptable Latent (Perlin) Script: Output tensor shape: {output_latent_samples.shape}, dtype: {output_latent_samples.dtype}, device: {output_latent_samples.device}")
# print(f"Scriptable Latent (Perlin) Script: Output latent stats: mean={torch.mean(output_latent_samples).item():.4f}, std={torch.std(output_latent_samples).item():.4f}")
# print(f"Scriptable Latent (Perlin) Script: Sample value (0,0,0,0): {output_latent_samples[0,0,0,0].item():.4f} (if shape allows)")
"""


class WolfScriptableEmptyLatent:
    def __init__(self):
        self.node_device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "width": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                    },
                ),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "Input sigmas. The first sigma is typically used for calibration."
                    },
                ),
                "perlin_blend_factor": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "0 = Gaussian, 1 = Perlin",
                    },
                ),
                "perlin_res_factor_h": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Higher val = lower freq/larger features (Height)",
                    },
                ),
                "perlin_res_factor_w": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 128,
                        "tooltip": "Higher val = lower freq/larger features (Width)",
                    },
                ),
                "perlin_frequency_factor": (
                    "FLOAT",
                    {
                        "default": 2.0,
                        "min": 1.01,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "Frequency multiplier per octave",
                    },
                ),
                "perlin_octaves": ("INT", {"default": 4, "min": 1, "max": 10}),
                "perlin_persistence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "perlin_contrast_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Scales Perlin noise before normalization to enhance its features.",
                    },
                ),
                "script": ("STRING", {"multiline": True, "default": DEFAULT_SCRIPT}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_latent"
    CATEGORY = "latent/noise"

    def generate_latent(
        self,
        model,
        width,
        height,
        batch_size,
        seed,
        sigmas,
        perlin_blend_factor,
        perlin_res_factor_h,
        perlin_res_factor_w,
        perlin_frequency_factor,
        perlin_octaves,
        perlin_persistence,
        perlin_contrast_scale,
        script,
    ):
        perlin_params = {
            "blend_factor": perlin_blend_factor,
            "res_factor_h": perlin_res_factor_h,
            "res_factor_w": perlin_res_factor_w,
            "frequency_factor": perlin_frequency_factor,
            "octaves": perlin_octaves,
            "persistence": perlin_persistence,
            "perlin_contrast_scale": perlin_contrast_scale,
        }

        script_globals = {
            "torch": torch,
            "math": math,
            "F": F,
            "rand_perlin_2d_octaves_fn": rand_perlin_2d_octaves,
            "rand_perlin_2d_fn": rand_perlin_2d,
            "_fade_fn": _fade,
        }

        if not isinstance(sigmas, torch.Tensor):
            print(
                f"{self.__class__.__name__}: Warning: Sigmas input was not a tensor. Attempting to convert."
            )
            try:
                sigmas = torch.tensor(sigmas, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"Could not convert sigmas to tensor: {e}")

        script_execution_device_str = "cpu"
        final_tensor_device = self.node_device

        try:
            actual_model_device = comfy.model_management.get_torch_device()
            if hasattr(model, "device") and model.device is not None:
                actual_model_device = model.device
            elif (
                hasattr(model, "model_devices")
                and model.model_devices
                and model.model_devices[0] is not None
            ):
                actual_model_device = model.model_devices[0]

            if actual_model_device is not None:
                script_execution_device_str = str(actual_model_device)
                final_tensor_device = actual_model_device
                # print(f"{self.__class__.__name__}: Script will target model device: {script_execution_device_str}")
            # else:
            # print(f"{self.__class__.__name__}: Could not reliably determine model device. Script targets CPU, final tensor on node_device ({self.node_device}).")
        except Exception as e_dev_fetch:
            print(
                f"{self.__class__.__name__}: Error fetching model device ({e_dev_fetch}). Script targets CPU, final tensor on node_device ({self.node_device})."
            )

        script_locals = {
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "seed": seed,
            "sigmas": sigmas.to(torch.device(script_execution_device_str)),
            "model": model,
            "device": script_execution_device_str,
            "perlin_params": perlin_params,
            "output_latent_samples": None,
        }

        # Python-side model detection
        py_model_is_flux = False
        py_detection_reason = "Unknown"
        if model is not None:
            actual_diffusion_model_host = None
            if hasattr(model, "model") and model.model is not None:
                actual_diffusion_model_host = model.model
            else:
                actual_diffusion_model_host = model

            if actual_diffusion_model_host is not None:
                if (
                    hasattr(actual_diffusion_model_host, "model_type")
                    and actual_diffusion_model_host.model_type is not None
                    # Check for .name attribute before trying to access it for FLUX string comparison
                    and hasattr(actual_diffusion_model_host.model_type, "name")
                    and actual_diffusion_model_host.model_type.name.upper() == "FLUX"
                ):
                    py_model_is_flux = True
                    py_detection_reason = (
                        "actual_diffusion_model.model_type.name == 'FLUX'"
                    )
                elif "Flux" in actual_diffusion_model_host.__class__.__name__:
                    py_model_is_flux = True
                    py_detection_reason = f"actual_diffusion_model.__class__.__name__ ('{actual_diffusion_model_host.__class__.__name__}') contains 'Flux'"

            if not py_model_is_flux:  # Fallback for top-level model object
                if (
                    hasattr(model, "model_type")
                    and model.model_type is not None
                    and hasattr(model.model_type, "name")
                    and model.model_type.name.upper() == "FLUX"
                ):
                    py_model_is_flux = True
                    py_detection_reason = "top-level model.model_type.name == 'FLUX'"
                elif "Flux" in model.__class__.__name__:
                    py_model_is_flux = True
                    py_detection_reason = f"top-level model.__class__.__name__ ('{model.__class__.__name__}') contains 'Flux'"

        num_channels_calc = 4  # Default to SDXL/standard channels
        if py_model_is_flux:
            num_channels_calc = 16
            print(
                f"{self.__class__.__name__}: Python host detected FLUX (reason: {py_detection_reason}). Channels for expected_shape: {num_channels_calc}"
            )
        else:
            print(
                f"{self.__class__.__name__}: Python host detected non-FLUX (reason: {py_detection_reason if model is not None else 'model not provided'}). Channels for expected_shape: {num_channels_calc}"
            )

        # Latent spatial dimensions are typically image_dimension / 8, assuming a VAE scale factor of 8.
        # This should be consistent for both FLUX and SDXL for VAE compatibility if they use 8x VAEs.
        latent_h_calc = height // 8
        latent_w_calc = width // 8

        if latent_h_calc <= 0 or latent_w_calc <= 0:
            raise ValueError(
                f"Calculated Python-side latent dimensions {latent_h_calc}x{latent_w_calc} are invalid."
            )

        expected_shape = (batch_size, num_channels_calc, latent_h_calc, latent_w_calc)
        print(
            f"{self.__class__.__name__}: Python-side host expected latent shape: {expected_shape} (for validation/fallback)"
        )

        try:
            exec(script, script_globals, script_locals)
            output_samples = script_locals.get("output_latent_samples")

            if output_samples is None:
                raise ValueError(
                    "Script did not assign a tensor to 'output_latent_samples'."
                )
            if not isinstance(output_samples, torch.Tensor):
                raise ValueError(
                    f"'output_latent_samples' must be a torch.Tensor, got {type(output_samples)}."
                )

            if output_samples.shape != expected_shape:
                # This is a critical mismatch if the Python host and script disagree on channels for the *same* model.
                raise ValueError(
                    f"CRITICAL MISMATCH: Output tensor shape {output_samples.shape} from script DISAGREES with Python-host expected shape {expected_shape}. Model detection is inconsistent!"
                )

            output_samples = output_samples.to(final_tensor_device, dtype=torch.float32)
            print(
                f"{self.__class__.__name__}: Final latent on device: {output_samples.device}, shape: {output_samples.shape}"
            )

        except Exception as e:
            print(f"Error executing script in {self.__class__.__name__}: {e}")
            traceback.print_exc()
            fallback_shape = (batch_size, num_channels_calc, height // 8, width // 8)
            print(
                f"Falling back to zeros latent in {self.__class__.__name__} with shape {fallback_shape} due to script error."
            )
            output_samples = torch.zeros(
                fallback_shape, dtype=torch.float32, device=final_tensor_device
            )

        return ({"samples": output_samples},)
