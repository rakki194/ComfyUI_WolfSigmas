import torch
import comfy.model_management
import traceback
import math  # For the default script or user scripts
import torch.nn.functional as F  # Needed for Perlin noise

# MAX_RESOLUTION fallback for Perlin noise functions if not otherwise available
# (though less critical here than in empty latent, good for robustness if functions are copied directly)
try:
    from nodes import MAX_RESOLUTION
except ImportError:
    try:
        from comfy.comfy_types import MAX_RESOLUTION
    except ImportError:
        MAX_RESOLUTION = 16384  # Fallback


# --- Perlin Noise Generation Functions (copied from wolf_scriptable_empty_latent.py) ---
def _fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def rand_perlin_2d(shape, res, fade_func=_fade, device="cpu", generator=None):
    # Ensure device is a torch.device object for linspace, not a string
    target_device = torch.device(device if isinstance(device, str) else "cpu")

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid_y = torch.linspace(0, res[0], shape[0], device=target_device)
    grid_x = torch.linspace(0, res[1], shape[1], device=target_device)
    grid = torch.stack(torch.meshgrid(grid_y, grid_x, indexing="ij"), dim=-1) % 1

    rand_opts = {"device": target_device}
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
    generator=None,
):
    # Ensure device is a torch.device object for noise tensor, not a string
    target_device = torch.device(device if isinstance(device, str) else "cpu")
    noise = torch.zeros(shape, device=target_device)
    frequency = 1.0
    amplitude = 1.0
    for _ in range(octaves):
        current_res_h = max(1, int(frequency * res[0]))
        current_res_w = max(1, int(frequency * res[1]))
        current_res = (current_res_h, current_res_w)
        # rand_perlin_2d itself handles the device string internally for its sub-components
        noise += amplitude * rand_perlin_2d(
            shape,
            current_res,
            fade_func=fade_func,
            device=device,  # Pass the original device string or torch.device here
            generator=generator,
        )
        frequency *= frequency_factor
        amplitude *= persistence
    return noise


# --- End Perlin Noise Generation Functions ---


class ExecutableScriptNoise:
    """
    An object that encapsulates the logic for generating noise via a user-defined script.
    The actual noise generation happens when the `generate_noise` method is called,
    typically by a sampler.
    """

    def __init__(
        self,
        script_code,
        seed,
        model,
        device_string,
        script_globals_template,
        sigmas,
        normalization_mode,
    ):
        self.script_code = script_code
        self.seed = seed  # This is the seed the script should primarily use
        self.model = model  # Optional model context
        self.device_string = (
            device_string  # Target device for tensor creation, e.g., "cuda:0"
        )
        self.script_globals_template = script_globals_template
        self.sigmas = sigmas  # Store sigmas tensor
        self.normalization_mode = normalization_mode  # Store normalization mode

    def apply_normalization(self, noise_tensor, mode):
        """
        Applies normalization to the noise tensor.
        Assumes noise_tensor is on self.device_string.
        """
        if mode == "None" or noise_tensor.numel() == 0:
            return noise_tensor

        original_device = noise_tensor.device
        # For simplicity and consistency with WolfDCTNoise, some ops might be easier on CPU
        # or to ensure reproducibility if specific implementations vary by device.
        # However, torch.mean/std are generally fine on GPU.
        # Let's keep it on original_device unless issues arise.

        processed_tensor = noise_tensor

        if mode == "Mean0Std1_channel":
            temp_normalized_channels = []
            for c in range(noise_tensor.shape[1]):  # NCHW format, C is dim 1
                channel_data = noise_tensor[:, c, :, :]
                if channel_data.numel() > 0:
                    mean = torch.mean(channel_data.float())  # Ensure float for mean/std
                    std = torch.std(channel_data.float())
                    if std > 1e-6:
                        temp_normalized_channels.append((channel_data - mean) / std)
                    else:
                        temp_normalized_channels.append(channel_data - mean)
                else:
                    temp_normalized_channels.append(channel_data)
            if temp_normalized_channels:
                processed_tensor = torch.stack(temp_normalized_channels, dim=1).to(
                    original_device
                )
        elif mode == "Mean0Std1_tensor":
            if noise_tensor.numel() > 0:
                mean = torch.mean(noise_tensor.float())
                std = torch.std(noise_tensor.float())
                if std > 1e-6:
                    processed_tensor = (noise_tensor - mean) / std
                else:
                    processed_tensor = noise_tensor - mean
        elif mode == "ScaleToStd1_channel":
            temp_scaled_channels = []
            for c in range(noise_tensor.shape[1]):
                channel_data = noise_tensor[:, c, :, :]
                if channel_data.numel() > 0:
                    std = torch.std(channel_data.float())
                    if std > 1e-6:
                        temp_scaled_channels.append(channel_data / std)
                    else:
                        temp_scaled_channels.append(channel_data)
                else:
                    temp_scaled_channels.append(channel_data)
            if temp_scaled_channels:
                processed_tensor = torch.stack(temp_scaled_channels, dim=1).to(
                    original_device
                )
        elif mode == "ScaleToStd1_tensor":
            if noise_tensor.numel() > 0:
                std = torch.std(noise_tensor.float())
                if std > 1e-6:
                    processed_tensor = noise_tensor / std

        return processed_tensor.to(noise_tensor.dtype)  # Ensure original dtype

    def generate_noise(self, latent_image_dict):
        """
        Executes the stored script to generate a noise tensor,
        then optionally normalizes and scales it by sigmas[0].
        Args:
            latent_image_dict (dict): A dictionary usually containing:
                'samples' (torch.Tensor): The latent tensor for which noise is needed.
                                           The generated noise should match its shape, dtype, and layout.
                'batch_index' (torch.Tensor, optional): Batch indices for noise preparation if needed by script.
        Returns:
            torch.Tensor: The generated noise tensor, on self.device_string.
        """
        target_torch_device = torch.device(self.device_string)

        script_locals = {
            "latent_image": latent_image_dict,
            "input_samples": latent_image_dict["samples"],  # Convenience alias
            "seed": self.seed,
            "model": self.model,
            "device": self.device_string,  # Pass the device string for torch.device() in script
            "sigmas": (  # Make full sigma schedule available to script if needed for context
                self.sigmas.to(target_torch_device) if self.sigmas is not None else None
            ),
            "output_noise_tensor": None,  # This is what the script should set (base noise)
        }

        current_script_globals = self.script_globals_template.copy()
        current_script_globals["torch"] = torch
        current_script_globals["math"] = math
        current_script_globals["F"] = F  # Add torch.nn.functional
        current_script_globals["rand_perlin_2d_octaves_fn"] = rand_perlin_2d_octaves
        current_script_globals["rand_perlin_2d_fn"] = rand_perlin_2d
        current_script_globals["_fade_fn"] = _fade
        # One could add more common modules here like 'numpy', 'random' if desired as globals.
        # current_script_globals['F'] = torch.nn.functional # Example

        try:
            # print(f"ExecutableScriptNoise: Executing script. Seed: {self.seed}, Device: {self.device_string}, Latent shape: {latent_image_dict['samples'].shape}")
            exec(self.script_code, current_script_globals, script_locals)
            base_noise_tensor = script_locals.get("output_noise_tensor")

            if base_noise_tensor is None:
                raise ValueError(
                    "Script did not assign a tensor to 'output_noise_tensor'."
                )
            if not isinstance(base_noise_tensor, torch.Tensor):
                raise ValueError(
                    f"'output_noise_tensor' must be a torch.Tensor, got {type(base_noise_tensor)}."
                )

            if base_noise_tensor.shape != latent_image_dict["samples"].shape:
                raise ValueError(
                    f"Generated noise tensor shape {base_noise_tensor.shape} does not match input latent shape {latent_image_dict['samples'].shape}."
                )

            # Ensure the script's output tensor is on the specified device string.
            base_noise_tensor_on_device = base_noise_tensor.to(target_torch_device)

            # Apply normalization if requested
            processed_noise = base_noise_tensor_on_device
            if self.normalization_mode != "None":
                print(f"  Applying normalization: {self.normalization_mode}")
                processed_noise = self.apply_normalization(
                    base_noise_tensor_on_device, self.normalization_mode
                )

            # Scale by sigmas[0] if sigmas are available
            final_output_noise = processed_noise
            if self.sigmas is not None and len(self.sigmas) > 0:
                # Ensure sigmas tensor is on a compatible device for item() or direct use
                # self.sigmas should ideally be on CPU or the target_torch_device
                # For safety, move the specific sigma value to CPU before .item() if it's not already scalar
                current_sigma_val = (
                    self.sigmas[0].cpu().item()
                    if self.sigmas[0].numel() == 1
                    else self.sigmas[0].item()
                )

                final_output_noise = processed_noise * current_sigma_val
                print(
                    f"  Script output (normalized if applicable) scaled by sigmas[0]: {current_sigma_val:.4f}"
                )
            else:
                print(
                    "  Using script output (normalized if applicable) as is (no sigmas[0] scaling)."
                )

            return final_output_noise.to(
                target_torch_device
            )  # Ensure final tensor is on target device

        except Exception as e:
            print(
                f"Error executing script in ExecutableScriptNoise.generate_noise: {e}"
            )
            traceback.print_exc()
            # Fallback: return zeros on the target device matching the latent_image_dict.
            return torch.zeros_like(
                latent_image_dict["samples"], device=torch.device(self.device_string)
            )


_DEFAULT_NOISE_SCRIPT = """# Script to generate custom noise for ComfyUI's WolfScriptableNoise
#
# This script is executed within the `generate_noise` method of a noise object.
#
# Available variables in the script's execution scope:
# - latent_image (dict): Contains 'samples' (the latent tensor, NCHW format) 
#                        and potentially 'batch_index'. Noise should match 'samples'.shape.
# - input_samples (tensor): Convenience alias for latent_image['samples'].
# - seed (int): The noise seed provided to the node. Use this for reproducible randomness.
# - model (ComfyUI model object): Optional, this is the model object if one was 
#                                 connected to the WolfScriptableNoise node. Can be None.
# - device (str): Target device string for tensor generation (e.g., "cuda:0", "cpu").
#                 Use this with torch.device(device) when creating new tensors.
# - torch (module): The PyTorch module.
# - math (module): Python's standard math module.
# - F (module): torch.nn.functional.
# - sigmas (tensor): The sigma schedule (if SIGMAS input is connected to the node).
#                    This is for context; scaling by sigmas[0] is now handled by the node post-script.
# - rand_perlin_2d_octaves_fn, rand_perlin_2d_fn, _fade_fn: Perlin noise functions.
#
# The script MUST assign the final "base" noise tensor to 'output_noise_tensor'.
# This tensor should:
#   1. Have the same shape as 'input_samples'.
#   2. Have the same dtype as 'input_samples'.
#   3. Be on the torch device specified by the 'device' string.
#   4. Represent the base noise before any sigmas[0] scaling (node handles this).
#
# Post-script processing by the node:
#   1. Optional normalization (if selected in node UI).
#   2. Scaling by sigmas[0] (if sigmas are available to the node).

print(f"WolfScriptableNoise Script: Generating base noise. Latent shape: {input_samples.shape}, Seed: {seed}, Device: {device}")
if sigmas is not None:
    print(f"  Sigmas available to script for context, first sigma: {sigmas[0].item() if len(sigmas) > 0 else 'empty'}")
else:
    print("  Sigmas not available to the script (SIGMAS input likely not connected or empty).")

# Default: Gaussian noise N(0,1)
noise_gen = torch.Generator(device=torch.device(device)).manual_seed(seed)
gaussian_noise = torch.randn(input_samples.shape, generator=noise_gen, dtype=input_samples.dtype, device=torch.device(device))

output_noise_tensor = gaussian_noise
print("  Script outputting N(0,1) Gaussian noise as 'output_noise_tensor'. Node will handle normalization and sigmas[0] scaling.")
"""


class WolfScriptableNoise:
    """
    A ComfyUI custom node that allows dynamic noise generation using a Python script.
    It returns a 'NOISE' object that, when invoked by a sampler, executes the
    provided script to generate a noise tensor.
    """

    def __init__(self):
        # Fallback device for the node's own logic if model/GPU not available for AUTO setting.
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
                        "tooltip": "Seed for the noise generation script.",
                    },
                ),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Sigma schedule for noise scaling and context."},
                ),
                "device_selection": (
                    ["AUTO", "CPU", "GPU"],
                    {
                        "default": "AUTO",
                        "tooltip": "Device for script execution and noise tensor creation. AUTO tries to use model's device or GPU.",
                    },
                ),
                "script": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": _DEFAULT_NOISE_SCRIPT,
                        "dynamicPrompts": False,
                        "tooltip": "Python script to generate noise.",
                    },
                ),
                "normalization_mode": (  # New input
                    [
                        "None",
                        "Mean0Std1_channel",
                        "Mean0Std1_tensor",
                        "ScaleToStd1_channel",
                        "ScaleToStd1_tensor",
                    ],
                    {
                        "default": "None",
                        "tooltip": "Normalization to apply to the script's base noise output.",
                    },
                ),
            },
            "optional": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "Optional model context for the script. Can influence AUTO device selection."
                    },
                ),
            },
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "create_noise_generator"
    CATEGORY = "sampling/custom_sampling/noise"  # Consistent with ComfyUI's RandomNoise, DisableNoise

    def create_noise_generator(
        self,
        seed,
        sigmas,
        device_selection,
        script,
        normalization_mode,
        model=None,  # Added normalization_mode
    ):

        target_device_str = ""
        if device_selection == "CPU":
            target_device_str = "cpu"
        elif device_selection == "GPU":
            if comfy.model_management.is_device_mps(
                self.node_device
            ):  # Check if primary device is MPS
                target_device_str = str(
                    comfy.model_management.get_torch_device()
                )  # Use MPS if that's what Comfy is using
            elif torch.cuda.is_available():
                target_device_str = str(
                    comfy.model_management.get_torch_device()
                )  # Usually cuda:0
            else:  # GPU selected but no CUDA, fallback to CPU
                print(
                    f"{self.__class__.__name__} Warning: GPU selected but no CUDA available. Falling back to CPU for script execution."
                )
                target_device_str = "cpu"
        else:  # AUTO
            # Prefer model's device if model is connected and device info is accessible
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
            elif comfy.model_management.is_device_mps(
                self.node_device
            ):  # Check if primary device is MPS
                target_device_str = str(self.node_device)
            elif torch.cuda.is_available():  # Fallback to GPU if available
                target_device_str = str(comfy.model_management.get_torch_device())
            else:  # Fallback to node's intermediate device (usually CPU if no GPU)
                target_device_str = str(self.node_device)

        print(
            f"{self.__class__.__name__}: Noise generator configured for device: '{target_device_str}', seed: {seed}, normalization: '{normalization_mode}'"
        )
        if sigmas is not None:
            print(
                f"  Sigmas provided to node, first sigma: {sigmas[0].item() if len(sigmas) > 0 else 'empty'}"
            )

        script_globals_template = {
            "__builtins__": __builtins__,
        }

        try:
            compile(script, "<WolfScriptableNoise_script>", "exec")
        except Exception as e:
            error_message = (
                f"Syntax error in provided script for {self.__class__.__name__}: {e}"
            )
            print(error_message)
            traceback.print_exc()

            class ErrorNoiseGenerator:
                def __init__(self, error_msg_prop, node_seed, node_sigmas):
                    self.error_message = error_msg_prop
                    self.seed = node_seed
                    self.sigmas = (
                        node_sigmas  # Store for completeness, though script failed
                    )

                def generate_noise(self, latent_image_dict):
                    print(
                        f"WolfScriptableNoise - ErrorNoiseGenerator: Script compilation failed. {self.error_message}"
                    )
                    # Fallback to zeros, on the device of the input latent.
                    target_dev = latent_image_dict["samples"].device
                    return torch.zeros_like(
                        latent_image_dict["samples"], device=target_dev
                    )

            return (ErrorNoiseGenerator(error_message, seed, sigmas),)

        noise_generator_object = ExecutableScriptNoise(
            script_code=script,
            seed=seed,
            model=model,
            device_string=target_device_str,
            script_globals_template=script_globals_template,
            sigmas=sigmas,  # Pass sigmas here
            normalization_mode=normalization_mode,  # Pass normalization_mode
        )
        return (noise_generator_object,)


NODE_CLASS_MAPPINGS = {"WolfScriptableNoise": WolfScriptableNoise}

NODE_DISPLAY_NAME_MAPPINGS = {"WolfScriptableNoise": "Scriptable Noise (🐺)"}
