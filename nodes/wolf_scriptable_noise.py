import torch
import comfy.model_management
import traceback
import math  # For the default script or user scripts


class ExecutableScriptNoise:
    """
    An object that encapsulates the logic for generating noise via a user-defined script.
    The actual noise generation happens when the `generate_noise` method is called,
    typically by a sampler.
    """

    def __init__(
        self, script_code, seed, model, device_string, script_globals_template
    ):
        self.script_code = script_code
        self.seed = seed  # This is the seed the script should primarily use
        self.model = model  # Optional model context
        self.device_string = (
            device_string  # Target device for tensor creation, e.g., "cuda:0"
        )
        self.script_globals_template = script_globals_template

    def generate_noise(self, latent_image_dict):
        """
        Executes the stored script to generate a noise tensor.
        Args:
            latent_image_dict (dict): A dictionary usually containing:
                'samples' (torch.Tensor): The latent tensor for which noise is needed.
                                           The generated noise should match its shape, dtype, and layout.
                'batch_index' (torch.Tensor, optional): Batch indices for noise preparation if needed by script.
        Returns:
            torch.Tensor: The generated noise tensor, on self.device_string.
        """

        script_locals = {
            "latent_image": latent_image_dict,
            "input_samples": latent_image_dict["samples"],  # Convenience alias
            "seed": self.seed,
            "model": self.model,
            "device": self.device_string,  # Pass the device string for torch.device() in script
            "output_noise_tensor": None,
        }

        current_script_globals = self.script_globals_template.copy()
        current_script_globals["torch"] = torch
        current_script_globals["math"] = math
        # One could add more common modules here like 'numpy', 'random' if desired as globals.
        # current_script_globals['F'] = torch.nn.functional # Example

        try:
            # print(f"ExecutableScriptNoise: Executing script. Seed: {self.seed}, Device: {self.device_string}, Latent shape: {latent_image_dict['samples'].shape}")
            exec(self.script_code, current_script_globals, script_locals)
            noise_tensor = script_locals.get("output_noise_tensor")

            if noise_tensor is None:
                raise ValueError(
                    "Script did not assign a tensor to 'output_noise_tensor'."
                )
            if not isinstance(noise_tensor, torch.Tensor):
                raise ValueError(
                    f"'output_noise_tensor' must be a torch.Tensor, got {type(noise_tensor)}."
                )

            if noise_tensor.shape != latent_image_dict["samples"].shape:
                raise ValueError(
                    f"Generated noise tensor shape {noise_tensor.shape} does not match input latent shape {latent_image_dict['samples'].shape}."
                )

            # Ensure the noise tensor is on the specified device string.
            # The script should ideally create it on this device, but this ensures it.
            final_noise_tensor = noise_tensor.to(torch.device(self.device_string))

            return final_noise_tensor

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
#
# The script MUST assign the final noise tensor to a variable named 'output_noise_tensor'.
# This tensor should:
#   1. Have the same shape as 'input_samples'.
#   2. Have the same dtype as 'input_samples'.
#   3. Be on the torch device specified by the 'device' string.

print(f"WolfScriptableNoise Script: Generating noise. Latent shape: {input_samples.shape}, Seed: {seed}, Device: {device}")

# --- Example Implementations ---

# Example 1: Standard Gaussian Noise (like torch.randn)
# This is often the default noise type used in diffusion models.
# generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
# noise = torch.randn(input_samples.shape, 
#                     generator=generator, 
#                     dtype=input_samples.dtype, 
#                     layout=input_samples.layout, 
#                     device=torch.device(device))

# Example 2: Zero Noise (results in no noise being added by this source)
noise = torch.zeros_like(input_samples, device=torch.device(device))

# Example 3: Uniform Noise
# Creates noise where each value is uniformly random between -1 and 1.
# generator = torch.Generator(device=torch.device(device)).manual_seed(seed)
# noise = (torch.rand(input_samples.shape, 
#                    generator=generator, 
#                    dtype=input_samples.dtype, 
#                    layout=input_samples.layout, 
#                    device=torch.device(device)) * 2.0) - 1.0

# --- Assign your chosen noise to output_noise_tensor ---
output_noise_tensor = noise
# print(f"WolfScriptableNoise Script: Generated noise tensor with shape {output_noise_tensor.shape}, mean {output_noise_tensor.mean():.4f}, std {output_noise_tensor.std():.4f}")
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

    def create_noise_generator(self, seed, device_selection, script, model=None):

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
            f"{self.__class__.__name__}: Noise generator configured for device: '{target_device_str}', seed: {seed}"
        )

        script_globals_template = {
            "__builtins__": __builtins__,
            # Specific modules like torch, math are added directly into the execution scope
            # within ExecutableScriptNoise.generate_noise for robustness.
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
                def __init__(self, error_msg_prop, node_seed):
                    self.error_message = error_msg_prop
                    self.seed = node_seed  # Store the original seed for API consistency

                def generate_noise(self, latent_image_dict):
                    print(
                        f"WolfScriptableNoise - ErrorNoiseGenerator: Script compilation failed. {self.error_message}"
                    )
                    # Fallback to zeros, on the device of the input latent.
                    target_dev = latent_image_dict["samples"].device
                    return torch.zeros_like(
                        latent_image_dict["samples"], device=target_dev
                    )

            return (ErrorNoiseGenerator(error_message, seed),)

        noise_generator_object = ExecutableScriptNoise(
            script_code=script,
            seed=seed,
            model=model,
            device_string=target_device_str,
            script_globals_template=script_globals_template,
        )
        return (noise_generator_object,)


NODE_CLASS_MAPPINGS = {"WolfScriptableNoise": WolfScriptableNoise}

NODE_DISPLAY_NAME_MAPPINGS = {"WolfScriptableNoise": "Scriptable Noise (🐺)"}
