import torch
import comfy.utils
import comfy.model_management
import traceback

try:
    from nodes import MAX_RESOLUTION
except ImportError:
    try:
        from comfy.comfy_types import MAX_RESOLUTION
    except ImportError:
        MAX_RESOLUTION = 16384  # Fallback

DEFAULT_SIMPLE_SCRIPT = """# Script to generate a basic latent, typically zeros.
# Available variables:
# - width, height, batch_size (integers)
# - device (string, e.g., "cuda:0" or "cpu") - target device for tensor generation
# - model (comfy model object, if connected to the node)
# - torch (module)
#
# The script MUST assign the final latent tensor (NCHW format, float32)
# to a variable named 'output_latent_samples'.

# Determine channel count based on model type (4 for SDXL/default, 16 for FLUX)
num_latent_channels = 4 # Default for SDXL etc.
if model is not None: # Check if model is provided to the node
    model_is_flux = False
    if hasattr(model, 'model_type') and model.model_type is not None and str(model.model_type).lower() == 'flux':
        model_is_flux = True
    elif 'Flux' in model.__class__.__name__ or (hasattr(model, 'diffusion_model') and 'Flux' in model.diffusion_model.__class__.__name__):
        model_is_flux = True

    if model_is_flux:
        num_latent_channels = 16
        print(f"Simple Scriptable Latent: FLUX model detected, channels set to {num_latent_channels}.")
    else:
        print(f"Simple Scriptable Latent: Non-FLUX model detected, channels set to {num_latent_channels}.")
else:
    print(f"Simple Scriptable Latent: Model not connected, defaulting channels to {num_latent_channels}.")

latent_h = height // 8
latent_w = width // 8
shape = (batch_size, num_latent_channels, latent_h, latent_w)

print(f"Simple Scriptable Latent: Generating {width}x{height} (batch: {batch_size}) on {device}. Target shape: {shape}")
output_latent_samples = torch.zeros(shape, dtype=torch.float32, device=torch.device(device))
"""


class WolfSimpleScriptableEmptyLatent:
    def __init__(self):
        self.node_device = (
            comfy.model_management.intermediate_device()
        )  # Device for node's own logic if script device differs

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
                "script": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_SIMPLE_SCRIPT},
                ),
            },
            "optional": {
                "model": ("MODEL",),  # Optional model input for script context
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate_simple_latent"
    CATEGORY = "latent/noise"  # Or "Wolf Custom Nodes/Latent"

    def generate_simple_latent(
        self, width, height, batch_size, device_selection, script, model=None
    ):
        script_globals = {
            "torch": torch,
            "math": __import__("math"),  # Make math module available too
        }

        # Determine target device for script execution and final tensor
        if device_selection == "CPU":
            target_device_str = "cpu"
            final_tensor_device = torch.device("cpu")
        elif device_selection == "GPU":
            target_device_str = str(comfy.model_management.get_torch_device())
            final_tensor_device = comfy.model_management.get_torch_device()
        else:  # AUTO
            if model is not None and hasattr(model, "device"):
                target_device_str = str(model.device)
                final_tensor_device = model.device
            elif (
                model is not None
                and hasattr(model, "model_devices")
                and model.model_devices
            ):
                target_device_str = str(model.model_devices[0])
                final_tensor_device = model.model_devices[0]
            else:
                target_device_str = str(self.node_device)
                final_tensor_device = self.node_device

        print(
            f"{self.__class__.__name__}: Script will execute targeting device string '{target_device_str}'. Final tensor on {final_tensor_device}"
        )

        script_locals = {
            "width": width,
            "height": height,
            "batch_size": batch_size,
            "model": model,  # Pass the model object to the script
            "device": target_device_str,  # Device string for script's internal torch.device() calls
            "output_latent_samples": None,
        }

        # Determine expected shape based on model type INSIDE Python for validation/fallback
        py_model_is_flux = False
        if model is not None:
            if (
                hasattr(model, "model_type")
                and model.model_type is not None
                and str(model.model_type).lower() == "flux"
            ):
                py_model_is_flux = True
            elif "Flux" in model.__class__.__name__ or (
                hasattr(model, "diffusion_model")
                and "Flux" in model.diffusion_model.__class__.__name__
            ):
                py_model_is_flux = True

        if py_model_is_flux:
            num_channels_calc = 16
        else:
            num_channels_calc = 4

        latent_h_calc = height // 8
        latent_w_calc = width // 8

        if latent_h_calc <= 0 or latent_w_calc <= 0:
            raise ValueError(
                f"Calculated Python-side latent dimensions {latent_h_calc}x{latent_w_calc} are invalid."
            )

        expected_shape = (batch_size, num_channels_calc, latent_h_calc, latent_w_calc)
        print(
            f"{self.__class__.__name__}: Python-side host expected latent shape: {expected_shape}"
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
                # This can happen if script logic for channel differs from python host logic (e.g. model not connected but script assumes flux)
                print(
                    f"{self.__class__.__name__}: Warning: Output tensor shape {output_samples.shape} from script differs from Python-host expected shape {expected_shape}. Using script's output shape for the return, but this may indicate a mismatch."
                )

            # Ensure the final tensor is on the designated final_tensor_device and is float32
            output_samples = output_samples.to(final_tensor_device, dtype=torch.float32)
            print(
                f"{self.__class__.__name__}: Final simple latent on device: {output_samples.device}, shape: {output_samples.shape}"
            )

        except Exception as e:
            print(f"Error executing script in {self.__class__.__name__}: {e}")
            traceback.print_exc()
            print(
                f"Falling back to zeros latent in {self.__class__.__name__} with shape {expected_shape} due to script error."
            )
            output_samples = torch.zeros(
                expected_shape, dtype=torch.float32, device=final_tensor_device
            )

        return ({"samples": output_samples},)
