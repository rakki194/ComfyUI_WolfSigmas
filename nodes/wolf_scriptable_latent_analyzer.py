import torch
import comfy.model_management
import traceback
import math

# MAX_RESOLUTION fallback (though less critical here)
try:
    from nodes import MAX_RESOLUTION
except ImportError:
    try:
        from comfy.comfy_types import MAX_RESOLUTION
    except ImportError:
        MAX_RESOLUTION = 16384  # Fallback

_DEFAULT_ANALYZER_SCRIPT = """# Default script for WolfScriptableLatentAnalyzer
# Available variables:
# - input_latent_tensor: The latent tensor (torch.Tensor) input to the node.
# - model: Optional model context, if provided to the node.
# - sigmas: Optional sigmas (torch.Tensor), if provided to the node.
# - torch: The PyTorch library.
# - math: The math library.
# - print_fn: A function to print messages to the console from the script.
#
# The script MUST assign a string to 'analysis_output_string'.
# Optionally, it can assign a modified tensor to 'output_latent_tensor'.
# If 'output_latent_tensor' is not assigned, the original input_latent_tensor is passed through.

if input_latent_tensor is None:
    analysis_output_string = "Error: No latent tensor provided."
    # output_latent_tensor = None # or some default
else:
    mean_val = torch.mean(input_latent_tensor).item()
    std_val = torch.std(input_latent_tensor).item()
    min_val = torch.min(input_latent_tensor).item()
    max_val = torch.max(input_latent_tensor).item()
    shape_val = list(input_latent_tensor.shape)
    dtype_val = input_latent_tensor.dtype
    device_val = input_latent_tensor.device

    analysis_output_string = (
        f"Latent Stats:\n"
        f"  Shape:  {shape_val}\n"
        f"  Dtype:  {dtype_val}\n"
        f"  Device: {device_val}\n"
        f"  Mean:   {mean_val:.6f}\n"
        f"  Std:    {std_val:.6f}\n"
        f"  Min:    {min_val:.6f}\n"
        f"  Max:    {max_val:.6f}"
    )

    # Example: Print to console
    # print_fn(analysis_output_string)

    # Example: To modify the latent and pass it on:
    # output_latent_tensor = input_latent_tensor * 2.0 

    # By default, the original latent is passed through if output_latent_tensor is not set.
    # To explicitly pass through the original latent:
    output_latent_tensor = input_latent_tensor
"""


class WolfScriptableLatentAnalyzer:
    """
    A ComfyUI custom node that allows dynamic analysis of a latent tensor using a Python script.
    It takes a latent tensor, executes a script, and outputs the (potentially modified)
    latent tensor and an analysis string.
    """

    def __init__(self):
        self.node_device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "script": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": _DEFAULT_ANALYZER_SCRIPT,
                        "dynamicPrompts": False,
                        "tooltip": "Python script to analyze the latent tensor.",
                    },
                ),
            },
            "optional": {
                "model": (
                    "MODEL",
                    {"tooltip": "Optional model context for the script."},
                ),
                "sigmas": (
                    "SIGMAS",
                    {"tooltip": "Optional sigmas for context in the script."},
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("LATENT", "analysis_log")
    FUNCTION = "execute_analysis_script"
    CATEGORY = "latent/analysis_wolf"

    def _script_print(self, message):
        print(f"[WolfScriptableLatentAnalyzer Script]: {message}")

    def execute_analysis_script(self, latent, script, model=None, sigmas=None):
        if not isinstance(latent, dict) or "samples" not in latent:
            error_msg = "Error: Input 'latent' is not a valid latent structure (must be a dict with 'samples' key)."
            self._script_print(error_msg)
            # Return original latent if possible, or None, and the error message.
            # It's tricky to decide what 'latent' should be if it's malformed.
            # For robustness, if 'latent' itself is the problem, we might not be able to form a valid LATENT output.
            # Returning None for latent output in this specific error case.
            return (None, error_msg)

        input_latent_tensor = latent["samples"]
        original_device = input_latent_tensor.device

        script_globals = {
            "torch": torch,
            "math": math,
            "print_fn": self._script_print,
        }
        script_locals = {
            "input_latent_tensor": input_latent_tensor,
            "model": model,
            "sigmas": sigmas,
            "analysis_output_string": None,
            "output_latent_tensor": None,
        }

        analysis_log = ""
        processed_latent_tensor = input_latent_tensor

        try:
            exec(script, script_globals, script_locals)

            analysis_log = script_locals.get("analysis_output_string")
            if analysis_log is None:
                analysis_log = (
                    "Error: Script did not assign a string to 'analysis_output_string'."
                )
                self._script_print(analysis_log)

            output_tensor_from_script = script_locals.get("output_latent_tensor")

            if output_tensor_from_script is not None:
                if not isinstance(output_tensor_from_script, torch.Tensor):
                    err = "Error: Script assigned a non-Tensor value to 'output_latent_tensor'."
                    self._script_print(err)
                    analysis_log += f"\n{err}"
                elif output_tensor_from_script.shape != input_latent_tensor.shape:
                    err = (
                        f"Error: Script's 'output_latent_tensor' shape {output_tensor_from_script.shape} "
                        f"does not match input latent shape {input_latent_tensor.shape}."
                    )
                    self._script_print(err)
                    analysis_log += f"\n{err}"
                else:
                    processed_latent_tensor = output_tensor_from_script.to(
                        original_device
                    )

        except Exception as e:
            error_details = (
                f"Error executing latent analysis script:\n{traceback.format_exc()}"
            )
            self._script_print(error_details)
            analysis_log = error_details

        output_latent_dict = (
            latent.copy()
        )  # Start with a copy to preserve other keys like 'batch_index'
        output_latent_dict["samples"] = processed_latent_tensor

        return (output_latent_dict, str(analysis_log))


NODE_CLASS_MAPPINGS = {
    "WolfScriptableLatentAnalyzer": WolfScriptableLatentAnalyzer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfScriptableLatentAnalyzer": "Wolf Scriptable Latent Analyzer",
}

# Example of how to use this node in a workflow:
# 1. KSampler -> WolfScriptableLatentAnalyzer (latent input)
# 2. View the `analysis_log` string (e.g., with a text display node or by printing from script)
# 3. The `LATENT` output can be passed to the next node (e.g., VAE Decode)
#
# To debug at various stages:
# - Insert this node after any node that outputs a LATENT.
# - The default script will give you basic stats (mean, std, min, max, shape).
# - You can customize the script to:
#   - Check for NaNs or Infs: `torch.isnan(input_latent_tensor).any()`, `torch.isinf(input_latent_tensor).any()`
#   - Log specific slices or values.
#   - Compare statistics against expected values (e.g., from sigmas if provided).
#   - Visualize parts if you connect to a node that can take custom image data (more complex).
#   - Modify the latent for testing (e.g., `output_latent_tensor = input_latent_tensor * 0 + N` to set to specific value N)
