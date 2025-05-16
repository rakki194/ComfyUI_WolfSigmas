import torch
import comfy.samplers
import comfy.utils
import comfy.model_management
import nodes  # Import nodes for access to KSampler's lists if needed for default/example script
import traceback
from comfy.k_diffusion import (
    sampling as k_diffusion_sampling_global,
)  # Import for fallback and script globals


class WolfSimpleSamplerScriptEvaluator:
    CATEGORY = "sampling/ksampler_wolf"
    RETURN_TYPES = ("SAMPLER", "STRING")
    RETURN_NAMES = ("SAMPLER", "status_message")
    FUNCTION = "evaluate_simple_sampler_script"

    _DEFAULT_SCRIPT_SIMPLE = """
# ComfyUI Wolf Simple Sampler Script Evaluator
#
# This script directly implements the sampling loop.
# The following variables are pre-defined and available for you to use:
#   model: The ComfyUI patched model object (e.g., comfy.model_patcher.ModelPatcher).
#          Call it like: model(current_latents, current_sigma_step_tensor, **extra_args_for_model_call)
#   x_initial: The initial latent tensor (noise for txt2img, noised image for img2img).
#   sigmas_schedule: A 1D tensor of sigma values for each step (e.g., [sigma_max, ..., sigma_min, 0.0]).
#   extra_args: Dictionary containing 'cond', 'uncond', 'cond_scale', 'noise_seed' (optional), 'image_cond' (optional), etc.
#   callback: The callback function for previews. Signature: callback({'i', 'denoised', 'x', 'sigma', ...})
#   disable: Boolean; if True, the progress bar should be suppressed (your script should respect this if it prints progress).
#   sampler_options: Dictionary with additional sampler-specific options from KSAMPLER's extra_options.
#
# Your script MUST assign the final denoised latents to a variable named 'latents'.
#
# Example: Custom Euler Sampler Implementation (Simplified)

import torch # Good practice, though 'torch' is usually available globally.

# These variables are passed in from the evaluator:
# model, x_initial, sigmas_schedule, extra_args, callback, disable, sampler_options

print(f"Wolf Simple Sampler: Initializing...")
print(f"Wolf Simple Sampler: Num sigma entries: {len(sigmas_schedule)}, x_initial shape: {x_initial.shape}")
# print(f"Wolf Simple Sampler: extra_args keys: {list(extra_args.keys())}")
# print(f"Wolf Simple Sampler: sampler_options keys: {list(sampler_options.keys())}")

# Make a copy of the initial latents to work on.
current_latents = x_initial.clone()

# The number of sampling steps is one less than the number of sigmas.
num_sampling_steps = len(sigmas_schedule) - 1

for i in range(num_sampling_steps):
    sigma_current = sigmas_schedule[i]
    sigma_next = sigmas_schedule[i+1]

    if not disable: # Respect the 'disable' flag for progress outputs
        print(f"Wolf Simple Sampler: Step {i+1}/{num_sampling_steps}, sigma: {sigma_current:.4f} -> {sigma_next:.4f}")

    # 1. Predict the denoised sample (x0_hat) using the model
    # The 'model' handles CFG internally. Ensure sigma_current is a tensor for the model call.
    denoised_prediction = model(current_latents, sigma_current.unsqueeze(0), **extra_args)

    # 2. Calculate the derivative 'd'
    d = (current_latents - denoised_prediction) / sigma_current
    
    # 3. Perform the Euler step
    dt = sigma_next - sigma_current # Change in sigma (will be negative)
    current_latents = current_latents + d * dt
    
    # Invoke the callback for previews
    if callback is not None:
        callback_payload = {
            'i': i,                            # Current step index (0-indexed)
            'denoised': denoised_prediction,   # The D(x) prediction
            'x': current_latents,              # The current latents x_t (after this step's update)
            'sigma': sigmas_schedule,          # The full sigma schedule tensor
            # 'sigma_hat': sigma_current       # Current sigma_t (optional, k_callback might not use it)
        }
        callback(callback_payload)

if not disable:
    print("Wolf Simple Sampler: Sampling loop completed.")

# CRITICAL: Assign the final result to the 'latents' variable.
# This is what the evaluator will return from the sampling process.
latents = current_latents
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": s._DEFAULT_SCRIPT_SIMPLE,
                        "dynamicPrompts": False,
                    },
                ),
            }
        }

    def evaluate_simple_sampler_script(self, script):
        status_message = "Script evaluation started."

        # This function will be wrapped by comfy.samplers.KSAMPLER
        # It must match the signature that KSAMPLER's internal logic expects to call.
        def actual_simple_sampler_function(
            model_patched_unet,  # Renamed for clarity in this scope
            initial_latent_x,  # Renamed for clarity
            sigmas_sched,  # Renamed for clarity
            *,
            extra_args,  # CORRECTED from extra_args_dict
            callback,  # CORRECTED from callback_fn
            disable,  # CORRECTED from disable_pbar
            **sampler_options,  # CORRECTED from sampler_options_dict
        ):
            script_locals = {
                "model": model_patched_unet,
                "x_initial": initial_latent_x,
                "sigmas_schedule": sigmas_sched,
                "extra_args": extra_args,  # USE CORRECTED NAME
                "callback": callback,  # USE CORRECTED NAME
                "disable": disable,  # USE CORRECTED NAME
                "sampler_options": sampler_options,  # USE CORRECTED NAME
                "latents": None,  # Initialize to ensure it exists
            }
            # Globals available to the script
            script_globals = {
                "torch": torch,
                "comfy": comfy,
                "nodes": nodes,
                "k_diffusion_sampling": k_diffusion_sampling_global,
                "__builtins__": __builtins__,  # Ensure basic builtins are available
            }

            exec(script, script_globals, script_locals)

            if "latents" not in script_locals or script_locals["latents"] is None:
                raise NameError(
                    "Script must define and assign a tensor to the 'latents' variable."
                )
            return script_locals["latents"]

        try:
            if script is None or not isinstance(script, str):
                raise ValueError("Script is None or not a string.")

            # We don't compile or exec here directly at the top level.
            # Instead, we pass the actual_simple_sampler_function (which will do the exec)
            # to KSAMPLER.

            # Test exec once to catch early syntax errors, though runtime errors inside the sampler
            # will be caught by the KSAMPLER or the execution environment.
            # This is a lightweight check.
            compile(script, "<string>", "exec")

            sampler_obj = comfy.samplers.KSAMPLER(
                actual_simple_sampler_function, extra_options={}, inpaint_options={}
            )
            status_message = "Script prepared successfully. SAMPLER object created."
            return (sampler_obj, status_message)

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = (
                f"Error preparing simple sampler script: {e}\nTraceback:\n{tb_str}"
            )
            print(error_message)
            try:
                # Fallback to standard Euler sampler
                euler_func = k_diffusion_sampling_global.sample_euler

                def default_sampler_func_wrapper(
                    model_patched_unet,
                    initial_latent_x,
                    sigmas_sched,
                    *,
                    extra_args_dict,
                    callback_fn,
                    disable_pbar,
                    **kwargs_options,
                ):
                    return euler_func(
                        model=model_patched_unet,
                        x=initial_latent_x,
                        sigmas=sigmas_sched,
                        extra_args=extra_args_dict,
                        callback=callback_fn,
                        disable=disable_pbar,
                        **kwargs_options,
                    )

                fallback_sampler = comfy.samplers.KSAMPLER(
                    default_sampler_func_wrapper, extra_options={}, inpaint_options={}
                )
                status_message = f"ERROR: {error_message}\nFALLBACK TO 'euler' SAMPLER."
                return (fallback_sampler, status_message)
            except Exception as fallback_e:
                final_error_message = f"CRITICAL ERROR: Failed to prepare script AND failed to load fallback 'euler': {fallback_e}\nOriginal error: {error_message}"
                print(final_error_message)

                def error_sampler_func(*args, **kwargs):
                    raise RuntimeError(
                        f"Simple sampler script failed and fallback also failed. Original error: {error_message}"
                    )

                if hasattr(comfy, "samplers") and hasattr(comfy.samplers, "KSAMPLER"):
                    error_sampler_obj = comfy.samplers.KSAMPLER(error_sampler_func)
                    return (error_sampler_obj, final_error_message)
                else:

                    class DummySampler:
                        def sample(*args, **kwargs):
                            raise RuntimeError(
                                f"Simple sampler script failed, fallback failed, KSAMPLER unavailable. Original error: {error_message}"
                            )

                    return (DummySampler(), final_error_message)


NODE_CLASS_MAPPINGS = {
    "WolfSimpleSamplerScriptEvaluator": WolfSimpleSamplerScriptEvaluator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfSimpleSamplerScriptEvaluator": "Wolf Simple Sampler Script (🐺)"
}
