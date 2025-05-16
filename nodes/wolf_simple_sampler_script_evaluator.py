import torch
import comfy.samplers
import comfy.utils
import comfy.model_management
import nodes
import traceback

from comfy.k_diffusion import (
    sampling as k_diffusion_sampling_global,
)


class WolfSimpleSamplerScriptEvaluator:
    CATEGORY = "sampling/ksampler_wolf"
    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("SAMPLER",)
    FUNCTION = "evaluate_simple_sampler_script"

    _DEFAULT_SCRIPT_SIMPLE = """
# ComfyUI Wolf Simple Sampler Script Evaluator
#
# This script directly implements the sampling loop.
# The following variables are pre-defined and available for you to use:
#   model, x_initial, sigmas_schedule, extra_args, callback, disable, sampler_options
#
# Your script MUST assign the final denoised latents to a variable named 'latents'.

import torch

current_latents = x_initial.clone()
num_sampling_steps = len(sigmas_schedule) - 1

for i in range(num_sampling_steps):
    sigma_current = sigmas_schedule[i]
    sigma_next = sigmas_schedule[i+1]
    # latents_for_denoising = current_latents.clone() # Not needed without debug
    denoised_prediction = model(current_latents, sigma_current.unsqueeze(0), **extra_args)
    d = (current_latents - denoised_prediction) / sigma_current
    dt = sigma_next - sigma_current
    current_latents = current_latents + d * dt
    
    if callback is not None:
        callback({'i': i, 'denoised': denoised_prediction, 'x': current_latents, 'sigma': sigmas_schedule})

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

        def actual_simple_sampler_function(
            model_patched_unet,
            initial_latent_x,
            sigmas_sched,
            *,
            extra_args,
            callback,
            disable,
            **sampler_options,
        ):
            script_locals = {
                "model": model_patched_unet,
                "x_initial": initial_latent_x,
                "sigmas_schedule": sigmas_sched,
                "extra_args": extra_args,
                "callback": callback,
                "disable": disable,
                "sampler_options": sampler_options,
                "latents": None,
            }
            script_globals = {
                "torch": torch,
                "comfy": comfy,
                "nodes": nodes,
                "k_diffusion_sampling": k_diffusion_sampling_global,
                "__builtins__": __builtins__,
            }
            exec(script, script_globals, script_locals)
            if "latents" not in script_locals or script_locals["latents"] is None:
                raise NameError("Script must assign to 'latents'.")
            return script_locals["latents"]

        try:
            compile(script, "<string>", "exec")
            # Create a standard KSAMPLER object for the first output
            sampler_object_for_ksampler_node = comfy.samplers.KSAMPLER(
                actual_simple_sampler_function
            )

            return (sampler_object_for_ksampler_node,)
        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = f"Error preparing script: {e}\\nTraceback:\\n{tb_str}"
            print(error_message)
            try:

                def fallback_sampler_function(*args, **kwargs):
                    return k_diffusion_sampling_global.sample_euler(
                        model=args[0],
                        x=args[1],
                        sigmas=args[2],
                        extra_args=kwargs.get("extra_args"),
                        callback=kwargs.get("callback"),
                        disable=kwargs.get("disable"),
                    )

                fallback_ksampler_obj = comfy.samplers.KSAMPLER(
                    fallback_sampler_function
                )
                print(f"ERROR: {error_message}\\nFALLBACK TO 'euler'.")
                return (fallback_ksampler_obj,)
            except Exception as fallback_e:
                final_err_msg = (  # Renamed for clarity
                    f"CRITICAL: Script error AND fallback Euler failed: {fallback_e}"
                )
                print(final_err_msg)

                def error_sampler_func(*args, **kwargs):
                    raise RuntimeError(
                        f"Sampler script failed, fallback failed. Original: {error_message}"
                    )

                error_ksampler_obj = comfy.samplers.KSAMPLER(error_sampler_func)
                return (error_ksampler_obj,)


NODE_CLASS_MAPPINGS = {
    "WolfSimpleSamplerScriptEvaluator": WolfSimpleSamplerScriptEvaluator
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfSimpleSamplerScriptEvaluator": "Wolf Simple Sampler Script (🐺)"
}
