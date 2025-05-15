import torch
import comfy.samplers
import comfy.utils
import comfy.model_management
import nodes  # Import nodes to access KSampler's SAMPLERS and SCHEDULERS if needed for default/example script
import traceback
from comfy.k_diffusion import (
    sampling as k_diffusion_sampling_global,
)  # Import for fallback and script globals


class WolfSamplerScriptEvaluator:
    CATEGORY = "sampling/ksampler_wolf"
    RETURN_TYPES = ("SAMPLER", "STRING")
    RETURN_NAMES = ("SAMPLER", "status_message")
    FUNCTION = "evaluate_sampler_script"

    _DEFAULT_SCRIPT = """
# ComfyUI Wolf Sampler Script Evaluator
#
# Define a function named 'wolf_sampler'.
# This function must return another function, the 'actual_sampler_function'.
#
# The 'actual_sampler_function' signature must be:
# def actual_sampler_function(model, x_initial, sigmas_schedule, *, extra_args, callback, disable, **sampler_options):
#   model: The ComfyUI patched model object (an instance of comfy.model_patcher.ModelPatcher or similar).
#          This is the UNet wrapper that you call like: model(x_current_latent, current_sigma_step, **extra_args_for_model_call).
#          The extra_args_for_model_call typically include 'cond', 'uncond', 'cond_scale' from the 'extra_args' dict.
#   x_initial: The initial latent tensor. For txt2img, this is usually pure noise.
#              For img2img/inpaint, this is the noised version of the initial image. This is your starting point for the sampling loop.
#   sigmas_schedule: A 1D tensor of sigma values for each step in the schedule (e.g., [sigma_max, ..., sigma_min, 0.0]).
#   extra_args (keyword-only): A dictionary containing:
#               'cond': Positive conditioning tensor.
#               'uncond': Negative conditioning tensor.
#               'cond_scale': The CFG (Classifier-Free Guidance) scale value (float).
#               'noise_seed' (optional): An integer seed for k-diffusion internal noise generation (if used by the sampler).
#               'image_cond' (optional): Image conditioning, e.g., for ControlNets.
#   callback (keyword-only): A function to call for previews, e.g., callback(current_step, x0_prediction, current_latents, total_steps).
#   disable (keyword-only): A boolean; if True, the progress bar should be suppressed by the sampler.
#   **sampler_options (keyword-only): A dictionary containing additional sampler-specific options passed from KSAMPLER's
#                                   extra_options (e.g., 's_churn', 's_tmin', 's_tmax', 's_noise', 'inpaint_options').
#
# The function must implement the sampling loop and return the final denoised latent tensor.
#
# Example using a built-in ComfyUI k-diffusion sampler (euler):

# Note: k_diffusion_sampling is typically available in the script's global scope if the node is set up correctly.
# from comfy.k_diffusion import sampling as k_diffusion_sampling # This line might be optional if provided globally

def wolf_sampler():
    def my_custom_euler_sampler(model, x_initial, sigmas_schedule, *, extra_args, callback, disable, **sampler_options):
        # model: Patched model from ComfyUI (the unet wrapper)
        # x_initial: Initial latent tensor (noise for txt2img, or noised image for img2img)
        # sigmas_schedule: The sigma schedule tensor
        # extra_args: Dict containing 'cond', 'uncond', 'cond_scale' (passed as keyword arg to k_diffusion samplers)
        # callback: Progress callback function
        # disable: Boolean to disable progress bar
        # sampler_options: Dict with sampler-specific options like s_churn, s_tmin, etc. (passed as **kwargs to k_diffusion samplers)

        print(f"Wolf Custom Sampler (Euler Example): Running with {len(sigmas_schedule)-1} steps.")
        print(f"Wolf Custom Sampler: extra_args keys: {extra_args.keys()}") # Should show cond, uncond, cond_scale etc.
        print(f"Wolf Custom Sampler: sampler_options keys: {sampler_options.keys()}")

        # Directly call the k_diffusion euler sampler with the provided arguments.
        # The arguments map directly to what k_diffusion samplers expect.
        
        # Ensure k_diffusion_sampling is available (it should be in globals from the node)
        if 'k_diffusion_sampling' not in globals():
            # Fallback import if not in globals (should not happen ideally)
            from comfy.k_diffusion import sampling as k_diffusion_sampling_local
            global k_diffusion_sampling
            k_diffusion_sampling = k_diffusion_sampling_local
            print("Wolf Sampler: Had to locally import k_diffusion_sampling")

        samples = k_diffusion_sampling.sample_euler(
            model,
            x_initial,
            sigmas_schedule,
            extra_args=extra_args,  # Pass the dictionary as a keyword argument
            callback=callback,
            disable=disable,
            **sampler_options     # Pass through other sampler-specific KWARGS like s_churn
        )
        return samples

    return my_custom_euler_sampler
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": s._DEFAULT_SCRIPT,
                        "dynamicPrompts": False,
                    },
                ),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    def evaluate_sampler_script(self, script, seed=0):
        script_locals = {}
        script_globals = {
            "torch": torch,
            "comfy": comfy,
            "nodes": nodes,
            "k_diffusion_sampling": k_diffusion_sampling_global,  # Make it available to the script
            "__script_seed__": seed,
        }

        status_message = "Script evaluation started."
        try:
            if script is None or not isinstance(script, str):
                raise ValueError("Script is None or not a string.")
            compiled_script = compile(script, "<string>", "exec")
            exec(compiled_script, script_globals, script_locals)

            if "wolf_sampler" not in script_locals:
                raise NameError("Script must define 'wolf_sampler'.")
            main_script_func = script_locals["wolf_sampler"]
            if not callable(main_script_func):
                raise TypeError("'wolf_sampler' must be a callable function.")
            actual_sampler_function = main_script_func()
            if not callable(actual_sampler_function):
                raise TypeError(
                    "The function returned by 'wolf_sampler' must also be a callable function (the actual sampler)."
                )
            sampler_obj = comfy.samplers.KSAMPLER(
                actual_sampler_function, extra_options={}, inpaint_options={}
            )
            status_message = "Script evaluated successfully. SAMPLER object created."
            return (sampler_obj, status_message)

        except Exception as e:
            tb_str = traceback.format_exc()
            error_message = (
                f"Error evaluating sampler script: {e}\nTraceback:\n{tb_str}"
            )
            print(error_message)
            try:
                # Use the globally imported k_diffusion_sampling for the fallback
                euler_func = k_diffusion_sampling_global.sample_euler

                # Need to wrap it slightly because KSAMPLER expects a function that matches its call signature,
                # not directly a k-diffusion sampler function. It passes model_wrap, sigmas, extra_args, callback etc.
                # The euler_func (k_diffusion_sampling.sampler_euler) expects model, x, sigmas, extra_args=, callback=, disable=, **kwargs

                # The default_sampler_func_wrapper is what KSAMPLER will call.
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
                final_error_message = f"CRITICAL ERROR: Failed to evaluate script AND failed to load fallback sampler 'euler': {fallback_e}\nOriginal error: {error_message}"
                print(final_error_message)

                def error_sampler_func(*args, **kwargs):
                    raise RuntimeError(
                        f"Sampler script failed and fallback also failed. Original error: {error_message}"
                    )

                if hasattr(comfy, "samplers") and hasattr(comfy.samplers, "KSAMPLER"):
                    error_sampler_obj = comfy.samplers.KSAMPLER(error_sampler_func)
                    return (error_sampler_obj, final_error_message)
                else:

                    class DummySampler:
                        def sample(*args, **kwargs):
                            raise RuntimeError(
                                f"Sampler script failed, fallback failed, and KSAMPLER unavailable. Original error: {error_message}"
                            )

                    return (DummySampler(), final_error_message)


NODE_CLASS_MAPPINGS = {"WolfSamplerScriptEvaluator": WolfSamplerScriptEvaluator}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfSamplerScriptEvaluator": "Wolf Sampler Script Evaluator (🐺)"
}
