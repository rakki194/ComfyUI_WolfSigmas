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
#                    The last sigma is usually 0.0. The loop runs for (len(sigmas_schedule) - 1) steps.
#   extra_args (keyword-only): A dictionary containing:
#               'cond': Positive conditioning tensor.
#               'uncond': Negative conditioning tensor.
#               'cond_scale': The CFG (Classifier-Free Guidance) scale value (float).
#               'noise_seed' (optional): An integer seed for k-diffusion internal noise generation (if used by the sampler).
#                                        (Note: this custom Euler does not use this directly unless you add noise per step)
#               'image_cond' (optional): Image conditioning, e.g., for ControlNets.
#   callback (keyword-only): A function to call for previews, e.g., callback(current_step_1_indexed, x0_prediction, current_latents_after_step, total_sampling_steps).
#   disable (keyword-only): A boolean; if True, the progress bar should be suppressed by the sampler.
#                           (Note: This custom Euler example does not currently implement progress bar suppression logic based on 'disable')
#   **sampler_options (keyword-only): A dictionary containing additional sampler-specific options passed from KSAMPLER's
#                                   extra_options (e.g., 's_churn', 's_tmin', 's_tmax', 's_noise', 'inpaint_options').
#                                   (Note: This custom Euler implementation does not use these options, but a more advanced one could)
#
# The function must implement the sampling loop and return the final denoised latent tensor.
#
# Example: Custom Euler Sampler Implementation

def wolf_sampler():
    # This function is called by the WolfSamplerScriptEvaluator.
    # It must return the actual sampling function.
    
    def wolf_custom_euler_sampler(model, x_initial, sigmas_schedule, *, extra_args, callback, disable, **sampler_options):
        # model: Patched model from ComfyUI (the UNet wrapper that handles CFG internally)
        # x_initial: Initial latent tensor (noise for txt2img, or noised image for img2img)
        # sigmas_schedule: The sigma schedule tensor (e.g., [sigma_max, ..., sigma_min, 0.0])
        # extra_args: Dict containing 'cond', 'uncond', 'cond_scale', 'image_cond' (optional), etc.
        # callback: Progress callback function
        # disable: Boolean to disable progress bar (currently not used by this example's print statements)
        # sampler_options: Dict with other sampler-specific options (currently not used by this example)

        # Ensure torch is available in the script's scope.
        # It's good practice, though 'torch' is usually injected into globals by the node.
        import torch

        print(f"Wolf Custom Euler Sampler: Initializing...")
        print(f"Wolf Custom Euler Sampler: Number of sigma entries: {len(sigmas_schedule)}")
        print(f"Wolf Custom Euler Sampler: extra_args keys: {list(extra_args.keys())}")
        print(f"Wolf Custom Euler Sampler: sampler_options keys: {list(sampler_options.keys())}")
        
        latents = x_initial.clone() # Start with a copy of the initial latents
        
        # The number of sampling steps is one less than the number of sigmas in the schedule.
        # The loop iterates from the first sigma up to the second-to-last sigma.
        # The last sigma (typically 0.0) is the target state after the final step.
        num_sampling_steps = len(sigmas_schedule) - 1

        # All necessary arguments for the model (cond, uncond, cond_scale, denoise_mask, image_cond, etc.)
        # are expected to be in the extra_args dictionary.
        # The model object (ModelPatcher) will correctly pick them up when extra_args is spread.
        # No need to manually construct model_call_kwargs by picking specific keys.
        
        # Filter out None values from model_call_kwargs, as the model might not expect them
        # if a conditioning arg was not provided (though KSampler usually provides defaults or actual values).
        # model_call_kwargs = {k: v for k, v in model_call_kwargs.items() if v is not None} # This filtering is handled by ModelPatcher or should be.

        for i in range(num_sampling_steps):
            sigma_current = sigmas_schedule[i]
            sigma_next = sigmas_schedule[i+1] # Target sigma for this step

            # sigma_current should generally not be zero here due to how schedules are typically formed.
            # If sigma_current is 0, division by it would cause issues.
            # ComfyUI's sigmas (from calculate_sigmas) are structured like [sigma_max, ..., sigma_min (>0), 0.0]
            # The loop for i from 0 to num_sampling_steps-1 ensures sigma_current is one of [sigma_max, ..., sigma_min].

            if not disable: # Basic check for progress bar suppression
                print(f"Wolf Custom Euler Sampler: Step {i+1}/{num_sampling_steps}, sigma_current: {sigma_current:.4f}, sigma_next: {sigma_next:.4f}")

            # 1. Predict the denoised sample (x0_hat) using the model
            # The 'model' object is a comfy.model_patcher.ModelPatcher instance which handles CFG.
            # It expects: model(current_latents, current_sigma_value_tensor, **all_extra_args)
            # Ensure sigma_current is at least 1D for model compatibility (especially with ControlNet paths)
            denoised_prediction = model(latents, sigma_current.unsqueeze(0), **extra_args)

            # 2. Calculate the derivative 'd' (direction pointing from noisy latents towards denoised prediction)
            # d = (x_t - x0_hat) / sigma_t
            d = (latents - denoised_prediction) / sigma_current
            
            # 3. Perform the Euler step to get latents for the next sigma level
            # x_{t+1} = x_t + d_t * (sigma_{t+1} - sigma_t)
            dt = sigma_next - sigma_current # Change in sigma for this step (will be negative)
            latents = latents + d * dt
            
            # Invoke the callback for previews (if provided)
            if callback is not None:
                # The 'callback' function provided by KSAMPLER (often named k_callback internally)
                # expects a single dictionary argument. This dictionary should contain keys like
                # 'i' (current step, 0-indexed), 'denoised' (the model's x0 prediction),
                # 'x' (current latents after the step), and 'sigma' (the full sigmas_schedule tensor).
                callback_payload = {
                    'i': i,                            # Current step index (0-indexed)
                    'denoised': denoised_prediction,   # The D(x) prediction
                    'x': latents,                      # The current latents x_t (after this step's update)
                    'sigma': sigmas_schedule,          # The full sigma schedule tensor
                    #'sigma_hat': sigma_current       # Optionally, current sigma_t (though k_callback might not use it directly)
                }
                callback(callback_payload) # Pass the dictionary to KSAMPLER's k_callback
        
        if not disable:
            print("Wolf Custom Euler Sampler: Sampling loop completed.")
        return latents # Return the final denoised latents

    return wolf_custom_euler_sampler
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
            }
        }

    def evaluate_sampler_script(self, script):  # seed parameter removed
        script_locals = {}
        script_globals = {
            "torch": torch,
            "comfy": comfy,
            "nodes": nodes,
            "k_diffusion_sampling": k_diffusion_sampling_global,  # Make it available to the script
            # "__script_seed__": seed, # Removed
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
