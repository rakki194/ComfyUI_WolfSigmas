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

    _LAST_RUN_STATS = None  # Class variable to store stats from the last run

    _DEFAULT_SCRIPT_SIMPLE = """
# ComfyUI Wolf Simple Sampler Script Evaluator
#
# This script directly implements the sampling loop.
# The following variables are pre-defined and available for you to use:
#   model, x_initial, sigmas_schedule, extra_args, callback, disable, sampler_options
#
# Your script MUST assign the final denoised latents to a variable named 'latents'.
# It SHOULD also populate a dictionary named 'collected_stats_output' with timeseries data for plotting.

import torch

# --- User Configuration for Logging ---
LOG_TO_CONSOLE = True  # Set to False to disable console logging of stats
CONSOLE_LOG_STEP_INTERVAL = 1    # Log stats every N steps. 1 means every step.
# --- End User Configuration ---

# --- Data Collection for Plotting ---
# This dictionary will be populated and made available to the calling node.
collected_stats_output = {
    "steps": [],
    "sigmas_current": [],
    "sigmas_next": [],
    "latent_mean_xi": [],       # Mean of x_i (input to model for current step)
    "latent_std_xi": [],        # Std of x_i
    "denoised_mean_x0_pred": [],# Mean of x0_pred_i (model's prediction for current step)
    "denoised_std_x0_pred": [], # Std of x0_pred_i
}
# --- End Data Collection ---

current_latents = x_initial.clone() # This is x_0 (initial noisy latent)
num_sampling_steps = len(sigmas_schedule) - 1

if LOG_TO_CONSOLE:
    print(f"[WolfSimpleSampler] Starting sampling. Total steps: {num_sampling_steps}")
    if num_sampling_steps > 0: # Only log initial if there are steps to take
        initial_mean = current_latents.mean().item()
        initial_std = current_latents.std().item()
        print(f"[WolfSimpleSampler] Initial Latent (x_0): Mean={initial_mean:.4f}, Std={initial_std:.4f}")

for i in range(num_sampling_steps):
    sigma_current = sigmas_schedule[i]    # sigma_i
    sigma_next = sigmas_schedule[i+1]     # sigma_{i+1}
    
    # current_latents is x_i at this point (latent input for model for step i)
    
    # Log stats for x_i (input to model for this step)
    latent_mean_val_xi = current_latents.mean().item()
    latent_std_val_xi = current_latents.std().item()

    # Get model's prediction for x0 based on x_i and sigma_i
    denoised_prediction = model(current_latents, sigma_current.unsqueeze(0), **extra_args) # x0_pred_i
    
    # Log stats for x0_pred_i
    denoised_mean_val_x0_pred_i = denoised_prediction.mean().item()
    denoised_std_val_x0_pred_i = denoised_prediction.std().item()
    
    current_step_for_log = i + 1 # 1-indexed step number for logging
    
    # Console Logging
    if LOG_TO_CONSOLE and current_step_for_log % CONSOLE_LOG_STEP_INTERVAL == 0:
        print(f"[WolfSimpleSampler] Step {current_step_for_log}/{num_sampling_steps}: Sigma_i={sigma_current.item():.4f} -> Sigma_{'{i+1}'}={sigma_next.item():.4f}")
        print(f"    Latent (x_i input): Mean={latent_mean_val_xi:.4f}, Std={latent_std_val_xi:.4f}")
        print(f"    Denoised (x0_pred_i): Mean={denoised_mean_val_x0_pred_i:.4f}, Std={denoised_std_val_x0_pred_i:.4f}")

    # Store data for plotting
    collected_stats_output["steps"].append(current_step_for_log)
    collected_stats_output["sigmas_current"].append(sigma_current.item())
    collected_stats_output["sigmas_next"].append(sigma_next.item())
    collected_stats_output["latent_mean_xi"].append(latent_mean_val_xi)
    collected_stats_output["latent_std_xi"].append(latent_std_val_xi)
    collected_stats_output["denoised_mean_x0_pred"].append(denoised_mean_val_x0_pred_i)
    collected_stats_output["denoised_std_x0_pred"].append(denoised_std_val_x0_pred_i)
    
    # Euler step: Calculate x_{i+1} from x_i and x0_pred_i
    d = (current_latents - denoised_prediction) / sigma_current 
    dt = sigma_next - sigma_current
    current_latents_next_step = current_latents + d * dt 
    
    current_latents = current_latents_next_step # current_latents is now x_{i+1} for the next iteration
    
    # Callback (if provided by KSampler)
    if callback is not None:
        callback_dict = {
            'i': i,                                 # Current step index (0 to N-1)
            'denoised': denoised_prediction,        # Model's prediction of x0 (x0_pred_i)
            'x': current_latents,                   # Current noisy latent (now x_{i+1})
            'sigma': sigmas_schedule,               # Full sigma schedule
            'sigma_hat': sigma_current,             # Current sigma_i used for denoising
            'sigma_next': sigma_next if i < num_sampling_steps -1 else torch.tensor(0.0) # Next sigma_{i+1}
        }
        callback(callback_dict)
    # --- End Logging and Data Collection for step i ---

if LOG_TO_CONSOLE and num_sampling_steps > 0: # Only log final if steps were taken
    final_mean = current_latents.mean().item()
    final_std = current_latents.std().item()
    print(f"[WolfSimpleSampler] Final Latent (output x_N): Mean={final_mean:.4f}, Std={final_std:.4f}")

if num_sampling_steps == 0: # If no steps (e.g. total_steps = 0 for some schedulers)
    if LOG_TO_CONSOLE:
        print(f"[WolfSimpleSampler] No sampling steps taken. Returning initial latents.")

print(f"[WolfSimpleSampler] Sampling finished.")
latents = current_latents # This is x_N (final denoised latent)
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
                "collected_stats_output": None,  # Ensure it's in locals for script to assign
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

            # Store collected stats
            WolfSimpleSamplerScriptEvaluator._LAST_RUN_STATS = script_locals.get(
                "collected_stats_output"
            )

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
