import torch
import numpy as np
import math
import traceback


class WolfSigmaScriptEvaluator:
    """
    Evaluates a Python script to generate a sigma schedule.
    The script has access to num_steps, sigma_max, sigma_min_positive, and min_epsilon_spacing.
    It must assign a list or NumPy array of num_steps floats to the variable 'active_sigmas'.
    The node then post-processes these sigmas.
    """

    RETURN_TYPES = ("SIGMAS", "STRING")
    RETURN_NAMES = ("SIGMAS", "error_log")
    FUNCTION = "evaluate_script"
    CATEGORY = "sampling/sigmas_wolf/script"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "script_code": (
                    "STRING",
                    {"multiline": True, "default": s.get_default_script()},
                ),
                "num_steps": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001},
                ),
                "sigma_min_positive": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0001, "max": 100.0, "step": 0.0001},
                ),
                "min_epsilon_spacing": (
                    "FLOAT",
                    {
                        "default": 1e-7,
                        "min": 1e-9,
                        "max": 0.1,
                        "step": 1e-7,
                        "precision": 9,
                    },
                ),
            }
        }

    @classmethod
    def get_default_script(cls):
        return """# Example script for WolfSigmaScriptEvaluator
# Available variables: num_steps, sigma_max, sigma_min_positive, min_epsilon_spacing
# Must assign a list or numpy array of length `num_steps` to `active_sigmas`.

import numpy as np

if num_steps == 1:
    # For a single active step, this value will be used.
    # The node ensures it's clamped between sigma_max and sigma_min_positive.
    active_sigmas = [sigma_max] 
else:
    # Linear spacing example:
    active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()

# Karras-like spacing example (comment out linear above if using this):
# rho = 7.0
# inv_rho_min = sigma_min_positive ** (1.0/rho)
# inv_rho_max = sigma_max ** (1.0/rho)
# t_values = np.linspace(0, 1, num_steps)
# active_sigmas = ((inv_rho_max + t_values * (inv_rho_min - inv_rho_max)) ** rho).tolist()
# if num_steps > 0:
#    active_sigmas[0] = sigma_max # Ensure endpoints if script logic might miss
#    active_sigmas[-1] = sigma_min_positive
"""

    def evaluate_script(
        self, script_code, num_steps, sigma_max, sigma_min_positive, min_epsilon_spacing
    ):
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        epsilon = float(min_epsilon_spacing)
        error_log = ""

        if N < 1:
            error_log = "Error: num_steps must be at least 1."
            return (torch.tensor([s_max, 0.0], dtype=torch.float32), error_log)

        # Ensure s_max is greater than s_min_pos, and s_min_pos is positive
        if s_min_pos <= epsilon:
            s_min_pos = epsilon
        if s_max <= s_min_pos:
            s_max = s_min_pos + N * epsilon
            if (
                s_max <= s_min_pos
            ):  # If s_min_pos was epsilon and N*epsilon is too small
                s_max = s_min_pos + epsilon * 10  # A bit more robust for small N

        script_globals = {"math": math, "np": np, "torch": torch}
        local_namespace = {
            "num_steps": N,
            "sigma_max": s_max,
            "sigma_min_positive": s_min_pos,
            "min_epsilon_spacing": epsilon,
            "active_sigmas": None,  # Script must populate this
        }

        try:
            exec(script_code, script_globals, local_namespace)
            script_generated_sigmas = local_namespace.get("active_sigmas")

            if script_generated_sigmas is None:
                raise ValueError("'active_sigmas' variable not set by the script.")

            if isinstance(script_generated_sigmas, np.ndarray):
                script_generated_sigmas = script_generated_sigmas.tolist()

            if not isinstance(script_generated_sigmas, list) or not all(
                isinstance(x, (int, float)) for x in script_generated_sigmas
            ):
                raise ValueError("'active_sigmas' must be a list of numbers.")

            if len(script_generated_sigmas) != N:
                raise ValueError(
                    f"'active_sigmas' length must be equal to num_steps ({N}), but got {len(script_generated_sigmas)}."
                )

            active_tensor = torch.tensor(script_generated_sigmas, dtype=torch.float32)
            error_log = "Script executed successfully."

        except Exception as e:
            error_log = f"Error during script execution or validation:\n{traceback.format_exc()}"
            # Fallback to a simple linear schedule on error
            active_tensor = (
                torch.linspace(s_max, s_min_pos, N)
                if N > 1
                else torch.tensor([s_max], dtype=torch.float32)
            )
            if N == 0:  # Should be caught by N < 1 check, but as fallback safety
                active_tensor = torch.empty(0)

        # Post-processing the active_tensor
        final_sigmas = torch.zeros(N + 1, dtype=torch.float32)

        if N == 1:
            val = active_tensor[0].item() if len(active_tensor) > 0 else s_max
            final_sigmas[0] = max(min(val, s_max), s_min_pos)
        elif N > 1:
            active_tensor[0] = s_max
            active_tensor[N - 1] = s_min_pos

            for i in range(N - 2, -1, -1):
                current_val = active_tensor[i].item()
                next_val = active_tensor[i + 1].item()
                active_tensor[i] = max(current_val, next_val + epsilon)
                active_tensor[i] = min(active_tensor[i].item(), s_max)

            active_tensor[0] = s_max
            for i in range(N - 1):
                current_val = active_tensor[i].item()
                next_val = active_tensor[i + 1].item()
                active_tensor[i + 1] = min(next_val, current_val - epsilon)
                active_tensor[i + 1] = max(active_tensor[i + 1].item(), s_min_pos)

            active_tensor[0] = s_max
            active_tensor[N - 1] = max(
                min(active_tensor[N - 1].item(), active_tensor[N - 2].item() - epsilon),
                s_min_pos,
            )

        if N > 0:
            final_sigmas[:N] = active_tensor
        final_sigmas[N] = 0.0

        return (final_sigmas, error_log)
