import torch

# import math


class WolfSigmaShiftAndScale:
    """
    Applies a global shift and scale to all sigmas in a schedule.
    Sigmas = (Sigmas + shift) * scale.
    The final 0.0 sigma, if present, remains unchanged.
    Monotonicity (decreasing order) and non-negativity are enforced.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "shift_and_scale_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transform"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_in": ("SIGMAS",),
                "shift": (
                    "FLOAT",
                    {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
                "scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
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

    def shift_and_scale_sigmas(self, sigmas_in, shift, scale, min_epsilon_spacing):
        sigmas = sigmas_in.clone().cpu()
        shift_val = float(shift)
        scale_val = float(scale)
        epsilon = float(min_epsilon_spacing)

        if len(sigmas) == 0:
            return (sigmas,)

        has_final_zero = sigmas[-1].item() == 0.0
        active_sigmas_count = (
            len(sigmas) - 1 if has_final_zero and len(sigmas) > 1 else len(sigmas)
        )

        for i in range(active_sigmas_count):
            sigmas[i] = (sigmas[i].item() + shift_val) * scale_val
            sigmas[i] = max(
                0.0, sigmas[i].item()
            )  # Ensure non-negative after transform

        # Restore final zero if it was there
        if has_final_zero and len(sigmas) > 1:
            sigmas[-1] = 0.0

        # Enforce monotonicity and ensure the value before 0.0 is > 0
        for i in range(len(sigmas) - 1):
            is_last_active_sigma = (i == len(sigmas) - 2) and has_final_zero

            if sigmas[i].item() <= sigmas[i + 1].item() + (
                0 if is_last_active_sigma and sigmas[i + 1].item() == 0 else epsilon
            ):
                if (
                    is_last_active_sigma and sigmas[i + 1].item() == 0.0
                ):  # Current is the one before final 0
                    sigmas[i] = max(epsilon, sigmas[i].item())  # Must be > 0
                    if sigmas[i].item() == 0.0:
                        sigmas[i] = epsilon  # if it was forced to 0 by shift/scale
                else:  # Not the one before final zero, or final zero not present
                    sigmas[i + 1] = sigmas[i].item() - epsilon

            sigmas[i + 1] = max(0.0, sigmas[i + 1].item())

        # Final check for the sigma before a potential 0.0
        if len(sigmas) > 1 and sigmas[-1].item() == 0.0 and sigmas[-2].item() <= 0.0:
            sigmas[-2] = epsilon
            if len(sigmas) > 2 and sigmas[-2].item() >= sigmas[-3].item() - epsilon:
                new_val = sigmas[-3].item() / 2.0
                sigmas[-2] = max(epsilon, new_val)  # ensure positive and smaller
            elif len(sigmas) == 2:  # only [val, 0], val must be > 0
                sigmas[-2] = max(epsilon, sigmas[-2].item())

        return (sigmas,)
