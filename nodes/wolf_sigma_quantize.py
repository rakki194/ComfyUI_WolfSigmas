import torch
import math  # For ceil/floor if used, or precision control


class WolfSigmaQuantize:
    """
    Quantizes sigmas in a schedule to a specific number of decimal places
    or to the nearest multiple of a quantization step.
    The final 0.0 sigma, if present, remains unchanged.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "quantize_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transform"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_in": ("SIGMAS",),
                "quantization_method": (
                    ["decimal_places", "step_multiple"],
                    {"default": "decimal_places"},
                ),
                "decimal_places": (
                    "INT",
                    {"default": 3, "min": 0, "max": 10, "step": 1},
                ),
                "quantization_step": (
                    "FLOAT",
                    {"default": 0.001, "min": 1e-7, "max": 10.0, "step": 0.0001},
                ),
                "rounding_mode": (["round", "floor", "ceil"], {"default": "round"}),
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

    def quantize_sigmas(
        self,
        sigmas_in,
        quantization_method,
        decimal_places,
        quantization_step,
        rounding_mode,
        min_epsilon_spacing,
    ):
        sigmas = sigmas_in.clone().cpu()
        epsilon = float(min_epsilon_spacing)

        if len(sigmas) == 0:
            return (sigmas,)

        has_final_zero = sigmas[-1].item() == 0.0
        active_sigmas_count = (
            len(sigmas) - 1 if has_final_zero and len(sigmas) > 1 else len(sigmas)
        )

        for i in range(active_sigmas_count):
            val = sigmas[i].item()
            quantized_val = val

            if quantization_method == "decimal_places":
                if decimal_places < 0:
                    decimal_places = 0
                multiplier = 10**decimal_places
                if rounding_mode == "round":
                    quantized_val = round(val * multiplier) / multiplier
                elif rounding_mode == "floor":
                    quantized_val = math.floor(val * multiplier) / multiplier
                elif rounding_mode == "ceil":
                    quantized_val = math.ceil(val * multiplier) / multiplier
            elif quantization_method == "step_multiple":
                if quantization_step <= 1e-9:
                    quantization_step = 1e-9  # Avoid division by zero or too small step
                if rounding_mode == "round":
                    quantized_val = round(val / quantization_step) * quantization_step
                elif rounding_mode == "floor":
                    quantized_val = (
                        math.floor(val / quantization_step) * quantization_step
                    )
                elif rounding_mode == "ceil":
                    quantized_val = (
                        math.ceil(val / quantization_step) * quantization_step
                    )

            sigmas[i] = max(0.0, quantized_val)  # Ensure non-negative

        if has_final_zero and len(sigmas) > 1:
            sigmas[-1] = 0.0

        # Enforce monotonicity
        for i in range(len(sigmas) - 1):
            is_last_active_sigma = (i == len(sigmas) - 2) and has_final_zero
            min_next_val_diff = (
                0 if is_last_active_sigma and sigmas[i + 1].item() == 0 else epsilon
            )

            if sigmas[i].item() <= sigmas[i + 1].item() + min_next_val_diff:
                if is_last_active_sigma and sigmas[i + 1].item() == 0.0:
                    current_val_quantized = sigmas[i].item()
                    # Must be > 0. If quantization made it 0, force to epsilon or original quantization step.
                    sigmas[i] = max(
                        current_val_quantized,
                        epsilon,
                        (
                            quantization_step
                            if quantization_method == "step_multiple"
                            else (
                                10**-decimal_places if decimal_places > 0 else epsilon
                            )
                        ),
                    )
                    if sigmas[i].item() == 0.0:  # If all were zero
                        sigmas[i] = epsilon

                else:
                    # Try to adjust the next one down, respecting its own quantization potential
                    # This is tricky. A simpler way: ensure s[i] > s[i+1] by pushing s[i+1] down.
                    new_next_val = sigmas[i].item() - epsilon
                    # Re-quantize new_next_val if needed, or just set it
                    # For simplicity, just set it and ensure it's non-negative
                    sigmas[i + 1] = max(0.0, new_next_val)

            sigmas[i + 1] = max(0.0, sigmas[i + 1].item())

        # Final check for the sigma before a potential 0.0
        if len(sigmas) > 1 and sigmas[-1].item() == 0.0 and sigmas[-2].item() <= 0.0:
            min_val_before_zero = max(
                epsilon,
                (
                    quantization_step
                    if quantization_method == "step_multiple"
                    else (10**-decimal_places if decimal_places >= 0 else epsilon)
                ),
            )
            sigmas[-2] = min_val_before_zero
            if (
                len(sigmas) > 2 and sigmas[-2].item() >= sigmas[-3].item() - epsilon
            ):  # If this broke previous monotonicity
                new_val = sigmas[-3].item() / 2.0  # Could try to quantize this too
                sigmas[-2] = max(min_val_before_zero, new_val)
            elif len(sigmas) == 2:  # only [val, 0]
                sigmas[-2] = max(min_val_before_zero, sigmas[-2].item())

        return (sigmas,)
