import torch


class WolfSigmaOneStepChromaImbalanced:
    """
    Provides an adjustable 1-step sigma schedule [start_sigma, end_sigma]
    with an 'imbalance_factor' to heuristically push the start or end sigma.
    Inspired by Unbalanced Optimal Transport concepts.
    - Positive imbalance: increases start_sigma (bias to "create more").
    - Negative imbalance: increases end_sigma (bias to "destroy less").
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_imbalanced_one_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_start_sigma": (
                    "FLOAT",
                    {"default": 0.992, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "base_end_sigma": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "imbalance_factor": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "scale_factor": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
            }
        }

    def get_imbalanced_one_step_sigmas(
        self, base_start_sigma, base_end_sigma, imbalance_factor, scale_factor
    ):
        start_sigma = float(base_start_sigma)
        end_sigma = float(base_end_sigma)

        if imbalance_factor > 0:
            increase = imbalance_factor * scale_factor
            start_sigma += increase
        elif imbalance_factor < 0:
            increase = abs(imbalance_factor) * scale_factor
            end_sigma += increase

        # Ensure sigmas are non-negative
        start_sigma = max(0.0, start_sigma)
        end_sigma = max(0.0, end_sigma)

        # Ensure start_sigma is strictly greater than end_sigma for a valid denoising step
        if start_sigma <= end_sigma:
            start_sigma = end_sigma + 0.0001  # Ensure a minimal positive step

        # Final check for [0,0] sigmas if all inputs somehow led to it
        if start_sigma == 0.0 and end_sigma == 0.0:
            start_sigma = 0.0001  # Default to a tiny valid step

        sigmas = torch.tensor([start_sigma, end_sigma], dtype=torch.float32)
        return (sigmas,)
