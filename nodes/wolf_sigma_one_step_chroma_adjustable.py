import torch


class WolfSigmaOneStepChromaAdjustable:
    """
    Provides an adjustable 1-step sigma schedule [start_sigma, end_sigma].
    Helpful for experimenting with 1-step inference for models like Chroma.
    Default start_sigma=0.992 is based on Chroma's typical initial noise level.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_adjustable_one_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_sigma": (
                    "FLOAT",
                    {"default": 0.992, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "end_sigma": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
            }
        }

    def get_adjustable_one_step_sigmas(self, start_sigma, end_sigma):
        start_sigma = float(start_sigma)
        end_sigma = float(end_sigma)

        if start_sigma <= end_sigma:
            # Ensure start_sigma is greater than end_sigma
            start_sigma = end_sigma + 0.001  # Make start sigma slightly larger
            if (
                start_sigma <= 0.001 and end_sigma == 0.0
            ):  # handle case where both were effectively zero
                start_sigma = 0.001
                end_sigma = 0.0

        # Final check, make sure start sigma is not zero if end_sigma is zero, to avoid [0,0]
        if start_sigma == 0.0 and end_sigma == 0.0:
            start_sigma = 0.001  # Default to a tiny step

        sigmas = torch.tensor([start_sigma, end_sigma], dtype=torch.float32)
        return (sigmas,)
