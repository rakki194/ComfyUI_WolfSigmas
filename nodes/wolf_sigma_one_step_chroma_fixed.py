import torch


class WolfSigmaOneStepChromaFixed:
    """
    Provides fixed sigmas [0.992, 0.0] specifically for 1-step Chroma inference.
    0.992 is based on the initial noise level for Chroma in OptimalStepsScheduler.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_fixed_one_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}  # No inputs needed as it's fixed

    def get_fixed_one_step_sigmas(self):
        sigmas = torch.tensor([0.992, 0.0], dtype=torch.float32)
        return (sigmas,)
