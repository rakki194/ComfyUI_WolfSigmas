import comfy.k_diffusion.sampling as k_diffusion_sampling


class WolfSigmaChromaKarras8Step:
    """
    Generates an 8-step Karras schedule using Chroma's default sigma parameters.
    Produces 9 sigma values for 8 sampling steps.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_chroma_karras_8_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/8_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.002,
                        "min": 0.0001,
                        "max": 100.0,
                        "step": 0.0001,
                        "round": False,
                    },
                ),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 1000.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "rho": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
            }
        }

    def get_chroma_karras_8_step_sigmas(self, sigma_min, sigma_max, rho):
        steps_intervals = 8
        sigmas = k_diffusion_sampling.get_sigmas_karras(
            n=steps_intervals,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device="cpu",
        )
        return (sigmas,)
