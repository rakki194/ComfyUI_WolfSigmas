import comfy.k_diffusion.sampling as k_diffusion_sampling


class WolfSigmaChromaKarras12Step:
    """
    Generates a 12-step Karras schedule using Chroma-like sigma parameters.
    Produces 13 sigma values for 12 sampling steps.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_chroma_karras_12_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step_chroma"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.002,  # Chroma default
                        "min": 0.0001,
                        "max": 100.0,
                        "step": 0.0001,
                        "round": False,
                    },
                ),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": 80.0,  # Chroma default
                        "min": 0.1,
                        "max": 1000.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
                "rho": (
                    "FLOAT",
                    {
                        "default": 7.0,  # Chroma default
                        "min": 0.1,
                        "max": 100.0,
                        "step": 0.1,
                        "round": False,
                    },
                ),
            }
        }

    def get_chroma_karras_12_step_sigmas(self, sigma_min, sigma_max, rho):
        steps_intervals = 12
        sigmas = k_diffusion_sampling.get_sigmas_karras(
            n=steps_intervals,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            rho=float(rho),
            device="cpu",
        )
        return (sigmas,)
