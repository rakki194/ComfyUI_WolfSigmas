import torch

# import math


class WolfSigmaBiasedKarras8Step:
    """
    Generates an 8-step (9 sigmas) Karras schedule with a bias_power.
    bias_power > 1.0 concentrates steps towards sigma_max (larger early steps).
    bias_power < 1.0 concentrates steps towards sigma_min (smaller early steps).
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_biased_karras_8_step_sigmas"
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
                "bias_power": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.2, "max": 5.0, "step": 0.05},
                ),
            }
        }

    def get_biased_karras_8_step_sigmas(self, sigma_min, sigma_max, rho, bias_power):
        steps_intervals = 8
        num_points = steps_intervals + 1

        current_sigma_min = float(sigma_min)
        current_sigma_max = float(sigma_max)
        current_rho = float(rho)
        current_bias_power = float(bias_power)

        if current_sigma_max <= current_sigma_min:
            current_sigma_max = current_sigma_min + 0.1  # Ensure max > min
        if current_sigma_min <= 1e-9:
            current_sigma_min = 1e-9  # Karras needs positive sigma_min for calculation

        # Original Karras `t` goes from 0 to 1 for N points (N-1 intervals)
        # `i / (num_points - 1)` gives 0 for first sigma (sigma_max) and 1 for last non-zero sigma (sigma_min)
        # For sigmas that are decreasing, linspace from 0 (high sigma) to 1 (low sigma)
        t_norm = torch.linspace(
            0, 1, steps_intervals, device="cpu"
        )  # for the N-1 intervals, so N non-zero sigmas

        # Apply bias to the normalized time
        # If bias_power > 1, t_biased is smaller for t_norm < 1 (concentrates early values)
        # If bias_power < 1, t_biased is larger for t_norm < 1 (spreads early values)
        t_biased = t_norm**current_bias_power

        # Karras formula components
        inv_rho_min = current_sigma_min ** (1 / current_rho)
        inv_rho_max = current_sigma_max ** (1 / current_rho)

        # Calculate sigmas based on biased t
        # We want sigmas to go from sigma_max down to sigma_min
        # So, (1.0 - t_biased) maps [0,1] biased to [1,0] biased
        # sigs = (inv_rho_min + (1.0 - t_biased) * (inv_rho_max - inv_rho_min)) ** current_rho # This would be sigma_min to sigma_max
        sigs_calc = (
            inv_rho_max + t_biased * (inv_rho_min - inv_rho_max)
        ) ** current_rho

        final_sigmas = torch.zeros(num_points, dtype=torch.float32, device="cpu")
        final_sigmas[:steps_intervals] = sigs_calc
        final_sigmas[steps_intervals] = 0.0

        # Ensure first sigma is sigma_max and the one before last is >= sigma_min
        final_sigmas[0] = current_sigma_max
        if steps_intervals > 0:
            final_sigmas[steps_intervals - 1] = max(
                final_sigmas[steps_intervals - 1].item(), current_sigma_min
            )

        # Ensure strict decrease for the non-zero part
        for i in range(steps_intervals - 1):
            if final_sigmas[i] <= final_sigmas[i + 1] + 1e-7:
                final_sigmas[i + 1] = final_sigmas[i] * 0.98  # Ensure some decrease
            if final_sigmas[i].item() <= 0:
                final_sigmas[i] = torch.tensor(1e-5)
        if steps_intervals > 0 and final_sigmas[steps_intervals - 1].item() <= 0:
            final_sigmas[steps_intervals - 1] = torch.tensor(
                max(1e-5, current_sigma_min)
            )

        final_sigmas[steps_intervals] = 0.0
        return (final_sigmas.cpu(),)
