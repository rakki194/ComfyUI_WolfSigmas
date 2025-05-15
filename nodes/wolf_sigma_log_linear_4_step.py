import torch
import math


class WolfSigmaLogLinear4Step:
    """
    Generates a 4-step log-linear schedule.
    Produces 5 sigma values: log-spaced from sigma_max to sigma_min_positive, then 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_log_linear_4_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/4_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
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
                "sigma_min_positive": (
                    "FLOAT",
                    {
                        "default": 0.002,
                        "min": 0.0001,
                        "max": 10.0,
                        "step": 0.0001,
                        "round": False,
                    },
                ),
            }
        }

    def get_log_linear_4_step_sigmas(self, sigma_max, sigma_min_positive):
        steps_intervals = 4
        num_sigmas_to_generate = steps_intervals

        current_sigma_max = float(sigma_max)
        current_sigma_min_positive = float(sigma_min_positive)

        if current_sigma_max <= current_sigma_min_positive:
            if current_sigma_min_positive > 1e-5:
                current_sigma_max = current_sigma_min_positive * 1.1
            else:
                current_sigma_min_positive = 1e-5
                current_sigma_max = 1e-4
        elif current_sigma_min_positive <= 1e-9:
            current_sigma_min_positive = 1e-5
            if current_sigma_max < current_sigma_min_positive:
                current_sigma_max = current_sigma_min_positive * 10

        log_sigma_max = math.log(current_sigma_max)
        log_sigma_min = math.log(current_sigma_min_positive)

        log_spaced_sigmas = torch.linspace(
            log_sigma_max, log_sigma_min, num_sigmas_to_generate
        )

        final_sigmas = torch.zeros(steps_intervals + 1, dtype=torch.float32)
        final_sigmas[:num_sigmas_to_generate] = torch.exp(log_spaced_sigmas)
        final_sigmas[steps_intervals] = 0.0

        for i in range(num_sigmas_to_generate - 1):
            if (
                final_sigmas[i] <= final_sigmas[i + 1] + 1e-7
            ):  # Add epsilon for strict decrease
                final_sigmas[i + 1] = final_sigmas[i] * 0.95
            if final_sigmas[i].item() <= 0:
                final_sigmas[i] = torch.tensor(1e-5)  # Ensure positive

        if num_sigmas_to_generate > 0:
            final_sigmas[num_sigmas_to_generate - 1] = max(
                final_sigmas[num_sigmas_to_generate - 1], current_sigma_min_positive
            )
            if (
                final_sigmas[num_sigmas_to_generate - 1].item() <= 0.0
                and current_sigma_min_positive > 0
            ):
                final_sigmas[num_sigmas_to_generate - 1] = torch.tensor(
                    current_sigma_min_positive
                )
            elif (
                final_sigmas[num_sigmas_to_generate - 1].item() <= 0.0
                and current_sigma_min_positive == 0.0
            ):  # If sigma_min is 0, last non-zero sigma must be > 0
                final_sigmas[num_sigmas_to_generate - 1] = torch.tensor(1e-5)

        final_sigmas[steps_intervals] = 0.0
        return (final_sigmas,)
