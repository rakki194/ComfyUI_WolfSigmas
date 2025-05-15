import torch
import math


class WolfSigmaCosineLogSNR8Step:
    """
    Generates an 8-step (9 sigmas) schedule where sigmas are spaced
    according to a cosine curve in the log-sigma domain.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_cosine_log_snr_8_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/8_step"

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

    def get_cosine_log_snr_8_step_sigmas(self, sigma_max, sigma_min_positive):
        steps_intervals = 8
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

        # t goes from 0 to 1
        t = torch.linspace(0, 1, num_sigmas_to_generate, device="cpu")
        # cosine_t maps t from 0..1 to 0..1 with cosine easing (0 at start, 1 at end)
        cosine_t = 0.5 * (1.0 - torch.cos(t * math.pi))

        # Interpolate in log space: log_sigma_max down to log_sigma_min
        # (1.0 - cosine_t) goes from 1 down to 0
        log_sigmas = log_sigma_min + (1.0 - cosine_t) * (log_sigma_max - log_sigma_min)

        final_sigmas = torch.zeros(steps_intervals + 1, dtype=torch.float32)
        final_sigmas[:num_sigmas_to_generate] = torch.exp(log_sigmas)
        final_sigmas[steps_intervals] = 0.0

        # Ensure strict decrease and positivity for the non-zero part
        for i in range(num_sigmas_to_generate - 1):
            if final_sigmas[i] <= final_sigmas[i + 1] + 1e-7:
                final_sigmas[i + 1] = final_sigmas[i] * 0.95
            if final_sigmas[i].item() <= 0:
                final_sigmas[i] = torch.tensor(1e-5)  # Ensure positive

        if num_sigmas_to_generate > 0:
            final_sigmas[num_sigmas_to_generate - 1] = max(
                final_sigmas[num_sigmas_to_generate - 1], current_sigma_min_positive
            )
            if (
                final_sigmas[num_sigmas_to_generate - 1].item() <= 1e-9
                and current_sigma_min_positive > 1e-9
            ):
                final_sigmas[num_sigmas_to_generate - 1] = torch.tensor(
                    current_sigma_min_positive
                )
            elif final_sigmas[num_sigmas_to_generate - 1].item() <= 1e-9:
                final_sigmas[num_sigmas_to_generate - 1] = torch.tensor(1e-5)

        final_sigmas[steps_intervals] = 0.0
        return (final_sigmas.cpu(),)
