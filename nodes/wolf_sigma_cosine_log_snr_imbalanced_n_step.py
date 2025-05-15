import torch
import math


class WolfSigmaCosineLogSNRImbalancedNStep:
    """
    Generates an N-step (N+1 sigmas) schedule with bias.
    Sigmas are spaced according to a biased cosine curve in log-sigma domain.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_imbalanced_cosine_log_snr_n_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/N_step_imbalanced"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_steps": (
                    "INT",
                    {"default": 12, "min": 1, "max": 1000},
                ),
                "sigma_max": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 1000.0,
                        "step": 0.1,
                    },
                ),
                "sigma_min_positive": (
                    "FLOAT",
                    {
                        "default": 0.002,
                        "min": 0.0001,
                        "max": 10.0,
                        "step": 0.0001,
                    },
                ),
                "bias_factor": (
                    "FLOAT",
                    {"default": 0.0, "min": -0.9, "max": 0.9, "step": 0.05},
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

    def get_imbalanced_cosine_log_snr_n_step_sigmas(
        self, num_steps, sigma_max, sigma_min_positive, bias_factor, min_epsilon_spacing
    ):
        steps_intervals = int(num_steps)
        num_active_sigmas = steps_intervals
        epsilon = float(min_epsilon_spacing)
        bias = float(bias_factor)

        current_sigma_max = float(sigma_max)
        current_sigma_min_positive = float(sigma_min_positive)

        if steps_intervals <= 0:
            return (torch.tensor([current_sigma_max, 0.0], dtype=torch.float32),)

        if current_sigma_max <= current_sigma_min_positive + epsilon * steps_intervals:
            current_sigma_max = current_sigma_min_positive + epsilon * (
                steps_intervals + 1
            )
        if current_sigma_min_positive <= epsilon:
            current_sigma_min_positive = epsilon
        if current_sigma_max <= current_sigma_min_positive:
            current_sigma_max = current_sigma_min_positive * 2
            if current_sigma_max <= current_sigma_min_positive:
                current_sigma_max = current_sigma_min_positive + 1.0

        log_sigma_max = math.log(current_sigma_max)
        log_sigma_min = math.log(current_sigma_min_positive)

        t_linear = torch.linspace(0, 1, num_active_sigmas, device="cpu")
        power = 1.0 - bias
        if power <= 0:
            power = epsilon
        t_biased = t_linear.pow(power)

        cosine_t = 0.5 * (1.0 - torch.cos(t_biased * math.pi))
        log_sigmas_calc = log_sigma_max + cosine_t * (log_sigma_min - log_sigma_max)

        final_sigmas = torch.zeros(
            steps_intervals + 1, dtype=torch.float32, device="cpu"
        )
        if num_active_sigmas > 0:
            final_sigmas[:num_active_sigmas] = torch.exp(log_sigmas_calc)
        final_sigmas[steps_intervals] = 0.0

        if num_active_sigmas > 0:
            final_sigmas[0] = current_sigma_max
            final_sigmas[num_active_sigmas - 1] = max(
                final_sigmas[num_active_sigmas - 1].item(), current_sigma_min_positive
            )

        final_sigmas[0] = current_sigma_max
        for i in range(steps_intervals - 1):
            lower_bound_for_next = max(current_sigma_min_positive, epsilon)
            final_sigmas[i + 1] = torch.clamp(
                final_sigmas[i + 1],
                min=lower_bound_for_next,
                max=final_sigmas[i].item() - epsilon,
            )

        if steps_intervals > 0:
            final_sigmas[steps_intervals - 1] = max(current_sigma_min_positive, epsilon)
            if steps_intervals > 1:
                final_sigmas[steps_intervals - 1] = min(
                    final_sigmas[steps_intervals - 1].item(),
                    final_sigmas[steps_intervals - 2].item() - epsilon,
                )
                final_sigmas[steps_intervals - 1] = max(
                    final_sigmas[steps_intervals - 1].item(),
                    current_sigma_min_positive,
                    epsilon,
                )

        final_sigmas[steps_intervals] = 0.0

        return (final_sigmas.cpu(),)
