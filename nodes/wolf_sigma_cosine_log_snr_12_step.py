import torch
import math


class WolfSigmaCosineLogSNR12Step:
    """
    Generates a 12-step (13 sigmas) schedule where sigmas are spaced
    according to a cosine curve in the log-sigma domain.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_cosine_log_snr_12_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step"

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

    def get_cosine_log_snr_12_step_sigmas(
        self, sigma_max, sigma_min_positive, min_epsilon_spacing
    ):
        steps_intervals = 12
        num_active_sigmas = steps_intervals  # N active sigmas for N intervals
        epsilon = float(min_epsilon_spacing)

        current_sigma_max = float(sigma_max)
        current_sigma_min_positive = float(sigma_min_positive)

        # Ensure valid ranges
        if current_sigma_max <= current_sigma_min_positive + epsilon * steps_intervals:
            current_sigma_max = current_sigma_min_positive + epsilon * (
                steps_intervals + 1
            )  # ensure gap
        if current_sigma_min_positive <= epsilon:
            current_sigma_min_positive = epsilon
        if (
            current_sigma_max <= current_sigma_min_positive
        ):  # final check if previous adjustment failed
            current_sigma_max = (
                current_sigma_min_positive * 2
            )  # arbitrary multiplier if still bad
            if current_sigma_max <= current_sigma_min_positive:  # if min_pos is huge
                current_sigma_max = current_sigma_min_positive + 1.0

        log_sigma_max = math.log(current_sigma_max)
        log_sigma_min = math.log(current_sigma_min_positive)

        t = torch.linspace(0, 1, num_active_sigmas, device="cpu")
        # cosine_t goes from 0 (at t=0) to 1 (at t=1)
        cosine_t = 0.5 * (1.0 - torch.cos(t * math.pi))

        # Interpolate in log space such that sigmas go from max down to min
        # So, when cosine_t is 0 (start), we want log_sigma_max.
        # When cosine_t is 1 (end), we want log_sigma_min.
        log_sigmas_calc = log_sigma_max + cosine_t * (log_sigma_min - log_sigma_max)
        # This means: log_sigmas_calc = (1-cosine_t)*log_sigma_max + cosine_t*log_sigma_min

        final_sigmas = torch.zeros(
            steps_intervals + 1, dtype=torch.float32, device="cpu"
        )
        if num_active_sigmas > 0:
            calculated_active_sigmas = torch.exp(log_sigmas_calc)
            # Ensure correct order due to potential floating point issues with exp
            # We expect calculated_active_sigmas[0] to be sigma_max and calculated_active_sigmas[-1] to be sigma_min_pos
            final_sigmas[:num_active_sigmas] = calculated_active_sigmas
        final_sigmas[steps_intervals] = 0.0  # Last sigma is 0.0

        # Initial boundary enforcement, can be important if exp caused slight deviations
        if num_active_sigmas > 0:
            final_sigmas[0] = current_sigma_max
            final_sigmas[num_active_sigmas - 1] = max(
                final_sigmas[num_active_sigmas - 1].item(), current_sigma_min_positive
            )

        # Robust post-processing loop (similar to other 12-step nodes)
        final_sigmas[0] = current_sigma_max
        for i in range(steps_intervals - 1):  # Iterate 0 to steps_intervals-2
            # Lower bound for next sigma is current_sigma_min_positive (or epsilon if larger)
            lower_bound_for_next = max(current_sigma_min_positive, epsilon)
            final_sigmas[i + 1] = torch.clamp(
                final_sigmas[i + 1],
                min=lower_bound_for_next,
                max=final_sigmas[i].item() - epsilon,
            )

        # Ensure last active sigma (index steps_intervals-1) is correctly bounded
        if steps_intervals > 0:
            final_sigmas[steps_intervals - 1] = max(current_sigma_min_positive, epsilon)
            if steps_intervals > 1:  # Must be less than the one before it
                final_sigmas[steps_intervals - 1] = min(
                    final_sigmas[steps_intervals - 1].item(),
                    final_sigmas[steps_intervals - 2].item() - epsilon,
                )
                # And re-assert lower bound after potential change
                final_sigmas[steps_intervals - 1] = max(
                    final_sigmas[steps_intervals - 1].item(),
                    current_sigma_min_positive,
                    epsilon,
                )

        final_sigmas[steps_intervals] = 0.0  # Final element must be 0

        return (final_sigmas.cpu(),)
