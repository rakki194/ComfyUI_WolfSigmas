import torch
import math  # For torch.exp if not using math.exp


class WolfSigmaSigmoid12Step:
    """
    Generates a 12-step schedule using a sigmoid curve.
    Produces 13 sigma values, from sigma_max down to sigma_min (potentially 0), then a final 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmoid_12_step_sigmas"
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
                "sigma_min_positive": (  # Renamed from sigma_min for clarity, must be > 0 for most uses
                    "FLOAT",
                    {
                        "default": 0.002,
                        "min": 0.0001,  # Avoid exactly 0 for this parameter if it implies the lowest active sigma
                        "max": 10.0,
                        "step": 0.0001,
                        "round": False,
                    },
                ),
                "steepness_factor": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1},
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

    def get_sigmoid_12_step_sigmas(
        self, sigma_max, sigma_min_positive, steepness_factor, min_epsilon_spacing
    ):
        steps_intervals = 12
        num_active_sigmas = steps_intervals
        epsilon = float(min_epsilon_spacing)

        current_sigma_max = float(sigma_max)
        # The sigmoid here is scaled between sigma_min_positive and sigma_max.
        # The original `sigma_min` could be 0. For active sigmas, we need a positive floor.
        current_sigma_min_active = float(sigma_min_positive)

        if current_sigma_max <= current_sigma_min_active + epsilon * steps_intervals:
            current_sigma_max = current_sigma_min_active + epsilon * (
                steps_intervals + 1
            )
        if current_sigma_min_active <= epsilon:
            current_sigma_min_active = epsilon

        # x_points for sigmoid: from steepness_factor down to -steepness_factor
        # This ensures sigmas go from near sigma_max down to near sigma_min_active
        x_points = torch.linspace(
            steepness_factor, -steepness_factor, num_active_sigmas, device="cpu"
        )
        sigmoid_values = 1 / (1 + torch.exp(-x_points))

        # Scale sigmoid (0..1 approx) to (sigma_min_active .. sigma_max)
        calculated_active_sigmas = (
            current_sigma_min_active
            + (current_sigma_max - current_sigma_min_active) * sigmoid_values
        )

        final_sigmas = torch.zeros(
            steps_intervals + 1, dtype=torch.float32, device="cpu"
        )
        if num_active_sigmas > 0:
            final_sigmas[:num_active_sigmas] = calculated_active_sigmas
        final_sigmas[steps_intervals] = 0.0

        if num_active_sigmas > 0:
            final_sigmas[0] = current_sigma_max
            final_sigmas[num_active_sigmas - 1] = max(
                final_sigmas[num_active_sigmas - 1].item(), current_sigma_min_active
            )

        final_sigmas[0] = current_sigma_max
        for i in range(steps_intervals - 1):
            lower_bound_for_next = max(current_sigma_min_active, epsilon)
            final_sigmas[i + 1] = torch.clamp(
                final_sigmas[i + 1],
                min=lower_bound_for_next,
                max=final_sigmas[i].item() - epsilon,
            )

        if steps_intervals > 0:
            final_sigmas[steps_intervals - 1] = max(current_sigma_min_active, epsilon)
            if steps_intervals > 1:
                final_sigmas[steps_intervals - 1] = min(
                    final_sigmas[steps_intervals - 1].item(),
                    final_sigmas[steps_intervals - 2].item() - epsilon,
                )
                final_sigmas[steps_intervals - 1] = max(
                    final_sigmas[steps_intervals - 1].item(),
                    current_sigma_min_active,
                    epsilon,
                )

        final_sigmas[steps_intervals] = 0.0

        return (final_sigmas.cpu(),)
