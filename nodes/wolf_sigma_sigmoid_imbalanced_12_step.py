import torch


class WolfSigmaSigmoidImbalanced12Step:
    """
    Generates a 12-step schedule using a sigmoid curve, with a bias factor
    to control step distribution. Produces 13 sigma values.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigma_max": (
                    "FLOAT",
                    {"default": 80.0, "min": 0.01, "max": 1000.0, "step": 0.1},
                ),
                "sigma_min": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0, "max": 10.0, "step": 0.001},
                ),
                "skew_factor": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),
                "bias_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),
                "min_epsilon_spacing": (
                    "FLOAT",
                    {
                        "default": 1e-5,
                        "min": 1e-7,
                        "max": 1e-2,
                        "step": 1e-7,
                        "round": False,
                    },
                ),
            }
        }

    def get_sigmas(
        self, sigma_max, sigma_min, skew_factor, bias_factor, min_epsilon_spacing
    ):
        num_steps = 12

        active_sigma_min = (
            sigma_min if sigma_min > min_epsilon_spacing else min_epsilon_spacing
        )
        current_sigma_max = float(sigma_max)

        if current_sigma_max <= active_sigma_min:
            current_sigma_max = active_sigma_min + min_epsilon_spacing * 10

        sigmas = torch.zeros(num_steps, dtype=torch.float32)

        t_linear = torch.linspace(0, 1, num_steps, dtype=torch.float32)
        t_biased = t_linear**bias_factor
        x_input = skew_factor * (1.0 - 2.0 * t_biased)
        sigmoid_normalized = 1.0 / (1.0 + torch.exp(-x_input))
        scaled_sigmas = (
            active_sigma_min
            + (current_sigma_max - active_sigma_min) * sigmoid_normalized
        )
        sigmas[:num_steps] = scaled_sigmas

        # Refined Post-processing:
        if num_steps > 0:  # num_steps is fixed at 12, so this is always true
            sigma_min_floor_for_last_active = sigma_min
            if sigma_min == 0.0:
                if num_steps > 1:
                    sigma_min_floor_for_last_active = min_epsilon_spacing
            else:
                sigma_min_floor_for_last_active = max(sigma_min, min_epsilon_spacing)

            sigmas[num_steps - 1] = max(
                sigmas[num_steps - 1], sigma_min_floor_for_last_active
            )

            for i in range(num_steps - 2, -1, -1):
                sigmas[i] = max(sigmas[i], sigmas[i + 1] + min_epsilon_spacing)
                sigmas[i] = max(sigmas[i], min_epsilon_spacing)

            sigmas[0] = min(sigmas[0], current_sigma_max)
            if num_steps > 1:
                sigmas[0] = max(sigmas[0], sigmas[1] + min_epsilon_spacing)
            sigmas[0] = max(sigmas[0], min_epsilon_spacing)

            sigmas[num_steps - 1] = max(
                sigmas[num_steps - 1], sigma_min_floor_for_last_active
            )
            if num_steps > 1:
                sigmas[num_steps - 2] = max(
                    sigmas[num_steps - 2], sigmas[num_steps - 1] + min_epsilon_spacing
                )

        final_sigmas = torch.zeros(num_steps + 1, dtype=torch.float32)
        final_sigmas[:num_steps] = sigmas

        return (final_sigmas,)
