import torch


class WolfSigmaSigmoid4Step:
    """
    Generates a 4-step schedule using a sigmoid curve.
    Produces 5 sigma values, from sigma_max down to sigma_min (potentially 0), then a final 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmoid_4_step_sigmas"
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
                "sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.029,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.001,
                        "round": False,
                    },
                ),
                "steepness_factor": (
                    "FLOAT",
                    {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1},
                ),
            }
        }

    def get_sigmoid_4_step_sigmas(self, sigma_max, sigma_min, steepness_factor):
        steps_intervals = 4
        num_sigmas_to_generate = steps_intervals

        current_sigma_max = float(sigma_max)
        current_sigma_min = float(sigma_min)

        if current_sigma_max <= current_sigma_min:
            current_sigma_max = current_sigma_min + 0.1

        x_points = torch.linspace(
            steepness_factor, -steepness_factor, num_sigmas_to_generate
        )
        sigmoid_values = 1 / (1 + torch.exp(-x_points))

        scaled_sigmas = (
            current_sigma_min + (current_sigma_max - current_sigma_min) * sigmoid_values
        )

        final_sigmas = torch.zeros(steps_intervals + 1, dtype=torch.float32)
        final_sigmas[:num_sigmas_to_generate] = scaled_sigmas
        final_sigmas[steps_intervals] = 0.0

        for i in range(num_sigmas_to_generate - 1):
            if final_sigmas[i] <= final_sigmas[i + 1] + 1e-7:
                final_sigmas[i + 1] = final_sigmas[i] * 0.95
            if (
                final_sigmas[i].item() <= 0 and i < num_sigmas_to_generate - 1
            ):  # Ensure non-final sigmas are positive
                final_sigmas[i] = torch.tensor(1e-5)

        if num_sigmas_to_generate > 0:
            final_sigmas[num_sigmas_to_generate - 1] = max(
                final_sigmas[num_sigmas_to_generate - 1], current_sigma_min
            )
            if (
                final_sigmas[num_sigmas_to_generate - 1].item() <= 0.0
                and current_sigma_min > 0.0
            ):
                final_sigmas[num_sigmas_to_generate - 1] = torch.tensor(
                    current_sigma_min
                )
            elif (
                final_sigmas[num_sigmas_to_generate - 1].item() <= 0.0
                and current_sigma_min == 0.0
            ):
                final_sigmas[num_sigmas_to_generate - 1] = torch.tensor(1e-5)

        final_sigmas[steps_intervals] = 0.0
        return (final_sigmas,)
