import torch


class WolfSigmaSigmoidImbalancedNStep:
    """
    Generates an N-step schedule using a sigmoid curve, with a bias factor
    to control step distribution. Produces N+1 sigma values.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/sigmas_wolf/n_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_steps": ("INT", {"default": 20, "min": 1, "max": 1000}),
                "sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.1},
                ),
                "sigma_min": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0, "max": 10.0, "step": 0.001},
                ),
                "skew_factor": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.1, "max": 10.0, "step": 0.1},
                ),  # Controls the 'steepness' or range of the sigmoid input
                "bias_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
                ),  # Power for t_norm
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
        self,
        num_steps,
        sigma_max,
        sigma_min,
        skew_factor,
        bias_factor,
        min_epsilon_spacing,
    ):
        if num_steps < 1:
            return (torch.tensor([sigma_max, 0.0], dtype=torch.float32),)

        # Define the effective minimum sigma for the active part of the schedule
        # This is sigma_min if >0, else a small epsilon.
        # The final list can still contain absolute 0 for sigma_min if user specified.
        active_sigma_min = (
            sigma_min if sigma_min > min_epsilon_spacing else min_epsilon_spacing
        )

        current_sigma_max = float(sigma_max)

        if current_sigma_max <= active_sigma_min:
            current_sigma_max = (
                active_sigma_min + min_epsilon_spacing * 10
            )  # Ensure max > min

        sigmas = torch.zeros(num_steps, dtype=torch.float32)

        # 1. Generate normalized time points
        t_linear = torch.linspace(0, 1, num_steps, dtype=torch.float32)

        # 2. Apply bias
        # If bias_factor is 1, t_biased is same as t_linear
        # If bias_factor > 1, more steps concentrated at the beginning (low t, high sigma)
        # If bias_factor < 1, more steps concentrated at the end (high t, low sigma)
        t_biased = t_linear**bias_factor

        # 3. Transform biased time to sigmoid input range
        # We want sigmas from high to low, so sigmoid input should go from high positive to high negative
        # x_input maps t_biased [0,1] to [skew_factor, -skew_factor]
        x_input = skew_factor * (1.0 - 2.0 * t_biased)

        # 4. Calculate sigmoid values (normalized from ~1 down to ~0)
        sigmoid_normalized = 1.0 / (1.0 + torch.exp(-x_input))

        # 5. Scale sigmas to [active_sigma_min, current_sigma_max]
        scaled_sigmas = (
            active_sigma_min
            + (current_sigma_max - active_sigma_min) * sigmoid_normalized
        )
        sigmas[:num_steps] = scaled_sigmas

        # Refined Post-processing:
        if num_steps > 0:
            # 1. Determine the floor for the last active sigma (sigmas[num_steps-1])
            #    It should respect the user's sigma_min. If sigma_min is 0.0,
            #    it can be 0.0 if it's the *only* step (num_steps=1).
            #    Otherwise, if sigma_min is 0.0 and there are multiple steps,
            #    the last active sigma should be at least min_epsilon_spacing.
            #    If sigma_min > 0, it must be at least sigma_min and also min_epsilon_spacing.

            sigma_min_floor_for_last_active = sigma_min  # Default to user's sigma_min
            if sigma_min == 0.0:
                if (
                    num_steps > 1
                ):  # Multiple steps, last active can't be 0 unless it's the only one
                    sigma_min_floor_for_last_active = min_epsilon_spacing
                # if num_steps == 1 and sigma_min == 0.0, it remains 0.0
            else:  # sigma_min > 0.0
                sigma_min_floor_for_last_active = max(sigma_min, min_epsilon_spacing)

            sigmas[num_steps - 1] = max(
                sigmas[num_steps - 1], sigma_min_floor_for_last_active
            )

            # 2. Iterate backwards from the second to last active sigma.
            #    Ensure strictly decreasing order and that each sigma is at least min_epsilon_spacing.
            for i in range(num_steps - 2, -1, -1):
                sigmas[i] = max(sigmas[i], sigmas[i + 1] + min_epsilon_spacing)
                sigmas[i] = max(
                    sigmas[i], min_epsilon_spacing
                )  # General positive floor

            # 3. Ensure the first sigma is no more than sigma_max (it might have been pushed up)
            #    and also adheres to positivity and relation to the next sigma.
            sigmas[0] = min(sigmas[0], current_sigma_max)
            if num_steps > 1:
                sigmas[0] = max(sigmas[0], sigmas[1] + min_epsilon_spacing)
            sigmas[0] = max(sigmas[0], min_epsilon_spacing)  # Ensure positive

            # 4. One final pass to ensure the last active sigma respects its specific floor,
            #    as the backward pass might have altered it based on min_epsilon_spacing only.
            sigmas[num_steps - 1] = max(
                sigmas[num_steps - 1], sigma_min_floor_for_last_active
            )
            # And re-ensure the one before it is sufficiently larger if it was adjusted
            if num_steps > 1:
                sigmas[num_steps - 2] = max(
                    sigmas[num_steps - 2], sigmas[num_steps - 1] + min_epsilon_spacing
                )

        # Final array with N+1 sigmas, ending with 0.0
        final_sigmas = torch.zeros(num_steps + 1, dtype=torch.float32)
        if num_steps > 0:
            final_sigmas[:num_steps] = sigmas
        # final_sigmas[num_steps] is already 0.0 due to torch.zeros initialization.

        return (final_sigmas,)
