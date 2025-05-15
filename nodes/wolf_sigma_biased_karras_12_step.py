import torch


class WolfSigmaBiasedKarras12Step:
    """
    Generates a 12-step (13 sigmas) Karras schedule with a bias_power.
    bias_power > 1.0 concentrates steps towards sigma_max (larger early steps).
    bias_power < 1.0 concentrates steps towards sigma_min (smaller early steps).
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_biased_karras_12_step_sigmas"
    CATEGORY = (
        "sampling/sigmas_wolf/12_step_imbalanced"  # Reflects it's biased/imbalanced
    )

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
                        "default": 1.0,  # Default for SD1.5 style
                        "min": 0.1,
                        "max": 1000.0,  # Max for Flux/Chroma style
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

    def get_biased_karras_12_step_sigmas(
        self, sigma_min, sigma_max, rho, bias_power, min_epsilon_spacing
    ):
        steps_intervals = 12  # Changed from 8 to 12
        num_points = steps_intervals + 1
        epsilon = float(min_epsilon_spacing)

        current_sigma_min = float(sigma_min)
        current_sigma_max = float(sigma_max)
        current_rho = float(rho)
        current_bias_power = float(bias_power)

        if current_sigma_max <= current_sigma_min:
            current_sigma_max = current_sigma_min + steps_intervals * epsilon
        if current_sigma_min <= epsilon:
            current_sigma_min = epsilon

        t_norm = torch.linspace(0, 1, steps_intervals, device="cpu")
        t_biased = t_norm**current_bias_power

        inv_rho_min = current_sigma_min ** (1 / current_rho)
        inv_rho_max = current_sigma_max ** (1 / current_rho)

        # Calculate sigmas: from sigma_max (t_biased=0) down to sigma_min (t_biased=1)
        sigs_calc = (
            inv_rho_max + t_biased * (inv_rho_min - inv_rho_max)
        ) ** current_rho

        final_sigmas = torch.zeros(num_points, dtype=torch.float32, device="cpu")
        final_sigmas[:steps_intervals] = sigs_calc
        final_sigmas[steps_intervals] = 0.0  # Last sigma is 0.0

        # Initial boundary enforcement
        final_sigmas[0] = current_sigma_max
        if steps_intervals > 0:
            final_sigmas[steps_intervals - 1] = max(
                final_sigmas[steps_intervals - 1].item(), current_sigma_min
            )

        # Post-processing loop to ensure strict decrease and minimum spacing for active sigmas
        final_sigmas[0] = current_sigma_max  # Re-assert first sigma
        for i in range(steps_intervals - 1):  # Iterate from 0 to steps_intervals-2
            # sigmas[i+1] should be <= sigmas[i] - epsilon
            # sigmas[i+1] should be >= current_sigma_min (if it's an active sigma)
            # sigmas[i+1] should be >= epsilon (always, if not 0.0)

            lower_bound_for_next = (
                current_sigma_min
                if (i + 1) < (steps_intervals - 1)
                else current_sigma_min
            )  # All active sigmas >= current_sigma_min
            lower_bound_for_next = max(lower_bound_for_next, epsilon)

            final_sigmas[i + 1] = torch.clamp(
                final_sigmas[i + 1],
                min=lower_bound_for_next,
                max=final_sigmas[i].item() - epsilon,
            )

        # Ensure last active sigma (index steps_intervals-1) is correctly bounded
        if steps_intervals > 0:
            final_sigmas[steps_intervals - 1] = max(current_sigma_min, epsilon)
            if steps_intervals > 1:  # Must be less than the one before it
                final_sigmas[steps_intervals - 1] = min(
                    final_sigmas[steps_intervals - 1].item(),
                    final_sigmas[steps_intervals - 2].item() - epsilon,
                )
                # And re-assert lower bound after potential change
                final_sigmas[steps_intervals - 1] = max(
                    final_sigmas[steps_intervals - 1].item(), current_sigma_min, epsilon
                )

        final_sigmas[steps_intervals] = 0.0  # Final element must be 0

        return (final_sigmas.cpu(),)
