import torch
import math


class WolfSigmaPolynomial:
    """
    Generates N_steps + 1 sigmas based on a polynomial function of normalized time.
    sigma = sigma_min_positive + (sigma_max - sigma_min_positive) * (1 - t_norm^power)
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_polynomial_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_steps": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.1},
                ),
                "sigma_min_positive": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0001, "max": 10.0, "step": 0.0001},
                ),
                "power": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.05,
                        "doc": "Power for the polynomial. 1.0=linear, >1.0 denser near sigma_max, <1.0 denser near sigma_min_positive.",
                    },
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

    def generate_polynomial_sigmas(
        self, num_steps, sigma_max, sigma_min_positive, power, min_epsilon_spacing
    ):
        num_active_steps = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        p = float(power)
        epsilon = float(min_epsilon_spacing)

        if num_active_steps < 1:
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)

        if s_max <= s_min_pos:
            s_max = s_min_pos + num_active_steps * epsilon
        if s_min_pos <= 0:
            s_min_pos = epsilon
        if p <= 0:
            p = epsilon  # Power must be positive

        active_sigmas = torch.zeros(num_active_steps, dtype=torch.float32)

        if num_active_steps == 1:
            active_sigmas[0] = (
                s_max  # Or (s_max + s_min_pos) / 2, but s_max is fine for one step
            )
        else:
            t_norm = torch.linspace(0, 1, num_active_steps, device="cpu")
            # Formula: s_min_pos + (s_max - s_min_pos) * (1 - t_norm^p)
            # This ensures sigma(t_norm=0) = s_max and sigma(t_norm=1) = s_min_pos
            active_sigmas = s_min_pos + (s_max - s_min_pos) * (1.0 - t_norm.pow(p))
            # Ensure boundaries explicitly after calculation due to potential float precision
            active_sigmas[0] = s_max
            active_sigmas[-1] = s_min_pos

        # Standard Post-processing
        final_sigmas = torch.zeros(num_active_steps + 1, dtype=torch.float32)
        if num_active_steps > 0:
            final_sigmas[:num_active_steps] = active_sigmas
            final_sigmas[0] = (
                s_max  # Re-assert, might have been changed by tensor assignment if active_sigmas was just one value
            )

            current_s_max = s_max
            current_s_min_positive = s_min_pos

            sigma_min_floor_for_last_active = current_s_min_positive
            if current_s_min_positive == 0.0:
                if num_active_steps > 1:
                    sigma_min_floor_for_last_active = epsilon
            else:
                sigma_min_floor_for_last_active = max(current_s_min_positive, epsilon)

            final_sigmas[num_active_steps - 1] = max(
                final_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
            )

            for i in range(num_active_steps - 2, -1, -1):
                final_sigmas[i] = max(final_sigmas[i], final_sigmas[i + 1] + epsilon)
                final_sigmas[i] = max(final_sigmas[i], epsilon)

            final_sigmas[0] = min(final_sigmas[0], current_s_max)
            if num_active_steps > 1:
                final_sigmas[0] = max(final_sigmas[0], final_sigmas[1] + epsilon)
            final_sigmas[0] = max(final_sigmas[0], epsilon)

            final_sigmas[num_active_steps - 1] = max(
                final_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
            )
            if num_active_steps > 1:
                final_sigmas[num_active_steps - 2] = max(
                    final_sigmas[num_active_steps - 2],
                    final_sigmas[num_active_steps - 1] + epsilon,
                )

        final_sigmas[num_active_steps] = 0.0

        return (final_sigmas,)
