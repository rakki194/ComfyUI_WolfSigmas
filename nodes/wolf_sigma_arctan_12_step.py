import torch
import math


class WolfSigmaArctan12Step:
    """
    Generates 12 + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive)
    are spaced such that arctan(sigma / c_factor) is linear. Last sigma is 0.0.
    Inspired by AYS paper Theorem 3.1.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_arctan_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "num_steps" is removed, hardcoded to 12
                "sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001},
                ),
                "sigma_min_positive": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0001, "max": 100.0, "step": 0.0001},
                ),
                "c_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 100.0, "step": 0.001},
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

    def generate_arctan_sigmas(
        self,
        sigma_max,
        sigma_min_positive,
        c_factor,
        min_epsilon_spacing,
    ):
        num_steps = 12  # Hardcoded
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        c = float(c_factor)
        epsilon = float(min_epsilon_spacing)

        if N <= 0:  # Should not happen
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)
        if s_max <= s_min_pos:
            s_max = s_min_pos + N * epsilon
        if s_min_pos <= 0:
            s_min_pos = epsilon
        if c <= 0:
            c = epsilon

        arctan_s_max = math.atan(s_max / c)
        arctan_s_min_pos = math.atan(s_min_pos / c)

        # N is always 12 here
        arctan_values = torch.linspace(arctan_s_max, arctan_s_min_pos, N)

        active_sigmas_list = [c * math.tan(val) for val in arctan_values.tolist()]

        active_sigmas_list[0] = s_max
        if N > 0:  # N = 12
            active_sigmas_list[-1] = s_min_pos

        final_sigmas_list = active_sigmas_list + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # Re-using the post-processing logic from the original WolfSigmaArctanNStep
        # This logic is quite specific and aims to preserve certain properties.
        sigmas[0] = s_max  # Ensure s_max is the first

        # Enforce monotonicity and positivity for the N active sigmas
        # And ensure the last active sigma is s_min_pos
        if N > 0:
            sigmas[N - 1] = s_min_pos  # Target for the last active sigma

        for i in range(N - 1):  # Iterate up to N-2 index (second to last active sigma)
            # Ensure current sigma is greater than next by at least epsilon
            current_sigma = sigmas[i].item()
            next_sigma = sigmas[i + 1].item()

            if current_sigma <= next_sigma + epsilon:
                sigmas[i + 1] = current_sigma - epsilon

            # Ensure active sigmas are at least s_min_pos (if not the absolute last) or epsilon
            # The arctan calculation can sometimes push values very low.
            # For active sigmas, they should not go below s_min_pos unless it's the last active sigma being set to s_min_pos.
            # More simply, all active sigmas must be >= s_min_pos
            sigmas[i + 1] = max(
                sigmas[i + 1].item(), s_min_pos if i + 1 < N - 1 else s_min_pos, epsilon
            )

        # Final check and enforcement of s_min_pos for the last active sigma and 0 for the final sigma
        if N > 0:
            sigmas[N - 1] = max(s_min_pos, epsilon)
            # Ensure it is less than the previous one if N > 1
            if N > 1:
                sigmas[N - 1] = min(
                    sigmas[N - 1].item(), sigmas[N - 2].item() - epsilon
                )
                sigmas[N - 1] = max(
                    sigmas[N - 1].item(), s_min_pos, epsilon
                )  # re-check lower bound
        sigmas[N] = 0.0

        # A simplified loop to ensure strict decrease and minimums after initial generation
        sigmas[0] = s_max
        for i in range(N - 1):  # Iterate N-1 times, from 0 to N-2
            # Ensure sigmas[i+1] is less than sigmas[i] by at least epsilon
            # and not less than s_min_pos (for active sigmas)
            lower_bound = (
                s_min_pos if (i + 1) < N else 0.0
            )  # This is not quite right. N-1 is last active
            actual_lower_bound = (
                s_min_pos if (i + 1) < N else epsilon
            )  # if sigmas[i+1] is an active sigma, its min is s_min_pos
            # last actual sigma is N, which is 0.
            # so this applies to sigmas[1]...sigmas[N-1]

            sigmas[i + 1] = torch.clamp(
                sigmas[i + 1], min=actual_lower_bound, max=sigmas[i].item() - epsilon
            )

        if N > 0:
            sigmas[N - 1] = max(
                s_min_pos, epsilon
            )  # Ensure last active sigma is at least s_min_pos
            if N > 1:  # And less than the one before it
                sigmas[N - 1] = min(
                    sigmas[N - 1].item(), sigmas[N - 2].item() - epsilon
                )
                sigmas[N - 1] = max(
                    sigmas[N - 1].item(), s_min_pos, epsilon
                )  # Re-check s_min_pos
        sigmas[N] = 0.0

        return (sigmas,)
