import torch
import math


class WolfSigmaLinearImbalancedNStep:
    """
    Generates N_steps + 1 sigmas with a bias factor.
    Active sigmas (sigma_max to sigma_min_positive) are spaced non-linearly based on bias.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_imbalanced_linear_sigmas"
    CATEGORY = "sampling/sigmas_wolf/N_step_imbalanced"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_steps": (
                    "INT",
                    {"default": 8, "min": 1, "max": 1000},
                ),
                "sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001},
                ),
                "sigma_min_positive": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0001, "max": 100.0, "step": 0.0001},
                ),
                "bias_factor": (
                    "FLOAT",
                    # bias_factor > 0: more steps near sigma_max (early diffusion)
                    # bias_factor < 0: more steps near sigma_min (late diffusion)
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

    def generate_imbalanced_linear_sigmas(
        self, num_steps, sigma_max, sigma_min_positive, bias_factor, min_epsilon_spacing
    ):
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        bias = float(bias_factor)
        epsilon = float(min_epsilon_spacing)

        if N <= 0:
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)
        if s_max <= s_min_pos:
            # Ensure s_max is actually larger, adjust if not to prevent issues with range
            s_max = s_min_pos + N * epsilon
        if s_min_pos <= 0.0:
            s_min_pos = epsilon

        if N == 1:
            # For 1 step, bias doesn't really apply in the same way,
            # just use the two points, similar to linear.
            active_sigmas = torch.tensor([s_max, s_min_pos], dtype=torch.float32)
        else:
            # Generate t values from 0 to 1
            t_linear = torch.linspace(0, 1, N, device="cpu")

            # Apply power based on bias_factor
            # exponent = 1.0 + bias # if bias > 0, exponent > 1, t^exp concentrates near 0
            # if bias < 0, exponent < 1, t^exp concentrates near 1
            # This means if bias > 0, t_biased values are smaller (more steps early in 0-1 range)
            # if bias < 0, t_biased values are larger (more steps late in 0-1 range)
            exponent = (
                1.0 - bias
            )  # Let's try this: if bias > 0 (early), exponent < 1 (e.g. 0.5 for sqrt like behavior)
            # if bias < 0 (late), exponent > 1 (e.g. 1.5 for squared like behavior)
            # A value of t^0.5 means values rise quickly from 0. A value of t^1.5 means values rise slowly from 0.

            if abs(exponent) < 1e-6:  # Avoid pow(0,0) or nearly zero exponent issues
                exponent = 1e-6 if exponent >= 0 else -1e-6

            # Correct logic for bias: bias > 0 means more steps at high sigma (early)
            # So we want the normalized steps (0..1) to be small for early steps.
            # x^(1/(1+bias)) if bias is positive strength, x^(1-bias) if bias is negative strength
            # Let's use a power 'p'. If p > 1, points cluster near 0 and 1. If p < 1, points cluster near middle.
            # We want to control clustering at start (sigma_max) or end (sigma_min_pos).

            # Alternative: Karras rho parameter controls density. Higher rho = denser near sigma_min.
            # For linear-like, let's use a simpler power curve on a 0-1 normalized space.

            # If bias > 0, we want steps to be denser near sigma_max.
            # This means the normalized step values (0 to 1, mapped to s_max to s_min) should increase slowly at first.
            # A function like x^p where p > 1 makes it increase slowly at first (e.g. x^2).
            # If bias < 0, we want steps denser near sigma_min.
            # This means normalized step values should increase quickly at first.
            # A function like x^p where 0 < p < 1 makes it increase quickly at first (e.g. sqrt(x)).

            power = (
                1.0 - bias
            )  # If bias is 0.5 (more steps early/sigma_max), power = 0.5 (sqrt like)
            # If bias is -0.5 (more steps late/sigma_min), power = 1.5 (square like)
            # This seems correct for t_biased mapping. t_biased = t_linear ** power

            if power <= 0:  # avoid issues with pow(0, non_positive)
                power = epsilon  # a small positive power as fallback

            t_biased = t_linear.pow(power)

            # Map t_biased (0 to 1) to sigmas (s_max down to s_min_pos)
            active_sigmas = s_max - t_biased * (s_max - s_min_pos)
            # Ensure the first is s_max and last is s_min_pos after biasing.
            active_sigmas[0] = s_max
            active_sigmas[-1] = s_min_pos

        final_sigmas_list = active_sigmas.tolist() + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # Apply the same robust post-processing as in the original LinearNStep
        # to ensure descending order, min_epsilon_spacing, and boundary conditions.
        sigmas[0] = s_max
        if N > 0:
            sigmas[N - 1] = max(s_min_pos, epsilon)

        for i in range(N - 1):  # 0 to N-2
            if sigmas[i].item() <= sigmas[i + 1].item() + epsilon:
                sigmas[i + 1] = sigmas[i].item() - epsilon
            sigmas[i + 1] = max(epsilon, sigmas[i + 1].item())

        if N > 0:
            sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos, epsilon)
            if sigmas[N - 1].item() <= sigmas[N].item() + epsilon:  # sigmas[N] is 0.0
                sigmas[N - 1] = max(s_min_pos, epsilon)

        sigmas[0] = s_max
        for i in range(N):  # 0 to N-1
            current_val = sigmas[i].item()
            next_val_target = sigmas[i + 1].item()

            if i < N - 1:  # i from 0 to N-2
                min_allowed_current = next_val_target + epsilon
                sigmas[i] = max(current_val, min_allowed_current)
                if i > 0 and sigmas[i].item() >= sigmas[i - 1].item() - epsilon:
                    sigmas[i] = sigmas[i - 1].item() - epsilon
            elif i == N - 1:  # i = N-1 (last active sigma)
                sigmas[i] = max(
                    current_val, s_min_pos, epsilon, next_val_target + epsilon
                )
                if i > 0 and sigmas[i].item() >= sigmas[i - 1].item() - epsilon:
                    sigmas[i] = sigmas[i - 1].item() - epsilon
                sigmas[i] = max(
                    sigmas[i].item(), s_min_pos, epsilon, next_val_target + epsilon
                )

        sigmas[N] = 0.0  # Ensure last element is 0

        return (sigmas,)
