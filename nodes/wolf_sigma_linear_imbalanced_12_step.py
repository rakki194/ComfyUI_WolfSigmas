import torch
import math


class WolfSigmaLinearImbalanced12Step:
    """
    Generates 12 + 1 sigmas with a bias factor.
    Active sigmas (sigma_max to sigma_min_positive) are spaced non-linearly based on bias.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_imbalanced_linear_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step_imbalanced"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # num_steps is hardcoded to 12
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
        self, sigma_max, sigma_min_positive, bias_factor, min_epsilon_spacing
    ):
        num_steps = 12  # Hardcoded
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        bias = float(bias_factor)
        epsilon = float(min_epsilon_spacing)

        if N <= 0:  # Should not happen with N=12
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)
        if s_max <= s_min_pos:
            s_max = s_min_pos + N * epsilon
        if s_min_pos <= 0.0:
            s_min_pos = epsilon

        # N is always 12, so N == 1 case is not strictly needed for this class
        # but the logic for N > 1 is general.
        # Generate t values from 0 to 1
        t_linear = torch.linspace(0, 1, N, device="cpu")

        power = 1.0 - bias
        if power <= 0:
            power = epsilon

        t_biased = t_linear.pow(power)

        active_sigmas = s_max - t_biased * (s_max - s_min_pos)
        active_sigmas[0] = s_max
        active_sigmas[-1] = s_min_pos

        final_sigmas_list = active_sigmas.tolist() + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # Apply the same robust post-processing
        sigmas[0] = s_max
        if N > 0:  # N is 12
            sigmas[N - 1] = max(s_min_pos, epsilon)

        for i in range(N - 1):
            if sigmas[i].item() <= sigmas[i + 1].item() + epsilon:
                sigmas[i + 1] = sigmas[i].item() - epsilon
            sigmas[i + 1] = max(epsilon, sigmas[i + 1].item())

        if N > 0:  # N is 12
            sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos, epsilon)
            if sigmas[N - 1].item() <= sigmas[N].item() + epsilon:
                sigmas[N - 1] = max(s_min_pos, epsilon)

        sigmas[0] = s_max
        for i in range(N):
            current_val = sigmas[i].item()
            next_val_target = sigmas[i + 1].item()

            if i < N - 1:
                min_allowed_current = next_val_target + epsilon
                sigmas[i] = max(current_val, min_allowed_current)
                if i > 0 and sigmas[i].item() >= sigmas[i - 1].item() - epsilon:
                    sigmas[i] = sigmas[i - 1].item() - epsilon
            elif i == N - 1:
                sigmas[i] = max(
                    current_val, s_min_pos, epsilon, next_val_target + epsilon
                )
                if i > 0 and sigmas[i].item() >= sigmas[i - 1].item() - epsilon:
                    sigmas[i] = sigmas[i - 1].item() - epsilon
                sigmas[i] = max(
                    sigmas[i].item(), s_min_pos, epsilon, next_val_target + epsilon
                )

        sigmas[N] = 0.0

        return (sigmas,)
