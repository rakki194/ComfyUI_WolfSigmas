import torch

# import math


class WolfSigmaLinearNStep:
    """
    Generates N_steps + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive)
    are linearly spaced. Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_linear_sigmas"
    CATEGORY = "sampling/sigmas_wolf/N_step"

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

    def generate_linear_sigmas(
        self, num_steps, sigma_max, sigma_min_positive, min_epsilon_spacing
    ):
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        epsilon = float(min_epsilon_spacing)

        if N <= 0:
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)  # Basic fallback
        if s_max <= s_min_pos:
            s_max = s_min_pos + N * epsilon
        if s_min_pos <= 0.0:
            s_min_pos = epsilon

        if N == 1:
            active_sigmas = torch.tensor([s_max, s_min_pos], dtype=torch.float32)
        else:
            active_sigmas = torch.linspace(s_max, s_min_pos, N, device="cpu")

        final_sigmas_list = active_sigmas.tolist() + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        sigmas[0] = s_max
        if N > 0:
            sigmas[N - 1] = max(s_min_pos, epsilon)

        for i in range(N - 1):
            if sigmas[i].item() <= sigmas[i + 1].item() + epsilon:
                sigmas[i + 1] = sigmas[i].item() - epsilon
            sigmas[i + 1] = max(epsilon, sigmas[i + 1].item())

        if N > 0:
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
                sigmas[i] = max(current_val, s_min_pos, epsilon)
                if i > 0 and sigmas[i].item() >= sigmas[i - 1].item() - epsilon:
                    sigmas[i] = sigmas[i - 1].item() - epsilon
                sigmas[i] = max(current_val, s_min_pos, epsilon)

        return (sigmas,)
