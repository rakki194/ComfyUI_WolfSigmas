import torch
import math


class WolfSigmaArctanNStep:
    """
    Generates N_steps + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive)
    are spaced such that arctan(sigma / c_factor) is linear. Last sigma is 0.0.
    Inspired by AYS paper Theorem 3.1.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_arctan_sigmas"
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
        num_steps,
        sigma_max,
        sigma_min_positive,
        c_factor,
        min_epsilon_spacing,
    ):
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        c = float(c_factor)
        epsilon = float(min_epsilon_spacing)

        if N <= 0:
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)  # Basic fallback
        if s_max <= s_min_pos:
            s_max = s_min_pos + N * epsilon
        if s_min_pos <= 0:
            s_min_pos = epsilon
        if c <= 0:
            c = epsilon

        arctan_s_max = math.atan(s_max / c)
        arctan_s_min_pos = math.atan(s_min_pos / c)

        # We want N intervals, so N+1 points. For active sigmas, N points.
        # If N=1 step, we need 2 sigmas: sigma_max, sigma_min_pos, then 0.0.
        # So, N sigmas from sigma_max down to sigma_min_pos.
        if N == 1:
            arctan_values = torch.tensor([arctan_s_max, arctan_s_min_pos])
        else:
            arctan_values = torch.linspace(arctan_s_max, arctan_s_min_pos, N)

        active_sigmas_list = [c * math.tan(val) for val in arctan_values.tolist()]

        # Ensure first and last active sigmas are set correctly
        active_sigmas_list[0] = s_max
        if N > 0:
            active_sigmas_list[-1] = s_min_pos

        final_sigmas_list = active_sigmas_list + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # Enforce monotonicity and positivity for the N active sigmas
        for i in range(N - 1):  # Iterate up to N-2 index (second to last active sigma)
            if sigmas[i].item() <= sigmas[i + 1].item() + epsilon:
                sigmas[i + 1] = sigmas[i].item() - epsilon
            sigmas[i + 1] = max(
                epsilon, sigmas[i + 1].item()
            )  # Ensure active sigmas > 0

        if (
            N > 0 and sigmas[N - 1].item() < s_min_pos
        ):  # last active sigma should be s_min_pos
            sigmas[N - 1] = s_min_pos
        if N > 0 and sigmas[N - 1].item() <= 0:
            sigmas[N - 1] = s_min_pos  # if it became zero, restore to s_min_pos

        # Ensure the one before last (0.0) is >= s_min_pos and > 0
        if N > 0 and sigmas[N - 1].item() <= 0.0:
            sigmas[N - 1] = s_min_pos
        if (
            N > 0 and sigmas[N - 1].item() <= sigmas[N].item() + epsilon
        ):  # sigmas[N] is 0.0
            sigmas[N - 1] = max(s_min_pos, epsilon)

        # Final check for sequence s0, s1, ..., s_{N-1}, 0.0
        for i in range(N):  # Iterate through all active sigmas
            sigmas[i] = max(sigmas[i].item(), epsilon if i < N else 0.0)
            if i < N - 1:  # Compare s_i with s_{i+1}
                if sigmas[i].item() <= sigmas[i + 1].item() + epsilon:
                    sigmas[i + 1] = sigmas[i].item() - epsilon
            elif i == N - 1:  # Compare s_{N-1} with 0.0
                if (
                    sigmas[i].item() <= sigmas[i + 1].item() + epsilon
                ):  # sigmas[i+1] is 0.0
                    sigmas[i] = max(
                        epsilon, s_min_pos, sigmas[i].item() - epsilon
                    )  # Ensure it's positive and greater than 0

        # One last pass to ensure s_{N-1} is correctly set and positive
        if N > 0:
            sigmas[N - 1] = max(s_min_pos, epsilon, sigmas[N - 1].item())
            if N > 1 and sigmas[N - 1].item() >= sigmas[N - 2].item() - epsilon:
                sigmas[N - 1] = sigmas[N - 2].item() - epsilon
            sigmas[N - 1] = max(s_min_pos, epsilon, sigmas[N - 1].item())

        return (sigmas,)
