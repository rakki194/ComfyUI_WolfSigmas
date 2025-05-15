import torch
import math


class WolfSigmaArctanImbalanced12Step:
    """
    Generates 12 + 1 sigmas with a bias factor applied in the arctan domain.
    Active sigmas (sigma_max to sigma_min_positive) are spaced non-linearly.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_imbalanced_arctan_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step_imbalanced"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # num_steps hardcoded to 12
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

    def generate_imbalanced_arctan_sigmas(
        self,
        sigma_max,
        sigma_min_positive,
        c_factor,
        bias_factor,
        min_epsilon_spacing,
    ):
        num_steps = 12  # Hardcoded
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        c = float(c_factor)
        bias = float(bias_factor)
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

        # N is always 12
        t_linear = torch.linspace(0, 1, N, device="cpu")
        power = 1.0 - bias
        if power <= 0:
            power = epsilon
        t_biased = t_linear.pow(power)

        arctan_biased_values_list = [
            (arctan_s_max - t_val * (arctan_s_max - arctan_s_min_pos))
            for t_val in t_biased.tolist()
        ]
        arctan_biased_values_tensor = torch.tensor(arctan_biased_values_list)
        arctan_biased_values_tensor[0] = arctan_s_max
        arctan_biased_values_tensor[-1] = arctan_s_min_pos

        active_sigmas_list = [
            c * math.tan(val) for val in arctan_biased_values_tensor.tolist()
        ]

        active_sigmas_list[0] = s_max
        if N > 0:  # N=12
            active_sigmas_list[-1] = s_min_pos

        final_sigmas_list = active_sigmas_list + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # Using the simplified clamp loop for post-processing for 12-step versions
        sigmas[0] = s_max
        for i in range(N - 1):  # Iterate N-1 times, from 0 to N-2 (0 to 10 for N=12)
            actual_lower_bound = s_min_pos if (i + 1) < N else epsilon
            sigmas[i + 1] = torch.clamp(
                sigmas[i + 1], min=actual_lower_bound, max=sigmas[i].item() - epsilon
            )

        if N > 0:  # N=12
            sigmas[N - 1] = max(s_min_pos, epsilon)
            if N > 1:
                sigmas[N - 1] = min(
                    sigmas[N - 1].item(), sigmas[N - 2].item() - epsilon
                )
                sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos, epsilon)
        sigmas[N] = 0.0

        return (sigmas,)
