import torch
import math


class WolfSigmaArctanImbalancedNStep:
    """
    Generates N_steps + 1 sigmas with a bias factor applied in the arctan domain.
    Active sigmas (sigma_max to sigma_min_positive) are spaced non-linearly.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_imbalanced_arctan_sigmas"
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
        num_steps,
        sigma_max,
        sigma_min_positive,
        c_factor,
        bias_factor,
        min_epsilon_spacing,
    ):
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        c = float(c_factor)
        bias = float(bias_factor)
        epsilon = float(min_epsilon_spacing)

        if N <= 0:
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)
        if s_max <= s_min_pos:
            s_max = s_min_pos + N * epsilon
        if s_min_pos <= 0:
            s_min_pos = epsilon
        if c <= 0:
            c = epsilon

        arctan_s_max = math.atan(s_max / c)
        arctan_s_min_pos = math.atan(s_min_pos / c)

        if N == 1:
            arctan_biased_values_tensor = torch.tensor([arctan_s_max, arctan_s_min_pos])
        else:
            t_linear = torch.linspace(0, 1, N, device="cpu")
            power = 1.0 - bias
            if power <= 0:
                power = epsilon
            t_biased = t_linear.pow(power)

            # Map t_biased (0 to 1) to arctan values (arctan_s_max down to arctan_s_min_pos)
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
        if N > 0:
            active_sigmas_list[-1] = s_min_pos

        final_sigmas_list = active_sigmas_list + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # Apply the same robust post-processing from WolfSigmaArctanNStep
        sigmas[0] = s_max
        if N > 0:
            sigmas[N - 1] = s_min_pos

        for i in range(N - 1):
            current_sigma = sigmas[i].item()
            next_sigma = sigmas[i + 1].item()
            if current_sigma <= next_sigma + epsilon:
                sigmas[i + 1] = current_sigma - epsilon
            sigmas[i + 1] = max(
                sigmas[i + 1].item(), s_min_pos if i + 1 < N - 1 else s_min_pos, epsilon
            )

        if N > 0:
            sigmas[N - 1] = max(s_min_pos, epsilon)
            if N > 1:
                sigmas[N - 1] = min(
                    sigmas[N - 1].item(), sigmas[N - 2].item() - epsilon
                )
                sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos, epsilon)
        sigmas[N] = 0.0

        # Fallback to simplified clamp loop from 12step if above isn't perfect
        sigmas[0] = s_max
        for i in range(N - 1):
            actual_lower_bound = s_min_pos if (i + 1) < N else epsilon
            sigmas[i + 1] = torch.clamp(
                sigmas[i + 1], min=actual_lower_bound, max=sigmas[i].item() - epsilon
            )
        if N > 0:
            sigmas[N - 1] = max(s_min_pos, epsilon)
            if N > 1:
                sigmas[N - 1] = min(
                    sigmas[N - 1].item(), sigmas[N - 2].item() - epsilon
                )
                sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos, epsilon)
        sigmas[N] = 0.0

        return (sigmas,)
