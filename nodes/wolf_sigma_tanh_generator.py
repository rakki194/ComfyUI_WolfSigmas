import torch
import math


class WolfSigmaTanhGenerator:
    """
    Generates N_steps + 1 sigmas based on a scaled and shifted tanh function
    of normalized time. Sigmas range from sigma_max to sigma_min_positive.
    The tanh function provides an S-shaped curve.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_tanh_sigmas"
    CATEGORY = "sampling/sigmas_wolf/generate"

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
                "tanh_scale": (
                    "FLOAT",
                    {
                        "default": 3.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "doc": "Scales the input to tanh, controlling steepness. Higher values = steeper.",
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

    def generate_tanh_sigmas(
        self, num_steps, sigma_max, sigma_min_positive, tanh_scale, min_epsilon_spacing
    ):
        num_active_steps = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        scale = float(tanh_scale)
        epsilon = float(min_epsilon_spacing)

        if num_active_steps < 1:
            return (torch.tensor([s_max, 0.0], dtype=torch.float32, device="cpu"),)

        if s_max <= s_min_pos:
            s_max = s_min_pos + max(epsilon * num_active_steps, epsilon * 10)
        if s_min_pos <= 0:
            s_min_pos = epsilon

        active_sigmas = torch.zeros(num_active_steps, dtype=torch.float32, device="cpu")

        if num_active_steps == 1:
            active_sigmas[0] = s_max
        else:
            t_points = torch.linspace(scale, -scale, num_active_steps, device="cpu")
            tanh_values = torch.tanh(t_points)

            min_tanh_val = torch.tanh(torch.tensor(-scale, device="cpu"))
            max_tanh_val = torch.tanh(torch.tensor(scale, device="cpu"))

            if (max_tanh_val - min_tanh_val).abs() < 1e-9:
                normalized_tanh = torch.linspace(0, 1, num_active_steps, device="cpu")
            else:
                normalized_tanh = (tanh_values - min_tanh_val) / (
                    max_tanh_val - min_tanh_val
                )

            active_sigmas = s_min_pos + normalized_tanh * (s_max - s_min_pos)

            active_sigmas[0] = s_max
            active_sigmas[-1] = s_min_pos

        # Standard Post-processing
        final_sigmas = torch.zeros(
            num_active_steps + 1, dtype=torch.float32, device="cpu"
        )
        if num_active_steps > 0:
            final_sigmas[:num_active_steps] = active_sigmas
            final_sigmas[0] = s_max  # Re-assert

            current_s_max = s_max
            current_s_min_positive = s_min_pos

            sigma_min_floor_for_last_active = current_s_min_positive
            if abs(current_s_min_positive) < epsilon:
                if num_active_steps > 1:
                    sigma_min_floor_for_last_active = epsilon
                else:
                    sigma_min_floor_for_last_active = max(
                        current_s_min_positive, epsilon
                    )
            else:
                sigma_min_floor_for_last_active = max(current_s_min_positive, epsilon)

            if num_active_steps > 0:
                current_val = final_sigmas[num_active_steps - 1].item()
                final_sigmas[num_active_steps - 1] = max(
                    current_val, sigma_min_floor_for_last_active
                )

            if num_active_steps > 1:
                for i in range(num_active_steps - 2, -1, -1):
                    val_i = final_sigmas[i].item()
                    val_i_plus_1 = final_sigmas[i + 1].item()
                    new_val_i = max(val_i, val_i_plus_1 + epsilon)
                    new_val_i = max(new_val_i, epsilon)
                    final_sigmas[i] = new_val_i

                val_0 = final_sigmas[0].item()
                val_1 = final_sigmas[1].item()
                new_val_0 = min(val_0, current_s_max)
                new_val_0 = max(new_val_0, val_1 + epsilon)
                new_val_0 = max(new_val_0, epsilon)
                final_sigmas[0] = new_val_0

            if num_active_steps > 0:
                current_val = final_sigmas[num_active_steps - 1].item()
                final_sigmas[num_active_steps - 1] = max(
                    current_val, sigma_min_floor_for_last_active
                )
                if num_active_steps == 1:
                    val_0 = final_sigmas[0].item()
                    final_sigmas[0] = max(val_0, sigma_min_floor_for_last_active)

        final_sigmas[num_active_steps] = 0.0

        return (final_sigmas,)
