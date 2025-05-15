import torch
import math
import comfy.k_diffusion.sampling as k_diffusion_sampling

try:
    from ..wolf_sigma_constants import AYS_CHROMA_SIGMAS_BASE
except ImportError:
    print(
        "Warning: AYS_CHROMA_SIGMAS_BASE not found in wolf_sigma_constants.py. Using a 1.0-max, 11-point AYS-derived placeholder."
    )
    AYS_CHROMA_SIGMAS_BASE = torch.tensor(
        [
            1.0,
            0.4320903,
            0.2579884,
            0.1492299,
            0.0918235,
            0.0589798,
            0.0379747,
            0.0259993,
            0.0160109,
            0.0077318,
            0.0019843,
        ],
        dtype=torch.float32,
    )


class WolfSigmaAYSImbalanced12Step:
    """
    Generates a 12-step schedule for Chroma, inspired by AYS, with a bias factor.
    Uses log-linear interpolation from AYS_CHROMA_SIGMAS_BASE, scaled and biased.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigma_max": (
                    "FLOAT",
                    {"default": 80.0, "min": 0.01, "max": 1000.0, "step": 0.1},
                ),
                "sigma_min": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0, "max": 10.0, "step": 0.001},
                ),
                "bias_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01},
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

    def _log_linear_interpolate(self, y_values, x_norm):
        if not isinstance(y_values, torch.Tensor):
            y_values = torch.tensor(y_values, dtype=torch.float32)
        num_points = len(y_values)
        if num_points == 0:
            return 0.0
        if num_points == 1:
            return y_values[0].item()
        x_abs = x_norm * (num_points - 1)
        idx0 = math.floor(x_abs)
        idx1 = min(idx0 + 1, num_points - 1)
        idx0 = min(idx0, num_points - 1)
        weight = x_abs - idx0
        y0 = y_values[idx0].item()
        y1 = y_values[idx1].item()
        log_y0 = math.log(max(y0, 1e-9))
        log_y1 = math.log(max(y1, 1e-9))
        log_y_interp = (1.0 - weight) * log_y0 + weight * log_y1
        return math.exp(log_y_interp)

    def get_sigmas(self, sigma_max, sigma_min, bias_factor, min_epsilon_spacing):
        num_active_steps = 12  # Fixed for this node

        base_schedule_sigmas = AYS_CHROMA_SIGMAS_BASE
        if not isinstance(base_schedule_sigmas, torch.Tensor):
            base_schedule_sigmas = torch.tensor(
                base_schedule_sigmas, dtype=torch.float32
            )

        if len(base_schedule_sigmas) == 0:
            print("Error: AYS_CHROMA_SIGMAS_BASE is empty. Returning Karras.")
            return (
                k_diffusion_sampling.get_sigmas_karras(
                    n=num_active_steps,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=7.0,
                    device="cpu",
                ),
            )

        original_base_max = base_schedule_sigmas[0].item()
        scaled_base_sigmas = base_schedule_sigmas.clone()
        if original_base_max > 1e-9 and sigma_max > 1e-9:
            scale_factor = sigma_max / original_base_max
            scaled_base_sigmas *= scale_factor
        else:
            print(
                f"Warning: Original AYS base max ({original_base_max}) is very small, scaling might be inaccurate."
            )

        active_sigmas = torch.zeros(num_active_steps, dtype=torch.float32)

        if num_active_steps == 1:  # Should not happen as num_active_steps is 12
            active_sigmas[0] = scaled_base_sigmas[0]
        else:
            for i in range(num_active_steps):
                norm_time_linear = i / (num_active_steps - 1)
                norm_time_biased = norm_time_linear**bias_factor
                norm_time_biased = max(0.0, min(1.0, norm_time_biased))
                active_sigmas[i] = self._log_linear_interpolate(
                    scaled_base_sigmas, norm_time_biased
                )

        # Post-processing
        sigma_min_floor_for_last_active = sigma_min
        if sigma_min == 0.0:
            # For 12 steps, last active sigma should be > min_epsilon_spacing if sigma_min is 0
            sigma_min_floor_for_last_active = min_epsilon_spacing
        else:
            sigma_min_floor_for_last_active = max(sigma_min, min_epsilon_spacing)

        active_sigmas[num_active_steps - 1] = max(
            active_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
        )

        for i in range(num_active_steps - 2, -1, -1):
            active_sigmas[i] = max(
                active_sigmas[i], active_sigmas[i + 1] + min_epsilon_spacing
            )
            active_sigmas[i] = max(active_sigmas[i], min_epsilon_spacing)

        active_sigmas[0] = min(active_sigmas[0], sigma_max)
        # num_active_steps is 12, so num_active_steps > 1 is true
        active_sigmas[0] = max(active_sigmas[0], active_sigmas[1] + min_epsilon_spacing)
        active_sigmas[0] = max(active_sigmas[0], min_epsilon_spacing)

        active_sigmas[num_active_steps - 1] = max(
            active_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
        )
        active_sigmas[num_active_steps - 2] = max(
            active_sigmas[num_active_steps - 2],
            active_sigmas[num_active_steps - 1] + min_epsilon_spacing,
        )

        final_sigmas = torch.zeros(num_active_steps + 1, dtype=torch.float32)
        final_sigmas[:num_active_steps] = active_sigmas

        return (final_sigmas,)
