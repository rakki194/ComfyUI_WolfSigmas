import torch
import math
import comfy.k_diffusion.sampling as k_diffusion_sampling

# Attempt to import the base AYS schedule for Chroma.
# This is expected to be a list or 1D tensor of sigma values, sorted descending.
try:
    from ..wolf_sigma_constants import AYS_CHROMA_SIGMAS_BASE
except ImportError:
    # Fallback if the constant is not found, using a placeholder.
    # This will likely cause issues if not replaced with the actual schedule.
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


class WolfSigmaAYS12Step:
    """
    Generates a 12-step (13 sigmas) schedule for Chroma, inspired by Align Your Steps (AYS).
    It uses log-linear interpolation from a pre-defined high-resolution AYS base schedule
    (AYS_CHROMA_SIGMAS_BASE), scaled to the target sigma_max and respecting sigma_min.
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
                    {
                        "default": 80.0,
                        "min": 0.01,
                        "max": 1000.0,
                        "step": 0.1,
                        "doc": "Target maximum sigma (e.g., 80.0 for Chroma)",
                    },
                ),
                "sigma_min": (
                    "FLOAT",
                    {
                        "default": 0.002,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.001,
                        "doc": "Target minimum sigma for the last active step (e.g., 0.002 for Chroma)",
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
                        "doc": "Minimum spacing between sigmas to ensure positivity and decrease.",
                    },
                ),
            }
        }

    def _log_linear_interpolate(self, y_values, x_norm):
        """
        Performs log-linear interpolation.
        y_values: list or 1D tensor of values (e.g., sigmas), sorted.
        x_norm: normalized query point (0 to 1).
        """
        if not isinstance(y_values, torch.Tensor):
            y_values = torch.tensor(y_values, dtype=torch.float32)

        num_points = len(y_values)
        if num_points == 0:
            return 0.0  # Should not happen with valid base schedule
        if num_points == 1:
            return y_values[0].item()

        # Denormalize x to fit the array indices
        x_abs = x_norm * (num_points - 1)

        idx0 = math.floor(x_abs)
        idx1 = min(idx0 + 1, num_points - 1)  # Clamp idx1 to not exceed bounds
        idx0 = min(
            idx0, num_points - 1
        )  # Clamp idx0 as well, esp. if x_abs is exactly num_points-1

        weight = x_abs - idx0

        y0 = y_values[idx0].item()
        y1 = y_values[idx1].item()

        # Ensure positivity for log
        log_y0 = math.log(max(y0, 1e-9))
        log_y1 = math.log(max(y1, 1e-9))

        log_y_interp = (1.0 - weight) * log_y0 + weight * log_y1
        return math.exp(log_y_interp)

    def get_sigmas(self, sigma_max, sigma_min, min_epsilon_spacing):
        num_active_steps = 12  # Fixed for this node

        base_schedule_sigmas = AYS_CHROMA_SIGMAS_BASE
        if not isinstance(base_schedule_sigmas, torch.Tensor):
            base_schedule_sigmas = torch.tensor(
                base_schedule_sigmas, dtype=torch.float32
            )

        if len(base_schedule_sigmas) == 0:
            print("Error: AYS_CHROMA_SIGMAS_BASE is empty. Returning Karras.")
            # Fallback to a Karras schedule for 12 steps
            return (
                k_diffusion_sampling.get_sigmas_karras(
                    n=num_active_steps,
                    sigma_min=sigma_min,
                    sigma_max=sigma_max,
                    rho=7.0,
                    device="cpu",
                ),
            )

        # Scale the base AYS schedule:
        # The AYS paper schedules are typically defined for a specific model (e.g. SD1.5 max ~14.6, SDXL max ~100s)
        # We need to scale it to the target_sigma_max for Chroma (e.g. 80.0)
        # WolfSigmaAYSPaperSchedulePicker scales based on target_sigma_max / original_max_from_paper.
        # We'll assume AYS_CHROMA_SIGMAS_BASE[0] is its original_max.
        original_base_max = base_schedule_sigmas[0].item()
        scaled_base_sigmas = base_schedule_sigmas.clone()  # Make a copy

        if original_base_max > 1e-9 and sigma_max > 1e-9:
            scale_factor = sigma_max / original_base_max
            scaled_base_sigmas *= scale_factor
        else:
            # if original_base_max is ~0, can't scale. Use as is or error.
            # if sigma_max is ~0, result is ~0.
            # For now, if original_base_max is too small, we don't scale, which might be problematic.
            print(
                f"Warning: Original AYS base max ({original_base_max}) is very small, scaling might be inaccurate."
            )

        active_sigmas = torch.zeros(num_active_steps, dtype=torch.float32)

        if num_active_steps == 1:
            active_sigmas[0] = scaled_base_sigmas[0]  # Use the max from scaled base
        else:
            for i in range(num_active_steps):
                norm_time = i / (num_active_steps - 1)  # Normalized time from 0 to 1
                active_sigmas[i] = self._log_linear_interpolate(
                    scaled_base_sigmas, norm_time
                )

        # Post-processing (common logic from other nodes)
        # Ensure the last active sigma respects sigma_min (or is at least min_epsilon_spacing).
        sigma_min_floor_for_last_active = sigma_min
        if sigma_min == 0.0:
            if num_active_steps > 1:
                sigma_min_floor_for_last_active = min_epsilon_spacing
        else:  # sigma_min > 0.0
            sigma_min_floor_for_last_active = max(sigma_min, min_epsilon_spacing)

        active_sigmas[num_active_steps - 1] = max(
            active_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
        )

        # Iterate backwards from the second to last active sigma.
        for i in range(num_active_steps - 2, -1, -1):
            active_sigmas[i] = max(
                active_sigmas[i], active_sigmas[i + 1] + min_epsilon_spacing
            )
            active_sigmas[i] = max(active_sigmas[i], min_epsilon_spacing)

        # Ensure the first sigma is no more than sigma_max and adheres to positivity/decrease.
        active_sigmas[0] = min(
            active_sigmas[0], sigma_max
        )  # Should be very close already
        if num_active_steps > 1:
            active_sigmas[0] = max(
                active_sigmas[0], active_sigmas[1] + min_epsilon_spacing
            )
        active_sigmas[0] = max(active_sigmas[0], min_epsilon_spacing)

        # Final pass for last active sigma's floor and the one before it.
        active_sigmas[num_active_steps - 1] = max(
            active_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
        )
        if num_active_steps > 1:
            active_sigmas[num_active_steps - 2] = max(
                active_sigmas[num_active_steps - 2],
                active_sigmas[num_active_steps - 1] + min_epsilon_spacing,
            )

        # Combine with the final 0.0
        final_sigmas = torch.zeros(num_active_steps + 1, dtype=torch.float32)
        final_sigmas[:num_active_steps] = active_sigmas
        # final_sigmas[num_active_steps] is already 0.0

        return (final_sigmas,)
