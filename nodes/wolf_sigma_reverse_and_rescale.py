import torch
import math


class WolfSigmaReverseAndRescale:
    """
    Takes an existing SIGMAS tensor, reverses the order of its active steps,
    and then rescales them to a new target sigma_max and sigma_min_positive.
    The last sigma (0.0) remains in place.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "reverse_and_rescale_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "new_sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 1000.0, "step": 0.1},
                ),
                "new_sigma_min_positive": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0001, "max": 10.0, "step": 0.0001},
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

    def reverse_and_rescale_sigmas(
        self, sigmas, new_sigma_max, new_sigma_min_positive, min_epsilon_spacing
    ):
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 2:
            # Invalid input, return a default 1-step schedule
            print(
                "Warning: Invalid input sigmas for ReverseAndRescale. Returning default."
            )
            return (torch.tensor([float(new_sigma_max), 0.0], dtype=torch.float32),)

        s_max_new = float(new_sigma_max)
        s_min_pos_new = float(new_sigma_min_positive)
        epsilon = float(min_epsilon_spacing)

        if s_max_new <= s_min_pos_new:
            s_max_new = s_min_pos_new + (len(sigmas) - 1) * epsilon
        if s_min_pos_new <= 0:
            s_min_pos_new = epsilon

        active_sigmas_original = sigmas[:-1].clone()  # Exclude the final 0.0
        num_active_steps = len(active_sigmas_original)

        if num_active_steps == 0:
            return (
                torch.tensor([s_max_new, 0.0], dtype=torch.float32),
            )  # Only had 0.0

        # 1. Reverse the active sigmas
        reversed_active_sigmas = torch.flip(active_sigmas_original, dims=[0])

        # 2. Rescale to new_sigma_min_positive and new_sigma_max
        # Current range of reversed sigmas
        old_min = reversed_active_sigmas.min().item()
        old_max = reversed_active_sigmas.max().item()

        rescaled_sigmas = torch.zeros_like(reversed_active_sigmas)

        if num_active_steps == 1:
            rescaled_sigmas[0] = s_max_new  # Single active step becomes the new max
        elif old_max - old_min < epsilon:  # All values were nearly the same
            # Create a linear spread in the new range
            rescaled_sigmas = torch.linspace(
                s_max_new, s_min_pos_new, num_active_steps, device="cpu"
            )
        else:
            # Linear rescale: y = y_min_new + (x - x_min_old) * (y_max_new - y_min_new) / (x_max_old - x_min_old)
            # Here, we want reversed_active_sigmas[0] (original smallest) to map to s_max_new
            # and reversed_active_sigmas[-1] (original largest) to map to s_min_pos_new.
            # This maintains the "shape" of the reversed distribution within the new bounds.
            rescaled_sigmas = s_min_pos_new + (reversed_active_sigmas - old_min) * (
                s_max_new - s_min_pos_new
            ) / (old_max - old_min)
            # The above maps min to min, max to max. We need to flip again if we want reversed order to map directly.
            # Simpler: rescale so first element is s_max_new, last is s_min_pos_new, preserving relative internal distances.
            # The previous formula did this if old_min was mapped to s_min_pos_new and old_max to s_max_new.
            # Let's make it explicit: reversed_active_sigmas[0] should become s_max_new, reversed_active_sigmas[-1] becomes s_min_pos_new.

            # Normalize reversed_active_sigmas to [0, 1] first
            normalized_reversed = (reversed_active_sigmas - old_min) / (
                old_max - old_min
            )
            # Then scale to new range: s_min_pos_new is at the end, s_max_new at the start.
            rescaled_sigmas = s_max_new - normalized_reversed * (
                s_max_new - s_min_pos_new
            )

            # Ensure boundary conditions due to floating point math
            rescaled_sigmas[0] = s_max_new
            rescaled_sigmas[-1] = s_min_pos_new

        # Standard Post-processing
        final_sigmas = torch.zeros(num_active_steps + 1, dtype=torch.float32)
        if num_active_steps > 0:
            final_sigmas[:num_active_steps] = rescaled_sigmas
            final_sigmas[0] = s_max_new  # Re-assert

            current_s_max = s_max_new
            current_s_min_positive = s_min_pos_new

            sigma_min_floor_for_last_active = current_s_min_positive
            if current_s_min_positive == 0.0:
                if num_active_steps > 1:
                    sigma_min_floor_for_last_active = epsilon
            else:
                sigma_min_floor_for_last_active = max(current_s_min_positive, epsilon)

            final_sigmas[num_active_steps - 1] = max(
                final_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
            )

            for i in range(num_active_steps - 2, -1, -1):
                final_sigmas[i] = max(final_sigmas[i], final_sigmas[i + 1] + epsilon)
                final_sigmas[i] = max(final_sigmas[i], epsilon)

            final_sigmas[0] = min(final_sigmas[0], current_s_max)
            if num_active_steps > 1:
                final_sigmas[0] = max(final_sigmas[0], final_sigmas[1] + epsilon)
            final_sigmas[0] = max(final_sigmas[0], epsilon)

            final_sigmas[num_active_steps - 1] = max(
                final_sigmas[num_active_steps - 1], sigma_min_floor_for_last_active
            )
            if num_active_steps > 1:
                final_sigmas[num_active_steps - 2] = max(
                    final_sigmas[num_active_steps - 2],
                    final_sigmas[num_active_steps - 1] + epsilon,
                )

        final_sigmas[num_active_steps] = 0.0

        return (final_sigmas,)
