import torch
import math


class WolfSigmaSlice:
    """
    Slices an existing sigma schedule, selecting a sub-sequence of active sigmas.
    The final 0.0 sigma, if present, is re-appended.
    Optionally re-ensures strictly decreasing order after slicing.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "slice_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_index": ("INT", {"default": -1, "min": -1, "max": 10000}),
                "step_size": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "ensure_strictly_decreasing": ("BOOLEAN", {"default": True}),
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

    def slice_sigmas(
        self,
        sigmas,
        start_index,
        end_index,
        step_size,
        ensure_strictly_decreasing,
        min_epsilon_spacing,
    ):
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
            print("Warning: Invalid input sigmas for Slice. Returning as is.")
            return (sigmas,)

        s_input = sigmas.clone()
        epsilon = float(min_epsilon_spacing)

        if len(s_input) == 0:
            return (s_input,)

        has_final_zero = s_input[-1].item() == 0.0

        active_sigmas_original = (
            s_input[:-1] if has_final_zero and len(s_input) > 1 else s_input.clone()
        )

        if len(active_sigmas_original) == 0:
            return (s_input,)

        actual_start_index = start_index
        actual_end_index = end_index

        if actual_start_index < 0:
            actual_start_index = 0  # Clamp negative start to 0

        # Python slice behavior: if end_index is -1, it means up to the second to last.
        # For user convenience, if end_index is -1, treat as 'to the very end'.
        # If end_index is a large positive, Python slice handles it by stopping at the end.
        if actual_end_index == -1:
            actual_end_index = len(active_sigmas_original)  # Slice up to length
        elif actual_end_index < -1:  # Other negative indices
            actual_end_index = (
                len(active_sigmas_original) + actual_end_index
            )  # Convert to positive index or 0
            if actual_end_index < 0:
                actual_end_index = 0

        # Ensure indices are within reasonable python slice bounds
        # start can be > end, resulting in empty. step must be positive.
        actual_step_size = max(1, step_size)

        sliced_active_sigmas = active_sigmas_original[
            actual_start_index:actual_end_index:actual_step_size
        ]

        active_sigmas_modified = sliced_active_sigmas.clone()

        if ensure_strictly_decreasing and len(active_sigmas_modified) > 0:
            active_sigmas_modified = torch.clamp(active_sigmas_modified, min=epsilon)
            if len(active_sigmas_modified) > 1:
                for i in range(len(active_sigmas_modified) - 2, -1, -1):
                    current_i_val = active_sigmas_modified[i].item()
                    next_i_val = active_sigmas_modified[i + 1].item()
                    if current_i_val <= next_i_val:
                        active_sigmas_modified[i] = next_i_val + epsilon
            # Ensure the first element is also clamped if it was the only one or became too small
            if (
                len(active_sigmas_modified) > 0
                and active_sigmas_modified[0].item() < epsilon
            ):
                active_sigmas_modified[0] = torch.tensor(
                    epsilon, dtype=s_input.dtype, device=s_input.device
                )

        # Re-attach final zero if it was present
        if has_final_zero:
            if len(active_sigmas_modified) == 0:
                return (
                    torch.tensor([0.0], dtype=s_input.dtype, device=s_input.device),
                )
            # Ensure last active sigma is positive before adding zero
            if active_sigmas_modified[-1].item() < epsilon:
                active_sigmas_modified[-1] = torch.tensor(
                    epsilon,
                    dtype=active_sigmas_modified.dtype,
                    device=active_sigmas_modified.device,
                )
                # Check if this adjustment caused an issue with the one before it
                if (
                    len(active_sigmas_modified) > 1
                    and active_sigmas_modified[-2].item()
                    <= active_sigmas_modified[-1].item()
                ):
                    active_sigmas_modified[-2] = (
                        active_sigmas_modified[-1].item() + epsilon
                    )
            s_output = torch.cat(
                (
                    active_sigmas_modified,
                    torch.tensor([0.0], dtype=s_input.dtype, device=s_input.device),
                )
            )
        else:  # No final zero originally
            if len(active_sigmas_modified) == 0:
                return (torch.empty(0, dtype=s_input.dtype, device=s_input.device),)
            s_output = active_sigmas_modified
            # If no final zero, ensure the last actual sigma is positive (if any exist)
            if len(s_output) > 0 and s_output[-1].item() < epsilon:
                s_output[-1] = torch.tensor(
                    epsilon,
                    dtype=active_sigmas_modified.dtype,
                    device=active_sigmas_modified.device,
                )
                if len(s_output) > 1 and s_output[-2].item() <= s_output[-1].item():
                    s_output[-2] = s_output[-1].item() + epsilon

        return (s_output,)
