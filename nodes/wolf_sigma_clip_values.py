import torch
import math


class WolfSigmaClipValues:
    """
    Clips individual active sigma values in a schedule to a specified minimum and maximum.
    The final 0.0 sigma, if present, remains unchanged.
    Optionally re-ensures strictly decreasing order after clipping.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "clip_sigma_values"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "min_clip_value": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1000.0, "step": 0.001},
                ),
                "max_clip_value": (
                    "FLOAT",
                    {"default": 150.0, "min": 0.0, "max": 1000.0, "step": 0.01},
                ),
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

    def clip_sigma_values(
        self,
        sigmas,
        min_clip_value,
        max_clip_value,
        ensure_strictly_decreasing,
        min_epsilon_spacing,
    ):
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
            print("Warning: Invalid input sigmas for ClipValues. Returning as is.")
            return (sigmas,)

        s_input = sigmas.clone()
        epsilon = float(min_epsilon_spacing)
        min_val = float(min_clip_value)
        max_val = float(max_clip_value)

        if min_val < 0:
            min_val = 0  # Sigmas must be non-negative
        if max_val < min_val:
            max_val = min_val  # Ensure max is not less than min

        if len(s_input) == 0:
            return (s_input,)

        has_final_zero = s_input[-1].item() == 0.0

        active_sigmas_original = (
            s_input[:-1] if has_final_zero and len(s_input) > 1 else s_input.clone()
        )

        if len(active_sigmas_original) == 0:
            return (s_input,)

        active_sigmas_modified = torch.clamp(
            active_sigmas_original, min=min_val, max=max_val
        )

        if ensure_strictly_decreasing and len(active_sigmas_modified) > 0:
            active_sigmas_modified = torch.sort(
                active_sigmas_modified, descending=True
            )[0]
            active_sigmas_modified = torch.clamp(active_sigmas_modified, min=epsilon)

            if len(active_sigmas_modified) > 1:
                for i in range(len(active_sigmas_modified) - 2, -1, -1):
                    current_i_val = active_sigmas_modified[i].item()
                    next_i_val = active_sigmas_modified[i + 1].item()
                    if current_i_val <= next_i_val:
                        active_sigmas_modified[i] = next_i_val + epsilon

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
            if active_sigmas_modified[-1].item() < epsilon:
                active_sigmas_modified[-1] = torch.tensor(
                    epsilon,
                    dtype=active_sigmas_modified.dtype,
                    device=active_sigmas_modified.device,
                )
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
        else:
            if len(active_sigmas_modified) == 0:
                return (torch.empty(0, dtype=s_input.dtype, device=s_input.device),)
            s_output = active_sigmas_modified
            if len(s_output) > 0 and s_output[-1].item() < epsilon:
                s_output[-1] = torch.tensor(
                    epsilon,
                    dtype=active_sigmas_modified.dtype,
                    device=active_sigmas_modified.device,
                )
                if len(s_output) > 1 and s_output[-2].item() <= s_output[-1].item():
                    s_output[-2] = s_output[-1].item() + epsilon

        return (s_output,)
