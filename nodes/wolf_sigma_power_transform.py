import torch

# import math


class WolfSigmaPowerTransform:
    """
    Applies a power transformation to an existing sigma schedule.
    Power > 1.0: concentrates steps towards the minimum of the schedule range.
    Power < 1.0: concentrates steps towards the maximum of the schedule range.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "transform_sigmas_power"
    CATEGORY = "sampling/sigmas_wolf/transform"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_in": ("SIGMAS",),
                "power": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.05},
                ),
                "override_input_min": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "override_input_max": (
                    "FLOAT",
                    {
                        "default": -1.0,
                        "min": -1.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
            }
        }

    def transform_sigmas_power(
        self, sigmas_in, power, override_input_min, override_input_max
    ):
        sigmas_tensor = sigmas_in.clone().cpu()
        current_power = float(power)

        if len(sigmas_tensor) == 0:
            return (sigmas_tensor,)
        if len(sigmas_tensor) == 1:
            # Cannot transform a single sigma in this way, or handle as special case? Return as is for now.
            if (
                sigmas_tensor[0].item() == 0.0
                and override_input_min >= 0
                and override_input_max > override_input_min
            ):
                # If input is just [0.0] but overrides are given, maybe create a point?
                # This is ambiguous, safer to return as is or error.
                pass  # Return as is
            return (sigmas_tensor,)

        # Separate the final 0.0 if it exists
        has_final_zero = sigmas_tensor[-1].item() == 0.0
        active_sigmas = (
            sigmas_tensor[:-1]
            if has_final_zero and len(sigmas_tensor) > 1
            else sigmas_tensor
        )

        if len(active_sigmas) == 0:  # Only contained [0.0] or was empty after slice
            return (sigmas_tensor,)
        if (
            len(active_sigmas) == 1 and not has_final_zero
        ):  # Single non-zero sigma, no range to transform
            return (sigmas_tensor,)
        if (
            len(active_sigmas) == 1 and has_final_zero
        ):  # e.g. [S, 0.0], S is max and min of active range. Power transform has no effect.
            return (sigmas_tensor,)

        # Determine actual min and max for normalization range
        actual_min = float(override_input_min)
        actual_max = float(override_input_max)

        if actual_min < 0:  # Auto-detect min from active_sigmas
            actual_min = active_sigmas[-1].item()
        if actual_max < 0:  # Auto-detect max from active_sigmas
            actual_max = active_sigmas[0].item()

        if (
            actual_max <= actual_min
        ):  # Range is zero or invalid, cannot transform meaningfully
            # print(f"WolfSigmaPowerTransform: Warning - max ({actual_max}) <= min ({actual_min}). Cannot transform, returning original.")
            return (sigmas_tensor,)

        transformed_active_sigmas = torch.zeros_like(active_sigmas)
        range_val = actual_max - actual_min
        if range_val < 1e-9:  # Effectively zero range
            # print(f"WolfSigmaPowerTransform: Warning - range is too small. Returning original.")
            return (sigmas_tensor,)

        for i, s_val_tensor in enumerate(active_sigmas):
            s_val = s_val_tensor.item()
            # Normalize s_val to [0,1] within the determined range
            norm_s = (s_val - actual_min) / range_val
            norm_s = max(0.0, min(1.0, norm_s))  # Clamp to [0,1]

            # Apply power. Note: (0^power is 0 for power>0; 1^power is 1)
            if norm_s == 0.0 and current_power <= 0:
                powered_s = 0.0  # Avoid 0 to non-positive power if norm_s is 0
            elif norm_s == 1.0 and current_power <= 0:
                powered_s = (
                    1.0  # Avoid 1 to non-positive power if norm_s is 1, result is 1
                )
            else:
                powered_s = norm_s**current_power

            # Denormalize back to [actual_min, actual_max] range
            new_s = actual_min + powered_s * range_val
            transformed_active_sigmas[i] = new_s

        # Reconstruct the full sigma array
        if has_final_zero:
            final_sigmas = torch.cat(
                (
                    transformed_active_sigmas,
                    torch.tensor([0.0], dtype=torch.float32, device="cpu"),
                )
            )
        else:
            final_sigmas = transformed_active_sigmas

        # Ensure descending order and non-negativity for safety, especially after power transform
        # This loop should ensure s_i >= s_{i+1} (or slightly > if not the final zero)
        for i in range(len(final_sigmas) - 1):
            # If it's the last two elements and the last one is 0.0, ensure previous is >= 0
            if i == len(final_sigmas) - 2 and final_sigmas[i + 1].item() == 0.0:
                final_sigmas[i] = max(0.0, final_sigmas[i].item())
            elif (
                final_sigmas[i].item() < final_sigmas[i + 1].item() + 1e-7
            ):  # Not descending or too close
                # Attempt to preserve the transformed value of s[i] if possible,
                # and push s[i+1] down. This can cascade.
                # A simpler approach: if order is broken, try to average or force spacing.
                # For now, a basic re-spacing or small reduction on s[i+1].
                if (
                    i > 0
                    and final_sigmas[i - 1].item() > final_sigmas[i + 1].item() + 2e-7
                ):  # if previous gives room
                    final_sigmas[i] = (
                        final_sigmas[i - 1].item() + final_sigmas[i + 1].item()
                    ) / 2.0
                else:  # Default to making next one smaller
                    final_sigmas[i + 1] = final_sigmas[i].item() * 0.99
            final_sigmas[i] = max(0.0, final_sigmas[i].item())  # Ensure non-negative

        if (
            len(final_sigmas) > 0
        ):  # Ensure last element (if not the special 0) is non-negative
            final_sigmas[-1] = max(0.0, final_sigmas[-1].item())

        # One final check to make sure the one before last is >=0 if last is 0
        if (
            len(final_sigmas) > 1
            and final_sigmas[-1].item() == 0.0
            and final_sigmas[-2].item() < 0.0
        ):
            final_sigmas[-2] = 0.0

        return (final_sigmas.cpu(),)
