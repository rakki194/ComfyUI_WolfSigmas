import torch

# import math
import comfy.k_diffusion.sampling as k_diffusion_sampling


class WolfSigmaNormalizeRange:
    """
    Normalizes an existing sigma schedule to a new min/max range.
    The relative spacing of sigmas is preserved as much as possible.
    The final 0.0 sigma, if present, remains 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "normalize_range_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transform"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_in": ("SIGMAS",),
                "new_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.001},
                ),
                "new_min_positive": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0001, "max": 100.0, "step": 0.0001},
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

    def normalize_range_sigmas(
        self, sigmas_in, new_max, new_min_positive, min_epsilon_spacing
    ):
        sigmas = sigmas_in.clone().cpu()
        target_max = float(new_max)
        target_min_pos = float(new_min_positive)
        epsilon = float(min_epsilon_spacing)

        if len(sigmas) == 0:
            return (sigmas,)
        if target_max <= target_min_pos:
            # print("Warning: WolfSigmaNormalizeRange - new_max must be greater than new_min_positive. Returning original.")
            return (sigmas,)

        has_final_zero = sigmas[-1].item() == 0.0
        active_sigmas_list = []
        if has_final_zero and len(sigmas) > 1:
            active_sigmas_list = [s.item() for s in sigmas[:-1]]
        elif not has_final_zero:
            active_sigmas_list = [s.item() for s in sigmas]

        if not active_sigmas_list:  # only contained [0.0] or was empty
            return (sigmas,)
        if len(active_sigmas_list) == 1:  # single active sigma
            active_sigmas_list[0] = (
                target_max  # or target_min_pos? or average? Let's set to target_max.
            )
            if has_final_zero:
                return (
                    torch.tensor([active_sigmas_list[0], 0.0], dtype=torch.float32),
                )
            else:
                return (torch.tensor(active_sigmas_list, dtype=torch.float32),)

        original_max = active_sigmas_list[0]
        original_min = active_sigmas_list[-1]

        # Handle cases where original range is zero or min > max (already unlikely for valid sigmas)
        if original_max <= original_min:
            # print("Warning: WolfSigmaNormalizeRange - Original sigmas range is zero or invalid. Attempting uniform distribution.")
            # Create a Karras-like distribution if the input is problematic
            num_active = len(active_sigmas_list)
            if num_active > 1:
                # Use Karras as a fallback if original range is degenerate
                temp_sigmas = k_diffusion_sampling.get_sigmas_karras(
                    n=num_active,
                    sigma_min=target_min_pos,
                    sigma_max=target_max,
                    rho=7.0,
                    device="cpu",
                )
                # k_diffusion_sampling.get_sigmas_karras returns N+1 sigmas for N intervals.
                # We want N sigmas here for the active part.
                # So if num_active is the number of sigmas (e.g. 4 for S0,S1,S2,S3), get_sigmas_karras needs n=num_active
                # this will give num_active+1 sigmas. The last one is usually sigma_min.
                # We need to map these.
                # This logic for fallback needs refinement if get_sigmas_karras output format is different from expectations.
                # For now, let's assume we can pick the first num_active elements if it's a common scheduler.
                # Fallback: linear space if Karras is problematic here
                normalized_sigmas = torch.linspace(
                    target_max, target_min_pos, num_active
                )

            elif (
                num_active == 1
            ):  # Should have been caught by len(active_sigmas_list) == 1
                normalized_sigmas = torch.tensor([target_max], dtype=torch.float32)
            else:  # no active sigmas, already returned
                normalized_sigmas = torch.tensor([], dtype=torch.float32)

        else:
            original_range = original_max - original_min
            target_range = target_max - target_min_pos
            normalized_values = []
            for s_val in active_sigmas_list:
                if (
                    original_range == 0
                ):  # Should be caught by original_max <= original_min
                    norm_s = 0.0  # Avoid division by zero, map all to min or max
                else:
                    norm_s = (
                        s_val - original_min
                    ) / original_range  # normalize to 0-1 (inverted: 1 for max, 0 for min)

                # We want sigmas to decrease, so if s_val is original_max, norm_s is 1.
                # If s_val is original_min, norm_s is 0.
                # So, new_val = target_min_pos + norm_s * target_range
                # This maps original_min -> target_min_pos and original_max -> target_max
                new_val = target_min_pos + norm_s * target_range
                normalized_values.append(new_val)
            normalized_sigmas = torch.tensor(normalized_values, dtype=torch.float32)
            # The list was high-to-low. Normalization might flip if not careful.
            # Correct logic: map [original_min, original_max] to [target_min_pos, target_max]
            # s_norm = (s - original_min) / (original_max - original_min)  [0,1]
            # new_s = target_min_pos + s_norm * (target_max - target_min_pos)
            # Since sigmas are decreasing:
            # original_sigmas_sorted_ascending = sorted(active_sigmas_list)
            # original_min_asc = original_sigmas_sorted_ascending[0]
            # original_max_asc = original_sigmas_sorted_ascending[-1]
            # if original_max_asc <= original_min_asc: return sigmas # cannot normalize
            # original_range_asc = original_max_asc - original_min_asc

            # Simpler: scale and shift the whole block
            # Shift to start at 0: active_sigmas - original_min
            # Scale to new range: (active_sigmas - original_min) * (target_range / original_range)
            # Shift to new min: previous_result + target_min_pos
            if original_range > 1e-9:  # Avoid division by zero
                current_tensor = torch.tensor(active_sigmas_list, dtype=torch.float32)
                normalized_sigmas = (
                    (current_tensor - original_min) * (target_range / original_range)
                ) + target_min_pos
            else:  # original_range is zero, all values were the same
                # Distribute them in the new range, e.g. linspace
                normalized_sigmas = torch.linspace(
                    target_max, target_min_pos, len(active_sigmas_list), device="cpu"
                )

        # Reconstruct the full sigma array
        final_sigmas_list = list(normalized_sigmas.numpy())

        # Ensure first element is target_max and last active is target_min_pos
        if final_sigmas_list:
            final_sigmas_list[0] = target_max
            final_sigmas_list[-1] = target_min_pos

        if has_final_zero:
            final_sigmas_list.append(0.0)

        final_sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # Enforce monotonicity and positivity
        for i in range(len(final_sigmas) - 1):
            is_last_active_sigma = (i == len(final_sigmas) - 2) and has_final_zero
            required_next_val = final_sigmas[i].item() - (
                0
                if is_last_active_sigma and final_sigmas[i + 1].item() == 0.0
                else epsilon
            )

            if final_sigmas[i + 1].item() >= required_next_val:
                if (
                    is_last_active_sigma and final_sigmas[i + 1].item() == 0.0
                ):  # Current is before final 0
                    final_sigmas[i] = max(
                        target_min_pos, epsilon, final_sigmas[i].item()
                    )  # ensure it's at least target_min_pos or epsilon
                    if final_sigmas[i].item() == 0.0:
                        final_sigmas[i] = epsilon  # if somehow forced to 0
                else:
                    final_sigmas[i + 1] = final_sigmas[i].item() - epsilon

            final_sigmas[i + 1] = max(0.0, final_sigmas[i + 1].item())

        if (
            len(final_sigmas) > 1
            and final_sigmas[-1].item() == 0.0
            and final_sigmas[-2].item() <= 0.0
        ):
            final_sigmas[-2] = max(
                epsilon, target_min_pos
            )  # Ensure it's at least epsilon and target_min_pos
            if (
                len(final_sigmas) > 2
                and final_sigmas[-2].item() >= final_sigmas[-3].item() - epsilon
            ):
                new_val = final_sigmas[-3].item() / 2.0
                final_sigmas[-2] = max(epsilon, target_min_pos, new_val)
            elif len(final_sigmas) == 2:
                final_sigmas[-2] = max(epsilon, target_min_pos, final_sigmas[-2].item())

        return (final_sigmas,)
