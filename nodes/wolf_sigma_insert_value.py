import torch
import math


class WolfSigmaInsertValue:
    """
    Inserts a custom sigma value into an existing schedule at a specified index.
    Optionally re-sorts the schedule afterwards to maintain strictly decreasing order
    and ensures the final 0.0 sigma (if present) remains last.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "insert_sigma_value"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "sigma_value": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.01},
                ),
                "index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1000},
                ),  # Max index could be dynamic
                "sort_after_insert": ("BOOLEAN", {"default": True}),
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

    def insert_sigma_value(
        self, sigmas, sigma_value, index, sort_after_insert, min_epsilon_spacing
    ):
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
            print("Warning: Invalid input sigmas for InsertValue. Returning as is.")
            return (sigmas,)

        s_input = sigmas.clone()
        val_to_insert = float(sigma_value)
        insert_idx = int(index)
        epsilon = float(min_epsilon_spacing)

        if val_to_insert <= 0:
            print(
                f"Warning: Sigma value to insert ({val_to_insert}) must be positive. Clamping to epsilon."
            )
            val_to_insert = epsilon

        has_final_zero = len(s_input) > 0 and s_input[-1].item() == 0.0

        active_sigmas_list = []
        if has_final_zero and len(s_input) > 1:
            active_sigmas_list = s_input[:-1].tolist()
        elif not has_final_zero and len(s_input) > 0:
            active_sigmas_list = s_input.tolist()

        # Clamp insert_idx to be within bounds of active_sigmas_list + 1 (to append)
        insert_idx = max(0, min(insert_idx, len(active_sigmas_list)))

        # Insert the value
        active_sigmas_list.insert(insert_idx, val_to_insert)

        # Remove duplicates after insertion, preserving order of first appearance for now.
        # This is more robust than set for preserving intended order if not sorting.
        seen = set()
        unique_list_after_insert = []
        for x in active_sigmas_list:
            if x not in seen:
                unique_list_after_insert.append(x)
                seen.add(x)

        active_sigmas_modified = torch.tensor(
            unique_list_after_insert, dtype=s_input.dtype, device=s_input.device
        )

        if sort_after_insert:
            # Sort in descending order
            active_sigmas_modified = torch.sort(
                active_sigmas_modified, descending=True
            )[0]

            # Post-process to ensure strictly decreasing and positive values
            if len(active_sigmas_modified) > 0:
                # Ensure all active sigmas are at least epsilon first
                active_sigmas_modified = torch.clamp(
                    active_sigmas_modified, min=epsilon
                )

                if len(active_sigmas_modified) > 1:
                    # Correct from right to left (smallest to largest magnitude)
                    for i in range(len(active_sigmas_modified) - 2, -1, -1):
                        if active_sigmas_modified[i] <= active_sigmas_modified[i + 1]:
                            active_sigmas_modified[i] = (
                                active_sigmas_modified[i + 1] + epsilon
                            )
                    # One more pass from left to right for safety, though usually not needed if above is correct
                    for i in range(len(active_sigmas_modified) - 1):
                        if active_sigmas_modified[i] <= active_sigmas_modified[i + 1]:
                            # This case should ideally not be hit if the backward pass was effective
                            # It might indicate an issue if s[i+1] was epsilon and s[i] became epsilon too
                            # A more robust way for the forward pass if needed:
                            active_sigmas_modified[i + 1] = (
                                active_sigmas_modified[i] - epsilon
                            )
                            if active_sigmas_modified[i + 1] < epsilon:
                                active_sigmas_modified[i + 1] = epsilon
                    # Final check on the very last active sigma if it got pushed down to zero by mistake
                    if active_sigmas_modified[-1] < epsilon:
                        active_sigmas_modified[-1] = epsilon
            elif len(active_sigmas_modified) == 1:  # Single active sigma
                active_sigmas_modified[0] = max(active_sigmas_modified[0], epsilon)

        # Re-attach final zero if it was present and there are active sigmas
        if has_final_zero:
            if (
                len(active_sigmas_modified) == 0
            ):  # All sigmas removed or started empty, only return [0.0]
                return (
                    torch.tensor([0.0], dtype=s_input.dtype, device=s_input.device),
                )
            # Ensure the last active sigma is > 0 before adding zero
            if active_sigmas_modified[-1] < epsilon:
                active_sigmas_modified[-1] = epsilon
                # Re-check previous if this change caused issues (rare if already sorted and spaced)
                if (
                    len(active_sigmas_modified) > 1
                    and active_sigmas_modified[-2] <= active_sigmas_modified[-1]
                ):
                    active_sigmas_modified[-2] = active_sigmas_modified[-1] + epsilon
            s_output = torch.cat(
                (
                    active_sigmas_modified,
                    torch.tensor([0.0], dtype=s_input.dtype, device=s_input.device),
                )
            )
        else:  # No final zero originally
            if len(active_sigmas_modified) == 0:  # All sigmas removed or started empty
                return (
                    torch.empty(0, dtype=s_input.dtype, device=s_input.device),
                )  # Or specific handling
            # Ensure last actual sigma is positive if not sorting and it became non-positive
            if not sort_after_insert and active_sigmas_modified[-1] < epsilon:
                active_sigmas_modified[-1] = epsilon
                if (
                    len(active_sigmas_modified) > 1
                    and active_sigmas_modified[-2] <= active_sigmas_modified[-1]
                ):
                    active_sigmas_modified[-2] = active_sigmas_modified[-1] + epsilon
            s_output = active_sigmas_modified

        return (s_output,)
