import torch
import comfy.k_diffusion.sampling as k_diffusion_sampling
from ..wolf_sigma_constants import AYS_CHROMA_SIGMAS_BASE


class WolfSigmaAYSInspired4Step:
    """
    Generates a 4-step schedule inspired by AYS (Align Your Steps) for SD1.5.
    Produces 5 sigma values for 4 sampling steps by picking points from a pre-defined AYS schedule.
    The selection of which AYS base sigmas to use is now configurable.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_ays_inspired_4_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/4_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index_s4": (
                    "INT",
                    {"default": 0, "min": 0, "max": 10},
                ),  # Index for the 1st (highest) sigma
                "index_s3": (
                    "INT",
                    {"default": 3, "min": 0, "max": 10},
                ),  # Index for the 2nd sigma
                "index_s2": (
                    "INT",
                    {"default": 6, "min": 0, "max": 10},
                ),  # Index for the 3rd sigma
                "index_s1": (
                    "INT",
                    {"default": 9, "min": 0, "max": 10},
                ),  # Index for the 4th (lowest non-zero) sigma
                "target_sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.001},
                ),
            }
        }

    def get_ays_inspired_4_step_sigmas(
        self, index_s4, index_s3, index_s2, index_s1, target_sigma_max
    ):
        num_steps = 4
        indices = [index_s4, index_s3, index_s2, index_s1]
        min_epsilon = 1e-7
        target_sigma_max = float(target_sigma_max)

        selected_base_sigmas = []
        for i in range(num_steps):
            idx = indices[i]
            if 0 <= idx < len(AYS_CHROMA_SIGMAS_BASE):
                selected_base_sigmas.append(AYS_CHROMA_SIGMAS_BASE[idx].item())
            else:
                # Fallback or error if index is out of bounds - use a default or raise error
                # For simplicity, let's use a nearby valid sigma or default, this could be refined
                safe_idx = max(0, min(idx, len(AYS_CHROMA_SIGMAS_BASE) - 1))
                selected_base_sigmas.append(AYS_CHROMA_SIGMAS_BASE[safe_idx].item())
                print(
                    f"Warning: AYS 4-Step index {idx} out of bounds, used {safe_idx}."
                )

        # Ensure sigmas are sorted high-to-low if user picks them out of order,
        # though for AYS the expectation is usually a specific sequence.
        # However, to prevent sampler errors, let's sort and then enforce strict decrease.
        s = sorted(selected_base_sigmas, reverse=True)

        # Scale the selected sigmas if a target_sigma_max is provided and valid
        if (
            len(s) > 0
            and target_sigma_max > 0
            and abs(s[0] - target_sigma_max) > 1e-6
            and s[0] > 1e-6
        ):
            current_max_selected = s[0]
            scale_factor = target_sigma_max / current_max_selected
            s = [val * scale_factor for val in s]

        final_sigmas_list = []
        if len(s) > 0:
            final_sigmas_list.append(s[0])
            for i in range(len(s) - 1):
                next_sigma = max(
                    s[i + 1], final_sigmas_list[-1] - min_epsilon
                )  # Ensure decrease
                # If s[i+1] was identical to s[i], this makes it slightly smaller.
                # If s[i+1] was much smaller, it uses s[i+1].
                # This also handles if user selected duplicate indices leading to same sigma value initially.
                if (
                    final_sigmas_list[-1] - next_sigma < min_epsilon
                ):  # If still not strictly decreasing
                    next_sigma = final_sigmas_list[-1] - min_epsilon
                final_sigmas_list.append(next_sigma)

        # Ensure all are positive before adding final zero
        for i in range(len(final_sigmas_list)):
            final_sigmas_list[i] = max(
                final_sigmas_list[i],
                min_epsilon if i < len(final_sigmas_list) - 1 else 0.0,
            )
            if i > 0 and final_sigmas_list[i] >= final_sigmas_list[i - 1]:
                final_sigmas_list[i] = final_sigmas_list[i - 1] - min_epsilon

        # Add the final 0.0 sigma for N+1 total sigmas
        final_sigmas_list.append(0.0)

        # Final check for positivity of the last non-zero sigma
        if len(final_sigmas_list) > 1 and final_sigmas_list[-2] <= 0.0:
            final_sigmas_list[-2] = min_epsilon
            if (
                final_sigmas_list[-2] >= final_sigmas_list[-3]
            ):  # Check against one before if exists
                final_sigmas_list[-2] = (
                    final_sigmas_list[-3] * 0.5
                    if len(final_sigmas_list) > 2
                    else min_epsilon
                )

        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)
        # Ensure we have num_steps + 1 sigmas
        if len(sigmas) != num_steps + 1:
            # This case should be handled by the logic above, but as a fallback:
            print(
                f"Warning: AYS 4-Step generated {len(sigmas)} sigmas, expected {num_steps+1}. Adjusting."
            )
            if len(sigmas) > num_steps + 1:
                sigmas = sigmas[: num_steps + 1]
            else:  # Reconstruct a Karras if something went very wrong
                sigmas = k_diffusion_sampling.get_sigmas_karras(
                    n=num_steps, sigma_min=0.002, sigma_max=1.0, rho=7, device="cpu"
                )

        return (sigmas,)
