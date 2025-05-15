import torch
import math


class WolfSigmaGeometricProgression:
    """
    Generates N_steps + 1 sigmas forming a geometric progression.
    Each active sigma is the previous one multiplied by a common_ratio.
    Sigmas are bounded by sigma_start (max) and sigma_min_positive.
    Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_geometric_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_steps": ("INT", {"default": 12, "min": 1, "max": 1000}),
                "sigma_start": (
                    "FLOAT",
                    {
                        "default": 80.0,
                        "min": 0.01,
                        "max": 1000.0,
                        "step": 0.1,
                        "doc": "Starting sigma value (effectively sigma_max).",
                    },
                ),
                "sigma_min_positive": (
                    "FLOAT",
                    {
                        "default": 0.002,
                        "min": 0.0001,
                        "max": 10.0,
                        "step": 0.0001,
                        "doc": "The smallest positive sigma allowed for active steps.",
                    },
                ),
                "common_ratio": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.01,
                        "max": 2.0,
                        "step": 0.01,
                        "doc": "Multiplier for each step. <1 for decreasing, >1 for increasing (unusual for diffusion).",
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

    def generate_geometric_sigmas(
        self,
        num_steps,
        sigma_start,
        sigma_min_positive,
        common_ratio,
        min_epsilon_spacing,
    ):
        num_active_steps = int(num_steps)
        s_start = float(sigma_start)
        s_min_pos = float(sigma_min_positive)
        ratio = float(common_ratio)
        epsilon = float(min_epsilon_spacing)

        if num_active_steps < 1:
            return (torch.tensor([s_start, 0.0], dtype=torch.float32),)

        if s_start <= s_min_pos:
            s_start = (
                s_min_pos + num_active_steps * epsilon
            )  # Ensure start is meaningfully larger
        if s_min_pos <= 0:
            s_min_pos = epsilon
        if ratio <= 0:
            ratio = epsilon  # Ratio must be positive

        active_sigmas = []
        current_sigma = s_start

        for _ in range(num_active_steps):
            active_sigmas.append(current_sigma)
            if (
                current_sigma <= s_min_pos + epsilon and ratio < 1.0
            ):  # Stop if we are small and decreasing
                break
            if (
                current_sigma * ratio < s_min_pos
                and len(active_sigmas) < num_active_steps
                and ratio < 1.0
            ):
                # If next step would be too small, and we haven't filled all steps, clamp last to s_min_pos
                current_sigma = s_min_pos
            elif (
                ratio >= 1.0
                and current_sigma >= s_start * (ratio ** (num_active_steps - 1))
                and len(active_sigmas) < num_active_steps
            ):  # Avoid exploding if ratio > 1
                current_sigma = s_start * (ratio ** len(active_sigmas))
            else:
                current_sigma *= ratio
            current_sigma = max(
                current_sigma, s_min_pos if ratio < 1.0 else epsilon
            )  # Ensure floor for decreasing, general epsilon for increasing

        # If loop broke early, fill remaining steps to ensure num_active_steps are present
        # For decreasing ratio, fill with s_min_pos. For increasing, this part might need refinement
        # but typically ratio < 1 for diffusion.
        while len(active_sigmas) < num_active_steps:
            if ratio < 1.0:
                active_sigmas.append(
                    max(
                        s_min_pos,
                        active_sigmas[-1] - epsilon if active_sigmas else s_min_pos,
                    )
                )
            else:  # ratio >= 1.0, geometric progression continues upwards, or plateaus if capped
                if active_sigmas:
                    next_s = active_sigmas[-1] * ratio
                    active_sigmas.append(
                        max(next_s, active_sigmas[-1] + epsilon)
                    )  # Ensure increase
                else:
                    active_sigmas.append(
                        s_start
                    )  # Should not happen if num_active_steps >= 1

        active_sigmas_tensor = torch.tensor(
            active_sigmas[:num_active_steps], dtype=torch.float32
        )

        # Standard Post-processing for strictly decreasing and positive sigmas
        final_sigmas = torch.zeros(num_active_steps + 1, dtype=torch.float32)

        if num_active_steps > 0:
            final_sigmas[0] = s_start  # Ensure first sigma is sigma_start
            current_s_max = s_start  # Alias for clarity in post-processing section
            current_s_min_positive = s_min_pos  # Alias

            # Transfer generated sigmas, ensuring first is s_start
            final_sigmas[:num_active_steps] = active_sigmas_tensor
            final_sigmas[0] = current_s_max

            sigma_min_floor_for_last_active = current_s_min_positive
            if (
                current_s_min_positive == 0.0
            ):  # Should be caught by s_min_pos = epsilon earlier
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
