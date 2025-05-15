import torch


class WolfSigmaLinear12Step:
    """
    Generates 12 + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive)
    are linearly spaced. Last sigma is 0.0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "generate_linear_sigmas"
    CATEGORY = "sampling/sigmas_wolf/12_step"  # Changed category

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # "num_steps" is removed, hardcoded to 12
                "sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001},
                ),
                "sigma_min_positive": (
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

    def generate_linear_sigmas(
        self, sigma_max, sigma_min_positive, min_epsilon_spacing
    ):
        num_steps = 12  # Hardcoded
        N = int(num_steps)
        s_max = float(sigma_max)
        s_min_pos = float(sigma_min_positive)
        epsilon = float(min_epsilon_spacing)

        if N <= 0:  # Should not happen with hardcoded 12
            return (torch.tensor([s_max, 0.0], dtype=torch.float32),)
        if s_max <= s_min_pos:
            s_max = s_min_pos + N * epsilon
        if s_min_pos <= 0.0:
            s_min_pos = epsilon

        # N is always 12 here, so N==1 case is not strictly needed but kept for robustness
        if N == 1:
            active_sigmas = torch.tensor([s_max, s_min_pos], dtype=torch.float32)
        else:
            active_sigmas = torch.linspace(s_max, s_min_pos, N, device="cpu")

        final_sigmas_list = active_sigmas.tolist() + [0.0]
        sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

        # The original code has a complex loop for ensuring descending order and min spacing.
        # This loop might be overly complex or have redundant checks.
        # For linear spacing, linspace should already guarantee descending order if s_max > s_min_pos.
        # The post-processing loop is to enforce minimum epsilon spacing and boundary conditions.

        sigmas[0] = s_max
        # Ensure the (N-1)th element (last active sigma) is at least s_min_pos
        if N > 0:  # N is 12
            sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos)

        # Corrected loop for ensuring descending order and min_epsilon_spacing
        # This simplifies the original complex loop logic.
        # Start from the first element and ensure it's s_max
        sigmas[0] = s_max
        # For all active sigmas up to the second to last active sigma (index N-2)
        for i in range(N - 1):
            # Ensure current sigma is greater than next sigma by at least epsilon
            # And also ensure next sigma is not less than s_min_pos (for elements up to N-1)
            # or 0.0 (for the last element which is sigmas[N])
            next_min_val = (
                s_min_pos if i < N - 1 else 0.0
            )  # This comparison point is tricky

            # Ensure sigmas[i] > sigmas[i+1] + epsilon
            # Adjust sigmas[i+1] if it's too close to sigmas[i] or too small
            # sigmas[i+1] should be at most sigmas[i] - epsilon
            # sigmas[i+1] should be at least s_min_pos (if i+1 < N) or 0.0 (if i+1 == N)

            if i + 1 < N:  # Active sigmas
                required_val_for_next = max(s_min_pos, epsilon)
                sigmas[i + 1] = max(
                    min(sigmas[i + 1].item(), sigmas[i].item() - epsilon),
                    required_val_for_next,
                )
            else:  # Last sigma (0.0) - should already be handled by list construction
                pass

        # Final check to ensure last active sigma (at N-1) is s_min_pos or greater, but also respects spacing from N-2
        if N > 1:  # N=12
            sigmas[N - 1] = max(s_min_pos, epsilon)  # Must be at least s_min_pos
            sigmas[N - 1] = min(
                sigmas[N - 1].item(), sigmas[N - 2].item() - epsilon
            )  # Respect spacing
            sigmas[N - 1] = max(
                sigmas[N - 1].item(), s_min_pos, epsilon
            )  # Re-check s_min_pos after spacing adjustment

        sigmas[N] = 0.0  # Ensure the last actual sigma is 0.0

        # A simpler loop to enforce strict decrease and minimums
        sigmas[0] = s_max
        for i in range(N - 1):  # Iterate 0 to N-2 (0 to 10 for N=12)
            # Ensure sigmas[i+1] is less than sigmas[i] by at least epsilon
            # and not less than s_min_pos
            sigmas[i + 1] = torch.clamp(
                sigmas[i + 1], min=s_min_pos, max=sigmas[i].item() - epsilon
            )

        # Ensure the last active sigma is exactly s_min_pos if it got pushed higher by clamp,
        # or if linspace ended above it, but respect spacing.
        if N > 0:  # N=12
            sigmas[N - 1] = max(s_min_pos, epsilon)
            if N > 1:
                sigmas[N - 1] = min(
                    sigmas[N - 1].item(), sigmas[N - 2].item() - epsilon
                )
                sigmas[N - 1] = max(
                    sigmas[N - 1].item(), s_min_pos, epsilon
                )  # re-assert s_min_pos

        sigmas[N] = 0.0  # Final element is 0

        # Let's use a structure closer to original's intent but simplified if possible
        # The original's loop is very defensive. Linspace should do most of the work.
        # The main concern is sigma_min_positive and epsilon spacing.

        # Re-evaluate the loop for correctness based on original logic
        # Original logic:
        # sigmas[0] = s_max
        # sigmas[N-1] = max(s_min_pos, epsilon) IF N > 0
        # Loop i from 0 to N-2:
        #   if sigmas[i] <= sigmas[i+1] + epsilon: sigmas[i+1] = sigmas[i] - epsilon
        #   sigmas[i+1] = max(epsilon, sigmas[i+1])
        # if N > 0:
        #   sigmas[N-1] = max(sigmas[N-1], s_min_pos, epsilon)
        #   if sigmas[N-1] <= sigmas[N] + epsilon: sigmas[N-1] = max(s_min_pos, epsilon)
        # sigmas[0] = s_max (redundant?)
        # Loop i from 0 to N-1:
        #   current_val = sigmas[i]
        #   next_val_target = sigmas[i+1]
        #   if i < N-1:
        #       min_allowed_current = next_val_target + epsilon
        #       sigmas[i] = max(current_val, min_allowed_current)
        #       if i > 0 and sigmas[i] >= sigmas[i-1] - epsilon: sigmas[i] = sigmas[i-1] - epsilon
        #   elif i == N-1: (last active sigma)
        #       sigmas[i] = max(current_val, s_min_pos, epsilon)
        #       if i > 0 and sigmas[i] >= sigmas[i-1] - epsilon : sigmas[i] = sigmas[i-1] - epsilon
        #       sigmas[i] = max(current_val, s_min_pos, epsilon)

        # Simplified approach:
        # 1. Generate linspace
        # 2. Set boundaries (s_max, s_min_pos, 0.0)
        # 3. Iterate backward to ensure spacing and minimums

        if N == 1:  # handles 1 step, N=12 will not hit this typically
            sigmas = torch.tensor([s_max, s_min_pos, 0.0], dtype=torch.float32)
            if sigmas[0].item() <= sigmas[1].item() + epsilon:
                sigmas[0] = sigmas[1].item() + epsilon
            sigmas[1] = max(sigmas[1].item(), epsilon)

            # Final check for 1-step
            sigmas[0] = max(s_max, sigmas[1].item() + epsilon)
            sigmas[1] = max(min(s_min_pos, sigmas[0].item() - epsilon), epsilon)
            sigmas = torch.tensor(
                [sigmas[0].item(), sigmas[1].item(), 0.0], dtype=torch.float32
            )

        else:  # N > 1 (N=12 case)
            steps = torch.linspace(s_max, s_min_pos, N, device="cpu")
            sigmas = torch.cat((steps, torch.tensor([0.0], device="cpu"))).to(
                dtype=torch.float32
            )

            # Ensure strict monotonic decrease and bounds
            sigmas[0] = s_max
            sigmas[N] = 0.0
            sigmas[N - 1] = max(
                s_min_pos, epsilon, sigmas[N].item() + epsilon
            )  # last active sigma

            # Iterate backwards from second to last active sigma
            for i in range(N - 2, -1, -1):  # from N-2 down to 0
                sigmas[i] = max(sigmas[i].item(), sigmas[i + 1].item() + epsilon)

            # Iterate forwards to ensure no value is too large / re-check s_max
            sigmas[0] = s_max
            for i in range(N - 1):  # from 0 up to N-2
                sigmas[i + 1] = min(sigmas[i + 1].item(), sigmas[i].item() - epsilon)

            # Final pass for lower bounds on active sigmas
            for i in range(1, N):  # from 1 up to N-1 (active sigmas except first)
                sigmas[i] = max(
                    sigmas[i].item(), s_min_pos if i < N else epsilon, epsilon
                )  # s_min_pos for active, else epsilon
            sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos, epsilon)

        # Ensure the number of elements is N+1
        if len(sigmas) != N + 1:
            # This case should ideally not be reached if logic is correct
            # Fallback to original calculation method if reshaping/reconstruction failed
            if N == 1:
                active_sigmas_fallback = torch.tensor(
                    [s_max, s_min_pos], dtype=torch.float32
                )
            else:
                active_sigmas_fallback = torch.linspace(
                    s_max, s_min_pos, N, device="cpu"
                )
            final_sigmas_list_fb = active_sigmas_fallback.tolist() + [0.0]
            sigmas = torch.tensor(final_sigmas_list_fb, dtype=torch.float32)
            # And re-apply original post-processing from WolfSigmaLinearNStep if necessary.
            # For now, assuming the simplified logic above is sufficient or I will copy the original loop.
            # Copying the original complex loop for safety until a simpler one is fully verified.
            # This ensures behavior parity for the post-processing.

            # Re-instating original post-processing logic, adapted for fixed N
            active_sigmas = (
                torch.linspace(s_max, s_min_pos, N, device="cpu")
                if N > 1
                else torch.tensor([s_max, s_min_pos], dtype=torch.float32)
            )
            final_sigmas_list = active_sigmas.tolist() + [0.0]
            sigmas = torch.tensor(final_sigmas_list, dtype=torch.float32)

            sigmas[0] = s_max
            if N > 0:  # N = 12
                sigmas[N - 1] = max(s_min_pos, epsilon)

            for i in range(N - 1):  # 0 to 10
                if sigmas[i].item() <= sigmas[i + 1].item() + epsilon:
                    sigmas[i + 1] = sigmas[i].item() - epsilon
                sigmas[i + 1] = max(epsilon, sigmas[i + 1].item())

            if N > 0:  # N = 12
                sigmas[N - 1] = max(sigmas[N - 1].item(), s_min_pos, epsilon)
                # Check against sigma[N] which is 0.0
                if sigmas[N - 1].item() <= sigmas[N].item() + epsilon:
                    sigmas[N - 1] = max(s_min_pos, epsilon)

            sigmas[0] = s_max  # Potentially redundant if s_max hasn't changed
            for i in range(N):  # 0 to 11
                current_val = sigmas[i].item()
                # sigmas has N+1 elements, so sigmas[i+1] is valid for i up to N-1
                next_val_target = sigmas[i + 1].item()

                if i < N - 1:  # i from 0 to 10
                    min_allowed_current = next_val_target + epsilon
                    sigmas[i] = max(current_val, min_allowed_current)
                    if i > 0 and sigmas[i].item() >= sigmas[i - 1].item() - epsilon:
                        sigmas[i] = sigmas[i - 1].item() - epsilon
                elif i == N - 1:  # i = 11 (last active sigma)
                    # next_val_target is sigmas[N] which is 0.0
                    sigmas[i] = max(
                        current_val, s_min_pos, epsilon, next_val_target + epsilon
                    )
                    if i > 0 and sigmas[i].item() >= sigmas[i - 1].item() - epsilon:
                        sigmas[i] = sigmas[i - 1].item() - epsilon
                    # Re-assert s_min_pos for the last active sigma after adjustments
                    sigmas[i] = max(
                        sigmas[i].item(), s_min_pos, epsilon, next_val_target + epsilon
                    )

        # Final check for total elements (N+1 elements)
        # Ensure last element is 0.0
        sigmas[N] = 0.0

        return (sigmas,)
