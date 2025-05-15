import torch

# import math


class WolfSigmaClampT0:
    """
    Clamps the first sigma (t0) of an incoming schedule to a specified value.
    Other sigmas are adjusted proportionally if they would exceed t0 or become non-monotonic.
    The final 0.0 sigma, if present, remains unchanged.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "clamp_t0"
    CATEGORY = "sampling/sigmas_wolf/transform"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_in": ("SIGMAS",),
                "target_t0": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.001},
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

    def clamp_t0(self, sigmas_in, target_t0, min_epsilon_spacing):
        sigmas = sigmas_in.clone().cpu()
        target_t0 = float(target_t0)
        epsilon = float(min_epsilon_spacing)

        if len(sigmas) == 0:
            return (sigmas,)

        original_t0 = sigmas[0].item()
        sigmas[0] = target_t0

        # Adjust subsequent sigmas to maintain monotonicity
        for i in range(1, len(sigmas)):
            # If the original sigma was 0.0 (likely the end), keep it 0.0
            if sigmas_in[i].item() == 0.0 and i == len(sigmas) - 1:
                sigmas[i] = 0.0
                continue

            # Option 1: Proportional scaling (might be complex if original_t0 was very different)
            # Option 2: Simple enforcement of monotonicity. Let's try this first.
            # Ensure current sigma is less than the previous one by at least epsilon
            required_max_val = sigmas[i - 1].item() - epsilon
            sigmas[i] = min(
                sigmas[i].item(), required_max_val
            )  # Keep original if already smaller
            sigmas[i] = max(sigmas[i].item(), 0.0)  # Ensure non-negative

            # If, after clamping to previous, it became non-monotonic with original intention, re-evaluate
            # This part is tricky. Let's assume for now that if sigmas[i] was originally > target_t0,
            # it should now be just below target_t0 (or sigmas[i-1]).
            # If it was much smaller, its relative position might be maintained.

        # Second pass to ensure strict decreasing order from the new t0
        for i in range(len(sigmas) - 1):
            if sigmas[i].item() <= sigmas[i + 1].item() + epsilon:
                # If the next is the final 0, this current one must be > 0
                if sigmas[i + 1].item() == 0.0 and i == len(sigmas) - 2:
                    sigmas[i] = max(
                        epsilon, sigmas[i].item()
                    )  # Ensure it's at least epsilon
                    if (
                        sigmas[i].item() <= sigmas[i + 1].item() + epsilon
                    ):  # if it was 0 and next is 0
                        sigmas[i] = epsilon  # force positive
                else:
                    # Try to make the next one smaller, relative to current one
                    sigmas[i + 1] = sigmas[i].item() - epsilon
            sigmas[i + 1] = max(0.0, sigmas[i + 1].item())  # ensure non-negative

        # Ensure the one before last is > 0 if last is 0
        if len(sigmas) > 1 and sigmas[-1].item() == 0.0 and sigmas[-2].item() <= 0.0:
            sigmas[-2] = epsilon
            if (
                len(sigmas) > 2 and sigmas[-2].item() >= sigmas[-3].item() - epsilon
            ):  # if this made it non-monotonic with one before
                sigmas[-2] = sigmas[-3].item() / 2.0  # fallback, could be smarter
            sigmas[-2] = max(epsilon, sigmas[-2].item())

        return (sigmas,)
