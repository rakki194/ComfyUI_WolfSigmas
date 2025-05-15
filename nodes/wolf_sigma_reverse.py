import torch


class WolfSigmaReverse:
    """
    Reverses the order of active sigmas in a schedule.
    The final 0.0 sigma, if present, remains unchanged.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "reverse_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
            }
        }

    def reverse_sigmas(self, sigmas):
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or len(sigmas) < 1:
            # Invalid input, return as is or a default
            print("Warning: Invalid input sigmas for Reverse. Returning as is.")
            return (sigmas,)

        if len(sigmas) == 1:  # Single value, nothing to reverse
            return (sigmas,)

        s_input = sigmas.clone()

        # Check if the last element is 0.0
        has_final_zero = s_input[-1].item() == 0.0

        if has_final_zero and len(s_input) > 1:
            active_sigmas = s_input[:-1]
            reversed_active_sigmas = torch.flip(active_sigmas, dims=[0])
            s_output = torch.cat((reversed_active_sigmas, s_input[-1:]), dim=0)
        else:
            s_output = torch.flip(s_input, dims=[0])

        return (s_output,)
