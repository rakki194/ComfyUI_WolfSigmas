import torch
import math


class WolfSigmaRespaceLogCosine:
    """
    Respaces an existing N-step sigma schedule (N+1 sigmas)
    to have its active points (sigma_max to sigma_min_positive)
    follow a cosine curve in the log-sigma domain.
    The number of steps in the output matches the input.
    The final 0.0 sigma, if present, remains unchanged.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "respace_log_cosine"
    CATEGORY = "sampling/sigmas_wolf/transform"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_in": ("SIGMAS",),
                "override_sigma_max": (
                    "FLOAT",
                    {
                        "default": -1.0,  # -1 means use from sigmas_in
                        "min": -1.0,
                        "max": 1000.0,
                        "step": 0.01,
                        "round": False,
                    },
                ),
                "override_sigma_min_positive": (
                    "FLOAT",
                    {
                        "default": -1.0,  # -1 means use from sigmas_in
                        "min": -1.0,
                        "max": 100.0,
                        "step": 0.0001,
                        "round": False,
                    },
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

    def respace_log_cosine(
        self,
        sigmas_in,
        override_sigma_max,
        override_sigma_min_positive,
        min_epsilon_spacing,
    ):
        sigmas_input = sigmas_in.clone().cpu()
        epsilon = float(min_epsilon_spacing)

        if len(sigmas_input) < 2:  # Needs at least [sigma_max, 0.0] or [s0, s1]
            return (sigmas_input,)

        has_final_zero = sigmas_input[-1].item() == 0.0
        num_active_sigmas = 0
        original_active_sigmas_list = []

        if has_final_zero:
            if len(sigmas_input) == 2 and sigmas_input[0].item() > 0:  # [S_max, 0.0]
                num_active_sigmas = 1  # Only s_max is active point for spacing
                original_active_sigmas_list = [sigmas_input[0].item()]
            elif len(sigmas_input) > 2:
                num_active_sigmas = len(sigmas_input) - 1
                original_active_sigmas_list = [s.item() for s in sigmas_input[:-1]]
            else:  # e.g. just [0.0] or invalid input like [0.0, 0.0]
                return (sigmas_input,)
        else:  # No final zero, all sigmas are considered active for spacing purposes
            num_active_sigmas = len(sigmas_input)
            original_active_sigmas_list = [s.item() for s in sigmas_input]

        if (
            num_active_sigmas == 0
        ):  # Should be caught by len(sigmas_input) < 2 or invalid cases
            return (sigmas_input,)

        # Determine sigma_max and sigma_min_positive for the cosine spacing
        s_max_eff = float(override_sigma_max)
        s_min_pos_eff = float(override_sigma_min_positive)

        if s_max_eff < 0:  # auto-detect from input
            s_max_eff = original_active_sigmas_list[0]
        if s_min_pos_eff < 0:  # auto-detect from input
            s_min_pos_eff = original_active_sigmas_list[-1]

        # Validate effective min/max
        if s_max_eff <= s_min_pos_eff:
            s_max_eff = s_min_pos_eff + num_active_sigmas * epsilon
        if s_min_pos_eff <= 0:
            s_min_pos_eff = epsilon
        if s_max_eff <= s_min_pos_eff:  # re-check after s_min_pos_eff adjustment
            s_max_eff = s_min_pos_eff + num_active_sigmas * epsilon

        log_s_max = math.log(s_max_eff)
        log_s_min_pos = math.log(s_min_pos_eff)

        new_active_sigmas = torch.zeros(num_active_sigmas, dtype=torch.float32)

        if num_active_sigmas == 1:  # Only one active sigma, it becomes s_max_eff
            new_active_sigmas[0] = s_max_eff
        else:
            # t goes from 0 to 1 for num_active_sigmas points
            t = torch.linspace(0, 1, num_active_sigmas, device="cpu")
            # cosine_t maps t from 0..1 to 0..1 with cosine easing (0 at start, 1 at end)
            cosine_t = 0.5 * (1.0 - torch.cos(t * math.pi))
            # Interpolate in log space: log_s_max down to log_s_min_pos
            # (1.0 - cosine_t) goes from 1 down to 0 as t goes 0 to 1.
            log_sigmas_spaced = log_s_min_pos + (1.0 - cosine_t) * (
                log_s_max - log_s_min_pos
            )
            new_active_sigmas = torch.exp(log_sigmas_spaced)

        # Ensure endpoints are met and strict decrease for active sigmas
        new_active_sigmas[0] = s_max_eff
        if num_active_sigmas > 1:
            new_active_sigmas[-1] = s_min_pos_eff

        for i in range(num_active_sigmas - 1):
            if new_active_sigmas[i].item() <= new_active_sigmas[i + 1].item() + epsilon:
                new_active_sigmas[i + 1] = new_active_sigmas[i].item() - epsilon
            new_active_sigmas[i + 1] = max(epsilon, new_active_sigmas[i + 1].item())

        if num_active_sigmas > 0:
            new_active_sigmas[-1] = max(
                s_min_pos_eff, epsilon, new_active_sigmas[-1].item()
            )

        # Reconstruct the full sigma array
        if has_final_zero:
            final_sigmas_tensor = torch.cat(
                (
                    new_active_sigmas,
                    torch.tensor([0.0], dtype=torch.float32, device="cpu"),
                )
            )
        else:
            final_sigmas_tensor = new_active_sigmas

        return (final_sigmas_tensor.cpu(),)
