import torch


class WolfSigmaFourStepChromaImbalanced:
    """
    Provides an imbalanced, Karras-style 4-sigma schedule
    for an 4-step sampling process.
    - Start sigma is derived from base_start_sigma and positive imbalance.
    - End sigma is derived from base_end_sigma and negative imbalance.
    - Intermediate sigmas are Karras-interpolated.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_imbalanced_four_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/3_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_start_sigma": (
                    "FLOAT",
                    {"default": 0.992, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "base_end_sigma": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "imbalance_factor": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01},
                ),
                "scale_factor": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01},
                ),
                "rho": (
                    "FLOAT",
                    {"default": 7.0, "min": 0.1, "max": 100.0, "step": 0.1},
                ),
                "sigma_spacing_epsilon": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.00001,
                        "max": 1.0,
                        "step": 0.00001,
                        "precision": 5,
                    },
                ),
            }
        }

    def get_imbalanced_four_step_sigmas(
        self,
        base_start_sigma,
        base_end_sigma,
        imbalance_factor,
        scale_factor,
        rho,
        sigma_spacing_epsilon,
    ):
        current_start_sigma = float(base_start_sigma)
        current_end_sigma = float(base_end_sigma)
        k_rho = float(rho)
        epsilon = float(sigma_spacing_epsilon)
        num_intervals = 3
        num_points = num_intervals + 1

        if imbalance_factor > 0:
            current_start_sigma += imbalance_factor * scale_factor
        elif imbalance_factor < 0:
            current_end_sigma += abs(imbalance_factor) * scale_factor

        current_start_sigma = max(0.0, current_start_sigma)
        current_end_sigma = max(0.0, current_end_sigma)

        min_total_span = num_intervals * epsilon
        if current_start_sigma < current_end_sigma + min_total_span:
            current_start_sigma = current_end_sigma + min_total_span

        if (
            current_start_sigma == 0.0
            and current_end_sigma == 0.0
            and min_total_span > 0
        ):
            current_start_sigma = min_total_span
        elif (
            current_start_sigma == 0.0
            and current_end_sigma == 0.0
            and min_total_span == 0.0
        ):
            current_start_sigma = 0.001

        interp_start_val = current_start_sigma
        interp_end_val_for_calc = max(
            current_end_sigma, epsilon if current_end_sigma == 0 else current_end_sigma
        )

        if interp_start_val <= interp_end_val_for_calc:
            interp_start_val = interp_end_val_for_calc + epsilon

        inv_rho_start = interp_start_val ** (1 / k_rho)
        inv_rho_end_calc = interp_end_val_for_calc ** (1 / k_rho)

        if interp_start_val == 0.0 and k_rho > 0:
            inv_rho_start = 0.0

        t = torch.linspace(0, 1, num_points, device="cpu")
        sigmas_generated = (
            inv_rho_start + t * (inv_rho_end_calc - inv_rho_start)
        ) ** k_rho

        sigmas_generated[0] = current_start_sigma
        sigmas_generated[num_points - 1] = current_end_sigma

        for i in range(num_points - 2):
            if sigmas_generated[i] <= sigmas_generated[i + 1] + epsilon:
                sigmas_generated[i + 1] = sigmas_generated[i] - epsilon

        for i in range(num_points - 1):
            if sigmas_generated[i] < 0:
                sigmas_generated[i] = epsilon

        sigmas_generated[num_points - 1] = max(0.0, sigmas_generated[num_points - 1])

        return (sigmas_generated.to(torch.float32),)
