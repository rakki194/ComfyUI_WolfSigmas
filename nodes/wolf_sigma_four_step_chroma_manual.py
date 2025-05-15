import torch


class WolfSigmaFourStepChromaManual:
    """
    Provides a 4-sigma schedule [s0, s1, s2, s3] for a 3-step sampling process,
    inspired by the Flux model's noise schedule function (sigma_fn).
    Allows adjustment of start/end sigmas and Flux shift parameters.
    Validation ensures s0 > s1 > s2 > s3 >= 0.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_flux_inspired_four_step_sigmas"
    CATEGORY = "sampling/sigmas_wolf/3_step"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_sigma": (
                    "FLOAT",
                    {"default": 0.992, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "end_sigma": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.001},
                ),
                "flux_shift1": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "flux_shift2": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "enforce_endpoints": ("BOOLEAN", {"default": True}),
                "min_step_epsilon": (
                    "FLOAT",
                    {
                        "default": 0.0001,
                        "min": 0.00001,
                        "max": 0.1,
                        "step": 0.00001,
                        "precision": 5,
                    },
                ),
            }
        }

    def _calculate_flux_sigma_at_t(
        self, t, user_sigma_max, user_sigma_min, shift1, shift2
    ):
        effective_calc_sigma_min = max(user_sigma_min, 1e-9)
        if user_sigma_max <= user_sigma_min:
            return (1.0 - t) * user_sigma_max + t * user_sigma_min
        denominator_part = user_sigma_max / effective_calc_sigma_min - 1.0
        val = user_sigma_max / (
            1.0 + (t * (1.0 - shift2) + shift2) ** shift1 * denominator_part
        )
        return val

    def get_flux_inspired_four_step_sigmas(
        self,
        start_sigma,
        end_sigma,
        flux_shift1,
        flux_shift2,
        enforce_endpoints,
        min_step_epsilon,
    ):
        s_user_max = float(start_sigma)
        s_user_min = float(end_sigma)
        f_shift1 = float(flux_shift1)
        f_shift2 = float(flux_shift2)
        epsilon = float(min_step_epsilon)
        num_intervals = 3
        num_points = num_intervals + 1

        time_points = torch.linspace(0.0, 1.0, num_points).tolist()
        sigmas_calculated = []

        if enforce_endpoints:
            sigmas_calculated.append(s_user_max)
            for i in range(1, num_points - 1):
                sigmas_calculated.append(
                    self._calculate_flux_sigma_at_t(
                        time_points[i], s_user_max, s_user_min, f_shift1, f_shift2
                    )
                )
            sigmas_calculated.append(s_user_min)
        else:
            for t_val in time_points:
                sigmas_calculated.append(
                    self._calculate_flux_sigma_at_t(
                        t_val, s_user_max, s_user_min, f_shift1, f_shift2
                    )
                )

        s = [float(val) for val in sigmas_calculated]

        # Validate and enforce monotonicity: s0 > s1 > ... > s_N >= 0
        s[num_points - 1] = max(0.0, s[num_points - 1])
        for i in range(num_points - 2, -1, -1):
            s[i] = max(s[i + 1] + epsilon, s[i])

        # Second pass to ensure order from top after adjustments
        for i in range(num_points - 1):
            s[i + 1] = min(s[i + 1], s[i] - epsilon)

        s[num_points - 1] = max(0.0, s[num_points - 1])

        sigmas_tensor = torch.tensor(s, dtype=torch.float32)
        return (sigmas_tensor,)
