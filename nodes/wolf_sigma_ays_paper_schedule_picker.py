import torch

# import math
from ..wolf_sigma_constants import AYS_PAPER_SCHEDULES  # MODIFIED IMPORT


class WolfSigmaAYSPaperSchedulePicker:
    """
    Picks one of the pre-defined schedules from the AYS Paper (Table 2).
    Outputs N_steps + 1 sigmas, where the last sigma is 0.0.
    The number of steps is determined by the chosen schedule.
    """

    SCHEDULE_NAMES = list(AYS_PAPER_SCHEDULES.keys())

    RETURN_TYPES = ("SIGMAS", "INT")
    RETURN_NAMES = ("SIGMAS", "num_steps_out")
    FUNCTION = "pick_ays_paper_schedule"
    CATEGORY = "sampling/sigmas_wolf/AYS_paper"

    @classmethod
    def INPUT_TYPES(s):
        default_schedule = s.SCHEDULE_NAMES[0] if s.SCHEDULE_NAMES else "None"
        schedule_names_list = s.SCHEDULE_NAMES if s.SCHEDULE_NAMES else ["None"]
        return {
            "required": {
                "schedule_name": (schedule_names_list, {"default": default_schedule}),
                "target_sigma_max": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.001},
                ),
            }
        }

    def pick_ays_paper_schedule(self, schedule_name, target_sigma_max):
        if schedule_name not in AYS_PAPER_SCHEDULES:
            raise ValueError(f"Unknown AYS Paper schedule: {schedule_name}")

        base_sigmas_list = AYS_PAPER_SCHEDULES[schedule_name]["sigmas"]
        original_max_from_paper = AYS_PAPER_SCHEDULES[schedule_name]["original_max"]

        num_steps = len(
            base_sigmas_list
        )  # The paper lists N sigmas for N steps, last is usually small not 0

        # Scale the base sigmas according to target_sigma_max relative to original paper's max
        # (e.g. SD1.5 original max is 14.615)
        scaled_sigmas_list = []
        if original_max_from_paper > 1e-9 and target_sigma_max > 1e-9:
            scale_factor = target_sigma_max / original_max_from_paper
            scaled_sigmas_list = [s * scale_factor for s in base_sigmas_list]
        else:  # Avoid division by zero or if target_sigma_max is effectively zero
            scaled_sigmas_list = list(base_sigmas_list)

        # Ensure the list is sorted in descending order and add the final 0.0
        # The paper schedules are already t_0 > t_1 > ... > t_{N-1} > 0
        # We need N+1 sigmas: s_0, s_1, ..., s_{N-1}, 0.0
        final_sigmas_values = sorted(scaled_sigmas_list, reverse=True) + [0.0]

        # Make sure the last active sigma (before 0.0) is positive, min epsilon if needed.
        if len(final_sigmas_values) > 1 and final_sigmas_values[-2] <= 0.0:
            final_sigmas_values[-2] = 1e-7  # Smallest positive value
            # and ensure it's less than previous if list is long enough
            if (
                len(final_sigmas_values) > 2
                and final_sigmas_values[-2] >= final_sigmas_values[-3]
            ):
                final_sigmas_values[-2] = final_sigmas_values[-3] * 0.5

        sigmas_tensor = torch.tensor(final_sigmas_values, dtype=torch.float32)

        # num_steps_out is the number of intervals, which is len(active_sigmas)
        # or len(final_sigmas_values) - 1
        num_steps_out = len(final_sigmas_values) - 1

        return (sigmas_tensor, num_steps_out)
