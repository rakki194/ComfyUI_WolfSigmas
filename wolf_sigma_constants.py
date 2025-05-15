import torch

# Align Your Steps Chroma Base Schedule
# Derived from SDXL 10-step AYS schedule (Table 3, ays.md), normalized to start at 1.0.
# This results in an 11-point schedule (10 intervals).
AYS_CHROMA_SIGMAS_BASE = torch.tensor(
    [
        1.0,
        0.4320903,
        0.2579884,
        0.1492299,
        0.0918235,
        0.0589798,
        0.0379747,
        0.0259993,
        0.0160109,
        0.0077318,
        0.0019843,
    ],
    dtype=torch.float32,
)

# Placeholder for Align Your Steps Paper Schedules (Table 2 from ays.md)
# User needs to populate this dictionary with the actual schedule data.
# Example structure:
# AYS_PAPER_SCHEDULES = {
# "AYS_SD15_5step": {"sigmas": [14.615, 3.535, 1.491, 0.742, 0.156], "original_max": 14.615},
# "AYS_SDXL_10step": {"sigmas": [700.669, 177.848, 72.967, ...], "original_max": 700.669},
# ... more schedules
# }
AYS_PAPER_SCHEDULES = {}
