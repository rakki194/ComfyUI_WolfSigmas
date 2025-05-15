import torch
import random


class WolfSigmaAddNoise:
    """
    Adds random noise to an existing sigma schedule.
    Ensures sigmas remain positive and optionally re-sorts them.
    The final 0.0 sigma (if present) remains unchanged by noise.
    """

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "add_noise_to_sigmas"
    CATEGORY = "sampling/sigmas_wolf/transformations"

    NOISE_TYPES = ["gaussian", "uniform"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS",),
                "noise_strength": (
                    "FLOAT",
                    {"default": 0.1, "min": 0.0, "max": 10.0, "step": 0.01},
                ),
                "noise_type": (s.NOISE_TYPES, {"default": "gaussian"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "ensure_strictly_decreasing": ("BOOLEAN", {"default": True}),
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

    def add_noise_to_sigmas(
        self,
        sigmas,
        noise_strength,
        noise_type,
        seed,
        ensure_strictly_decreasing,
        min_epsilon_spacing,
    ):
        if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1:
            print("Warning: Invalid input sigmas for AddNoise. Returning as is.")
            return (sigmas,)

        s_input = sigmas.clone()
        strength = float(noise_strength)
        epsilon = float(min_epsilon_spacing)

        if strength == 0.0:
            return (s_input,)

        # Set seed for reproducibility
        rng = random.Random(seed)
        # torch_rng = torch.Generator(device=s_input.device).manual_seed(seed) # Not used yet, but good for future torch-based random

        has_final_zero = len(s_input) > 0 and s_input[-1].item() == 0.0

        active_sigmas = (
            s_input[:-1].clone()
            if has_final_zero and len(s_input) > 1
            else s_input.clone()
        )

        if len(active_sigmas) == 0:
            return (s_input,)

        for i in range(len(active_sigmas)):
            original_sigma = active_sigmas[i].item()
            noise_val = 0.0
            if noise_type == "gaussian":
                noise_std = max(
                    strength * 0.01, min(original_sigma * strength, strength)
                )  # Apply strength relative to sigma, but capped
                noise_val = rng.gauss(
                    0, noise_std if noise_std > 0 else 1e-6
                )  # Ensure std_dev is positive for gauss
            elif noise_type == "uniform":
                noise_range = max(
                    strength * 0.01, min(original_sigma * strength, strength)
                )
                noise_val = rng.uniform(-noise_range, noise_range)

            active_sigmas[i] = original_sigma + noise_val
            if active_sigmas[i].item() < epsilon:
                active_sigmas[i] = torch.tensor(
                    epsilon, dtype=active_sigmas.dtype, device=active_sigmas.device
                )

        if ensure_strictly_decreasing and len(active_sigmas) > 0:
            active_sigmas = torch.sort(active_sigmas, descending=True)[0]
            active_sigmas = torch.clamp(active_sigmas, min=epsilon)
            if len(active_sigmas) > 1:
                for i in range(len(active_sigmas) - 2, -1, -1):
                    if active_sigmas[i] <= active_sigmas[i + 1]:
                        active_sigmas[i] = active_sigmas[i + 1] + epsilon
            elif len(active_sigmas) == 1:
                active_sigmas[0] = max(active_sigmas[0].item(), epsilon)
                active_sigmas[0] = torch.tensor(
                    active_sigmas[0], dtype=s_input.dtype, device=s_input.device
                )

        if has_final_zero:
            if len(active_sigmas) == 0:
                return (
                    torch.tensor([0.0], dtype=s_input.dtype, device=s_input.device),
                )
            if active_sigmas[-1].item() < epsilon:
                active_sigmas[-1] = torch.tensor(
                    epsilon, dtype=active_sigmas.dtype, device=active_sigmas.device
                )
                if (
                    len(active_sigmas) > 1
                    and active_sigmas[-2].item() <= active_sigmas[-1].item()
                ):
                    active_sigmas[-2] = active_sigmas[-1] + epsilon
            s_output = torch.cat(
                (
                    active_sigmas,
                    torch.tensor([0.0], dtype=s_input.dtype, device=s_input.device),
                )
            )
        else:
            if len(active_sigmas) == 0:
                return (torch.empty(0, dtype=s_input.dtype, device=s_input.device),)
            s_output = active_sigmas
            if len(s_output) > 0 and s_output[-1].item() < epsilon:
                s_output[-1] = torch.tensor(
                    epsilon, dtype=active_sigmas.dtype, device=active_sigmas.device
                )
                if len(s_output) > 1 and s_output[-2].item() <= s_output[-1].item():
                    s_output[-2] = s_output[-1] + epsilon

        return (s_output,)
