import torch
import comfy.samplers
import comfy.sample
import latent_preview
import comfy.utils
import comfy.model_management

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server-side use
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE_WOLF_PLOTTER = True
except ImportError:
    MATPLOTLIB_AVAILABLE_WOLF_PLOTTER = False
    print(
        "ComfyUI_WolfSigmas: Matplotlib not found. WolfSamplerCustomAdvancedPlotter plotting will be disabled."
    )


class WolfSamplerCustomAdvancedPlotter:
    @classmethod
    def INPUT_TYPES(s):
        plot_options_extra = {}
        matplotlib_warning_text = "Matplotlib not installed. Plotting will be disabled."
        if not MATPLOTLIB_AVAILABLE_WOLF_PLOTTER:
            plot_options_extra = {"forceInput": True, "label": matplotlib_warning_text}

        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
            },
            "optional": {
                "plot_timesteps": ("BOOLEAN", {"default": False}),
                "plot_max_steps": (
                    "INT",
                    {"default": 16, "min": 1, "max": 1000, "step": 1},
                ),
                "plot_batch_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 63, "step": 1},
                ),
                "plot_channel_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1023, "step": 1},
                ),
                "plot_colormap": (
                    [
                        "default",
                        "gray",
                        "viridis",
                        "plasma",
                        "inferno",
                        "magma",
                        "cividis",
                    ],
                    {"default": "gray"},
                ),
                "plot_grid_cols": (
                    "INT",
                    {"default": 4, "min": 1, "max": 16, "step": 1},
                ),
                "plot_title_text": (
                    "STRING",
                    {"default": "Denoised Latent Timesteps (🐺)"},
                ),
                "plot_figure_width": (
                    "FLOAT",
                    {"default": 15.0, "min": 5.0, "max": 50.0, "step": 0.5},
                ),
                "plot_figure_height_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Scaling factor for plot height based on width and rows",
                    },
                ),
                "plot_dpi": (
                    "INT",
                    {"default": 100, "min": 50, "max": 300, "step": 10},
                ),
                # This input is only to make the warning visible in the UI if matplotlib is missing
                "matplotlib_status_check": (
                    "STRING",
                    {
                        "default": (
                            ""
                            if MATPLOTLIB_AVAILABLE_WOLF_PLOTTER
                            else matplotlib_warning_text
                        ),
                        "visible": False,
                        **plot_options_extra,
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "IMAGE")
    RETURN_NAMES = ("output", "denoised_output", "timestep_plot_wolf")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling/wolf"

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        plot_timesteps=False,
        plot_max_steps=16,
        plot_batch_index=0,
        plot_channel_index=0,
        plot_colormap="gray",
        plot_grid_cols=4,
        plot_title_text="Denoised Latent Timesteps (🐺)",
        plot_figure_width=15.0,
        plot_figure_height_scale=1.0,
        plot_dpi=100,
        matplotlib_status_check=None,
    ):

        latent = latent_image
        # Keep a reference for shape info and use it for fixing channels
        latent_image_samples_ref = latent["samples"]
        latent = latent.copy()  # Important to copy before modifying
        # Fix empty channels based on the model patcher from the guider
        latent_image_samples_fixed = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher, latent_image_samples_ref
        )
        latent["samples"] = latent_image_samples_fixed

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = (
            {}
        )  # For the original latent preview (denoised output of the whole process)

        collected_denoised_latents_for_plot = []
        collected_sigmas_for_plot = []

        num_preview_steps = 0
        if (
            sigmas is not None
            and hasattr(sigmas, "shape")
            and len(sigmas.shape) > 0
            and sigmas.shape[-1] > 0
        ):
            num_preview_steps = sigmas.shape[-1] - 1
            if num_preview_steps < 0:  # handles single sigma case
                num_preview_steps = 0

        # Standard callback for x0 output (final denoised)
        original_callback_for_x0 = latent_preview.prepare_callback(
            guider.model_patcher, num_preview_steps, x0_output
        )

        # Progress bar setup
        pbar = comfy.utils.ProgressBar(
            num_preview_steps if num_preview_steps > 0 else 1
        )  # ensure pbar has at least 1 total step

        def combined_callback(*args, **kwargs):
            # Handle both possible callback signatures
            if len(args) == 1 and isinstance(args[0], dict):
                S_dict = args[0]
            elif len(args) == 4:
                i, denoised, x, total_steps = args
                S_dict = {
                    "i": i,
                    "denoised": denoised,
                    "x": x,
                    "total_steps": total_steps,
                    # Optionally add more keys if needed
                }
            else:
                raise ValueError(f"Unexpected callback arguments: {args} {kwargs}")

            # Call original preview callback for final x0
            if original_callback_for_x0 is not None:
                original_callback_for_x0(
                    S_dict["i"],
                    S_dict["denoised"],
                    S_dict["x"],
                    S_dict.get("total_steps", num_preview_steps),
                )

            current_step_index = S_dict["i"]
            total_steps_for_pbar = num_preview_steps if num_preview_steps > 0 else 1
            pbar.update_absolute(current_step_index + 1, total_steps_for_pbar, None)

            # Collect data for plotting if enabled
            if (
                plot_timesteps
                and MATPLOTLIB_AVAILABLE_WOLF_PLOTTER
                and len(collected_denoised_latents_for_plot) < plot_max_steps
            ):
                # 'denoised' in S_dict is the model's prediction of x0 at the current step
                current_step_denoised_latent = S_dict["denoised"]

                b, c, h, w = current_step_denoised_latent.shape
                actual_batch_idx = min(plot_batch_index, b - 1) if b > 0 else 0
                actual_channel_idx = min(plot_channel_index, c - 1) if c > 0 else 0
                actual_batch_idx = max(0, actual_batch_idx)
                actual_channel_idx = max(0, actual_channel_idx)

                if (
                    current_step_denoised_latent.nelement() > 0
                    and actual_batch_idx < b
                    and actual_channel_idx < c
                ):
                    denoised_slice = current_step_denoised_latent[
                        actual_batch_idx, actual_channel_idx, :, :
                    ].cpu()
                    collected_denoised_latents_for_plot.append(denoised_slice)
                    # Try to get sigma if present, else fallback
                    sigma = S_dict.get("sigma")
                    if sigma is not None:
                        if hasattr(sigma, "item"):
                            collected_sigmas_for_plot.append(sigma.item())
                        else:
                            collected_sigmas_for_plot.append(float(sigma))
                    else:
                        collected_sigmas_for_plot.append(0.0)

        disable_pbar_flag = not comfy.utils.PROGRESS_BAR_ENABLED

        # Generate the initial noise (respects noise.seed from the RandomNoise or DisableNoise node)
        # The latent passed to generate_noise provides shape and device context.
        generated_noise_tensor = noise.generate_noise(latent)

        # Perform sampling
        # The callback receives a dictionary S_dict with keys like 'x', 'i', 'sigma', 'sigma_hat', 'denoised'
        sampled_latent_tensor = guider.sample(
            generated_noise_tensor,
            latent_image_samples_fixed,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=combined_callback,
            disable_pbar=disable_pbar_flag,
            seed=noise.seed,
        )  # Pass seed for reproducibility within sampler if it uses it

        sampled_latent_tensor = sampled_latent_tensor.to(
            comfy.model_management.intermediate_device()
        )

        # Prepare main LATENT outputs
        output_latent_dict = latent.copy()
        output_latent_dict["samples"] = sampled_latent_tensor

        denoised_output_latent_dict = latent.copy()
        if "x0" in x0_output and x0_output["x0"] is not None:
            # process_latent_out typically handles VAE scaling if applicable
            final_denoised_processed = guider.model_patcher.model.process_latent_out(
                x0_output["x0"].cpu()
            )
            denoised_output_latent_dict["samples"] = final_denoised_processed
        else:
            # Fallback: if x0_output didn't get populated (e.g. 0 steps, or sampler doesn't use x0 key in callback dict)
            # Return the final sampled output as the "denoised_output" as a best effort.
            # This ensures the node always provides a valid latent in this slot.
            denoised_output_latent_dict["samples"] = sampled_latent_tensor.clone()

        # Initialize plot image tensor (default empty/black)
        plot_image_tensor = torch.zeros(
            (1, 64, 64, 3), dtype=torch.float32, device="cpu"
        )

        if (
            plot_timesteps
            and MATPLOTLIB_AVAILABLE_WOLF_PLOTTER
            and collected_denoised_latents_for_plot
        ):
            try:
                num_plots = len(collected_denoised_latents_for_plot)
                grid_cols_actual = (
                    min(plot_grid_cols, num_plots) if num_plots > 0 else 1
                )
                num_rows = (
                    (num_plots + grid_cols_actual - 1) // grid_cols_actual
                    if num_plots > 0
                    else 1
                )

                h_sample, w_sample = (
                    collected_denoised_latents_for_plot[0].shape
                    if num_plots > 0
                    else (64, 64)
                )
                aspect_ratio = w_sample / h_sample if h_sample > 0 else 1.0

                base_subplot_width = (
                    plot_figure_width / grid_cols_actual
                    if grid_cols_actual > 0
                    else plot_figure_width
                )

                fig_actual_width = plot_figure_width
                fig_actual_height = (
                    num_rows
                    * (
                        base_subplot_width / aspect_ratio
                        if aspect_ratio > 0
                        else base_subplot_width
                    )
                    * plot_figure_height_scale
                )
                fig_actual_height += (
                    1.5 if plot_title_text else 0.5
                )  # Padding for title

                fig_actual_width = max(5, fig_actual_width)
                fig_actual_height = max(5, fig_actual_height)

                fig, axes = plt.subplots(
                    num_rows,
                    grid_cols_actual,
                    figsize=(fig_actual_width, fig_actual_height),
                    dpi=plot_dpi,
                )
                if num_plots == 1:
                    axes = np.array([axes])  # Ensure axes is always iterable
                axes = axes.flatten()

                for i in range(num_plots):
                    ax = axes[i]
                    latent_slice_np = collected_denoised_latents_for_plot[i].numpy()

                    vmin, vmax = (
                        np.percentile(latent_slice_np, [1, 99])
                        if latent_slice_np.size > 0
                        else (0, 1)
                    )

                    cmap_to_use = (
                        plot_colormap
                        if plot_colormap != "default"
                        else plt.rcParams["image.cmap"]
                    )

                    im = ax.imshow(
                        latent_slice_np,
                        cmap=cmap_to_use,
                        aspect="auto",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    ax.set_title(
                        f"Step {i}, σ: {collected_sigmas_for_plot[i]:.3f}", fontsize=8
                    )
                    ax.axis("off")

                for j in range(num_plots, len(axes)):  # Hide unused subplots
                    axes[j].axis("off")

                if plot_title_text:
                    fig.suptitle(plot_title_text, fontsize=12)

                fig.tight_layout(
                    rect=[0, 0.03, 1, 0.95] if plot_title_text else None
                )  # Adjust for suptitle

                fig.canvas.draw()
                buf = fig.canvas.buffer_rgba()
                plot_image_np_rgba = np.asarray(buf)
                plt.close(fig)

                plot_image_np_rgb = plot_image_np_rgba[:, :, :3]  # RGBA to RGB
                plot_image_tensor = torch.from_numpy(
                    plot_image_np_rgb.astype(np.float32) / 255.0
                ).unsqueeze(0)
            except Exception as e:
                print(
                    f"Error during timestep plotting in WolfSamplerCustomAdvancedPlotter: {e}"
                )
                # plot_image_tensor remains the default black image

        return (output_latent_dict, denoised_output_latent_dict, plot_image_tensor)
