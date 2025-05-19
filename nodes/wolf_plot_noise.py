import torch
import numpy as np
import comfy.model_management
import comfy.utils

# Try to import matplotlib and set backend
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("WolfPlotNoise: Matplotlib not found. This node will not be available.")


class WolfPlotNoise:
    @classmethod
    def INPUT_TYPES(cls):
        if not MATPLOTLIB_AVAILABLE:
            return {
                "required": {
                    "error": (
                        "STRING",
                        {
                            "default": "Matplotlib not installed. This node is disabled.",
                            "multiline": True,
                        },
                    )
                }
            }

        return {
            "required": {
                "noise_provider": ("NOISE",),
                "sigmas": (
                    "SIGMAS",
                    {
                        "tooltip": "Full sigma schedule, primarily for context to the noise provider and for title info (sigmas[0])."
                    },
                ),
                "latent_template": ("LATENT",),
                "plot_mode": (
                    ["Single Channel (Colormapped)", "RGB (Channels 0,1,2)"],
                    {
                        "default": "Single Channel (Colormapped)",
                        "tooltip": "Mode for plotting the noise.",
                    },
                ),
                "batch_item_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 63, "step": 1},
                ),
                "channel_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1023,
                        "step": 1,
                        "tooltip": "Channel to plot in Single Channel mode.",
                    },
                ),
                "colormap": (
                    ["gray", "viridis", "plasma", "inferno", "magma", "cividis"],
                    {"default": "gray", "tooltip": "Colormap for Single Channel mode."},
                ),
                "plot_title": (
                    "STRING",
                    {"default": "Output of Noise Provider"},
                ),
                "figure_width_inches": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 2.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Width of the output figure in inches.",
                    },
                ),
                "figure_aspect_ratio": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 10.0,
                        "step": 0.1,
                        "tooltip": "Aspect ratio (width/height) of the plot area.",
                    },
                ),
                "dpi": (
                    "INT",
                    {
                        "default": 100,
                        "min": 50,
                        "max": 300,
                        "step": 10,
                        "tooltip": "Dots per inch for plot resolution.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot_noise"
    CATEGORY = "debug/visualize_wolf"

    def plot_noise(
        self,
        noise_provider,
        sigmas,
        latent_template,
        plot_mode,
        batch_item_index,
        channel_index,
        colormap,
        plot_title,
        figure_width_inches,
        figure_aspect_ratio,
        dpi,
    ):
        if not MATPLOTLIB_AVAILABLE:
            print(
                "Error: WolfPlotNoise (plot_noise) called but Matplotlib is not available."
            )
            dummy_black = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
            return (dummy_black,)

        context_sigma_val = "N/A"
        sigmas_for_provider = sigmas
        if not isinstance(sigmas, torch.Tensor) or sigmas.numel() == 0:
            print(
                "WolfPlotNoise: Sigmas input is empty or not a tensor. Plotting will proceed; context for provider might be limited."
            )
            sigmas_for_provider = torch.tensor(
                [1.0], device=comfy.model_management.get_torch_device()
            )  # Minimal sigma for provider
        else:
            context_sigma_val = f"{sigmas[0].item():.3f}"

        if "samples" not in latent_template or not isinstance(
            latent_template["samples"], torch.Tensor
        ):
            raise ValueError(
                "latent_template must be a dictionary with a 'samples' torch.Tensor."
            )

        template_samples = latent_template["samples"]
        b, c, h, w = template_samples.shape
        dtype = template_samples.dtype
        device = comfy.model_management.get_torch_device()

        actual_batch_idx = max(0, min(batch_item_index, b - 1 if b > 0 else 0))

        if h == 0 or w == 0:
            print(
                f"WolfPlotNoise: Latent template has zero height ({h}) or width ({w}). Cannot generate plot."
            )
            dummy_black = torch.zeros((1, 1, 1, 3), dtype=torch.float32, device="cpu")
            return (dummy_black,)

        dummy_samples_for_noise_gen = torch.zeros(
            (b, c, h, w), dtype=dtype, device=device
        )
        latent_image_dict_for_provider = {
            "samples": dummy_samples_for_noise_gen,
            "sigmas": sigmas_for_provider,
        }

        print(
            f"WolfPlotNoise: Generating single noise instance. Mode: {plot_mode}. Template shape: {(b,c,h,w)}, Device for noise gen: {device}"
        )

        noise_tensor_to_plot = noise_provider.generate_noise(
            latent_image_dict_for_provider
        ).to(device)

        if actual_batch_idx >= noise_tensor_to_plot.shape[0]:
            print(
                f"WolfPlotNoise: Batch index {actual_batch_idx} out of bounds for generated noise tensor (shape {noise_tensor_to_plot.shape}). Clamping to 0."
            )
            actual_batch_idx = 0

        if noise_tensor_to_plot.numel() == 0 or noise_tensor_to_plot.shape[0] == 0:
            print(
                f"WolfPlotNoise: Noise provider returned an empty or invalid tensor. Shape: {noise_tensor_to_plot.shape}"
            )
            dummy_black = torch.zeros(
                (1, h if h > 0 else 1, w if w > 0 else 1, 3),
                dtype=torch.float32,
                device="cpu",
            )
            return (dummy_black,)

        # Figure size calculation
        fig_height_inches = (
            figure_width_inches / figure_aspect_ratio
            if figure_aspect_ratio > 0
            else figure_width_inches
        )
        fig_height_inches = max(2.0, fig_height_inches)
        figure_width_inches = max(2.0, figure_width_inches)
        fig, ax = plt.subplots(
            1, 1, figsize=(figure_width_inches, fig_height_inches), dpi=dpi
        )

        title_with_context = (
            f"{plot_title} (Mode: {plot_mode}, σ[0] Ctx: {context_sigma_val})"
        )
        plot_aspect_ratio = w / h if h > 0 else 1.0

        if plot_mode == "RGB (Channels 0,1,2)":
            num_channels_available = noise_tensor_to_plot.shape[1]
            selected_noise_batch = noise_tensor_to_plot[actual_batch_idx]

            if num_channels_available >= 3:
                rgb_slices = selected_noise_batch[0:3, :, :]  # (3, H, W)
            elif num_channels_available == 2:
                print(
                    "WolfPlotNoise (RGB Mode): 2 channels available. Using Ch0 for Red, Ch1 for Green, Blue=0."
                )
                slice_r = selected_noise_batch[0, :, :]
                slice_g = selected_noise_batch[1, :, :]
                slice_b = torch.zeros_like(slice_r)  # Blue channel as zeros
                rgb_slices = torch.stack([slice_r, slice_g, slice_b], dim=0)
            elif num_channels_available == 1:
                print(
                    "WolfPlotNoise (RGB Mode): 1 channel available. Plotting as grayscale."
                )
                slice_gray = selected_noise_batch[0, :, :]
                rgb_slices = torch.stack([slice_gray, slice_gray, slice_gray], dim=0)
            else:  # 0 channels in noise_tensor_to_plot[actual_batch_idx] (should be caught earlier by noise_tensor_to_plot.numel() == 0)
                print(
                    "WolfPlotNoise (RGB Mode): 0 channels available in noise tensor. Plotting black."
                )
                rgb_slices = torch.zeros(
                    (3, h if h > 0 else 1, w if w > 0 else 1),
                    device=device,
                    dtype=dtype,
                )

            rgb_display_np = (
                rgb_slices.cpu().float().numpy().transpose(1, 2, 0)
            )  # (H, W, 3)

            for i in range(rgb_display_np.shape[2]):  # Normalize each channel
                channel_data = rgb_display_np[:, :, i]
                if channel_data.size > 0:
                    min_v = np.percentile(channel_data, 1)
                    max_v = np.percentile(channel_data, 99)
                    if max_v > min_v:
                        rgb_display_np[:, :, i] = (channel_data - min_v) / (
                            max_v - min_v
                        )
                    elif max_v == min_v:  # Handle flat channel (all same values)
                        rgb_display_np[:, :, i] = np.full_like(
                            channel_data, 0.5
                        )  # Mid-gray
                    else:  # Should ideally not be reached if max_v == min_v is handled
                        rgb_display_np[:, :, i] = np.zeros_like(channel_data)
                else:  # Should not happen if h,w > 0
                    rgb_display_np[:, :, i] = np.zeros_like(channel_data)

            rgb_display_np = np.clip(rgb_display_np, 0, 1)
            ax.imshow(rgb_display_np, aspect=plot_aspect_ratio)
            title_with_context += f" (Ch 0,1,2 as RGB)"

        else:  # Single Channel (Colormapped)
            actual_channel_idx = max(
                0,
                min(
                    channel_index,
                    (
                        noise_tensor_to_plot.shape[1] - 1
                        if noise_tensor_to_plot.shape[1] > 0
                        else 0
                    ),
                ),
            )
            if actual_channel_idx >= noise_tensor_to_plot.shape[1]:
                print(
                    f"WolfPlotNoise (Single Channel): Channel index {actual_channel_idx} out of bounds for noise tensor (shape {noise_tensor_to_plot.shape}). Clamping to 0."
                )
                actual_channel_idx = 0

            if noise_tensor_to_plot.shape[1] == 0:  # No channels to plot
                print(
                    "WolfPlotNoise (Single Channel): Noise tensor has 0 channels. Plotting black."
                )
                noise_slice_2d_for_plot = np.zeros(
                    (h if h > 0 else 1, w if w > 0 else 1), dtype=np.float32
                )
            else:
                noise_slice_2d_for_plot = (
                    noise_tensor_to_plot[actual_batch_idx, actual_channel_idx, :, :]
                    .cpu()
                    .float()
                    .numpy()
                )

            ax.imshow(noise_slice_2d_for_plot, cmap=colormap, aspect=plot_aspect_ratio)
            title_with_context += f" (Ch: {actual_channel_idx})"

        ax.set_title(title_with_context, fontsize=10)
        ax.axis("off")
        fig.tight_layout(rect=[0, 0, 1, 0.95] if title_with_context else None)

        fig.canvas.draw()
        plot_image_np_rgba = np.array(fig.canvas.buffer_rgba())
        plt.close(fig)

        plot_image_np_rgb = plot_image_np_rgba[:, :, :3]
        plot_image_tensor = torch.from_numpy(
            plot_image_np_rgb.astype(np.float32) / 255.0
        )
        plot_image_tensor = plot_image_tensor.unsqueeze(0)

        return (plot_image_tensor,)


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if MATPLOTLIB_AVAILABLE:
    NODE_CLASS_MAPPINGS["WolfPlotNoise"] = WolfPlotNoise
    NODE_DISPLAY_NAME_MAPPINGS["WolfPlotNoise"] = "Visualize Provided Noise (🐺)"
else:
    print(
        "Wolf Custom Nodes: WolfPlotNoise/VisualizeProvidedNoise is unavailable because Matplotlib is not installed."
    )
