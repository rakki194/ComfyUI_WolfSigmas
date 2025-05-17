# wolf_plot_sampler_stats.py
import torch
import numpy as np
import io

# Try to import WolfSimpleSamplerScriptEvaluator to access its class variable
# This creates a dependency but is a common way to share data between nodes if not passing directly.
try:
    from .wolf_simple_sampler_script_evaluator import WolfSimpleSamplerScriptEvaluator
except ImportError:
    # This might happen if files are moved or during certain ComfyUI startups.
    # Fallback or error handling might be needed for robustness in a real package.
    print(
        "Warning: Could not import WolfSimpleSamplerScriptEvaluator. Plotting node may not find data."
    )
    WolfSimpleSamplerScriptEvaluator = None

# Matplotlib import
MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-GUI backend
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print(
        "Warning: Matplotlib is not installed. WolfPlotSamplerStatsNode will not be able to plot."
    )
    print("Please install Matplotlib: pip install matplotlib")


class WolfPlotSamplerStatsNode:
    CATEGORY = "debug/visualize"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot_image",)
    FUNCTION = "plot_stats"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trigger": (
                    "LATENT",
                    {"forceInput": True},
                ),
                "plot_width": ("INT", {"default": 800, "min": 200, "max": 4096}),
                "plot_height": ("INT", {"default": 1000, "min": 200, "max": 4096}),
                "font_size": ("INT", {"default": 10, "min": 5, "max": 20}),
            },
            "optional": {
                "title_override": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    def _create_placeholder_image(self, width, height, message):
        img = np.full((height, width, 3), 200, dtype=np.uint8)  # Light gray
        # Basic text rendering with PIL if available, otherwise just gray
        try:
            from PIL import Image, ImageDraw, ImageFont

            pil_img = Image.fromarray(img)
            draw = ImageDraw.Draw(pil_img)
            try:
                font = ImageFont.truetype("arial.ttf", size=max(15, width // 40))
            except IOError:
                font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), message, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (width - text_width) // 2
            text_y = (height - text_height) // 2
            draw.text((text_x, text_y), message, fill=(0, 0, 0), font=font)
            img = np.array(pil_img)
        except ImportError:
            pass  # Keep it simple if PIL is not there
        return (torch.from_numpy(img).unsqueeze(0).float() / 255.0,)

    def plot_stats(
        self, trigger, plot_width, plot_height, font_size, title_override=None
    ):
        if not MATPLOTLIB_AVAILABLE:
            return self._create_placeholder_image(
                plot_width, plot_height, "Matplotlib not available."
            )

        if WolfSimpleSamplerScriptEvaluator is None or not hasattr(
            WolfSimpleSamplerScriptEvaluator, "_LAST_RUN_STATS"
        ):
            return self._create_placeholder_image(
                plot_width, plot_height, "Sampler data source not found."
            )

        stats_data = WolfSimpleSamplerScriptEvaluator._LAST_RUN_STATS

        if not stats_data or not stats_data.get("steps"):
            return self._create_placeholder_image(
                plot_width, plot_height, "No statistics data found from sampler."
            )

        plt.style.use("seaborn-v0_8-darkgrid")  # Using a seaborn style
        plt.rcParams.update({"font.size": font_size})

        fig, axs = plt.subplots(
            4, 1, figsize=(plot_width / 100, plot_height / 100), dpi=100
        )
        fig.tight_layout(pad=4.0)  # Add padding between subplots and for title

        plot_title = (
            title_override if title_override else "Sampler Statistics Over Steps"
        )
        fig.suptitle(plot_title, fontsize=font_size + 4)

        steps = stats_data["steps"]

        # Plot 1: Sigmas
        axs[0].plot(
            steps,
            stats_data["sigmas_current"],
            label="Sigma Current (sigma_i)",
            color="blue",
            marker=".",
        )
        # axs[0].plot(steps, stats_data["sigmas_next"], label="Sigma Next (sigma_i+1)", color='lightblue', linestyle='--') # Can be noisy
        axs[0].set_ylabel("Sigma Value")
        axs[0].set_title("Sigma Schedule")
        axs[0].legend()
        axs[0].grid(True)

        # Plot 2: Latent Stats (x_i)
        axs[1].plot(
            steps,
            stats_data["latent_mean_xi"],
            label="Latent x_i Mean",
            color="green",
            marker=".",
        )
        axs[1].plot(
            steps,
            stats_data["latent_std_xi"],
            label="Latent x_i Std",
            color="lightgreen",
            linestyle="--",
            marker=".",
        )
        axs[1].set_ylabel("Value")
        axs[1].set_title("Input Latent (x_i) Statistics")
        axs[1].legend()
        axs[1].grid(True)

        # Plot 3: Denoised Prediction Stats (x0_pred_i)
        axs[2].plot(
            steps,
            stats_data["denoised_mean_x0_pred"],
            label="Denoised x0_pred_i Mean",
            color="red",
            marker=".",
        )
        axs[2].plot(
            steps,
            stats_data["denoised_std_x0_pred"],
            label="Denoised x0_pred_i Std",
            color="salmon",
            linestyle="--",
            marker=".",
        )
        axs[2].set_ylabel("Value")
        axs[2].set_title("Denoised Prediction (x0_pred_i) Statistics")
        axs[2].legend()
        axs[2].grid(True)

        # Plot 4: Combined Standard Deviations for scale comparison
        axs[3].plot(
            steps,
            stats_data["sigmas_current"],
            label="Sigma Current (sigma_i)",
            color="blue",
            marker=".",
        )
        axs[3].plot(
            steps,
            stats_data["latent_std_xi"],
            label="Latent x_i Std",
            color="lightgreen",
            linestyle="--",
            marker=".",
        )
        axs[3].plot(
            steps,
            stats_data["denoised_std_x0_pred"],
            label="Denoised x0_pred_i Std",
            color="salmon",
            linestyle=":",
            marker=".",
        )
        axs[3].set_xlabel("Sampling Step")
        axs[3].set_ylabel("Value")
        axs[3].set_title("Combined Standard Deviations & Sigmas")
        axs[3].legend()
        axs[3].grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        # Convert to PIL Image then to NumPy array
        from PIL import Image  # Matplotlib depends on Pillow usually

        pil_image = Image.open(buf)
        numpy_image = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
        plt.close(fig)
        buf.close()

        return (torch.from_numpy(numpy_image).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {"WolfPlotSamplerStatsNode": WolfPlotSamplerStatsNode}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfPlotSamplerStatsNode": "Wolf Plot Sampler Stats (🐺)"
}
