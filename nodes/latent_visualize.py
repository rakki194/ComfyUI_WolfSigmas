import torch


class LatentVisualizeDirect:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "latent/visualize"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
            }
        }

    def visualize(self, latent):
        latent_samples = latent["samples"]  # Shape (B, C, H, W)
        b, c, h, w = latent_samples.shape

        if c == 0:
            # Handle empty latent case? Return black image.
            print(
                "Warning: LatentVisualizeDirect input has 0 channels. Returning black image."
            )
            return (
                torch.zeros(
                    (b, h, w, 3), dtype=torch.float32, device=latent_samples.device
                ),
            )

        if c < 3:
            # Handle cases with fewer than 3 channels (e.g., grayscale VAEs?)
            # Visualize the first channel as grayscale repeated across RGB
            print(
                f"Warning: LatentVisualizeDirect input has {c} channel(s). Visualizing first channel as grayscale."
            )
            image_data = latent_samples[
                :, 0:1, :, :
            ]  # Take first channel -> (B, 1, H, W)
            image_data = image_data.repeat(1, 3, 1, 1)  # -> (B, 3, H, W)
        else:
            # Take the first 3 channels for RGB visualization
            image_data = latent_samples[:, :3, :, :]  # Shape (B, 3, H, W)

        # Normalize the data to [0, 1] range for visualization
        # Perform min-max normalization across H, W dims for each channel independently
        # Keep batch and channel dims separate during normalization
        min_val = torch.amin(image_data, dim=(-2, -1), keepdim=True)
        max_val = torch.amax(image_data, dim=(-2, -1), keepdim=True)
        range_val = max_val - min_val

        # Avoid division by zero if range is zero (flat channel/image)
        # Add small epsilon to range_val denominator
        normalized_data = (image_data - min_val) / (range_val + 1e-6)

        # Clip just in case of numerical issues or epsilon interaction
        normalized_data = torch.clamp(normalized_data, 0.0, 1.0)

        # Permute from (B, C, H, W) -> (B, H, W, C) for ComfyUI IMAGE format
        image_output = normalized_data.permute(0, 2, 3, 1).contiguous()

        # Ensure float32 (ComfyUI expects this)
        image_output = image_output.float()

        return (image_output,)
