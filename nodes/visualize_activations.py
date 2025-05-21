import torch
import comfy.model_management
import torch.nn.functional as F
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_target_module(
    model: torch.nn.Module, target_name: str
) -> torch.nn.Module | None:
    """Recursively searches for a submodule by its dot-separated name."""
    module = model
    for name in target_name.split("."):
        if not hasattr(module, name):
            # Handle cases like ModuleList/Sequential where indices are used
            if name.isdigit() and isinstance(
                module, (torch.nn.ModuleList, torch.nn.Sequential)
            ):
                try:
                    module = module[int(name)]
                    continue
                except IndexError:
                    pass  # Fall through to return None
            return None
        module = getattr(module, name)
    return module


def prepare_sdxl_conditioning(
    conditioning: list,
    model_wrapper: torch.nn.Module,
    latent_shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    Prepares the conditioning dictionary required for SDXL UNet forward pass,
    including the 'y' vector derived from pooled embeddings.

    Args:
        conditioning: The conditioning list, typically from CLIPTextEncodeSDXL.
                      Expected format: [[positive_embedding, {'pooled_output': positive_pooled}], ...]
        model_wrapper: The ComfyUI model wrapper (e.g., BaseModel).
        latent_shape: The shape of the latent tensor (B, C, H, W).
        device: The target device for the tensors.
        dtype: The target dtype for the tensors.

    Returns:
        A dictionary containing 'context' (embedding tensor) and 'y' (ADM conditioning vector).

    Raises:
        TypeError, ValueError: If conditioning format is incorrect or pooled_output is missing.
        AttributeError: If the model_wrapper lacks the 'encode_adm' method.
        RuntimeError: If 'encode_adm' fails.
    """
    if not isinstance(conditioning, list) or len(conditioning) == 0:
        raise TypeError("Conditioning input must be a non-empty list.")

    positive_conditioning = conditioning[0]
    if not isinstance(positive_conditioning, list) or len(positive_conditioning) < 2:
        raise TypeError(
            "Positive conditioning element must be a list [embedding_tensor, info_dict]."
        )

    embedding_tensor = positive_conditioning[0]
    pooled_dict = positive_conditioning[1]

    if not isinstance(pooled_dict, dict):
        raise TypeError(
            "Positive conditioning info (conditioning[0][1]) must be a dictionary."
        )
    if "pooled_output" not in pooled_dict:
        raise ValueError(
            "Could not find 'pooled_output' in the positive conditioning dictionary."
        )

    pooled_tensor = pooled_dict["pooled_output"]

    # Ensure tensors are on the correct device and dtype
    embedding_tensor = embedding_tensor.to(device=device, dtype=dtype)
    pooled_tensor = pooled_tensor.to(device=device, dtype=dtype)

    # Extract latent dimensions for ADM parameters
    _, _, latent_h, latent_w = latent_shape
    img_height = latent_h * 8  # Standard VAE factor
    img_width = latent_w * 8
    crop_h, crop_w = 0, 0  # Assuming no cropping for simplicity here
    target_h, target_w = img_height, img_width  # Assuming target matches original size

    # Generate the 'y' vector using the model's encode_adm method
    y_vector = None
    if hasattr(model_wrapper.model, "encode_adm"):
        try:
            y_vector = model_wrapper.model.encode_adm(
                width=img_width,
                height=img_height,
                crop_w=crop_w,
                crop_h=crop_h,
                target_width=target_w,
                target_height=target_h,
                pooled_output=pooled_tensor,
            ).to(device=device, dtype=dtype)
        except Exception as e:
            print(
                f"Error calling model.model.encode_adm: {e}. Cannot proceed without 'y'."
            )
            raise RuntimeError(
                f"Failed to generate 'y' vector via encode_adm: {e}"
            ) from e
    else:
        raise AttributeError(
            "model.model does not have encode_adm method. Cannot generate required 'y' vector for SDXL UNet."
        )

    if (
        y_vector is None
    ):  # Should be redundant given error handling above, but belt-and-suspenders
        raise RuntimeError("Failed to generate y_vector for unknown reasons.")

    # Return the conditioning dictionary expected by the UNet forward pass
    return {"context": embedding_tensor, "y": y_vector}


def prepare_flux_conditioning(
    conditioning: list,
    model_wrapper: torch.nn.Module,
    latent_shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """
    Prepares the conditioning dictionary required for FLUX/Chroma UNet forward pass.
    Expects conditioning to be [context, y] or [context] (in which case y is mean-pooled context).
    """
    if isinstance(conditioning, list):
        if len(conditioning) == 1:
            context = conditioning[0]
            if isinstance(context, list):
                # If context is a list, take the first tensor
                context = context[0]
            # Use mean pooling over tokens for y
            y = context.mean(dim=1, keepdim=True)
            print("INFO: Using mean-pooled context as y for Chroma.")
        elif len(conditioning) >= 2:
            context, y = conditioning[:2]
        else:
            raise TypeError(
                "Conditioning input for FLUX/Chroma must be a list: [context, y] or [context]."
            )

    return {
        "context": context.to(device=device, dtype=dtype),
        "y": y.to(device=device, dtype=dtype),
    }


def ensure_float32(tensor):
    if not tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(
            f"[Quantized Support] Converting tensor from {tensor.dtype} to float32 for compatibility."
        )
        return tensor.float()
    return tensor


def run_forward_pass_and_capture_activation(
    model_wrapper: torch.nn.Module,
    target_block_name: str,
    latent: dict,
    sigmas: torch.Tensor,
    sigma_index: int,
    conditioning: list,
    add_noise: bool,  # New parameter
) -> torch.Tensor:
    """
    Runs a single forward pass of the UNet model to capture the activation
    from a specified target module using a hook. Optionally adds noise to the latent input.

    Args:
        model_wrapper: The ComfyUI model wrapper.
        target_block_name: Dot-separated name of the target module within the UNet.
        latent: The latent input dictionary {'samples': tensor}.
        sigmas: The tensor of noise sigmas for the sampling schedule.
        sigma_index: The index into the sigmas tensor to use.
        conditioning: The conditioning list for the UNet.
        add_noise: If True, add Gaussian noise scaled by the selected sigma to the latent input.

    Returns:
        The captured activation tensor (moved to CPU, dtype float32).

    Raises:
        AttributeError: If diffusion_model or target_module is not found.
        TypeError: If diffusion_model is not a Module or conditioning is invalid.
        ValueError: If target_block_name is empty, sigmas are empty, or index is out of bounds.
        RuntimeError: If hook fails to capture activation or SDXL conditioning fails.
    """
    device = comfy.model_management.get_torch_device()
    offload_device = comfy.model_management.unet_offload_device()
    activation_storage = {}  # Temporary storage for the hook

    if not target_block_name:
        raise ValueError("target_block_name cannot be empty.")

    # --- Access UNet and Target Module ---
    try:
        unet_model = model_wrapper.model.diffusion_model
    except AttributeError:
        raise AttributeError("Could not find 'diffusion_model' within model.model.")
    if not isinstance(unet_model, torch.nn.Module):
        raise TypeError("model.model.diffusion_model is not a valid PyTorch Module.")

    target_module = get_target_module(unet_model, target_block_name)
    if target_module is None:
        available = "\n".join(
            [name for name, _ in unet_model.named_modules()][:30]
        )  # Show first 30
        raise ValueError(
            f"Target block '{target_block_name}' not found. Available starts with:\n{available}..."
        )

    # --- Determine Model Dtype ---
    try:
        actual_model_dtype = next(unet_model.parameters()).dtype
        if not isinstance(actual_model_dtype, torch.dtype):
            print(
                f"Warning: Detected UNet dtype is not a torch.dtype ({type(actual_model_dtype)}). Defaulting to float32 for forward pass."
            )
            actual_model_dtype = torch.float32
    except StopIteration:
        print(
            "Warning: Could not determine UNet dtype (no parameters found?). Defaulting to float32."
        )
        actual_model_dtype = torch.float32
    print(f"Using model dtype {actual_model_dtype} for forward pass.")

    # --- Hook Function Definition ---
    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        if isinstance(act, torch.Tensor):
            activation_storage["output"] = act.detach().cpu().to(torch.float32)
        else:
            print(
                f"Warning: Hook captured non-tensor output of type {type(act)} from {target_block_name}. Skipping."
            )

    # --- Prepare Inputs for Forward Pass ---
    if sigmas is None or len(sigmas) == 0:
        raise ValueError("Sigmas input cannot be empty.")
    if not (0 <= sigma_index < len(sigmas)):
        raise ValueError(
            f"sigma_index {sigma_index} out of bounds for sigmas list of length {len(sigmas)}."
        )
    target_sigma = sigmas[sigma_index].item()  # Get the specific sigma value
    print(f"Using sigma at index {sigma_index}: {target_sigma:.4f}")

    latent_samples = (
        latent["samples"].clone().to(device=device, dtype=actual_model_dtype)
    )
    latent_samples = ensure_float32(latent_samples)

    # --- Optional Noise Injection ---
    input_latent = latent_samples  # Default to original latent
    if add_noise:
        # Seed the generator for reproducible noise per step
        torch.manual_seed(0)
        # Generate Gaussian noise with same shape, device, dtype as latent
        noise = torch.randn_like(latent_samples)
        # Scale noise by the target sigma for this step
        scaled_noise = noise * target_sigma
        # Add noise to the latent samples
        input_latent = latent_samples + scaled_noise
        input_latent = ensure_float32(input_latent)
        print(f"Added noise with std dev (sigma) {target_sigma:.4f} to latent input.")
    else:
        print("Skipping noise injection.")
    # -----------------------------

    # Adjust latent channels for Chroma/Flux if needed
    model_type = type(model_wrapper.model).__name__
    if model_type in ["Chroma", "Flux"]:
        chroma_in_channels = getattr(unet_model, "in_channels", None)
        chroma_patch_size = getattr(unet_model, "patch_size", None)
        if chroma_in_channels is not None and chroma_patch_size is not None:
            expected_channels = chroma_in_channels // (
                chroma_patch_size * chroma_patch_size
            )
            if input_latent.shape[1] != expected_channels:
                import torch.nn as nn

                input_latent = ensure_float32(input_latent)
                conv = nn.Conv2d(
                    input_latent.shape[1], expected_channels, kernel_size=1
                )
                conv = conv.to(device=input_latent.device, dtype=torch.float32)
                input_latent = conv(input_latent)
                # print(
                #     f"DEBUG: Adjusted latent channels from {latent_samples.shape[1]} to {expected_channels} for Chroma/Flux."
                # )
                # print("DEBUG: New input_latent shape:", input_latent.shape)

    # Create sigma tensor matching batch size, on device, with correct dtype
    sigma_tensor = torch.full(
        (input_latent.shape[0],), target_sigma, device=device, dtype=actual_model_dtype
    )
    sigma_tensor = ensure_float32(sigma_tensor)

    # Prepare conditioning for SDXL or FLUX/Chroma
    model_type = type(model_wrapper.model).__name__
    if model_type in ["Chroma", "Flux"]:
        cond_dict = prepare_flux_conditioning(
            conditioning, model_wrapper, input_latent.shape, device, actual_model_dtype
        )
        cond_dict["context"] = ensure_float32(cond_dict["context"])
        cond_dict["y"] = ensure_float32(cond_dict["y"])
    else:
        cond_dict = prepare_sdxl_conditioning(
            conditioning, model_wrapper, input_latent.shape, device, actual_model_dtype
        )
        cond_dict["context"] = ensure_float32(cond_dict["context"])
        cond_dict["y"] = ensure_float32(cond_dict["y"])

    # --- Execute Forward Pass with Hook ---
    handle = None
    try:
        handle = target_module.register_forward_hook(hook_fn)
        comfy.model_management.load_model_gpu(
            model_wrapper
        )  # Ensure model is on execution device
        unet_model.to(device)

        with torch.no_grad():
            # Use the (potentially noisy) input_latent
            if model_type in ["Chroma", "Flux"]:
                # Provide a dummy guidance tensor (zeros) with the same batch size as input_latent
                batch_size = input_latent.shape[0]
                device_ = input_latent.device
                dtype_ = input_latent.dtype
                guidance = torch.zeros(batch_size, device=device_, dtype=dtype_)
                # Ensure all tensors are on the same device as the model
                model_device = next(unet_model.parameters()).device
                input_latent = input_latent.to(model_device)
                sigma_tensor = sigma_tensor.to(model_device)
                cond_dict["context"] = cond_dict["context"].to(model_device)
                cond_dict["y"] = cond_dict["y"].to(model_device)
                guidance = guidance.to(model_device)
                # print(
                #     "DEBUG: input_latent shape:", input_latent.shape, type(input_latent)
                # )
                # print(
                #     "DEBUG: sigma_tensor shape:", sigma_tensor.shape, type(sigma_tensor)
                # )
                # print(
                #     "DEBUG: context shape:",
                #     cond_dict["context"].shape,
                #     type(cond_dict["context"]),
                # )
                # print("DEBUG: y shape:", cond_dict["y"].shape, type(cond_dict["y"]))
                # print("DEBUG: guidance shape:", guidance.shape, type(guidance))
                # print("DEBUG: guidance value:", guidance)
                _ = unet_model.forward(
                    input_latent,
                    sigma_tensor,
                    cond_dict["context"],
                    cond_dict["y"],
                    guidance,
                )
            else:
                _ = unet_model.forward(input_latent, sigma_tensor, **cond_dict)

    except Exception as e:
        print(f"Error during UNet forward pass: {e}")
        raise  # Re-raise the exception after printing
    finally:
        # CRITICAL: Always remove the hook
        if handle is not None:
            handle.remove()
        # Attempt to offload the model
        unet_model.to(offload_device)

    # --- Retrieve Captured Activation ---
    if "output" not in activation_storage:
        raise RuntimeError(
            f"Hook failed to capture any activation from '{target_block_name}'. Check layer type/output."
        )

    activation = activation_storage["output"]  # Return CPU float32 tensor
    activation = ensure_float32(activation)
    # activation shape: (B, C, H, W) or (B, N, C) etc., dtype=float32, device=cpu
    # print(f"Captured activation shape: {activation.shape}, dtype: {activation.dtype}")
    # print("DEBUG: Activation min:", activation.min().item())
    # print("DEBUG: Activation max:", activation.max().item())
    # print("DEBUG: Activation mean:", activation.mean().item())
    # print("DEBUG: Activation std:", activation.std().item())
    # print("DEBUG: Activation contains NaN:", bool((activation != activation).any()))
    # print(
    #     "DEBUG: Activation contains Inf:",
    #     bool((activation == float("inf")).any() or (activation == float("-inf")).any()),
    # )
    return activation


def _normalize_image_tensor(
    tensor: torch.Tensor, percentile: float = 100.0
) -> torch.Tensor:
    """
    Normalizes a tensor (H, W) or (C, H, W) to the range [0, 1].
    If percentile < 100, values are clipped to [-p, p] where p is the
    percentile-th percentile of the absolute values, then mapped to [0, 1].
    Otherwise, standard min/max normalization is used.
    """
    tensor = ensure_float32(tensor)
    # print("DEBUG: _normalize_image_tensor input min:", tensor.min().item())
    # print("DEBUG: _normalize_image_tensor input max:", tensor.max().item())
    # print("DEBUG: _normalize_image_tensor input mean:", tensor.mean().item())
    # print("DEBUG: _normalize_image_tensor input std:", tensor.std().item())
    if percentile < 0.0 or percentile > 100.0:
        print(f"Warning: Invalid percentile {percentile}. Using 100.0 (min/max).")
        percentile = 100.0

    if percentile >= 99.99:  # Use standard min/max normalization
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        range_val = max_val - min_val
        normalized = (tensor - min_val) / (
            range_val + 1e-6
        )  # Epsilon for constant tensors
    else:
        # Percentile normalization
        abs_tensor = torch.abs(tensor)
        q = torch.quantile(
            abs_tensor.to(torch.float64).cpu(), percentile / 100.0
        ).item()
        q = max(q, 1e-6)  # Ensure q is not zero to avoid division by zero
        clipped_tensor = torch.clamp(tensor, -q, q)
        normalized = (clipped_tensor + q) / (2 * q)

    out = torch.clamp(normalized, 0.0, 1.0)
    # print("DEBUG: _normalize_image_tensor output min:", out.min().item())
    # print("DEBUG: _normalize_image_tensor output max:", out.max().item())
    # print("DEBUG: _normalize_image_tensor output mean:", out.mean().item())
    # print("DEBUG: _normalize_image_tensor output std:", out.std().item())
    return out


def _whiten_tensor(
    tensor: torch.Tensor, mean: bool = True, std: bool = True
) -> torch.Tensor:
    """
    Whitens a tensor (H, W) or (C, H, W) by subtracting the mean and dividing by the standard deviation.
    """
    tensor = ensure_float32(tensor)
    if mean:
        tensor = tensor - tensor.mean()
    if std:
        tensor = tensor / (tensor.std() + 1e-6)
    return tensor


def _apply_colormap(tensor_hw: torch.Tensor, colormap: str) -> torch.Tensor:
    """Applies a colormap to a normalized single-channel tensor (H, W) -> (H, W, 3)."""
    tensor_hw = ensure_float32(tensor_hw)
    if tensor_hw.ndim != 2:
        raise ValueError(
            f"Input tensor for colormap must be 2D (H, W), got {tensor_hw.ndim}D"
        )
    if colormap == "greyscale":
        return tensor_hw.unsqueeze(-1).repeat(1, 1, 3)
    elif colormap == "coolwarm":
        # Manual Coolwarm: 0 -> Blue, 0.5 -> White, 1 -> Red
        x = tensor_hw
        # Calculate components based on interpolating between Blue-White and White-Red
        t1 = 2 * x  # Ramp from 0 to 2
        t2 = 2 * (x - 0.5)  # Ramp from -1 to 1

        R = torch.where(x < 0.5, t1, torch.ones_like(x))
        G = torch.where(x < 0.5, t1, 1 - t2)
        B = torch.where(x < 0.5, torch.ones_like(x), 1 - t2)

        # Combine and clamp
        rgb_tensor = torch.stack([R, G, B], dim=-1)
        return torch.clamp(rgb_tensor, 0.0, 1.0)
    else:
        try:
            import matplotlib
            import matplotlib.cm

            cmap = matplotlib.cm.get_cmap(colormap)
            np_img = tensor_hw.cpu().numpy()
            colored = cmap(np_img)[:, :, :3]  # Drop alpha
            # print(
            #     f"DEBUG: Colormap '{colormap}' output min: {colored.min()}, max: {colored.max()}, dtype: {colored.dtype}, shape: {colored.shape}"
            # )
            torch_img = torch.from_numpy(colored).float()
            torch_img = torch.clamp(torch_img, 0.0, 1.0)
            if torch_img.shape[-1] != 3:
                # print(
                #     f"Warning: Colormap output has {torch_img.shape[-1]} channels, expected 3."
                # )
                torch_img = torch_img[..., :3]
            # print(
            #     f"DEBUG: torch_img min: {torch_img.min().item()}, max: {torch_img.max().item()}, mean: {torch_img.mean().item()}, std: {torch_img.std().item()}, shape: {torch_img.shape}, dtype: {torch_img.dtype}"
            # )
            return torch_img
        except Exception as e:
            print(
                f"Warning: Failed to apply colormap '{colormap}': {e}. Defaulting to greyscale."
            )
            return tensor_hw.unsqueeze(-1).repeat(1, 1, 3)


def _draw_text_on_image(
    img_tensor_hwc: torch.Tensor, text: str, font_size_divisor: int = 6
) -> torch.Tensor:
    """
    Draws text onto a single image tensor (H, W, C) using PIL.
    Returns original tensor if font loading or drawing fails.
    """
    img_tensor_hwc = ensure_float32(img_tensor_hwc)
    H, W, C = img_tensor_hwc.shape
    if H < 10 or W < 10:
        return img_tensor_hwc

    font = None
    try:
        font_size = max(8, H // font_size_divisor)
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            try:
                # Pillow 10.0.0+ allows size argument for default font
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                # Fallback for older Pillow versions
                font = ImageFont.load_default()
                if font_size > 10:
                    print(
                        "Warning: Using small default bitmap font. Install 'arial.ttf' or Pillow 10+ for better results."
                    )
    except Exception as e:
        print(
            f"Warning: Could not load font. Text '{text}' will not be drawn. Error: {e}"
        )
        return img_tensor_hwc

    try:
        img_np = (img_tensor_hwc * 255).byte().numpy()
        pil_image = Image.fromarray(img_np, "RGB")
        draw = ImageDraw.Draw(pil_image)
        text_pos = (2, 1)
        fill_color = (255, 255, 255)
        draw.text(text_pos, text, font=font, fill=fill_color)
        img_tensor_out = torch.from_numpy(np.array(pil_image)).float() / 255.0
        return img_tensor_out
    except Exception as e:
        print(f"Warning: Failed to draw text '{text}'. Error: {e}")
        return img_tensor_hwc


def _arrange_images_in_grid(
    image_tensors_hwc: list[torch.Tensor],
    padding_x: int = 0,
    padding_y: int = 0,
    padding_color_rgb: tuple[int, int, int] = (0, 0, 0),  # Default black
) -> torch.Tensor:
    """
    Arranges a list of (H, W, C) image tensors into a square grid.
    Adds optional padding between images.
    """
    image_tensors_hwc = [ensure_float32(img) for img in image_tensors_hwc]
    if not image_tensors_hwc:
        return torch.zeros(
            (1, 1, 3), dtype=torch.float32
        )  # Return tiny black image as float32

    num_images = len(image_tensors_hwc)
    grid_size = math.ceil(math.sqrt(num_images))
    img_h, img_w, img_c = image_tensors_hwc[0].shape

    # Ensure valid padding values
    padding_x = max(0, padding_x)
    padding_y = max(0, padding_y)

    # Calculate total grid dimensions including padding
    total_h = grid_size * img_h + (grid_size - 1) * padding_y
    total_w = grid_size * img_w + (grid_size - 1) * padding_x

    # Convert padding color to tensor format (0-1 float)
    pad_color_tensor = torch.tensor(padding_color_rgb, dtype=torch.float32) / 255.0

    # Create the grid tensor filled with padding color, float32
    grid_tensor = torch.zeros((total_h, total_w, img_c), dtype=torch.float32)
    grid_tensor[:, :] = pad_color_tensor  # Fill with padding color

    current_row, current_col = 0, 0
    for i, img_tensor in enumerate(image_tensors_hwc):
        if img_tensor.shape != (img_h, img_w, img_c):
            print(
                f"Warning: Image {i} has shape {img_tensor.shape}, expected {(img_h, img_w, img_c)}. Skipping."
            )
            continue

        # Ensure image is float32 in [0, 1]
        img_tensor = img_tensor.to(dtype=torch.float32)
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # Calculate top-left corner for pasting the image
        paste_y = current_row * (img_h + padding_y)
        paste_x = current_col * (img_w + padding_x)

        # Paste the image
        grid_tensor[paste_y : paste_y + img_h, paste_x : paste_x + img_w, :] = (
            img_tensor
        )

        current_col += 1
        if current_col >= grid_size:
            current_col = 0
            current_row += 1

    return grid_tensor


def _create_visualization_grid(
    normalized_images_hw: list[torch.Tensor],
    colormap: str,
    render_ids: bool,
    padding_x: int,
    padding_y: int,
    padding_color: str,
    id_prefix: str = "",
) -> torch.Tensor:
    """
    Applies colormap, optionally draws IDs, and arranges images into a grid with padding.
    """
    normalized_images_hw = [ensure_float32(img) for img in normalized_images_hw]
    processed_images_hwc = []
    for i, img_hw in enumerate(normalized_images_hw):
        # 1. Apply Colormap
        img_hwc_color = _apply_colormap(img_hw, colormap)

        # 2. Draw ID (Optional)
        if render_ids:
            img_hwc_color = _draw_text_on_image(img_hwc_color, f"{id_prefix}{i}")

        processed_images_hwc.append(img_hwc_color)

    # Define padding colors
    color_map_rgb = {
        "Magenta": (255, 0, 255),
        "Black": (0, 0, 0),
        "White": (255, 255, 255),
    }
    padding_color_rgb = color_map_rgb.get(
        padding_color, (0, 0, 0)
    )  # Default to black if invalid

    # 3. Arrange in Grid with Padding
    final_grid = _arrange_images_in_grid(
        processed_images_hwc,
        padding_x=padding_x,
        padding_y=padding_y,
        padding_color_rgb=padding_color_rgb,
    )
    return final_grid


def _perform_pca(
    activation_item_chw: torch.Tensor, num_components: int, percentile: float
) -> list[torch.Tensor]:
    """
    Performs PCA on a (C, H, W) activation tensor.

    Args:
        activation_item_chw: The activation tensor for a single batch item (C, H, W).
        num_components: The number of principal components to compute.
        percentile: The percentile to use for normalization.

    Returns:
        A list of normalized principal component score tensors [(H, W), (H, W), ...],
        each normalized individually to [0, 1] and on CPU.

    Raises:
        ValueError: If C < 2 or num_components <= 0.
        RuntimeError: If SVD fails during PCA calculation.
    """
    activation_item_chw = ensure_float32(activation_item_chw)
    C, H, W = activation_item_chw.shape
    if C < 2:
        raise ValueError(f"PCA requires at least 2 channels, but got {C}.")
    if num_components <= 0:
        print("Warning: num_components for PCA is <= 0, returning empty list.")
        return []

    num_components_to_compute = min(num_components, C)
    print(f"Performing PCA to compute top {num_components_to_compute} components.")

    # --- PCA Calculation ---
    # Ensure data is on CPU for SVD if it wasn't already
    X = activation_item_chw.cpu().permute(1, 2, 0).reshape(H * W, C)
    X_centered = _whiten_tensor(X, mean=True, std=False)

    try:
        # SVD on CPU is generally more stable for large matrices
        U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        Scores = U * S  # Shape: (N, C), where N = H*W
    except RuntimeError as e:
        if "svd" in str(e).lower():
            raise RuntimeError(
                f"SVD failed during PCA: {e}. Check for NaNs/Infs in activations."
            )
        else:
            raise RuntimeError(f"Runtime error during SVD/PCA: {e}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during SVD/PCA: {e}")

    # --- Extract and Normalize PC Scores ---
    normalized_pcs_hw = []
    for pc_idx in range(num_components_to_compute):
        pc_scores = Scores[:, pc_idx]  # Shape (N,)
        pc_img_data_hw = pc_scores.reshape(H, W)
        # Normalize each component individually to [0, 1] using the provided method
        normalized_pc_hw = _normalize_image_tensor(pc_img_data_hw, percentile)
        normalized_pcs_hw.append(normalized_pc_hw)  # Already on CPU

    return normalized_pcs_hw


def infer_transformer_spatial_shape(
    num_tokens: int, latent_shape: tuple
) -> tuple[int, int]:
    """
    Infers spatial dimensions (H', W') for transformer tokens based on the
    aspect ratio of the original latent space. Aims for H' * W' = num_tokens.
    """
    _, _, H_lat, W_lat = latent_shape
    target_aspect_ratio = 1.0 if H_lat == 0 or W_lat == 0 else H_lat / W_lat

    best_h, best_w = 1, num_tokens
    min_aspect_diff = abs((best_h / best_w) - target_aspect_ratio)

    for h_prime in range(1, int(math.sqrt(num_tokens)) + 1):
        if num_tokens % h_prime == 0:
            w_prime = num_tokens // h_prime
            aspect_ratio = h_prime / w_prime
            aspect_diff = abs(aspect_ratio - target_aspect_ratio)

            # Check if this pair (h', w') is better
            if aspect_diff < min_aspect_diff:
                min_aspect_diff = aspect_diff
                best_h, best_w = h_prime, w_prime

            # Check the other pair (w', h') as well
            aspect_ratio_inv = w_prime / h_prime
            aspect_diff_inv = abs(aspect_ratio_inv - target_aspect_ratio)
            if aspect_diff_inv < min_aspect_diff:
                min_aspect_diff = aspect_diff_inv
                best_h, best_w = w_prime, h_prime  # Use the swapped pair

    print(
        f"Inferred spatial shape ({best_h}x{best_w}) for {num_tokens} tokens based on latent aspect ratio {target_aspect_ratio:.2f}."
    )
    return best_h, best_w


def process_transformer_activation(
    activation: torch.Tensor,  # Shape: (B, N, C) or (B, C, N)
    latent_shape: tuple,  # Shape: (B, C, H_lat, W_lat)
    visualization_mode: str,  # "Channel Grid" or "PCA Grid"
    num_components: int,  # Used for both limiting channels and PCA
    render_ids: bool,
    colormap: str,
    percentile: float,
) -> torch.Tensor:
    """
    Processes 3D transformer activations by inferring spatial dimensions,
    reshaping, and then applying the chosen visualization grid method.
    """
    activation = ensure_float32(activation)
    B = activation.shape[0]
    if B > 1:
        print("Warning: Batch size > 1. Visualizing first batch item only.")
    act_item = activation[0]  # -> (N, C) or (C, N)

    # Determine N (tokens) and C (features) dimensions
    if act_item.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor after batch selection, got {act_item.ndim}D"
        )
    dim0, dim1 = act_item.shape
    # Heuristic: Assume the larger dimension is Tokens (N) if not square
    if dim0 > dim1:
        N, C = dim0, dim1
        act_item_nc = act_item
        print(f"Assuming transformer shape (N={N}, C={C}).")
    elif dim1 > dim0:
        C, N = dim0, dim1
        act_item_nc = act_item.transpose(0, 1)
        print(f"Assuming transformer shape (C={C}, N={N}). Transposing to (N, C).")
    else:
        N, C = dim0, dim1
        act_item_nc = act_item
        print(f"Ambiguous square activation ({N}x{C}). Assuming (Tokens, Features).")

    if N == 0 or C == 0:
        raise ValueError(
            f"Transformer activation has zero dimension ({N=}, {C=}). Cannot process."
        )

    # Infer the spatial dimensions H', W' based on latent aspect ratio
    H_prime, W_prime = infer_transformer_spatial_shape(N, latent_shape)

    # Reshape the activation from (N, C) -> (H', W', C) -> (C, H', W')
    try:
        activation_reshaped_chw = (
            act_item_nc.reshape(H_prime, W_prime, C).permute(2, 0, 1).contiguous()
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to reshape transformer activation (N={N}, C={C}) to ({C}, {H_prime}, {W_prime}): {e}. Check factors."
        )

    print(
        f"Reshaped transformer activation to spatial format: ({C}, {H_prime}, {W_prime})"
    )

    # --- Apply the visualization logic to the reshaped tensor ---
    grid_tensor_hwc = None
    normalized_images_hw = []
    id_prefix = ""

    if visualization_mode == "Channel Grid":
        num_to_show = min(num_components, C)
        print(
            f"Visualizing first {num_to_show} (reshaped) channels with colormap '{colormap}'."
        )
        if num_to_show == 0:
            print("Warning: num_components is 0.")
        # Normalize the selected channels
        for c_idx in range(num_to_show):
            channel_data = activation_reshaped_chw[c_idx]  # Shape (H', W')
            normalized_images_hw.append(
                _normalize_image_tensor(channel_data.cpu(), percentile)
            )
        id_prefix = ""

    elif visualization_mode == "PCA Grid":
        print(
            f"Visualizing top {num_components} PCA components (reshaped) with colormap '{colormap}'."
        )
        # Perform PCA on the reshaped (C, H', W') tensor
        # _perform_pca returns normalized (H', W') tensors on CPU
        normalized_images_hw = _perform_pca(
            activation_reshaped_chw, num_components, percentile
        )
        id_prefix = "PC "
    else:
        raise ValueError(
            f"Unsupported visualization_mode '{visualization_mode}' for reshaped transformer activation."
        )

    # Create the grid from the prepared list of normalized H'xW' images
    grid_tensor_hwc = _create_visualization_grid(
        normalized_images_hw, colormap, render_ids, 2, 2, "Black", id_prefix
    )

    return grid_tensor_hwc


def apply_upscaling(image_tensor: torch.Tensor, factor: int) -> torch.Tensor:
    """Upscales an image tensor (1, H, W, 3) using nearest-neighbor interpolation."""
    if factor <= 1:
        return image_tensor

    b, h, w, c = image_tensor.shape
    image_tensor_bchw = image_tensor.permute(0, 3, 1, 2).contiguous()
    upscaled_output = F.interpolate(
        image_tensor_bchw, scale_factor=factor, mode="nearest"
    )
    image_output = upscaled_output.permute(0, 2, 3, 1).contiguous()
    print(
        f"Upscaled output by {factor}x to {image_output.shape[1]}x{image_output.shape[2]}"
    )
    return image_output


# --- DePatchify for DiT/Chroma/Flux ---
def depatchify(tokens, latent_shape, patch_size, in_channels):
    """
    tokens: [B, N, C*P^2]
    latent_shape: (B, C, H, W)
    patch_size: int
    in_channels: int
    Returns: [B, C, H, W]
    """
    B, N, CP2 = tokens.shape
    C = in_channels
    P = patch_size
    H = latent_shape[2]
    W = latent_shape[3]
    assert CP2 == C * P * P, f"Token last dim {CP2} != C*P^2 ({C}*{P}^2)"
    assert N == (H // P) * (
        W // P
    ), f"Token count {N} != (H/P)*(W/P) ({H}//{P}*{W}//{P})"
    # Step 1: [B, N, C*P^2] -> [B, N, C, P, P]
    x = tokens.view(B, N, C, P, P)
    # Step 2: [B, N, C, P, P] -> [B, H//P, W//P, C, P, P]
    x = x.view(B, H // P, W // P, C, P, P)
    # Step 3: [B, H//P, W//P, C, P, P] -> [B, C, H//P, P, W//P, P]
    x = x.permute(0, 3, 1, 4, 2, 5)
    # Step 4: [B, C, H//P, P, W//P, P] -> [B, C, H, W]
    x = x.contiguous().view(B, C, H, W)
    print(
        f"[DePatchify] Converted tokens [B, N, C*P^2]={tokens.shape} to [B, C, H, W]={x.shape}"
    )
    return x


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Unified ComfyUI Node Class                                                   #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class VisualizeActivation:
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize_activation"
    CATEGORY = "debug/visualize"

    @classmethod
    def INPUT_TYPES(cls):
        # Infer available blocks dynamically (if possible) or provide a default list
        # This part remains complex and might need user input or refinement
        # For now, providing common examples:
        common_blocks = [
            "input_blocks.1.1.transformer_blocks.0.attn1",  # SDXL Attention
            "input_blocks.4.1.transformer_blocks.0.attn1",
            "middle_block.1.transformer_blocks.0.attn1",
            "output_blocks.3.1.transformer_blocks.0.attn1",
            "output_blocks.7.1.transformer_blocks.0.attn1",
            "input_blocks.0.0.weight",  # SD 1.5 Conv
            "input_blocks.1.0.emb_layers.1",  # SD 1.5 Time Embedding
            "middle_block.1.proj_out",  # SD 1.5 ResBlock projection
            "diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1",  # Explicit diffusion_model path
            "diffusion_model.middle_block.1.transformer_blocks.0.attn1",
            "diffusion_model.output_blocks.5.1.transformer_blocks.0.attn1",
            "Not Listed (Enter Manually)",
        ]
        # TODO: Potentially add a way to dynamically list modules from a loaded model

        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "conditioning": ("CONDITIONING",),
                "sigmas": ("SIGMAS",),
                "target_block_name": (common_blocks,),
                "custom_block_name": ("STRING", {"default": "", "multiline": False}),
                "visualization_mode": (["Channel Grid", "PCA Grid"],),
                "num_components": (
                    "INT",
                    {"default": 16, "min": 1, "max": 4096, "step": 1},
                ),  # Used for both modes
                "colormap": (
                    [
                        "viridis",
                        "plasma",
                        "inferno",
                        "magma",
                        "cividis",
                        "gray",
                        "hot",
                        "cool",
                    ],
                    {"default": "viridis"},
                ),
            },
            "optional": {
                "sigma_index": ("INT", {"default": 0, "min": 0, "max": 1000}),
                "add_noise": ("BOOLEAN", {"default": False}),
                "upscale_factor": ("INT", {"default": 1, "min": 1, "max": 8}),
                "render_ids": ("BOOLEAN", {"default": True}),
                "normalization_percentile": (
                    "FLOAT",
                    {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
                # New Padding Inputs
                "padding_x": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "padding_y": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
                "padding_color": (["Magenta", "Black", "White"], {"default": "Black"}),
            },
        }

    def visualize_activation(
        self,
        model,
        latent,
        conditioning,
        sigmas,
        target_block_name,
        custom_block_name,
        visualization_mode,
        num_components,  # Unified parameter
        colormap,  # New parameter
        sigma_index=0,
        add_noise=False,  # New parameter
        upscale_factor=1,
        render_ids=True,
        normalization_percentile=100.0,
        padding_x=2,  # Added
        padding_y=2,  # Added
        padding_color="Black",  # Added
    ):
        # Use custom_block_name if 'Not Listed (Enter Manually)' is selected
        if target_block_name == "Not Listed (Enter Manually)" and custom_block_name:
            target_block_name = custom_block_name

        # --- 1. Run Forward Pass and Capture Activation (potentially with added noise) ---
        activation = run_forward_pass_and_capture_activation(
            model_wrapper=model,
            target_block_name=target_block_name,
            latent=latent,
            sigmas=sigmas,
            sigma_index=sigma_index,
            conditioning=conditioning,
            add_noise=add_noise,  # Pass the flag
        )
        # activation shape: (B, C, H, W) or (B, N, C) etc., dtype=float32, device=cpu
        print(
            f"Captured activation shape: {activation.shape}, dtype: {activation.dtype}"
        )
        print("DEBUG: Activation min:", activation.min().item())
        print("DEBUG: Activation max:", activation.max().item())
        print("DEBUG: Activation mean:", activation.mean().item())
        print("DEBUG: Activation std:", activation.std().item())
        print("DEBUG: Activation contains NaN:", bool((activation != activation).any()))
        print(
            "DEBUG: Activation contains Inf:",
            bool(
                (activation == float("inf")).any()
                or (activation == float("-inf")).any()
            ),
        )

        # --- DePatchify for Chroma/DiT/Flux if needed ---
        model_type = type(model.model).__name__
        unet_model = (
            model.model.diffusion_model
            if hasattr(model.model, "diffusion_model")
            else None
        )
        if (
            model_type in ["Chroma", "Flux", "DiT"]
            and activation.ndim == 3
            and unet_model is not None
        ):
            patch_size = getattr(unet_model, "patch_size", None)
            in_channels = getattr(unet_model, "in_channels", None)
            latent_shape = latent["samples"].shape
            if patch_size is not None and in_channels is not None:
                B, N, CP2 = activation.shape
                expected_CP2 = in_channels * patch_size * patch_size
                expected_N = (latent_shape[2] // patch_size) * (
                    latent_shape[3] // patch_size
                )
                if CP2 == expected_CP2 and N == expected_N:
                    print(
                        f"[DePatchify] Detected DiT/Chroma/Flux activation, depatchifying..."
                    )
                    activation = depatchify(
                        activation, latent_shape, patch_size, in_channels
                    )

        # --- 2. Process Activation Based on Shape and Mode ---
        grid_tensor_hwc = None  # Tensor for the final grid (H_grid, W_grid, 3)

        if activation.ndim == 4:
            # --- Handle 4D Spatial Activations (B, C, H, W) ---
            B, C, H, W = activation.shape
            print(
                f"Processing 4D spatial activation ({B=}, {C=}, {H=}, {W=}) with mode '{visualization_mode}' and colormap '{colormap}'."
            )
            if B > 1:
                print("Warning: Batch size > 1. Visualizing first batch item only.")
            act_item_chw = activation[0]  # Take first batch item -> (C, H, W), CPU

            normalized_images_hw = []
            id_prefix = ""

            if visualization_mode == "Channel Grid":
                num_to_show = min(num_components, C)
                if num_to_show == 0:
                    print("Warning: num_components is 0.")
                # Normalize the selected channels
                for c_idx in range(num_to_show):
                    channel_data = act_item_chw[c_idx]  # Shape (H, W)
                    normalized_images_hw.append(
                        _normalize_image_tensor(channel_data, normalization_percentile)
                    )
                id_prefix = ""

            elif visualization_mode == "PCA Grid":
                # Perform PCA on the (C, H, W) tensor
                # _perform_pca returns normalized (H, W) tensors on CPU
                normalized_images_hw = _perform_pca(
                    act_item_chw, num_components, normalization_percentile
                )
                id_prefix = "PC "

            else:
                raise ValueError(
                    f"Unsupported visualization_mode: {visualization_mode}"
                )

            # Create the grid from the prepared list of normalized HxW images
            grid_tensor_hwc = _create_visualization_grid(
                normalized_images_hw,
                colormap,
                render_ids,
                padding_x,
                padding_y,
                padding_color,
                id_prefix,
            )

        elif activation.ndim == 3:
            # --- Handle 3D Transformer Activations (B, N, C) or (B, C, N) ---
            print(
                f"Processing 3D transformer activation (shape: {activation.shape}) with mode '{visualization_mode}' and colormap '{colormap}'."
            )
            # Reshape and apply grid/PCA logic using num_components
            grid_tensor_hwc = process_transformer_activation(
                activation=activation,  # CPU tensor
                latent_shape=latent["samples"].shape,  # Pass original latent shape
                visualization_mode=visualization_mode,  # "Channel Grid" or "PCA Grid"
                num_components=num_components,  # Limits channels or PCA components after reshape
                render_ids=render_ids,
                colormap=colormap,
                percentile=normalization_percentile,  # Pass percentile
            )

        else:
            raise ValueError(
                f"Unsupported activation shape: {activation.shape}. Only 3D and 4D tensors are supported."
            )

        # Add batch dimension for ComfyUI
        image_output_no_upscale = grid_tensor_hwc.unsqueeze(
            0
        )  # Shape: (1, H_grid, W_grid, 3)

        # --- 3. Apply Optional Upscaling ---
        image_output = apply_upscaling(image_output_no_upscale, upscale_factor)
        # Shape: (1, H_out, W_out, 3), dtype=float32, device=cpu

        # --- 4. Return Result ---
        # ComfyUI expects image tensors in BHWC format, float type
        return (image_output,)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# Node Mappings for ComfyUI                                                    #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

NODE_CLASS_MAPPINGS = {
    "VisualizeActivation": VisualizeActivation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VisualizeActivation": "Visualize Activation",
}
