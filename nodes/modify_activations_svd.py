import torch
import comfy.model_management


# Helper function
def get_target_module(model, target_name):
    module = model
    for name in target_name.split("."):
        if not hasattr(module, name):
            if name.isdigit() and isinstance(
                module, (torch.nn.ModuleList, torch.nn.Sequential)
            ):
                try:
                    module = module[int(name)]
                    continue
                except IndexError:
                    pass
            return None
        module = getattr(module, name)
    return module


class ModifyActivationsSVD:
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/experimental"
    OPERATION_MODES = ["keep_top", "zero_bottom"]

    @classmethod
    def INPUT_TYPES(cls):
        # Max SV value is indicative, real limit depends on tensor size
        max_sv_limit = 8192
        return {
            "required": {
                "model": ("MODEL",),
                "target_block_name": ("STRING", {"default": "middle_block.1"}),
                "n_singular_values": (
                    "INT",
                    {"default": -1, "min": -1, "max": max_sv_limit},
                ),
                "operation_mode": (cls.OPERATION_MODES, {"default": "keep_top"}),
            }
        }

    def patch(
        self, model, target_block_name, n_singular_values, operation_mode="keep_top"
    ):
        new_model = model.clone()

        if not target_block_name:
            print("Warning: target_block_name is empty. No SVD patch applied.")
            return (new_model,)

        try:
            unet_model = new_model.model.diffusion_model
            if not isinstance(unet_model, torch.nn.Module):
                raise TypeError(
                    "model.model.diffusion_model is not a valid PyTorch Module"
                )
        except AttributeError:
            print(
                "Error: Could not access model.model.diffusion_model. Cannot apply SVD patch."
            )
            return (model,)
        except TypeError as e:
            print(f"Error accessing diffusion model: {e}. Cannot apply SVD patch.")
            return (model,)

        target_module = get_target_module(unet_model, target_block_name)

        if target_module is None:
            print(
                f"Warning: Target block '{target_block_name}' not found for SVD patch. No patch applied."
            )
            return (new_model,)

        # --- Hook Function Definition ---
        def hook_fn(module, input, output):
            is_tuple_output = isinstance(output, tuple)
            output_tensor = output[0] if is_tuple_output else output

            if not isinstance(output_tensor, torch.Tensor):
                return None

            if output_tensor.ndim == 4:
                B, C, H, W = output_tensor.shape
                original_dtype = output_tensor.dtype

                # Reshape for SVD: (B, C, H*W)
                reshaped_tensor = output_tensor.view(B, C, H * W)
                modified_batch = []

                for i in range(B):
                    matrix = reshaped_tensor[i].float()  # SVD often needs float32

                    try:
                        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
                        num_sv = len(S)

                        n_effect = (
                            num_sv
                            if n_singular_values < 0
                            else min(n_singular_values, num_sv)
                        )

                        modified_matrix = None
                        if (
                            n_effect == num_sv and operation_mode != "zero_bottom"
                        ):  # No change needed unless zeroing bottom 0
                            modified_matrix = matrix
                        elif n_effect == 0 and operation_mode == "keep_top":
                            modified_matrix = torch.zeros_like(matrix)
                        else:
                            if operation_mode == "keep_top":
                                s_k = S[:n_effect]
                                modified_matrix = (
                                    U[:, :n_effect] @ torch.diag(s_k) @ Vh[:n_effect, :]
                                )
                            elif operation_mode == "zero_bottom":
                                S_new = S.clone()
                                S_new[num_sv - n_effect :] = (
                                    0.0  # Zero out the bottom n_effect values
                                )
                                modified_matrix = U @ torch.diag(S_new) @ Vh

                        if modified_matrix is None:
                            modified_matrix = matrix  # Fallback if logic fails

                        # Reshape back and append
                        modified_batch.append(
                            modified_matrix.view(C, H, W).to(original_dtype)
                        )

                    except Exception as e:
                        print(
                            f"  Batch {i}: SVD failed for '{target_block_name}'. Error: {e}. Using original."
                        )
                        modified_batch.append(matrix.view(C, H, W).to(original_dtype))

                final_modified_tensor = torch.stack(modified_batch, dim=0)
                print(
                    f"Applied SVD hook ({operation_mode}, n={n_singular_values}) to '{target_block_name}'"
                )

                if is_tuple_output:
                    new_output_tuple = list(output)
                    new_output_tuple[0] = final_modified_tensor
                    return tuple(new_output_tuple)
                else:
                    return final_modified_tensor
            else:
                print(
                    f"Warning: SVD Hook target '{target_block_name}' output not 4D (B,C,H,W), shape={output_tensor.shape}. Skipped."
                )
                return None

        # --- End Hook ---

        handle = target_module.register_forward_hook(hook_fn)
        # We might need a mechanism to track/remove these hooks later if applied multiple times
        # For now, rely on the hook being attached to the clone

        print(
            f"Registered SVD modification hook on block '{target_block_name}' for the cloned model."
        )
        return (new_model,)
