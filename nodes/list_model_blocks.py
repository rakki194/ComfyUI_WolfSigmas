import torch


class ListModelBlocks:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("block_names_string",)
    FUNCTION = "list_blocks"
    CATEGORY = "debug/model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    def list_blocks(self, model):
        try:
            unet_model = model.model  # Access the underlying diffusion model
            if not isinstance(unet_model, torch.nn.Module):
                raise TypeError("model.model is not a valid PyTorch Module")

            block_names = [name for name, module in unet_model.named_modules()]

            if not block_names:
                return ("No named modules found within model.model.",)

            # Join names with newlines for readability
            output_string = "\n".join(block_names)

            return (output_string,)

        except AttributeError:
            return (
                "Error: Could not access model.model. Is the input a valid ComfyUI model?",
            )
        except TypeError as e:
            return (f"Error: {e}",)
        except Exception as e:
            return (f"An unexpected error occurred: {str(e)}",)
