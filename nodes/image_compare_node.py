import folder_paths
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import json
import os
import random
import sys
import torch


# Note: Removed inheritance from PreviewImage
class ComfyUIImageCompare:
    def __init__(self):
        # Mimic PreviewImage initialization for save_images context
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1  # Preview uses lower compression

    NAME = "Image Compare (🐺)"  # Directly define name
    CATEGORY = "ComfyUI-ImageCompare"  # Directly define category
    FUNCTION = "compare_images"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    # Copied and adapted from SaveImage/PreviewImage in nodes.py
    # Make sure all dependencies like 'folder_paths', 'Image', 'PngInfo', 'np', 'json', 'os', 'args' are available
    def save_images(
        self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(images):
            # Ensure image is on CPU before converting to numpy
            if image.device != torch.device("cpu"):
                image = image.cpu()

            i = 255.0 * image.numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            # Assume metadata is enabled unless globally disabled (which we can't easily check here)
            # If it *is* disabled globally, this will save metadata anyway, which is acceptable.
            # Check if prompt or extra_pnginfo is provided before creating PngInfo object
            if prompt is not None or extra_pnginfo is not None:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            # Preserve batch number if filename contains placeholder, otherwise use counter
            if "%batch_num%" in filename:
                filename_with_batch_num = filename.replace(
                    "%batch_num%", str(batch_number)
                )
                file = f"{filename_with_batch_num}_{counter:05}_.png"
            else:
                file = f"{filename}_{counter:05}_.png"

            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}

    # The main function for this node
    def compare_images(
        self,
        image_a=None,
        image_b=None,
        # filename_prefix="ComfyUI-ImageCompare.compare.", # Use default from save_images or keep custom?
        prompt=None,
        extra_pnginfo=None,
    ):
        print("[ComfyUI-ImageCompare] Entering compare_images")  # DEBUG
        # Use the node's filename prefix logic
        filename_prefix = "ComfyUI-ImageCompare.compare."

        result = {"ui": {"a_images": [], "b_images": []}}
        try:
            if image_a is not None and len(image_a) > 0:
                print(
                    f"[ComfyUI-ImageCompare] Processing image_a (shape: {image_a.shape})"
                )  # DEBUG
                # Call the copied save_images method
                saved_a = self.save_images(
                    image_a, filename_prefix, prompt, extra_pnginfo
                )
                print(
                    f"[ComfyUI-ImageCompare] Result from save_images (A): {saved_a}"
                )  # DEBUG
                if saved_a and "ui" in saved_a and "images" in saved_a["ui"]:
                    result["ui"]["a_images"] = saved_a["ui"]["images"]
                else:
                    print(
                        "[ComfyUI-ImageCompare] WARNING: Unexpected format from save_images (A)"
                    )  # DEBUG

            if image_b is not None and len(image_b) > 0:
                print(
                    f"[ComfyUI-ImageCompare] Processing image_b (shape: {image_b.shape})"
                )  # DEBUG
                # Call the copied save_images method
                saved_b = self.save_images(
                    image_b, filename_prefix, prompt, extra_pnginfo
                )
                print(
                    f"[ComfyUI-ImageCompare] Result from save_images (B): {saved_b}"
                )  # DEBUG
                if saved_b and "ui" in saved_b and "images" in saved_b["ui"]:
                    result["ui"]["b_images"] = saved_b["ui"]["images"]
                else:
                    print(
                        "[ComfyUI-ImageCompare] WARNING: Unexpected format from save_images (B)"
                    )  # DEBUG
        except Exception as e:
            print(
                f"[ComfyUI-ImageCompare] ERROR during compare_images: {e}",
                file=sys.stderr,
            )  # DEBUG
            import traceback

            traceback.print_exc()
            # Optionally re-raise or return an error structure?
            # For now, just return the potentially empty result

        # Ensure the output format matches what the JS expects
        # The JS now expects { ui: { a_images: [...], b_images: [...] } }
        print(f"[ComfyUI-ImageCompare] Returning result: {result}")  # DEBUG
        return result
