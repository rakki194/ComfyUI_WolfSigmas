class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "execute"
    CATEGORY = "image utils"

    def execute(self, image):
        return (image.shape[2], image.shape[1], image.shape[0])


NODE_CLASS_MAPPINGS = {"GetImageSize": GetImageSize}

NODE_DISPLAY_NAME_MAPPINGS = {"GetImageSize": "Get Image Size"}
