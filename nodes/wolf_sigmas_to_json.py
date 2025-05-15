import torch
import json

# import math


class WolfSigmasToJSON:
    """
    Converts an input SIGMAS object to its JSON string representation.
    Useful for debugging or exporting sigma schedules.
    """

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sigmas_json",)
    FUNCTION = "convert_to_json"
    CATEGORY = "sampling/sigmas_wolf/util"  # New category for utility functions

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"sigmas": ("SIGMAS",)}}

    def convert_to_json(self, sigmas):
        if not isinstance(sigmas, torch.Tensor):
            # This case should ideally not happen if type system is enforced
            # but good to have a check.
            print("Warning: WolfSigmasToJSON received non-Tensor input for sigmas.")
            return (json.dumps([]),)

        sigmas_list = sigmas.tolist()
        try:
            sigmas_json_string = json.dumps(sigmas_list, indent=2)
        except TypeError as e:
            print(f"Error converting sigmas to JSON: {e}. Sigmas: {sigmas_list}")
            sigmas_json_string = json.dumps([])  # Return empty list on error
        return (sigmas_json_string,)
