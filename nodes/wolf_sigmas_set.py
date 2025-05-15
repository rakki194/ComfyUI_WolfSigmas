import torch
import json


class WolfSigmasSet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas_json": (
                    "STRING",
                    {"multiline": True, "default": "[14.61, ..., 0.0]"},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "set_sigmas"
    CATEGORY = "sampling/sigmas_wolf"

    def set_sigmas(self, sigmas_json):
        try:
            sigmas_list = json.loads(sigmas_json)
            if not isinstance(sigmas_list, list) or not all(
                isinstance(x, (int, float)) for x in sigmas_list
            ):
                raise ValueError("JSON must be a list of numbers.")
            sigmas = torch.tensor(sigmas_list, dtype=torch.float32)
            return (sigmas,)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            # Return a default or error-indicating tensor might be an option,
            # but ComfyUI usually expects the correct type or it will error out.
            # For now, re-raise or handle by returning an empty tensor / specific error signal if desired.
            raise ValueError(f"Invalid JSON format for sigmas: {e}")
        except Exception as e:
            print(f"Error processing sigmas: {e}")
            raise ValueError(f"Error converting JSON to sigmas tensor: {e}")
