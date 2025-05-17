import torch
import json
import traceback  # For more detailed error logging

print("WolfProbe: wolf_probe.py loaded")


# Helper function to process tensor data based on capture mode
def process_tensor_data(tensor, capture_mode, slice_size):
    if not isinstance(tensor, torch.Tensor):
        return {"type": str(type(tensor))}

    processed_data = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }

    if capture_mode == "Metadata Only":
        return processed_data

    # For actual data, detach and move to CPU
    tensor_cpu = tensor.detach().clone().cpu()

    if capture_mode == "Full Tensor (CPU)":
        try:
            processed_data["tensor_data"] = tensor_cpu.tolist()
            # Add a note about potential size
            processed_data["info"] = "Full tensor data. Can be very large."
        except Exception as e:
            processed_data["error"] = f"Error converting full tensor to list: {e}"
    elif capture_mode == "Sample Slice (CPU)":
        try:
            processed_data["tensor_sample"] = tensor_cpu.flatten()[:slice_size].tolist()
        except Exception as e:
            processed_data["error"] = f"Error getting tensor slice: {e}"
    elif capture_mode == "Mean/Std (CPU)":
        try:
            # Ensure tensor is float for mean/std calculations
            tensor_float = tensor_cpu.float()
            processed_data["mean"] = tensor_float.mean().item()
            processed_data["std"] = tensor_float.std().item()
        except Exception as e:
            processed_data["error"] = f"Error calculating mean/std: {e}"

    return processed_data


class WolfProbeNode:
    captured_data = {}  # Class variable to store data from hooks
    capture_handles = []  # Class variable to store hook handles for removal

    @classmethod
    def INPUT_TYPES(s):
        print("WolfProbeSetup: INPUT_TYPES called")
        return {
            "required": {
                "model": ("MODEL",),
                "target_module_name": (
                    "STRING",
                    {"default": "double_blocks.0.img_attn.proj"},
                ),
                "hook_type": (
                    ["Forward", "Backward - Output Grads", "Backward - Input Grads"],
                    {"default": "Forward"},
                ),
                "capture_mode": (
                    [
                        "Metadata Only",
                        "Full Tensor (CPU)",
                        "Sample Slice (CPU)",
                        "Mean/Std (CPU)",
                    ],
                    {"default": "Metadata Only"},
                ),
            },
            "optional": {
                "slice_size": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "clear_previous_hooks": ("BOOLEAN", {"default": True}),
                "enabled": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "STRING",
    )
    RETURN_NAMES = (
        "MODEL",
        "status",
    )
    FUNCTION = "setup_probe"
    CATEGORY = "Debug/WolfProbe"

    @classmethod
    def clear_all_hooks(cls):
        print(
            f"WolfProbe: Clearing {len(cls.capture_handles)} hooks and all captured data."
        )
        for handle in cls.capture_handles:
            try:
                handle.remove()
            except Exception as e:
                print(f"WolfProbe: Error removing a hook: {e}")
        cls.capture_handles.clear()
        cls.captured_data.clear()
        print("WolfProbe: All hooks and captured data cleared.")

    def setup_probe(
        self,
        model,
        target_module_name,
        hook_type,
        capture_mode,
        slice_size=10,
        clear_previous_hooks=True,
        enabled=True,
    ):
        print(
            f"WolfProbeSetup: setup_probe called. Target: '{target_module_name}', Hook: '{hook_type}', Capture: '{capture_mode}', Slice: {slice_size}, Clear: {clear_previous_hooks}, Enabled: {enabled}"
        )

        if not enabled:
            if clear_previous_hooks:
                self.clear_all_hooks()
            message = "WolfProbe: Probing disabled. No hooks registered."
            print(message)
            return (model, message)

        if clear_previous_hooks:
            self.clear_all_hooks()

        try:
            import torch
        except ImportError:
            print("WolfProbe: PyTorch not found!")
            return (model, "PyTorch not found!")

        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            message = "WolfProbe: Model structure not as expected (model.model.diffusion_model not found)."
            print(message)
            return (model, message)

        base_model_to_search = model.model.diffusion_model
        print(f"WolfProbeSetup: Base model for search: {type(base_model_to_search)}")

        module_parts = target_module_name.split(".")
        current_module = base_model_to_search
        found_module = True
        for i, part in enumerate(module_parts):
            if hasattr(current_module, part):
                current_module = getattr(current_module, part)
            else:
                print(
                    f"WolfProbeSetup: Part '{part}' (in '{target_module_name}') not found in module type '{type(current_module)}'. Path traced: {'.'.join(module_parts[:i])}"
                )
                found_module = False
                break

        if not found_module:
            message = f"WolfProbe: Module '{target_module_name}' not found."
            print(message)
            return (model, message)

        target_module = current_module
        print(
            f"WolfProbeSetup: Module '{target_module_name}' found. Type: {type(target_module)}"
        )

        active_target_module_name = target_module_name
        active_capture_mode = capture_mode
        active_slice_size = slice_size
        active_hook_type = hook_type  # For backward hook to know what to capture

        def forward_hook_fn(module, input_tensors, output_tensors):
            print(
                f"WolfProbeHook (Forward): *** HOOK FIRED for '{active_target_module_name}' ***"
            )
            capture_key = active_target_module_name

            data_to_store = {
                "module_class": module.__class__.__name__,
                "hook_type": "Forward",
            }
            print(
                f"WolfProbeHook (Forward) [{capture_key}]: Module class: {module.__class__.__name__}"
            )

            # Handle inputs
            if isinstance(input_tensors, tuple):
                data_to_store["inputs"] = [
                    process_tensor_data(t, active_capture_mode, active_slice_size)
                    for t in input_tensors
                ]
            else:  # Assuming single tensor if not tuple
                data_to_store["inputs"] = [
                    process_tensor_data(
                        input_tensors, active_capture_mode, active_slice_size
                    )
                ]
            print(f"WolfProbeHook (Forward) [{capture_key}]: Processed inputs.")

            # Handle outputs
            if isinstance(output_tensors, tuple):
                data_to_store["outputs"] = [
                    process_tensor_data(t, active_capture_mode, active_slice_size)
                    for t in output_tensors
                ]
            else:  # Assuming single tensor if not tuple
                data_to_store["outputs"] = [
                    process_tensor_data(
                        output_tensors, active_capture_mode, active_slice_size
                    )
                ]
            print(f"WolfProbeHook (Forward) [{capture_key}]: Processed outputs.")

            WolfProbeNode.captured_data[capture_key] = data_to_store
            print(
                f"WolfProbeHook (Forward) [{capture_key}]: Data stored. Current captured keys: {list(WolfProbeNode.captured_data.keys())}"
            )

        def backward_hook_fn(module, grad_input_tuple, grad_output_tuple):
            print(
                f"WolfProbeHook (Backward): *** HOOK FIRED for '{active_target_module_name}' ({active_hook_type}) ***"
            )
            capture_key = active_target_module_name

            # Initialize data_to_store for this event, possibly merging if hooks fire multiple times for same key (though current design clears)
            data_for_this_hook = {
                "module_class": module.__class__.__name__,
                "hook_type": active_hook_type,
            }
            print(
                f"WolfProbeHook (Backward) [{capture_key}]: Module class: {module.__class__.__name__}"
            )

            if active_hook_type == "Backward - Input Grads":
                # grad_input_tuple contains gradients w.r.t. inputs of the module's forward pass
                data_for_this_hook["grad_inputs"] = [
                    (
                        process_tensor_data(g, active_capture_mode, active_slice_size)
                        if g is not None
                        else None
                    )
                    for g in grad_input_tuple
                ]
                print(
                    f"WolfProbeHook (Backward) [{capture_key}]: Processed grad_inputs."
                )

            if active_hook_type == "Backward - Output Grads":
                # grad_output_tuple contains gradients w.r.t. outputs of the module's forward pass
                data_for_this_hook["grad_outputs"] = [
                    (
                        process_tensor_data(g, active_capture_mode, active_slice_size)
                        if g is not None
                        else None
                    )
                    for g in grad_output_tuple
                ]
                print(
                    f"WolfProbeHook (Backward) [{capture_key}]: Processed grad_outputs."
                )

            # Store or merge the data
            # If multiple backward hooks could target the same module with different sub-types (input/output grads),
            # we might need a more complex key or merge strategy. For now, one event overwrites or adds its specific grad type.
            if (
                capture_key not in WolfProbeNode.captured_data
                or WolfProbeNode.captured_data[capture_key].get("hook_type")
                != "Forward"
            ):
                # If no prior data or prior data isn't forward, set/overwrite
                WolfProbeNode.captured_data[capture_key] = data_for_this_hook
            else:
                # If prior data was 'Forward', merge backward data into it
                WolfProbeNode.captured_data[capture_key].update(data_for_this_hook)
                # Ensure hook_type reflects that both might be present or prioritize backward if that's the latest event type for the key
                WolfProbeNode.captured_data[capture_key][
                    "hook_type"
                ] = f"Forward then {active_hook_type}"

            print(
                f"WolfProbeHook (Backward) [{capture_key}]: Data stored/merged. Current captured keys: {list(WolfProbeNode.captured_data.keys())}"
            )
            # Note: The backward hook might be called multiple times for a single .backward() call in complex graphs.
            # The above store/merge logic is a basic attempt to handle it for one key.

        message = (
            f"WolfProbe: Setup for '{target_module_name}' - NO HOOK REGISTERED YET."
        )
        if target_module:
            try:
                if hook_type == "Forward":
                    handle = target_module.register_forward_hook(forward_hook_fn)
                    message_verb = "Forward hook registered"
                elif hook_type in ["Backward - Output Grads", "Backward - Input Grads"]:
                    # register_module_full_backward_hook is preferred for modules
                    handle = target_module.register_module_full_backward_hook(
                        backward_hook_fn
                    )
                    message_verb = f"{hook_type} (module full backward) hook registered"
                else:
                    raise ValueError(f"Unknown hook_type: {hook_type}")

                WolfProbeNode.capture_handles.append(handle)
                message = f"WolfProbe: {message_verb} on '{target_module_name}' ({target_module.__class__.__name__})."
                print(message)
                print(f"WolfProbeSetup: Handles: {WolfProbeNode.capture_handles}")
            except Exception as e:
                detailed_error = traceback.format_exc()
                message = f"WolfProbe: Error registering hook ('{hook_type}') on '{target_module_name}': {e}. Details: {detailed_error}"
                print(message)
                return (model, message)
        else:
            message = f"WolfProbe: Module '{target_module_name}' was None. No hook registered."
            print(message)

        return (model, message)


class WolfProbeGetDataNode:
    @classmethod
    def INPUT_TYPES(s):
        print("WolfProbeGetData: INPUT_TYPES called")
        return {
            "required": {
                "trigger": (
                    "LATENT",
                ),  # Changed to LATENT, user needs to connect KSampler output
                "data_key": (
                    "STRING",
                    {"default": "double_blocks.0.img_attn.proj", "multiline": False},
                ),
            },
            "optional": {
                "clear_data_after_read": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captured_data_json",)
    FUNCTION = "get_data"
    CATEGORY = "Debug/WolfProbe"
    OUTPUT_NODE = True

    def get_data(self, trigger, data_key, clear_data_after_read=False):
        print(
            f"WolfProbeGetData: get_data called. Trigger type: {type(trigger)}, Data Key: '{data_key}', Clear: {clear_data_after_read}"
        )

        try:
            import json
        except ImportError:
            print("WolfProbeGetData: json module not found!")
            return ("json module not found!",)

        print(
            f"WolfProbeGetData: Current global captured_data keys: {list(WolfProbeNode.captured_data.keys())}"
        )
        # Print full content carefully, can be huge if tensor data is stored
        # For debugging, maybe just print keys or a summary if actual tensor data is present

        data_to_log = {}
        for key, value in WolfProbeNode.captured_data.items():
            # Create a shallow copy to avoid modifying the original, and summarize tensor data
            item_summary = {}
            for k, v in value.items():
                if k in [
                    "inputs",
                    "outputs",
                    "grad_inputs",
                    "grad_outputs",
                ] and isinstance(v, list):
                    item_summary[k] = []
                    for tensor_info in v:
                        if isinstance(tensor_info, dict):
                            summary = {
                                tk: tv
                                for tk, tv in tensor_info.items()
                                if tk not in ["tensor_data", "tensor_sample"]
                            }
                            if "tensor_data" in tensor_info:
                                summary["has_full_tensor_data"] = True
                            if "tensor_sample" in tensor_info:
                                summary["has_tensor_sample"] = True
                            item_summary[k].append(summary)
                        else:
                            item_summary[k].append(tensor_info)
                else:
                    item_summary[k] = v
            data_to_log[key] = item_summary
        try:
            print(
                f"WolfProbeGetData: Summarized captured_data content: {json.dumps(data_to_log, indent=2, default=str)}"
            )
        except Exception as e:
            print(
                f"WolfProbeGetData: Error serializing summarized captured_data for logging: {e}"
            )

        retrieved_data = WolfProbeNode.captured_data.get(data_key, None)

        output_str = ""
        if retrieved_data:
            print(f"WolfProbeGetData: Data FOUND for key '{data_key}'.")
            try:
                # For the actual output, serialize the full retrieved_data
                output_str = json.dumps(retrieved_data, indent=2, default=str)
            except Exception as e:
                output_str = f"Error serializing captured data to JSON: {e}. Raw data (summary): {data_to_log.get(data_key)}"
                print(f"WolfProbeGetData: {output_str}")

            if clear_data_after_read:
                if data_key in WolfProbeNode.captured_data:
                    del WolfProbeNode.captured_data[data_key]
                    print(
                        f"WolfProbeGetData: Data for key '{data_key}' cleared after read."
                    )
        else:
            output_str = f"No data found for key: '{data_key}'. Available keys: {list(WolfProbeNode.captured_data.keys())}"
            print(f"WolfProbeGetData: {output_str}")

        return (output_str,)
