# ComfyUI Wolf Sigmas

## ⚠️ Security Warning: Remote Code Execution ⚠️

**This custom node pack includes nodes that can execute arbitrary Python code provided by the user or embedded within imported workflows/scripts. Specifically, the following nodes pose a risk if you load untrusted content:**

- **`Wolf Sigma Script Evaluator (🐺)`**
- **`Wolf Sampler Script Evaluator (🐺)`**
- **`Wolf Simple Sampler Script (🐺)`**
- **`Scriptable Empty Latent (🐺)`**
- **`Wolf Simple Sampler Script (🐺)`**

**Executing scripts from untrusted sources can lead to security vulnerabilities, including remote code execution on your machine.**

**Please exercise extreme caution:**

- **NEVER load or run workflows, scripts, or JSON files containing these nodes from sources you do not fully trust.**
- **Always review any Python code within these nodes before execution if you are unsure of its origin or purpose.**

**The authors of this node pack are not responsible for any damage or security incidents that may occur from the misuse of these script evaluation features.**

This custom node pack for ComfyUI provides a suite of tools for generating and manipulating sigma schedules for diffusion models. These nodes are particularly useful for fine-tuning the sampling process, experimenting with different step counts, and adapting schedules for specific models.

## Table of Contents

- [ComfyUI Wolf Sigmas](#comfyui-wolf-sigmas)
  - [⚠️ Security Warning: Remote Code Execution ⚠️](#️-security-warning-remote-code-execution-️)
  - [Table of Contents](#table-of-contents)
  - [Model Introspection Utilities](#model-introspection-utilities)
    - [List Model Blocks (🐺)](#list-model-blocks-)
    - [Latent Visualize (Direct) (🐺)](#latent-visualize-direct-)
    - [Visualize Activation (🐺)](#visualize-activation-)
      - [Key Features](#key-features)
      - [Inputs](#inputs)
      - [Outputs](#outputs)
      - [How it Works](#how-it-works)
  - [Experimental Model Patching](#experimental-model-patching)
    - [Modify Activations (SVD) (🐺)](#modify-activations-svd-)
  - [WolfProbe (Model Debugging)](#wolfprobe-model-debugging)
    - [Features](#features)
    - [Nodes](#nodes)
    - [How to Use](#how-to-use)
      - [Example Workflow Structure (Forward Hook)](#example-workflow-structure-forward-hook)
    - [Finding `target_module_name`](#finding-target_module_name)
    - [Debugging WolfProbe](#debugging-wolfprobe)
    - [Important Notes for Backward Hooks](#important-notes-for-backward-hooks)
    - [Known Limitations / Future Ideas](#known-limitations--future-ideas)
  - [Advanced](#advanced)
    - [Scriptable Sigma Generator](#scriptable-sigma-generator)
    - [Scriptable Sampler Generator](#scriptable-sampler-generator)
    - [Simple Scriptable Sampler](#simple-scriptable-sampler)
    - [Scriptable Latent Analyzer (🐺)](#scriptable-latent-analyzer-)
      - [⚠️ Security Warning](#️-security-warning)
      - [Scriptable Latent Analyzer Inputs](#scriptable-latent-analyzer-inputs)
      - [Scriptable Latent Analyzer Outputs](#scriptable-latent-analyzer-outputs)
      - [Scripting Environment](#scripting-environment)
      - [Default Script: Basic Latent Statistics](#default-script-basic-latent-statistics)
      - [Use Cases](#use-cases)
    - [Scriptable Noise Generator (🐺)](#scriptable-noise-generator-)
      - [⚠️ One More Security Warning](#️-one-more-security-warning)
      - [The Inputs](#the-inputs)
      - [The Outputs](#the-outputs)
      - [Scripting Environment (`generate_noise` method execution)](#scripting-environment-generate_noise-method-execution)
      - [Default Script: Basic Gaussian Noise](#default-script-basic-gaussian-noise)
      - [Potential Use Cases](#potential-use-cases)
    - [DCT Noise (🐺)](#dct-noise-)
      - [DCT Noise Inputs](#dct-noise-inputs)
      - [DCT Noise Outputs](#dct-noise-outputs)
      - [How it Works](#how-it-works-1)
      - [Potential Use Cases](#potential-use-cases-1)
    - [Scriptable Empty Latent (🐺)](#scriptable-empty-latent-)
      - [⚠️ Scriptable Security Warning](#️-scriptable-security-warning)
      - [Scriptable Empty Latent Inputs](#scriptable-empty-latent-inputs)
      - [Scriptable Empty Latent Outputs](#scriptable-empty-latent-outputs)
      - [The Scripting Environment](#the-scripting-environment)
      - [Default Script: Calibrated Structured Noise Mathematics](#default-script-calibrated-structured-noise-mathematics)
    - [Simple Scriptable Empty Latent (🐺)](#simple-scriptable-empty-latent-)
      - [⚠️ Simple Security Warning](#️-simple-security-warning)
      - [Simple Inputs](#simple-inputs)
      - [Simple Outputs](#simple-outputs)
      - [Simple Scripting Environment](#simple-scripting-environment)
      - [Default Script: Model-Aware Zero Latent](#default-script-model-aware-zero-latent)
    - [DCT Noise Latent (🐺)](#dct-noise-latent-)
      - [DCT Noise Latent Inputs](#dct-noise-latent-inputs)
      - [DCT Noise Latent Outputs](#dct-noise-latent-outputs)
      - [How it Works](#how-it-works-2)
  - [General Sigma Utilities](#general-sigma-utilities)
    - [Get Sigmas (🐺)](#get-sigmas-)
    - [Set Sigmas from JSON (🐺)](#set-sigmas-from-json-)
    - [Wolf Sigmas to JSON](#wolf-sigmas-to-json)
  - [Sigma Transformation](#sigma-transformation)
    - [Wolf Sigma Power Transform](#wolf-sigma-power-transform)
    - [Wolf Sigma Transform (Clamp T0)](#wolf-sigma-transform-clamp-t0)
    - [Wolf Sigma Transform (Shift \& Scale)](#wolf-sigma-transform-shift--scale)
    - [Wolf Sigma Transform (Normalize Range)](#wolf-sigma-transform-normalize-range)
    - [Wolf Sigma Transform (Quantize)](#wolf-sigma-transform-quantize)
    - [Wolf Sigma Transform (Respace Log-Cosine)](#wolf-sigma-transform-respace-log-cosine)
    - [Sigma Clip Values (🐺)](#sigma-clip-values-)
    - [Sigma Schedule Reverser (🐺)](#sigma-schedule-reverser-)
    - [Wolf Sigma Reverse and Rescale](#wolf-sigma-reverse-and-rescale)
    - [Sigma Slice (🐺)](#sigma-slice-)
    - [Sigma Insert Value (🐺)](#sigma-insert-value-)
    - [Sigma Add Noise (🐺)](#sigma-add-noise-)
  - [General Purpose Sigma Generators](#general-purpose-sigma-generators)
    - [Wolf Sigma Geometric Progression](#wolf-sigma-geometric-progression)
    - [Wolf Sigma Polynomial](#wolf-sigma-polynomial)
    - [Sigma Tanh Generator (🐺)](#sigma-tanh-generator-)
  - [Image Utilities](#image-utilities)
    - [Get Image Size](#get-image-size)
      - [Get Image Size Description](#get-image-size-description)
      - [Get Image Size Features](#get-image-size-features)
      - [Usage](#usage)
      - [Example Workflow](#example-workflow)

---

## Model Introspection Utilities

### List Model Blocks (🐺)

- **Class:** `ListModelBlocks`
- **Display Name:** `List Model Blocks`
- **Category:** `debug/model`
- **Description:** This utility node facilitates the inspection of a diffusion model's architecture by listing all named modules within its core structure. It iterates through the `named_modules()` of the `model.model` attribute (typically the UNet or primary diffusion model component) and returns a comprehensive list of their hierarchical names.
- **Inputs:**
  - `model`: `MODEL` - The ComfyUI model object to inspect.
- **Outputs:**
  - `block_names_string`: `STRING` - A single string where each discovered module name is listed on a new line. This format is suitable for direct display or for parsing by other tools/scripts.
- **Technical Details & Use Case:** The `ListModelBlocks` node is primarily designed as a debugging and development aid. Its main purpose is to help users identify the precise string paths for specific layers or blocks within a model. These paths are often required by other advanced nodes that perform targeted operations on model components, such as:
  - `WolfProbeSetup`: For specifying the `target_module_name` to attach forward or backward hooks.
  - `ModifyActivationsSVD`: For identifying the `target_block_name` for SVD modification.
  - Custom script nodes that need to access or modify particular layers.

The node attempts to access `model.model`. If this attribute doesn't exist or is not a `torch.nn.Module`, an error message will be returned in the output string. The output provides a direct way to visualize the hierarchy of modules, for example, `input_blocks.0.0.weight`, `middle_block.1.resnets.0.norm1`, etc., making it easier to understand the model's structure and specify correct paths for other operations.

### Latent Visualize (Direct) (🐺)

- **Class:** `LatentVisualizeDirect`
- **Display Name:** `Latent Visualize (Direct)`
- **Category:** `latent/visualize`
- **Description:** Provides a direct method to visualize the content of a latent tensor as an image. This is useful for debugging and understanding the intermediate states of latent representations in a diffusion workflow.
- **Inputs:**
  - `latent`: `LATENT` - The latent tensor dictionary (expects `{"samples": torch.Tensor}` with shape B, C, H, W).
- **Outputs:**
  - `IMAGE`: `IMAGE` - An image tensor (B, H, W, 3) normalized to the [0, 1] range, suitable for display with standard image preview nodes.
- **Technical Details & Use Case:** The node processes the input latent tensor as follows:
    1. Extracts the `samples` tensor.
    2. If the number of channels (`C`) is 0, it returns a black image.
    3. If `C` is less than 3 (e.g., 1 or 2), it takes the first channel and repeats it across three RGB channels to form a grayscale visualization.
    4. If `C` is 3 or more, it takes the first three channels for an RGB visualization.
    5. The selected image data (B, 3, H, W) is then normalized per-channel (independently for R, G, B across their respective H, W dimensions) to the [0, 1] range using min-max normalization. A small epsilon ($10^{-6}$) is added to the range denominator to prevent division by zero for flat channels.
    6. The tensor is permuted from (B, C, H, W) to (B, H, W, C) and converted to `float32` to match ComfyUI's standard IMAGE format.

This node is helpful for qualitatively assessing what features or noise patterns are present in a latent at various stages, such as after initial noise generation, after a few sampling steps, or before VAE decoding.

### Visualize Activation (🐺)

- **Class:** `VisualizeActivation`
- **Display Name:** `Visualize Activation (🐺)`
- **Category:** `debug/visualize`
- **Description:** Captures and visualizes the activation tensor from a specified target module within a UNet (or similar diffusion model component) after a single forward pass. This node is highly configurable and supports various activation shapes, visualization modes (channel grid, PCA), colormaps, and optional noise injection.

#### Key Features

- **Targeted Activation Capture:** Specify any module within the model's `diffusion_model` (e.g., UNet) using its dot-separated name (e.g., `input_blocks.1.1.transformer_blocks.0.attn1`).
- **Handles Multiple Activation Shapes:**
  - **4D Spatial Activations (B, C, H, W):** Visualizes channels or PCA components.
  - **3D Transformer Activations (B, N, C or B, C, N):** Infers spatial dimensions (H', W') from token count (N) and latent aspect ratio, reshapes to (C, H', W'), then visualizes channels or PCA components.
- **Visualization Modes:**
  - `Channel Grid`: Displays the first `num_components` channels of the activation tensor.
  - `PCA Grid`: Performs Principal Component Analysis (PCA) on the activation channels and displays the top `num_components` principal component scores as images.
- **SDXL Compatibility:** Includes logic (`prepare_sdxl_conditioning`) to correctly format conditioning for SDXL UNet forward passes, including the `y` vector from pooled embeddings.
- **Optional Noise Injection:** Can add Gaussian noise (scaled by the selected `sigmas[sigma_index]`) to the input latent before the forward pass to observe its effect on activations (`add_noise` parameter).
- **Customizable Appearance:**
  - **Colormaps:** Multiple colormap options (`viridis`, `plasma`, `gray`, `hot`, etc.) for rendering single-channel visualizations.
  - **Normalization:** Activations are normalized to [0, 1] for visualization, with an option to use `normalization_percentile` for robust scaling (clips to [-p, p] based on percentile of absolute values before mapping to [0,1]).
  - **IDs:** Optionally render channel numbers or PCA component numbers on each sub-image (`render_ids`).
  - **Grid Layout:** Configurable padding (`padding_x`, `padding_y`) and padding color (`padding_color`) for the output grid.
  - **Upscaling:** The final grid image can be upscaled using nearest-neighbor interpolation (`upscale_factor`).

#### Inputs

- **Required:**
  - `model`: `MODEL` - The input diffusion model.
  - `latent`: `LATENT` - The input latent dictionary (`{"samples": tensor}`).
  - `conditioning`: `CONDITIONING` - Conditioning data (e.g., from CLIPTextEncodeSDXL).
  - `sigmas`: `SIGMAS` - Sigma schedule tensor.
  - `target_block_name`: `COMBO` (or manual string) - Dot-separated path to the target module inside `model.model.diffusion_model`.
  - `visualization_mode`: `COMBO["Channel Grid", "PCA Grid"]` - How to visualize the activation channels.
  - `num_components`: `INT` (default: 16) - Max channels to show (Channel Grid) or PCA components to compute (PCA Grid).
  - `colormap`: `COMBO` (default: "viridis") - Colormap for single-channel visualizations.
- **Optional:**
  - `sigma_index`: `INT` (default: 0) - Index of the sigma from `sigmas` to use for the forward pass (and noise if enabled).
  - `add_noise`: `BOOLEAN` (default: False) - If True, adds noise scaled by `sigmas[sigma_index]` to the input latent.
  - `upscale_factor`: `INT` (default: 1) - Factor to upscale the final visualization grid.
  - `render_ids`: `BOOLEAN` (default: True) - Whether to draw IDs on sub-images.
  - `normalization_percentile`: `FLOAT` (default: 100.0) - Percentile for robust normalization. 100.0 uses standard min/max.
  - `padding_x`: `INT` (default: 2) - Horizontal padding between grid images.
  - `padding_y`: `INT` (default: 2) - Vertical padding between grid images.
  - `padding_color`: `COMBO["Magenta", "Black", "White"]` (default: "Black") - Color for padding between grid images.

#### Outputs

- `IMAGE`: `IMAGE` - A single image tensor (1, H_grid, W_grid, 3) representing the visualized activations arranged in a grid.

#### How it Works

1. The node takes a model, latent, conditioning, and sigmas as input.
2. It registers a forward hook on the `target_block_name` within the model's UNet (`model.model.diffusion_model`).
3. If `add_noise` is True, Gaussian noise (scaled by `sigmas[sigma_index]`) is added to a copy of the input `latent['samples']`.
4. For SDXL models, `prepare_sdxl_conditioning` is used to create the `context` and `y` inputs required by the UNet.
5. A single forward pass of the UNet is performed using the (potentially noisy) latent, the selected `sigmas[sigma_index]`, and the prepared conditioning.
6. The hook captures the output tensor of the `target_block_name`.
7. The captured activation is processed based on its dimensionality:
    - **4D (B,C,H,W):** The first batch item is taken. If `Channel Grid`, the first `num_components` channels are normalized and colormapped. If `PCA Grid`, PCA is run on the channels, and the top `num_components` scores are normalized and colormapped.
    - **3D (e.g., B,N,C for transformers):** The first batch item is taken. Spatial dimensions (H', W') are inferred for the N tokens based on the original latent's aspect ratio. The (N,C) tensor is reshaped to (C,H',W') and then processed like a 4D activation using `Channel Grid` or `PCA Grid` mode.
8. The resulting individual visualizations (normalized, colormapped, optionally with IDs) are arranged into a grid with specified padding.
9. This grid is optionally upscaled and returned as a standard ComfyUI IMAGE tensor.

This node is invaluable for debugging model behavior, understanding what features different layers are learning, and verifying the impact of noise or conditioning at specific points in the UNet architecture.

---

## Experimental Model Patching

### Modify Activations (SVD) (🐺)

- **Class:** `ModifyActivationsSVD`
- **Display Name:** `Modify Activations (SVD)`
- **Category:** `model_patches/experimental`
- **Description:** This node applies a patch to a diffusion model to modify the activations of a specified target module using Singular Value Decomposition (SVD). It operates by registering a forward hook on the target module of a *cloned* model. During the forward pass, the output tensor of the hooked module is processed by SVD to either keep only the top `n_singular_values` (low-rank approximation) or zero out the bottom `n_singular_values`.
- **Inputs:**
  - `model`: `MODEL` - The input diffusion model.
  - `target_block_name`: `STRING` (default: `"middle_block.1"`) - The string path to the target module within the model\'s UNet (e.g., `model.model.diffusion_model`). Use the `List Model Blocks` node to find appropriate names.
  - `n_singular_values`: `INT` (default: -1, min: -1, max: 8192) - The number of singular values to affect.
    - If -1, all singular values are kept (no change unless `operation_mode` is `zero_bottom` and `n_singular_values` is effectively 0 for that mode meaning all values zeroed).
    - For `keep_top`: This many singular values (and corresponding components) are retained. If 0, the output becomes a zero tensor.
    - For `zero_bottom`: The components corresponding to this many smallest singular values are zeroed out.
  - `operation_mode`: `COMBO["keep_top", "zero_bottom"]` (default: `"keep_top"`) -
    - `keep_top`: Reconstructs the activation tensor using only the `n_singular_values` largest singular values (low-rank approximation).
    - `zero_bottom`: Zeros out the `n_singular_values` smallest singular values, effectively removing high-frequency or less significant components.

- **Outputs:**
  - `MODEL`: `MODEL` - A new, cloned model instance with the SVD modification hook applied to the specified target module.

- **Technical Details & Use Case:**
  1. The node first clones the input `model` to avoid modifying the original.
  2. It attempts to locate the `target_module` within `model.model.diffusion_model` using the provided `target_block_name`.
  3. A forward hook is registered on the found `target_module`.
  4. Inside the hook, if the output is a 4D tensor (B, C, H, W), it is reshaped to (B, C, H*W). For each item in the batch, the (C, H*W) matrix is processed:
     a. Singular Value Decomposition ($U, S, V_h = \text{svd}(\text{matrix})$) is performed (matrix is cast to `float32` for SVD).
     b. Based on `operation_mode` and `n_singular_values`:
        - **`keep_top`**: The matrix is reconstructed as $U[:, :n_{eff}] \cdot \text{diag}(S[:n_{eff}]) \cdot V_h[:n_{eff}, :]$. If $n_{singular_values}$ (and thus $n_{eff}$) is 0, a zero matrix is produced. If $n_{singular_values}$ is -1 or greater than or equal to the actual number of singular values, the original matrix is used.
        - **`zero_bottom`**: A new singular value vector $S_{new}$ is created by taking the original $S$ and setting the last $n_{eff}$ values to 0.0. The matrix is then reconstructed as $U \cdot \text{diag}(S_{new}) \cdot V_h$.
     c. The modified matrix is reshaped back to (C, H, W) and converted to its original `dtype`.
  5. The batch of modified activations is stacked and returned as the new output of the hooked module.

  This technique can be used for various experimental purposes, such as:
  - Model compression research (by retaining only dominant singular values).
  - Analyzing the impact of specific components of activations.
  - Potentially influencing generation style by filtering activation frequencies.
  - Debugging or understanding information flow through specific layers.

  **Note:** The SVD operation is performed on the CPU after casting to `float32` for stability and compatibility with `torch.linalg.svd`. The modified tensor is then moved back to the original device and dtype.

---

## WolfProbe (Model Debugging)

WolfProbe is a custom node set for ComfyUI designed to help developers and researchers inspect the internal workings of diffusion models. It allows you to attach hooks to specific modules within a model and capture information about the tensors passing through them (activations) or the gradients flowing backward through them.

This can be invaluable for debugging, understanding model behavior, or extracting intermediate activations and gradients for analysis.

### Features

- **Targeted Probing**: Specify the exact module within a diffusion model you want to inspect using its string path (e.g., `double_blocks.0.img_attn.proj`).
- **Forward and Backward Hooks**:
  - Capture forward pass activations (inputs and outputs of a module).
  - Capture backward pass gradients (gradients with respect to module inputs or outputs).
- **Flexible Data Capture Modes**: Choose what to capture from tensors:
  - `Metadata Only`: Shapes, dtypes, and device (minimal overhead).
  - `Mean/Std (CPU)`: Mean and standard deviation of tensors.
  - `Sample Slice (CPU)`: A small slice of the flattened tensor.
  - `Full Tensor (CPU)`: The entire tensor data (use with caution due to memory/performance implications).
- **Flexible Workflow Integration**: Designed to be inserted into existing ComfyUI workflows.
- **Clear Data Retrieval**: Captured data can be retrieved as a JSON string for easy viewing or further processing.
- **Debug-Friendly**: Includes verbose console logging (currently indicated by "(Debug V2)" in node names) to trace the probing process.

### Nodes

This package provides two main nodes:

1. **`Wolf Probe Setup (Debug V2)`**: Configures and attaches the probe (a forward or backward hook) to the specified module in the model and defines what data to capture.
2. **`Wolf Probe Get Data (Debug V2)`**: Retrieves the data captured by the hooks after the model has been executed (and potentially after a backward pass for gradients).

### How to Use

Here's a typical workflow for using WolfProbe:

1. **Load Your Model**: Use a standard model loader to load your diffusion model.

2. **Set Up the Probe (`Wolf Probe Setup (Debug V2)`)**:
    - Connect the `MODEL` output from your loader to the `model` input of the `Wolf Probe Setup (Debug V2)` node.
    - **`target_module_name` (STRING)**: The exact path to the submodule you want to probe (e.g., `double_blocks.0.img_attn.proj`).
    - **`hook_type` (Dropdown)**: Choose the type of hook:
        - `Forward`: Captures inputs and outputs of the module during the forward pass.
        - `Backward - Output Grads`: Captures gradients with respect to the module's output(s) during a backward pass. **Requires a workflow that performs a backward pass (e.g., calculates loss and calls `loss.backward()`).**
        - `Backward - Input Grads`: Captures gradients with respect to the module's input(s) during a backward pass. **Requires a workflow that performs a backward pass.**
    - **`capture_mode` (Dropdown)**: Select what data to extract from the tensors:
        - `Metadata Only`: Captures shape, dtype, device. Recommended for general use.
        - `Mean/Std (CPU)`: Captures mean and standard deviation. Tensors are moved to CPU.
        - `Sample Slice (CPU)`: Captures the first N elements (defined by `slice_size`) of the flattened tensor. Tensors are moved to CPU.
        - `Full Tensor (CPU)`: Captures the entire tensor data as a list. **Warning: This can consume significant memory and slow down processing, especially for large tensors. Use sparingly.** Tensors are moved to CPU.
    - **`slice_size` (INT, optional, default: 10)**: Number of elements to capture if `capture_mode` is `Sample Slice (CPU)`.
    - **`clear_previous_hooks` (BOOLEAN, default: True)**: Recommended. Clears previously registered hooks by this node before setting new ones.
    - **`enabled` (BOOLEAN, default: True)**: Easily disable probing without removing the node.
    - **Output `MODEL`**: The model with the hook attached. This **must** be passed to the node that will execute the model (e.g., KSampler for forward hooks, or a graph that includes a backward pass for backward hooks).
    - **Output `status` (STRING)**: A message indicating hook registration status.

3. **Execute the Model (and potentially Backward Pass)**:
    - **For Forward Hooks**: Connect the `MODEL` output from `Wolf Probe Setup` to your `KSampler` (or similar). Configure and run the KSampler.
    - **For Backward Hooks**: This is more advanced. You need a workflow that:
        a. Uses the hooked `MODEL` in a forward pass (e.g., via KSampler or direct model call).
        b. Calculates a scalar `loss` based on the model's output.
        c. Calls `loss.backward()` (e.g., within a custom script node).
        *The hooks will fire during the `loss.backward()` execution.*

4. **Retrieve Captured Data (`Wolf Probe Get Data (Debug V2)`)**:
    - **`trigger` (LATENT, IMAGE, etc.)**: Critically important for timing.
        - For forward hooks: Connect an output from your KSampler (e.g., its `LATENT` or `IMAGE` output) to this `trigger` input.
        - For backward hooks: Connect a signal that executes *after* your `loss.backward()` call has completed.
    - **`data_key` (STRING)**: Must **exactly match** the `target_module_name` from `Wolf Probe Setup`.
    - **`clear_data_after_read` (BOOLEAN, default: False)**: If True, removes data for this key after retrieval.
    - **Output `captured_data_json` (STRING)**: JSON string with captured data. For backward hooks, this will include `grad_inputs` or `grad_outputs` fields.

#### Example Workflow Structure (Forward Hook)

```plaintext
[Model Loader] ----MODEL----> [Wolf Probe Setup]
                                     |
                                     |----MODEL (hooked)----> [KSampler]
                                                                 |
                                                                 |----LATENT/IMAGE----> [Wolf Probe Get Data]
                                                                                           (trigger input)
```

### Finding `target_module_name`

This is often the trickiest part.

- **Standard UNets**: Module names often follow patterns like `input_blocks.X.Y...`, `middle_block...`, `output_blocks.X.Y...`. You might need to inspect the model's Python code (e.g., in `comfy/ldm/modules/diffusionmodules/openaimodel.py` for standard UNets) or use debugging techniques to print module names.
- **FLUX/Chroma Models (via `ComfyUI-FluxMod`)**: As explored, the path starts from `model.model.diffusion_model` (which is a `FluxMod` or `Chroma` instance). Examples include:
  - `img_in`
  - `txt_in`
  - `double_blocks.0.img_attn.qkv`
  - `single_blocks.3.linear1`
  - `final_layer.linear`
    You can find these names by inspecting the `model.py` and `layers.py` files within the `ComfyUI-FluxMod/flux_mod/` directory.
- **General Technique**: You can temporarily add `print(name, type(module))` in the `named_modules()` loop of a model's `__init__` or by iterating over `model.model.diffusion_model.named_modules()` in a debug script.

### Debugging WolfProbe

- **Console Output**: The "(Debug V2)" nodes print extensively. Check for:
  - Setup messages: `WolfProbeSetup: setup_probe called... Hook: '<type>', Capture: '<mode>', ...`
  - Hook registration: `WolfProbe: <Forward/Backward> hook registered...`
  - **Hook Firing**:
    - `WolfProbeHook (Forward): *** HOOK FIRED ... ***` (during KSampler/forward pass)
    - `WolfProbeHook (Backward): *** HOOK FIRED ... ***` (during `loss.backward()`)
  - Data storage messages from within the hooks.
  - `WolfProbeGetData` messages detailing keys and retrieved data.
- **Status Output**: From `WolfProbeSetup`.
- **Correct `data_key` and `trigger` Connection**.

### Important Notes for Backward Hooks

- **Requires `loss.backward()`**: Gradients are only computed if a backward pass is explicitly performed in your workflow. Standard KSampler workflows do not do this.
- **Workflow Complexity**: You will need to create a graph that includes loss calculation and the backward call, likely using custom script nodes for PyTorch operations.
- **Trigger Timing**: Ensure the `trigger` for `WolfProbeGetData` fires *after* the `loss.backward()` call.

### Known Limitations / Future Ideas

- The merging logic for when both forward and backward hooks populate the same `data_key` is basic. More sophisticated handling might be needed for complex scenarios.
- `Full Tensor (CPU)` capture mode can be very resource-intensive.
- Could add more sophisticated tensor summarization options (e.g., histograms, specific percentile values).
- User interface for selecting modules directly from the model graph would be a major enhancement but is complex to implement.

---

## Advanced

### Scriptable Sigma Generator

For advanced users who need maximum flexibility, the `Wolf Sigma Script Evaluator (🐺)` node allows for the creation of custom sigma schedules by executing a user-provided Python script. This provides a powerful way to define complex or experimental schedules.

For detailed information on how to use the script evaluator, its inputs, outputs, scripting capabilities, and example scripts, please see the [Wolf Sigma Script Evaluator Documentation](./SIGMA_EVALUATOR.md).

### Scriptable Sampler Generator

Similarly, for users wishing to implement entirely custom sampling algorithms, the `Wolf Sampler Script Evaluator (🐺)` node enables the definition of new samplers through Python scripts. This node integrates with ComfyUI's KSampler system, allowing custom samplers to be used like any built-in sampler.

For detailed information on how to use the sampler script evaluator, its scripting requirements, parameters, and example sampler implementations (like Euler and Euler Ancestral), please see the [Wolf Sampler Script Evaluator Documentation](./SAMPLER_EVALUATOR.md).

### Simple Scriptable Sampler

The `Wolf Simple Sampler Script (🐺)` node offers a streamlined way to define custom sampling logic by executing a user-provided Python script for the core denoising loop. This is simpler than implementing a full KSampler-compatible class and is well-suited for direct experimentation with sampling algorithms. The provided script is given access to necessary variables like the model, initial latents, and the sigma schedule, and is expected to return the final denoised latents.

- **Class:** `WolfSimpleSamplerScriptEvaluator`
- **Display Name:** `Wolf Simple Sampler Script (🐺)`
- **Category:** `sampling/ksampler_wolf`
- **Description:** Executes a user-provided Python script that defines the core sampling loop. The script receives variables like the model, initial latents (`x_initial`), and the sigma schedule (`sigmas_schedule`), and must assign the final denoised latents to a variable named `latents`. A default script implementing a basic Euler sampler is provided as a template.
- **Inputs:**
  - `script`: `STRING` (multiline, default: Python script for basic Euler sampling) - The Python script implementing the sampling loop.
- **Outputs:**
  - `SAMPLER`: `SAMPLER` - A ComfyUI compatible sampler object that can be directly used with KSampler nodes.

For detailed information on the scripting environment, available variables, requirements, and example scripts, please refer to the [Wolf Simple Sampler Script Evaluator Documentation](./SIMPLE_SAMPLER_EVALUATOR.md).

### Scriptable Latent Analyzer (🐺)

This node provides a flexible way to inspect, analyze, and optionally modify latent tensors at any point in a workflow using a user-provided Python script. It's a powerful tool for debugging latent states, calculating custom statistics, or experimentally altering latents.

- **Class:** `WolfScriptableLatentAnalyzer`
- **Display Name:** `Wolf Scriptable Latent Analyzer (🐺)`
- **Category:** `latent/analysis_wolf`
- **Description:** Executes a user-provided Python script to analyze an input latent tensor. The script can calculate statistics, check for issues, or even modify the latent. The node outputs the (potentially modified) latent and a string log from the script.

#### ⚠️ Security Warning

This node executes arbitrary Python code. **Only use scripts from trusted sources.** Review any script before execution if you are unsure of its origin.

#### Scriptable Latent Analyzer Inputs

- `latent`: `LATENT` - The input latent dictionary (expects `{"samples": torch.Tensor, ...}`).
- `script`: `STRING` (multiline, default: Python script for basic latent stats) - The Python script to execute for latent analysis.
- `model` (optional): `MODEL` - An optional model context that can be passed to the script.
- `sigmas` (optional): `SIGMAS` - Optional sigmas (tensor) that can be passed to the script for context (e.g., comparing latent std to current sigma).

#### Scriptable Latent Analyzer Outputs

- `LATENT`: `LATENT` - The (potentially modified) latent dictionary, with the processed tensor in `latent["samples"]`.
- `analysis_log`: `STRING` - A string output from the script, typically used for logging statistics or messages.

#### Scripting Environment

The Python script executed by this node has access to the following:

**Local Variables:**

- `input_latent_tensor`: `torch.Tensor` - The samples tensor extracted from the input `latent` dictionary.
- `model`: The `MODEL` object, if connected.
- `sigmas`: The `SIGMAS` tensor, if connected.
- `torch`: The PyTorch module.
- `math`: The Python math module.
- `print_fn(message)`: A function to print messages to the ComfyUI console, prefixed with `[WolfScriptableLatentAnalyzer Script]:`.

**Script Requirements & Behavior:**

- The script **must** assign a string value to a variable named `analysis_output_string`. This string will be output by the node.
- The script **may** assign a `torch.Tensor` to a variable named `output_latent_tensor`.
  - If assigned, this tensor must have the same shape as `input_latent_tensor` and will become the new `latent["samples"]` output. The node ensures it's moved to the original device of the input latent.
  - If `output_latent_tensor` is not assigned (or assigned `None`), the original `input_latent_tensor` is passed through unchanged.

#### Default Script: Basic Latent Statistics

The default script calculates and formats basic statistics of the `input_latent_tensor` into the `analysis_output_string`:

- Shape
- Dtype
- Device
- Mean
- Standard Deviation (Std)
- Minimum Value (Min)
- Maximum Value (Max)

It also demonstrates how to pass through the original latent by assigning `output_latent_tensor = input_latent_tensor`.

#### Use Cases

- **Debugging:** Insert at various points in a workflow to check the numerical properties (mean, std, min, max, presence of NaNs/Infs) of latents.
- **Analysis:** Calculate custom metrics or statistics from latent tensors.
- **Conditional Modification:** Implement logic to alter latents based on their properties or external inputs (e.g., clamping values, adding conditional noise).
- **Understanding Sampler Behavior:** Observe how latent statistics change step-by-step through a sampling process by placing this node after each step (if the sampler allows intermediate output).

### Scriptable Noise Generator (🐺)

This node allows for the creation of custom noise objects that can be used by ComfyUI samplers (e.g., KSampler). Instead of relying on standard Gaussian noise, users can provide a Python script that defines how the noise tensor is generated when the sampler requests it. This enables advanced noise manipulation, such as structured noise, patterned noise, or noise with specific statistical properties tailored to the sampling process.

- **Class:** `WolfScriptableNoise`
- **Display Name:** `Scriptable Noise (🐺)`
- **Category:** `sampling/custom_sampling/noise`
- **Description:** Returns a custom NOISE object. When a sampler calls the `generate_noise(latent_image_dict)` method of this object, the provided Python script is executed to produce a noise tensor matching the shape of `latent_image_dict["samples"]`.

#### ⚠️ One More Security Warning

This node executes arbitrary Python code. **Only use scripts from trusted sources.** Review any script before execution if you are unsure of its origin.

#### The Inputs

- `seed`: `INT` (default: 0, `control_after_generate`: True) - Seed for random number generation within the script.
- `sigmas`: `SIGMAS` - The sigma schedule tensor. This is passed to the script and can be used for context, e.g., scaling noise according to `sigmas[0]`.
- `device_selection`: `COMBO["AUTO", "CPU", "GPU"]` (default: "AUTO") - Determines the target device for script execution and tensor creation. "AUTO" attempts to use the model's device or GPU.
- `script`: `STRING` (multiline, default: Python script for basic Gaussian noise) - The Python script to execute for noise generation.
- `model` (optional): `MODEL` - An optional model context passed to the script. Can influence `AUTO` device selection.

#### The Outputs

- `NOISE`: `NOISE` - A custom noise object compatible with ComfyUI samplers. This object has a `generate_noise(latent_image_dict)` method that, when called, executes the user's script.

#### Scripting Environment (`generate_noise` method execution)

The Python script provided to the `script` input is executed when the `generate_noise` method of the output `NOISE` object is called by a sampler. The script has access to:

**Local Variables:**

- `latent_image`: `dict` - The dictionary passed by the sampler, typically containing:
  - `latent_image["samples"]`: `torch.Tensor` - The latent tensor for which noise is needed. The generated noise must match its shape, dtype, and layout.
  - `latent_image["batch_index"]` (optional): `torch.Tensor` - Batch indices, if provided by the sampler.
- `input_samples`: `torch.Tensor` - A direct alias to `latent_image["samples"]`.
- `seed`: `INT` - The seed value from the node's input.
- `model`: The `MODEL` object, if connected to the node.
- `device`: `str` - The target device string (e.g., "cuda:0", "cpu") for tensor creation, determined by `device_selection`.
- `sigmas`: `torch.Tensor` - The sigmas tensor from the node's input, moved to the execution `device`.
- `output_noise_tensor`: Initially `None`. The script **must** assign the final generated noise tensor (matching shape, dtype, layout of `input_samples`, and on the correct `device`) to this variable.

**Global Helper Modules & Functions:**

- `torch`: The PyTorch module.
- `math`: The Python math module.
- `F`: `torch.nn.functional` (e.g., for Perlin noise if implemented in script).
- `rand_perlin_2d_octaves_fn`, `rand_perlin_2d_fn`, `_fade_fn`: Helper functions for Perlin noise generation (copied from `wolf_scriptable_empty_latent.py` and made available in the script's global scope if you wish to use them).

#### Default Script: Basic Gaussian Noise

The default script demonstrates simple Gaussian noise generation:

```python
# Default script for WolfScriptableNoise
# Available: latent_image, input_samples, seed, model, device, sigmas, torch, math, F, 
#            rand_perlin_2d_octaves_fn, rand_perlin_2d_fn, _fade_fn
# Must assign to output_noise_tensor

# Ensure a torch.Generator object is created on the correct device for reproducibility
# Create a new generator for each call if seed needs to be re-applied per call, 
# or manage a single generator if that's intended for sequential calls from a sampler.
generator = torch.Generator(device=device)
generator.manual_seed(seed)

noise = torch.randn_like(input_samples, generator=generator, device=device)

# Example: Scale noise by the first sigma value (common practice)
# if sigmas is not None and sigmas.numel() > 0:
#     noise = noise * sigmas[0]

output_noise_tensor = noise
```

This script generates standard Gaussian noise $\mathcal{N}(0, 1)$ matching the shape of the input latent samples. An example of how to scale this noise by `sigmas[0]` is commented out.

#### Potential Use Cases

- **Custom Noise Distributions:** Implement noise from uniform, Laplace, or other distributions.
- **Structured Noise:** Generate Perlin noise, simplex noise, or other procedural patterns to guide the initial stages of diffusion.
- **Seedable & Deterministic Noise:** Full control over the random number generator for noise creation.
- **Noise Debugging:** Insert print statements or calculations within the noise generation script to check its statistical properties (e.g., mean, std) before it's used by the sampler.
- **Conditional Noise:** Generate different types of noise based on inputs like `sigmas`, `model` properties, or even `batch_index`.

### DCT Noise (🐺)

- **Class:** `WolfDCTNoise`
- **Display Name:** `DCT Noise (🐺)`
- **Category:** `sampling/custom_sampling/noise`
- **Description:** Returns a custom `NOISE` object that, when invoked by a sampler, generates DCT (Discrete Cosine Transform)-based noise. This noise is characterized by JPEG-like compression artifacts and can be controlled by various parameters. The generated noise is automatically scaled by the current sigma value (typically `sigmas[0]`) provided by the sampler.

#### DCT Noise Inputs

- **Required:**
  - `seed`: `INT` (default: 0, `control_after_generate`: True) - Seed for the random number generator used in DCT noise creation.
  - `sigmas`: `SIGMAS` - The sigma schedule. The first sigma (`sigmas[0]`) from this schedule (or the one passed by the sampler) is typically used to scale the generated noise.
  - `device_selection`: `COMBO["AUTO", "CPU", "GPU"]` (default: "AUTO") - Target device for the noise tensor. "AUTO" attempts to use the model's device or GPU.
  - `dc_map_base_min`: `FLOAT` (default: -800.0) - Minimum value for the initial DC coefficient map.
  - `dc_map_base_max`: `FLOAT` (default: 800.0) - Maximum value for the initial DC coefficient map.
  - `dc_map_smooth_sigma`: `FLOAT` (default: 2.0) - Sigma for Gaussian smoothing of the DC coefficient map.
  - `ac_coeff_laplacian_scale`: `FLOAT` (default: 30.0) - Scale for the Laplacian distribution of pre-quantized AC coefficients.
  - `q_table_multiplier`: `FLOAT` (default: 1.0) - Multiplier for the default JPEG quantization table.
  - `normalization`: `COMBO["None", "Mean0Std1_channel", "Mean0Std1_tensor", "ScaleToStd1_channel", "ScaleToStd1_tensor"]` (default: "Mean0Std1_channel") - Normalization method for the raw DCT noise before scaling by sigma.
- **Optional:**
  - `model`: `MODEL` - Optional model context, can influence `AUTO` device selection.

#### DCT Noise Outputs

- `NOISE`: `NOISE` - A custom noise object compatible with ComfyUI samplers. This object's `generate_noise(latent_image_dict)` method produces the DCT noise.

#### How it Works

The `WolfDCTNoise` node creates an `ExecutableDCTNoise` object that stores the configuration. When a sampler requests noise (by calling `generate_noise(latent_image_dict)` on this object):

1. The shape of the required noise is taken from `latent_image_dict["samples"]`.
2. The target device is determined based on the node's `device_selection` and `model` input.
3. DCT noise is generated for each channel and batch item using the same core logic as `WolfDCTNoiseScriptableLatent` (see its "How it Works" section for details on DC/AC coefficient generation, quantization, and IDCT). The parameters `dc_map_base_min/max`, `dc_map_smooth_sigma`, `ac_coeff_laplacian_scale`, and `q_table_multiplier` control this process.
4. The raw generated noise (as a NumPy array) is converted to a PyTorch tensor.
5. The specified `normalization` method is applied.
6. The noise tensor is then scaled by the appropriate sigma value, typically `latent_image_dict['sigmas'][0]`. If `latent_image_dict['sigmas']` is not available, it falls back to `sigmas[0]` from the node's input.
7. The final, scaled noise tensor is moved to the target device and returned to the sampler.

#### Potential Use Cases

- **Textured Initial Noise:** Introduce initial noise with JPEG-like artifacts for stylized image generation.
- **Alternative to Gaussian Noise:** Experiment with different noise characteristics in the sampling process.
- **Controllable Artifacts:** Fine-tune the appearance of compression-like patterns in the noise.

### Scriptable Empty Latent (🐺)

The `WolfScriptableEmptyLatent` node provides a highly flexible way to generate initial latent noise for the diffusion process. It allows users to define the generation logic using a Python script, enabling complex noise patterns, structured noise, or any custom initialization beyond simple Gaussian noise or zeros. The default script implements a sophisticated "Calibrated Structured Noise" combining Perlin and Gaussian noise, scaled by the initial sigma of a provided schedule and the VAE's scaling factor.

- **Class:** `WolfScriptableEmptyLatent`
- **Display Name:** `Scriptable Empty Latent (🐺)`
- **Category:** `latent/noise` (or your preferred category like `Wolf Custom Nodes/Latent`)
- **Description:** Executes a user-provided Python script to generate an initial latent tensor. The script has access to various parameters like dimensions, seed, the VAE model, a sigma schedule, and Perlin noise configuration. It must produce a latent tensor of the correct NCHW shape.

#### ⚠️ Scriptable Security Warning

This node executes arbitrary Python code. **Only use scripts from trusted sources.** Review any script before execution if you are unsure of its origin.

#### Scriptable Empty Latent Inputs

- `model`: `MODEL` - The main diffusion model, used primarily to access VAE properties like `latent_format.scale_factor`.
- `width`: `INT` (default: 1024, min: 64, step: 8) - Target width of the image (latent width will be `width // 8`).
- `height`: `INT` (default: 1024, min: 64, step: 8) - Target height of the image (latent height will be `height // 8`).
- `batch_size`: `INT` (default: 1, min: 1, max: 64) - Number of latent images to generate in the batch.
- `seed`: `INT` (default: 0, `control_after_generate`: True) - Seed for random number generation within the script.
- `sigmas`: `SIGMAS` - An input sigma schedule (tensor). The default script uses the first sigma (`sigmas[0]`) for calibration.
- `perlin_blend_factor`: `FLOAT` (default: 0.5, min: 0.0, max: 1.0) - Blend factor between Gaussian and Perlin noise in the default script. 0.0 for pure Gaussian, 1.0 for pure Perlin.
- `perlin_res_factor_h`: `INT` (default: 8, min: 1, max: 128) - Resolution factor for Perlin noise height. Higher values mean lower frequency / larger features. Latent height is divided by this.
- `perlin_res_factor_w`: `INT` (default: 8, min: 1, max: 128) - Resolution factor for Perlin noise width.
- `perlin_frequency_factor`: `FLOAT` (default: 2.0, min: 1.01, max: 4.0) - Frequency multiplier for each Perlin octave.
- `perlin_octaves`: `INT` (default: 4, min: 1, max: 10) - Number of Perlin noise octaves to sum.
- `perlin_persistence`: `FLOAT` (default: 0.5, min: 0.01, max: 1.0) - Amplitude persistence factor between Perlin octaves.
- `perlin_contrast_scale`: `FLOAT` (default: 1.0, min: 0.1, max: 20.0, step: 0.1, tooltip: "Scales Perlin noise before normalization to enhance its features.") - Multiplies the raw Perlin noise before it's normalized, to amplify its structural features.
- `script`: `STRING` (multiline, default: Python script for calibrated structured noise) - The Python script to execute for latent generation.

#### Scriptable Empty Latent Outputs

- `LATENT`: `LATENT` - A dictionary containing the generated latent tensor under the key `"samples"`.

#### The Scripting Environment

The Python script executed by this node has access to the following:

**Local Variables:**

- `width`, `height`, `batch_size`, `seed`: Integers as provided to the node.
- `sigmas`: A `torch.Tensor` of sigma values, moved to the execution `device`.
- `model`: The `MODEL` object passed to the node.
- `device`: A string representing the torch device for execution (e.g., `"cuda:0"`, `"cpu"`).
- `perlin_params`: A dictionary containing all `perlin_*` input values:
  `{'blend_factor', 'res_factor_h', 'res_factor_w', 'frequency_factor', 'octaves', 'persistence', 'perlin_contrast_scale'}`.
- `output_latent_samples`: Initially `None`. The script **must** assign the final generated latent tensor (NCHW float32 tensor on `device`) to this variable.

**Global Helper Modules & Functions:**

- `torch`: The PyTorch module.
- `math`: The Python math module.
- `F`: `torch.nn.functional`.
- `rand_perlin_2d_octaves_fn(shape, res, octaves, persistence, frequency_factor, device)`: Function to generate multi-octave 2D Perlin noise. `device` argument should be the string from `script_locals`.
- `rand_perlin_2d_fn(shape, res, fade_func, device)`: Function to generate single-octave 2D Perlin noise.
- `_fade_fn(t)`: The quintic fade function $6t^5 - 15t^4 + 10t^3$ used in Perlin noise.

#### Default Script: Calibrated Structured Noise Mathematics

The default script generates initial noise by blending Perlin noise with Gaussian noise and then calibrates its magnitude using the provided `sigmas` and VAE properties.

1. **Parameter Setup:**
    - Latent dimensions: $H_L = \text{height} // 8$, $W_L = \text{width} // 8$. (Note: For FLUX models, the script uses $H_L = \text{height} // 8, W_L = \text{width} // 8$ as well, but with 16 channels).
    - Target sigma $\sigma_{target}$ is taken as the first value from the input `sigmas` tensor.
    - VAE scaling factor $s_{VAE}$ is determined based on the model type (e.g., 0.18215 for typical SDXL, 1.0 for FLUX). This value is logged for informational purposes.
    - The **effective sigma** for scaling the final $\mathcal{N}(0, 1)$ noise is:
        $$ \sigma_{eff} = \sigma_{target} $$
    This means the standard deviation of the final output noise tensor is intended to match the `target_sigma` from the input schedule.

2. **Gaussian Noise Generation ($G$):**
    - Standard Gaussian noise is generated for each element in the latent tensor $(B, C, H_L, W_L)$, where $B$ is batch size and $C=4$ channels.
    - $G_{b,c,h,w} \sim \mathcal{N}(0, 1)$

3. **Perlin Noise Generation ($P$):**
    - Generated independently for each batch item and channel, using `rand_perlin_2d_octaves_fn`.
    - **Base Perlin Noise (`rand_perlin_2d_fn`):**
        - A grid of random 2D gradient vectors is created. For a point $(x,y)$ in the latent space, its position within a grid cell $(u,v)$ (where $u,v \in [0,1]$) is determined.
        - Dot products are computed between the gradient vectors at the cell's four corners ($g_{00}, g_{10}, g_{01}, g_{11}$) and the displacement vectors from the point to these corners.
            - $dp_{00} = g_{00} \cdot (u, v)$
            - $dp_{10} = g_{10} \cdot (u-1, v)$
            - $dp_{01} = g_{01} \cdot (u, v-1)$
            - $dp_{11} = g_{11} \cdot (u-1, v-1)$
        - These dot products are interpolated using a **fade function** $\text{fade}(t) = 6t^5 - 15t^4 + 10t^3$ to ensure smooth transitions. Let $t_u = \text{fade}(u)$ and $t_v = \text{fade}(v)$.
        - Lerp (Linear Interpolation): $\text{lerp}(a, b, w) = a + w(b-a)$.
            - $v_0 = \text{lerp}(dp_{00}, dp_{10}, t_u)$
            - $v_1 = \text{lerp}(dp_{01}, dp_{11}, t_u)$
        - The final Perlin value for that point is $\text{lerp}(v_0, v_1, t_v)$, typically scaled by $\sqrt{2}$.
    - **Octaves:** Multiple layers (octaves) of Perlin noise are summed. For each octave $i$:
        - Frequency $f_i = \text{frequency\_factor}^i$. The resolution `res` for `rand_perlin_2d_fn` is scaled by $f_i$.
        - Amplitude $A_i = \text{persistence}^i$.
        - The total Perlin noise is $P_{raw} = \sum_{i=0}^{\text{octaves}-1} A_i \times \text{Perlin}_i(\text{shape}, \text{res} \times f_i)$.
    - The base resolution for Perlin noise is determined by `perlin_res_factor_h` and `perlin_res_factor_w`:
        - $\text{res}_h = \max(1, H_L // \text{perlin\_res\_factor\_h})$
        - $\text{res}_w = \max(1, W_L // \text{perlin\_res\_factor\_w})$

4. **Perlin Contrast Scaling:**
    - Let $c_{scale}$ be the `perlin_contrast_scale` input parameter.
    - The raw Perlin noise $P_{raw}$ (generated in step 3) is multiplied by $c_{scale}$ to enhance its features before normalization:
        $$ P_{scaled} = P_{raw} \times c_{scale} $$

5. **Normalization:**
    - Both the Gaussian noise $G$ and the contrast-scaled Perlin noise $P_{scaled}$ are independently normalized to have zero mean and unit standard deviation across their spatial dimensions (height and width) for each channel and batch item. For any noise tensor $X$ (here $X$ would be $G$ or $P_{scaled}$):
        $$ X_{norm} = \frac{X - \mu_X}{\sigma_X + \epsilon} $$
        where $\mu_X$ is the mean of $X$, $\sigma_X$ is its standard deviation, and $\epsilon$ is a small constant (e.g., $10^{-5}$) to prevent division by zero. So, we get $G_{norm}$ from $G$, and $P_{norm}$ from $P_{scaled}$.

6. **Blending:**
    - The normalized Gaussian noise $G_{norm}$ and normalized (contrast-enhanced) Perlin noise $P_{norm}$ are blended using the `perlin_blend_factor` ($\alpha_{blend}$):
        $$ N_{blend} = (1 - \alpha_{blend}) \cdot G_{norm} + \alpha_{blend} \cdot P_{norm} $$
        This is equivalent to `torch.lerp(G_norm, P_norm, perlin_blend_factor)`.

7. **Re-Normalization of Blended Noise:**
    - The blended noise $N_{blend}$ is normalized again to ensure it has zero mean and unit standard deviation:
        $$ N_{blend, norm} = \frac{N_{blend} - \mu_{N_{blend}}}{\sigma_{N_{blend}} + \epsilon} $$

8. **Final Scaling:**
    - The re-normalized blended noise $N_{blend, norm}$ (which is $\mathcal{N}(0, 1)$) is scaled by the `effective_sigma` (which is $\sigma_{target}$) calculated in step 1:
        $$ \text{Output Latent} = N_{blend, norm} \times \sigma_{eff} $$
    This ensures the initial latent noise has a standard deviation equal to $\sigma_{target}$, which is typically expected by samplers using schedules like Karras.

### Simple Scriptable Empty Latent (🐺)

The `WolfSimpleScriptableEmptyLatent` node provides a straightforward way to generate initial latent noise using a Python script. It is simpler than the `Scriptable Empty Latent (🐺)` node and is well-suited for basic latent generation or when custom logic is needed for channel count or initial values, especially with models like FLUX. The default script generates a zero-filled latent tensor, automatically adjusting the number of channels based on the connected model.

- **Class:** `WolfSimpleScriptableEmptyLatent`
- **Display Name:** `Simple Scriptable Empty Latent (🐺)`
- **Category:** `latent/noise`
- **Description:** Executes a user-provided Python script to generate an initial latent tensor. The script has access to parameters like dimensions and an optional model connection. It must produce a latent tensor of the correct NCHW shape.

#### ⚠️ Simple Security Warning

This node executes arbitrary Python code. **Only use scripts from trusted sources.** Review any script before execution if you are unsure of its origin.

#### Simple Inputs

- `width`: `INT` (default: 1024, min: 64, step: 8) - Target width of the image (latent width will be `width // 8`).
- `height`: `INT` (default: 1024, min: 64, step: 8) - Target height of the image (latent height will be `height // 8`).
- `batch_size`: `INT` (default: 1, min: 1, max: 64) - Number of latent images to generate in the batch.
- `device_selection`: `COMBO["AUTO", "CPU", "GPU"]` (default: "AUTO") - Specifies the target device for the script execution and the final latent tensor. "AUTO" attempts to use the model\'s device or falls back to an intermediate device.
- `script`: `STRING` (multiline, default: Python script for model-aware zero latent) - The Python script to execute for latent generation.
- `model`: `MODEL` (optional) - An optional model input. The default script uses this to determine if the model is FLUX-based to set the latent channel count to 16 (otherwise 4).

#### Simple Outputs

- `LATENT`: `LATENT` - A dictionary containing the generated latent tensor under the key `"samples"`.

#### Simple Scripting Environment

The Python script executed by this node has access to the following:

**Local Variables:**

- `width`, `height`, `batch_size`: Integers as provided to the node.
- `model`: The `MODEL` object passed to the node (can be `None`).
- `device`: A string representing the torch device for execution (e.g., `"cuda:0"`, `"cpu"`), determined by the `device_selection` input and available hardware/model.
- `output_latent_samples`: Initially `None`. The script **must** assign the final generated latent tensor (NCHW float32 tensor on `device`) to this variable.

**Global Helper Modules & Functions:**

- `torch`: The PyTorch module.
- `math`: The Python math module.

#### Default Script: Model-Aware Zero Latent

The default script generates a zero-filled latent tensor with dimensions appropriate for the specified `width`, `height`, and `batch_size`.

1. **Channel Determination:**
    - It inspects the connected `model` (if any).
    - If the model is identified as a FLUX model (by checking `model.model_type` or class name), `num_latent_channels` is set to 16.
    - Otherwise, `num_latent_channels` defaults to 4 (standard for SD, SDXL, etc.).
    - If no model is connected, it defaults to 4 channels.
    - Prints the detected model type and chosen channel count.

2. **Latent Dimensions:**
    - Latent height: $H_L = \text{height} // 8$.
    - Latent width: $W_L = \text{width} // 8$.

3. **Tensor Generation:**
    - The target shape is $(B, C, H_L, W_L)$, where $B$ is `batch_size` and $C$ is the determined `num_latent_channels`.
    - A `torch.zeros` tensor of this `shape` is created with `dtype=torch.float32` on the specified `device`.
    - This tensor is assigned to `output_latent_samples`.
    - Prints the generation parameters and target shape.

This provides a basic, flexible starting point for latent generation, particularly useful for ensuring compatibility with models requiring different latent channel depths.

### DCT Noise Latent (🐺)

- **Class:** `WolfDCTNoiseScriptableLatent`
- **Display Name:** `DCT Noise Latent (🐺)`
- **Category:** `latent/noise`
- **Description:** Generates an initial latent tensor using DCT (Discrete Cosine Transform)-based noise synthesis. This method aims to produce noise with characteristics similar to JPEG compression artifacts. The properties of the generated noise, such as the strength and smoothness of block-like patterns, can be controlled through various input parameters. This node is useful for initializing the diffusion process with a specific kind of textured noise rather than standard Gaussian noise or zeros.

#### DCT Noise Latent Inputs

- **Required:**
  - `width`: `INT` (default: 1024, min: 64, max: MAX_RESOLUTION, step: 8) - Target width of the image (latent width will be `width // 8`).
  - `height`: `INT` (default: 1024, min: 64, max: MAX_RESOLUTION, step: 8) - Target height of the image (latent height will be `height // 8`).
  - `batch_size`: `INT` (default: 1, min: 1, max: 64) - Number of latent images to generate in the batch.
  - `device_selection`: `COMBO["AUTO", "CPU", "GPU"]` (default: "AUTO") - Specifies the target device for the latent tensor.
  - `seed`: `INT` (default: 0) - Seed for the random number generator used in noise creation.
  - `dc_map_base_min`: `FLOAT` (default: -800.0) - Minimum value for the initial (pre-smoothing) DC coefficient map.
  - `dc_map_base_max`: `FLOAT` (default: 800.0) - Maximum value for the initial DC coefficient map.
  - `dc_map_smooth_sigma`: `FLOAT` (default: 2.0) - Sigma for Gaussian smoothing of the DC coefficient map. Higher values create smoother transitions between DC blocks.
  - `ac_coeff_laplacian_scale`: `FLOAT` (default: 30.0) - Scale parameter for the Laplacian distribution from which pre-quantized AC (Alternating Current) coefficients are drawn.
  - `q_table_multiplier`: `FLOAT` (default: 1.0) - Multiplier for the default JPEG quantization table. Values > 1.0 increase quantization (stronger artifacts), < 1.0 decrease it.
  - `normalization`: `COMBO["None", "Mean0Std1_channel", "Mean0Std1_tensor", "ScaleToStd1_channel", "ScaleToStd1_tensor"]` (default: "Mean0Std1_channel") - Method to normalize the generated raw DCT noise before outputting it as a latent.
- **Optional:**
  - `model`: `MODEL` - Optional model input. Used by `device_selection="AUTO"` and to determine the number of latent channels (e.g., 4 for SDXL, 16 for FLUX).

#### DCT Noise Latent Outputs

- `LATENT`: `LATENT` - A dictionary containing the generated DCT noise latent tensor under the key `"samples"`. The tensor will have the shape (batch_size, num_latent_channels, height // 8, width // 8).

#### How it Works

The `WolfDCTNoiseScriptableLatent` node directly generates a latent tensor using an internal implementation of DCT-based noise synthesis. Unlike the "Scriptable" latent nodes, it does not execute a user-provided Python script for this core generation.

1. It determines the target device and number of latent channels (4 for standard models, 16 for FLUX, based on the optional `model` input).
2. It initializes a NumPy random number generator with the provided `seed`.
3. For each channel in each batch item:
    a. An initial map of DC (Direct Current / average) coefficients for 8x8 blocks is created using random values within `dc_map_base_min` and `dc_map_base_max`.
    b. This DC map is smoothed using a Gaussian filter with `dc_map_smooth_sigma` to create spatial correlation.
    c. For each 8x8 block, AC (Alternating Current / detail) coefficients are generated from a Laplacian distribution scaled by `ac_coeff_laplacian_scale`.
    d. Both DC and AC coefficients are quantized using a standard JPEG quantization table (scaled by `q_table_multiplier`) and then dequantized.
    e. An Inverse DCT (`idctn`) is applied to each 8x8 block of coefficients to produce a spatial domain block.
    f. These spatial blocks are assembled into a single channel of noise for the latent.
4. The resulting NumPy array (B, C, H_latent, W_latent) is converted to a PyTorch tensor.
5. The specified `normalization` method is applied to the tensor.
6. The final tensor is moved to the target device and returned.

This node allows for creating initial latents that inherently possess compression-like artifacts, which can influence the style and texture of the generated images.

---

## General Sigma Utilities

### Get Sigmas (🐺)

- **Class:** `WolfSigmasGet`
- **Display Name:** `Get Sigmas (🐺)`
- **Category:** `sampling/sigmas_wolf`
- **Description:** Retrieves the sigma schedule from a given model based on the specified sampler settings (steps, denoise, scheduler type). This is a utility to extract the sigmas that ComfyUI would normally calculate internally.
- **Inputs:**
  - `model`: `MODEL` - The diffusion model.
  - `steps`: `INT` (default: 20, min: 1, max: 10000) - The total number of sampling steps.
  - `denoise`: `FLOAT` (default: 1.0, min: 0.0, max: 1.0, step: 0.01) - The denoising strength. A value of 1.0 uses all steps, while a smaller value uses a fraction of the steps from the end of the schedule.
  - `scheduler`: `COMBO[comfy.samplers.KSampler.SCHEDULERS]` - The ComfyUI native scheduler type to use for sigma calculation.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The calculated sigma tensor.
  - `sigmas_json`: `STRING` - A JSON string representation of the sigmas list.

### Set Sigmas from JSON (🐺)

- **Class:** `WolfSigmasSet`
- **Display Name:** `Set Sigmas from JSON (🐺)`
- **Category:** `sampling/sigmas_wolf`
- **Description:** Converts a JSON string representing a list of sigma values into a SIGMAS tensor that can be used by samplers. Useful for manually defining or importing custom sigma schedules.
- **Inputs:**
  - `sigmas_json`: `STRING` (multiline, default: `"[14.61, ..., 0.0]"`) - A JSON string representing a list of numbers (e.g., `[14.615, 8.0, 0.029, 0.0]`).
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The sigma tensor created from the JSON input.

### Wolf Sigmas to JSON

- **Class:** `WolfSigmasToJSON`
- **Display Name:** `Wolf Sigmas to JSON`
- **Category:** `sampling/sigmas_wolf/util`
- **Description:** Converts an input SIGMAS object to its JSON string representation. Useful for debugging or exporting sigma schedules.
- **Inputs:**
  - `sigmas`: `SIGMAS` - The input SIGMAS object.
- **Outputs:**
  - `sigmas_json`: `STRING` - A JSON string representation of the sigmas list.

---

## Sigma Transformation

### Wolf Sigma Power Transform

- **Class:** `WolfSigmaPowerTransform`
- **Display Name:** `Wolf Sigma Power Transform`
- **Category:** `sampling/sigmas_wolf/transform`
- **Description:** Applies a power transformation to an existing sigma schedule.
    The active sigmas (excluding a potential final 0.0) are normalized to the range $[0,1]$.
    Let $s$ be an active sigma, $s_{min}$ and $s_{max}$ be the min/max of the active range.
    Normalized sigma: $s_{norm} = (s - s_{min}) / (s_{max} - s_{min})$.
    Powered sigma: $s_{powered} = s_{norm}^{\text{power}}$.
    Denormalized sigma: $s_{new} = s_{min} + s_{powered} \times (s_{max} - s_{min})$.
    `power > 1.0`: concentrates steps towards the minimum of the schedule range (smaller steps later).
    `power < 1.0`: concentrates steps towards the maximum of the schedule range (smaller steps earlier).
- **Inputs:**
  - `sigmas_in`: `SIGMAS` - The input sigma schedule.
  - `power`: `FLOAT` (default: 1.0, min: 0.1, max: 10.0) - The power exponent.
  - `override_input_min`: `FLOAT` (default: -1.0) - If >= 0, overrides auto-detected min sigma for normalization.
  - `override_input_max`: `FLOAT` (default: -1.0) - If >= 0, overrides auto-detected max sigma for normalization.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The transformed sigma schedule.

### Wolf Sigma Transform (Clamp T0)

- **Class:** `WolfSigmaClampT0`
- **Display Name:** `Wolf Sigma Transform (Clamp T0)`
- **Category:** `sampling/sigmas_wolf/transform`
- **Description:** Clamps the first sigma (t0) of an incoming schedule to a specified value. Other sigmas are adjusted proportionally if they would exceed t0 or become non-monotonic. The final 0.0 sigma, if present, remains unchanged.
- **Inputs:**
  - `sigmas_in`: `SIGMAS` - The input sigma schedule.
  - `target_t0`: `FLOAT` (default: 1.0, min: 0.0001, max: 1000.0, step: 0.001) - The target value for the first sigma.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing to maintain between sigmas if adjustments are needed.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The transformed sigma schedule with t0 clamped.

### Wolf Sigma Transform (Shift & Scale)

- **Class:** `WolfSigmaShiftAndScale`
- **Display Name:** `Wolf Sigma Transform (Shift & Scale)`
- **Category:** `sampling/sigmas_wolf/transform`
- **Description:** Applies a global shift and scale to all sigmas in a schedule: `Sigmas = (Sigmas + shift) * scale`. The final 0.0 sigma, if present, remains unchanged. Monotonicity (decreasing order) and non-negativity are enforced.
- **Inputs:**
  - `sigmas_in`: `SIGMAS` - The input sigma schedule.
  - `shift`: `FLOAT` (default: 0.0, min: -100.0, max: 100.0, step: 0.01) - Value to add to each sigma before scaling.
  - `scale`: `FLOAT` (default: 1.0, min: 0.01, max: 100.0, step: 0.01) - Value to multiply each shifted sigma by.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing for monotonicity enforcement.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The transformed sigma schedule.

### Wolf Sigma Transform (Normalize Range)

- **Class:** `WolfSigmaNormalizeRange`
- **Display Name:** `Wolf Sigma Transform (Normalize Range)`
- **Category:** `sampling/sigmas_wolf/transform`
- **Description:** Normalizes an existing sigma schedule to a new min/max range. The relative spacing of sigmas is preserved as much as possible. The final 0.0 sigma, if present, remains 0.0.
- **Inputs:**
  - `sigmas_in`: `SIGMAS` - The input sigma schedule.
  - `new_max`: `FLOAT` (default: 1.0, min: 0.0001, max: 1000.0, step: 0.001) - The target maximum sigma value (first sigma).
  - `new_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - The target minimum positive sigma value (last active sigma).
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing for monotonicity enforcement.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The normalized sigma schedule.

### Wolf Sigma Transform (Quantize)

- **Class:** `WolfSigmaQuantize`
- **Display Name:** `Wolf Sigma Transform (Quantize)`
- **Category:** `sampling/sigmas_wolf/transform`
- **Description:** Quantizes sigmas in a schedule to a specific number of decimal places or to the nearest multiple of a quantization step. The final 0.0 sigma, if present, remains unchanged.
- **Inputs:**
  - `sigmas_in`: `SIGMAS` - The input sigma schedule.
  - `quantization_method`: `COMBO["decimal_places", "step_multiple"]` (default: "decimal_places") - Method for quantization.
  - `decimal_places`: `INT` (default: 3, min: 0, max: 10) - Number of decimal places if method is `decimal_places`.
  - `quantization_step`: `FLOAT` (default: 0.001, min: 1e-7, max: 10.0, step: 0.0001) - Step multiple if method is `step_multiple`.
  - `rounding_mode`: `COMBO["round", "floor", "ceil"]` (default: "round") - Rounding method to use.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing for post-quantization monotonicity enforcement.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The quantized sigma schedule.

### Wolf Sigma Transform (Respace Log-Cosine)

- **Class:** `WolfSigmaRespaceLogCosine`
- **Display Name:** `Wolf Sigma Transform (Respace Log-Cosine)`
- **Category:** `sampling/sigmas_wolf/transform`
- **Description:** Respaces an existing N-step sigma schedule (N+1 sigmas) to have its active points (sigma_max to sigma_min_positive) follow a cosine curve in the log-sigma domain. The number of steps in the output matches the input. The final 0.0 sigma, if present, remains unchanged.
- **Inputs:**
  - `sigmas_in`: `SIGMAS` - The input sigma schedule.
  - `override_sigma_max`: `FLOAT` (default: -1.0) - If >= 0, overrides auto-detected max sigma for spacing. -1 uses max from `sigmas_in`.
  - `override_sigma_min_positive`: `FLOAT` (default: -1.0) - If >= 0, overrides auto-detected min positive sigma. -1 uses min from `sigmas_in`.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing for monotonicity enforcement.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The respaced sigma schedule.

### Sigma Clip Values (🐺)

- **Class:** `WolfSigmaClipValues`
- **Display Name:** `Sigma Clip Values (🐺)`
- **Category:** `sampling/sigmas_wolf/transformations`
- **Description:** Clips individual active sigma values in a schedule to a specified minimum and maximum. The final 0.0 sigma, if present, remains unchanged. Optionally re-ensures strictly decreasing order after clipping.
- **Inputs:**
  - `sigmas`: `SIGMAS` - The input sigma schedule.
  - `min_clip_value`: `FLOAT` (default: 0.001, min: 0.0, max: 1000.0, step: 0.001) - Minimum value to clip sigmas to.
  - `max_clip_value`: `FLOAT` (default: 150.0, min: 0.0, max: 1000.0, step: 0.01) - Maximum value to clip sigmas to.
  - `ensure_strictly_decreasing`: `BOOLEAN` (default: True) - Whether to enforce strictly decreasing order after clipping.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing if order is re-enforced.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The sigma schedule with values clipped.

### Sigma Schedule Reverser (🐺)

- **Class:** `WolfSigmaReverse`
- **Display Name:** `Sigma Schedule Reverser (🐺)`
- **Category:** `sampling/sigmas_wolf/transformations`
- **Description:** Reverses the order of active sigmas in a schedule. The final 0.0 sigma, if present, remains unchanged.
- **Inputs:**
  - `sigmas`: `SIGMAS` - The input sigma schedule.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The reversed sigma schedule.

### Wolf Sigma Reverse and Rescale

- **Class:** `WolfSigmaReverseAndRescale`
- **Display Name:** `Wolf Sigma Reverse and Rescale`
- **Category:** `sampling/sigmas_wolf/transformations`
- **Description:** Takes an existing SIGMAS tensor, reverses the order of its active steps, and then rescales them to a new target sigma_max and sigma_min_positive. The last sigma (0.0) remains in place.
- **Inputs:**
  - `sigmas`: `SIGMAS` - The input sigma schedule.
  - `new_sigma_max`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1) - The target maximum sigma for the reversed and rescaled schedule.
  - `new_sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001) - The target minimum positive sigma.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing for monotonicity enforcement.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The reversed and rescaled sigma schedule.

### Sigma Slice (🐺)

- **Class:** `WolfSigmaSlice`
- **Display Name:** `Sigma Slice (🐺)`
- **Category:** `sampling/sigmas_wolf/transformations`
- **Description:** Slices an existing sigma schedule, selecting a sub-sequence of active sigmas. The final 0.0 sigma, if present, is re-appended. Optionally re-ensures strictly decreasing order after slicing.
- **Inputs:**
  - `sigmas`: `SIGMAS` - The input sigma schedule.
  - `start_index`: `INT` (default: 0, min: 0, max: 10000) - Start index for slicing active sigmas.
  - `end_index`: `INT` (default: -1, min: -1, max: 10000) - End index for slicing active sigmas (-1 means to the end).
  - `step_size`: `INT` (default: 1, min: 1, max: 1000) - Step size for slicing.
  - `ensure_strictly_decreasing`: `BOOLEAN` (default: True) - Whether to enforce strictly decreasing order after slicing.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing if order is re-enforced.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The sliced sigma schedule.

### Sigma Insert Value (🐺)

- **Class:** `WolfSigmaInsertValue`
- **Display Name:** `Sigma Insert Value (🐺)`
- **Category:** `sampling/sigmas_wolf/transformations`
- **Description:** Inserts a custom sigma value into an existing schedule at a specified index. Optionally re-sorts the schedule afterwards to maintain strictly decreasing order and ensures the final 0.0 sigma (if present) remains last.
- **Inputs:**
  - `sigmas`: `SIGMAS` - The input sigma schedule.
  - `sigma_value`: `FLOAT` (default: 1.0, min: 0.0001, max: 1000.0, step: 0.01) - The sigma value to insert.
  - `index`: `INT` (default: 0, min: 0, max: 1000) - Index in the active sigmas list where the value should be inserted.
  - `sort_after_insert`: `BOOLEAN` (default: True) - Whether to sort the schedule (descending) after insertion.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing if sorted.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The sigma schedule with the value inserted.

### Sigma Add Noise (🐺)

- **Class:** `WolfSigmaAddNoise`
- **Display Name:** `Sigma Add Noise (🐺)`
- **Category:** `sampling/sigmas_wolf/transformations`
- **Description:** Adds random noise to an existing sigma schedule. Ensures sigmas remain positive and optionally re-sorts them. The final 0.0 sigma (if present) remains unchanged by noise.
- **Inputs:**
  - `sigmas`: `SIGMAS` - The input sigma schedule.
  - `noise_strength`: `FLOAT` (default: 0.1, min: 0.0, max: 10.0, step: 0.01) - Strength of the noise to add.
  - `noise_type`: `COMBO["gaussian", "uniform"]` (default: "gaussian") - Type of noise distribution.
  - `seed`: `INT` (default: 0, min: 0, max: 0xFFFFFFFFFFFFFFFF) - Seed for the random noise generation.
  - `ensure_strictly_decreasing`: `BOOLEAN` (default: True) - Whether to sort the schedule (descending) and enforce spacing after adding noise.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing if sorted.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The sigma schedule with noise added.

---

## General Purpose Sigma Generators

These nodes provide flexible ways to generate sigma schedules based on mathematical functions.

### Wolf Sigma Geometric Progression

- **Class:** `WolfSigmaGeometricProgression`
- **Display Name:** `Wolf Sigma Geometric Progression`
- **Category:** `sampling/sigmas_wolf/transformations` (Note: Category in code is `transformations`, seems more like a generator. Using code value.)
- **Description:** Generates N_steps + 1 sigmas forming a geometric progression. Each active sigma is the previous one multiplied by a `common_ratio`. Sigmas are bounded by `sigma_start` (max) and `sigma_min_positive`. Last sigma is 0.0.
- **Inputs:**
  - `num_steps`: `INT` (default: 12, min: 1, max: 1000) - Number of active sigmas to generate.
  - `sigma_start`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1) - Starting sigma value (sigma_max).
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001) - Smallest positive sigma allowed.
  - `common_ratio`: `FLOAT` (default: 0.75, min: 0.01, max: 2.0, step: 0.01) - Multiplier for each step. <1 for decreasing sigmas.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 geometrically spaced sigmas.

### Wolf Sigma Polynomial

- **Class:** `WolfSigmaPolynomial`
- **Display Name:** `Wolf Sigma Polynomial`
- **Category:** `sampling/sigmas_wolf/transformations` (Note: Category in code is `transformations`, seems more like a generator. Using code value.)
- **Description:** Generates N_steps + 1 sigmas based on a polynomial function of normalized time: `sigma = sigma_min_positive + (sigma_max - sigma_min_positive) * (1 - t_norm^power)`. Last sigma is 0.0.
- **Inputs:**
  - `num_steps`: `INT` (default: 12, min: 1, max: 1000) - Number of active sigmas.
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001) - Smallest positive sigma.
  - `power`: `FLOAT` (default: 1.0, min: 0.1, max: 10.0, step: 0.05) - Power for the polynomial. 1.0=linear, >1.0 denser near sigma_max, <1.0 denser near sigma_min_positive.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 polynomially spaced sigmas.

### Sigma Tanh Generator (🐺)

- **Class:** `WolfSigmaTanhGenerator`
- **Display Name:** `Sigma Tanh Generator (🐺)`
- **Category:** `sampling/sigmas_wolf/generate`
- **Description:** Generates N_steps + 1 sigmas based on a scaled and shifted tanh function of normalized time. Sigmas range from `sigma_max` to `sigma_min_positive`. The tanh function provides an S-shaped curve. Last sigma is 0.0.
- **Inputs:**
  - `num_steps`: `INT` (default: 12, min: 1, max: 1000) - Number of active sigmas.
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001) - Smallest positive sigma.
  - `tanh_scale`: `FLOAT` (default: 3.0, min: 0.1, max: 10.0, step: 0.1) - Scales the input to tanh, controlling steepness. Higher values = steeper.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 tanh-spaced sigmas.

---

## Image Utilities

### Get Image Size

- **Class:** `GetImageSize`
- **Display Name:** `Get Image Size`
- **Category:** `image utils`

#### Get Image Size Description

The "Get Image Size" node is a simple utility that takes an image (or a batch of images) as input and returns its dimensions (width, height) and the number of images in the batch.

This node was originally part of the ComfyUI_essentials pack and has been separated into this standalone repository for more granular control and easier management.

#### Get Image Size Features

- Extracts width, height, and image count from a given image tensor.
- Supports batch processing (provides the count of images in the batch).
- Simple and efficient.

#### Usage

Once installed, you can find the "Get Image Size" node in ComfyUI under the "image utils" category (or by searching for "Get Image Size").

**Inputs:**

- `image` (`IMAGE`): The image or batch of images to get the size from.

**Outputs:**

- `width` (`INT`): The width of the input image(s) in pixels.
- `height` (`INT`): The height of the input image(s) in pixels.
- `count` (`INT`): The number of images in the batch.

Simply connect an image output from another node to the `image` input of the "Get Image Size" node. The `width`, `height`, and `count` outputs can then be used in other nodes that require this information.

#### Example Workflow

```plaintext
+-------------------+     +-------------------+
| Load Image        | --> | Get Image Size    | --> [width, height, count]
+-------------------+     +-------------------+
                         |
                         +--> (Connect width/height/count to other nodes)
```
