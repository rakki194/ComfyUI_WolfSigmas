# Wolf Custom Sampler Nodes Documentation

This document provides an overview and usage instructions for the `WolfSimpleSamplerScriptEvaluator` and `WolfPlotSamplerStatsNode` custom nodes for ComfyUI, found within the ComfyUI_WolfSigmas pack.

## Overview

These two nodes work in tandem to provide a highly flexible and introspective sampling experience:

1. **`WolfSimpleSamplerScriptEvaluator`**: Allows you to define a custom sampling loop using a Python script. This gives you direct control over the denoising process, step-by-step. It also collects statistics about the sampling process.
2. **`WolfPlotSamplerStatsNode`**: Visualizes the statistics collected by the `WolfSimpleSamplerScriptEvaluator` from its most recent run, offering insights into the sampler's behavior.

This combination is particularly useful for:

* Experimenting with novel sampling algorithms.
* Debugging existing samplers.
* Understanding the dynamics of the denoising process (e.g., how latents and predictions evolve, sigma schedules).
* Fine-tuning sampling parameters based on observed statistics.

## 1. `WolfSimpleSamplerScriptEvaluator`

**Node Name:** `Wolf Simple Sampler Script (🐺)`
**Category:** `sampling/ksampler_wolf`

### Purpose

This node enables users to replace the standard ComfyUI sampling logic with their own Python script. It essentially wraps your script into a `SAMPLER` object that can be used with KSampler nodes (or any node expecting a sampler).

### Inputs

* **`script`**: (STRING, multiline)
  * This is the core input where you provide the Python code for your custom sampler.
  * A default script implementing a basic Euler sampler with statistics collection is provided.

### Outputs

* **`SAMPLER`**: (SAMPLER)
  * A ComfyUI compatible sampler object. When a KSampler node uses this sampler, the provided Python script is executed to perform the sampling.

### Scripting Environment

When your script is executed, several variables are pre-defined and available in its local scope:

* **`model`**: The patched UNet model (callable). You call this with `model(current_latents, current_sigma, **extra_args)` to get the denoised prediction.
* **`x_initial`**: The initial noisy latent tensor (usually `x_0` if noise is added before, or `x_T` if starting from pure noise).
* **`sigmas_schedule`**: A 1D tensor containing the sigma values for each step of the sampling process. `sigmas_schedule[0]` is the sigma for the first step, and `sigmas_schedule[-1]` is typically close to 0.
* **`extra_args`**: A dictionary containing extra arguments to be passed to the `model` callable (e.g., `cond`, `uncond`, `cond_scale`).
* **`callback`**: A callback function (if provided by the KSampler) that can be called at each step for things like preview generation. The default script shows how to use this.
  * The `callback` expects a dictionary with specific keys: `'i'` (step index), `'denoised'` (model's `x0_pred`), `'x'` (current noisy latent for next step), `'sigma'` (full schedule), `'sigma_hat'` (current sigma), `'sigma_next'` (next sigma).
* **`disable`**: A boolean flag. The script should respect this if needed (though often handled by the KSampler itself before calling the sampler).
* **`sampler_options`**: A dictionary containing additional options passed from the KSampler.

### Mandatory Script Output Variable

Your script **MUST** define a variable named `latents` and assign the final denoised latent tensor to it. This is the value that will be returned by the sampler.

```python
# Example: At the end of your script
latents = final_denoised_result 
```

### Statistics Collection

To enable visualization with `WolfPlotSamplerStatsNode`, your script **SHOULD** populate a dictionary named `collected_stats_output`. This dictionary should contain lists of values, where each list represents a time series over the sampling steps.

The default script populates the following keys:

* `"steps"`: List of step numbers (e.g., `[1, 2, ..., N]`).
* `"sigmas_current"`: List of `sigma_i` values used at each step.
* `"sigmas_next"`: List of `sigma_{i+1}` values for each step.
* `"latent_mean_xi"`: Mean of the input latent `x_i` before model prediction at each step.
* `"latent_std_xi"`: Standard deviation of the input latent `x_i` at each step.
* `"denoised_mean_x0_pred"`: Mean of the model's denoised prediction `x0_pred_i` at each step.
* `"denoised_std_x0_pred"`: Standard deviation of the model's denoised prediction `x0_pred_i` at each step.

You can add custom metrics to this dictionary if you wish to plot them (though you would need to modify the plotting node or create a new one to visualize custom keys).

This `collected_stats_output` dictionary is automatically stored by the `WolfSimpleSamplerScriptEvaluator` node in a class-level variable `_LAST_RUN_STATS` after the script execution. The `WolfPlotSamplerStatsNode` then reads from this variable.

### Default Script (Simplified Euler Sampler)

The default script provides a good starting point. It implements a standard Euler ancestral sampler:

1. Initializes `current_latents` with `x_initial`.
2. Loops through the `sigmas_schedule`:
    a.  Calculates `sigma_current` and `sigma_next`.
    b.  Records statistics for the current `current_latents` (which is `x_i`).
    c.  Calls `model(current_latents, sigma_current, **extra_args)` to get `denoised_prediction` (which is `x0_pred_i`).
    d.  Records statistics for `denoised_prediction`.
    e.  Performs the Euler step:
        ```python
        d = (current_latents - denoised_prediction) / sigma_current
        dt = sigma_next - sigma_current
        current_latents_next_step = current_latents + d * dt
        ```
    f.  Updates `current_latents` to `current_latents_next_step`.
    g.  Calls the `callback` function.
3. Assigns the final `current_latents` to the `latents` variable.

It also includes basic console logging for monitoring progress.

### Error Handling

If the script fails to compile or raises an error during execution:

1. The error and traceback are printed to the console.
2. The node attempts to fall back to a standard `sample_euler` sampler.
3. If the fallback also fails, a critical error is reported, and the sampler will raise a `RuntimeError` if used.

## 2. `WolfPlotSamplerStatsNode`

**Node Name:** `Wolf Plot Sampler Stats (🐺)`
**Category:** `debug/visualize`

### Purpose

This node generates an image containing several plots that visualize the statistics collected by the `WolfSimpleSamplerScriptEvaluator` during its last sampling run.

### Inputs

* **`trigger`**: (`*`, forceInput: True)
  * This input accepts any data type. Its primary purpose is to act as a trigger to ensure this node executes *after* the sampling process (which uses the `WolfSimpleSamplerScriptEvaluator`) has completed.
  * Typically, you would connect an output from the KSampler node (e.g., `LATENT` or `IMAGE`) to this input.
* **`plot_width`**: (INT, default: 800)
  * The width of the generated plot image in pixels.
* **`plot_height`**: (INT, default: 1000)
  * The height of the generated plot image in pixels.
* **`font_size`**: (INT, default: 10)
  * The base font size used for text in the plots.
* **`title_override`**: (STRING, optional, default: "")
  * If provided, this string will be used as the main title for the plot image. Otherwise, a default title "Sampler Statistics Over Steps" is used.

### Outputs

* **`plot_image`**: (IMAGE)
  * An image tensor containing the generated plots. This can be connected to a `Preview Image` node or `Save Image` node to view the statistics.

### Data Source

The `WolfPlotSamplerStatsNode` retrieves its data from the class variable `WolfSimpleSamplerScriptEvaluator._LAST_RUN_STATS`. This means it always plots the data from the most recent execution of *any* `WolfSimpleSamplerScriptEvaluator` node in your workflow.

### Dependencies

* **Matplotlib**: This node requires the `matplotlib` Python library to be installed in your ComfyUI's Python environment.
  * If `matplotlib` is not found, the node will output a placeholder image with a warning message: "Matplotlib not available."
  * You can typically install it with: `pip install matplotlib` (ensure you are using the pip associated with ComfyUI's Python environment).

### Generated Plots

The node generates an image with four subplots:

1. **Sigma Schedule**:
    * Plots `Sigma Current (sigma_i)` over the sampling steps.
    * Useful for understanding the noise schedule used by the sampler.
2. **Input Latent (x_i) Statistics**:
    * Plots the `Mean` and `Standard Deviation (Std)` of the latent tensor `x_i` (the input to the model at each step).
    * Helps visualize how the characteristics of the noisy latent evolve.
3. **Denoised Prediction (x0_pred_i) Statistics**:
    * Plots the `Mean` and `Standard Deviation (Std)` of the model's denoised prediction `x0_pred_i` at each step.
    * Shows how the model's prediction of the clean image changes throughout the sampling process.
4. **Combined Standard Deviations & Sigmas**:
    * Plots `Sigma Current (sigma_i)`, `Latent x_i Std`, and `Denoised x0_pred_i Std` on the same axes.
    * Useful for comparing the scales of these important quantities.

### Error Handling

* If `matplotlib` is not available, a placeholder image is returned with a message.
* If the `WolfSimpleSamplerScriptEvaluator` or its `_LAST_RUN_STATS` attribute cannot be found (e.g., due to import issues or if the sampler hasn't run), a placeholder image with "Sampler data source not found." is returned.
* If `_LAST_RUN_STATS` is empty or doesn't contain the expected "steps" key, a placeholder with "No statistics data found from sampler." is returned.

## How to Use Them Together (Example Workflow)

1. **Add Nodes**:
    * Add a `WolfSimpleSamplerScriptEvaluator` node.
    * Add a KSampler node (e.g., `KSampler` or `KSamplerAdvanced`).
    * Add a `WolfPlotSamplerStatsNode`.
    * Add a `Preview Image` node (or `Save Image`).

2. **Connections**:
    * **Sampler Script to KSampler**: Connect the `SAMPLER` output of `WolfSimpleSamplerScriptEvaluator` to the `sampler` input of your KSampler node.
    * **KSampler to Plot Trigger**: Connect an output from your KSampler node (e.g., `LATENT` or `IMAGE`) to the `trigger` input of the `WolfPlotSamplerStatsNode`. This ensures the plot node only updates after the sampling is complete.
    * **Plot to Preview**: Connect the `plot_image` output of `WolfPlotSamplerStatsNode` to the `images` input of a `Preview Image` node.

3. **Configure `WolfSimpleSamplerScriptEvaluator`**:
    * Modify the `script` input if you want to use a custom sampling logic. For initial testing, the default script is fine.
    * Ensure your script (if custom) populates the `collected_stats_output` dictionary correctly if you want meaningful plots.

4. **Configure KSampler**:
    * Set up your model, positive/negative prompts, latent image, seed, steps, cfg, etc., as you normally would. The `sampler_name` will effectively be ignored as you are providing a custom sampler object. The `scheduler` on the KSampler will still be used to generate the `sigmas_schedule` passed to your script.

5. **Run Workflow**:
    * Queue the prompt.
    * The KSampler will use the script from `WolfSimpleSamplerScriptEvaluator` to generate the image.
    * After the KSampler finishes, the `WolfPlotSamplerStatsNode` will trigger, read the collected statistics, and generate the plot image, which will then be displayed by the `Preview Image` node.

```
Workflow Example:

(Load Checkpoint) -> [MODEL] ------> (KSampler)
(CLIP Text Encode) -> [CONDITIONING] -> (KSampler)
(EmptyLatentImage) -> [LATENT] -----> (KSampler)

(WolfSimpleSamplerScriptEvaluator) -> [SAMPLER] -> (KSampler) [SAMPLER input]

(KSampler) -> [LATENT output] -> (WolfPlotSamplerStatsNode) [trigger input]
(WolfPlotSamplerStatsNode) -> [plot_image] -> (PreviewImage)
```

## Advanced Usage & Customization

The primary power of `WolfSimpleSamplerScriptEvaluator` lies in its flexibility. You can:

* **Implement any sampling algorithm**: Go beyond Euler and try DDIM, DPM-Solver, Heun, ancestral variants, etc. You'll need to understand the mathematical formulations and translate them into Python, using the provided variables.
* **Modify existing algorithms**: Tweak parts of an existing algorithm, for example, by adding noise at intermediate steps, changing how `d` or `dt` is calculated, or implementing custom guidance.
* **Experiment with sigma schedules**: While the `sigmas_schedule` is provided by the KSampler's scheduler, your script can choose to interpret or modify it (though this is less common).
* **Log more data**: Add more keys to the `collected_stats_output` dictionary to track other variables of interest within your custom sampling loop. You would then need to either modify `WolfPlotSamplerStatsNode` or create a new plotting node to visualize this additional data.

When writing custom scripts, pay close attention to:

* The shapes and dtypes of tensors.
* The device (CPU/GPU) on which tensors reside (ComfyUI generally handles this well, but be mindful if creating new tensors).
* The exact meaning of `sigma` in the context of your chosen model and sampling theory.

By using these tools, you can gain a deeper understanding of diffusion models and innovate on sampling techniques directly within the ComfyUI environment.
