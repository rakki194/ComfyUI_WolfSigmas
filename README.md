# ComfyUI Wolf Sigmas

## ⚠️ Security Warning: Remote Code Execution ⚠️

**This custom node pack includes nodes that can execute arbitrary Python code provided by the user or embedded within imported workflows/scripts. Specifically, the following nodes pose a risk if you load untrusted content:**

- **`Wolf Sigma Script Evaluator (🐺)`**
- **`Wolf Sampler Script Evaluator (🐺)`**
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
  - [Advanced](#advanced)
    - [Scriptable Sigma Generator](#scriptable-sigma-generator)
    - [Scriptable Sampler Generator](#scriptable-sampler-generator)
    - [Simple Scriptable Sampler](#simple-scriptable-sampler)
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
