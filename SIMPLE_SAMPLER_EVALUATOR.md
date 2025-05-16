# Wolf Simple Sampler Script Evaluator (🐺) Documentation

The `Wolf Simple Sampler Script (🐺)` node offers a streamlined way to define custom sampling logic by executing a user-provided Python script for the core denoising loop. This is simpler than implementing a full KSampler-compatible class and is well-suited for direct experimentation with sampling algorithms.

## Node Details

- **Class:** `WolfSimpleSamplerScriptEvaluator`
- **Display Name:** `Wolf Simple Sampler Script (🐺)`
- **Category:** `sampling/ksampler_wolf`
- **Description (from README):** Executes a user-provided Python script that defines the core sampling loop. The script receives variables like the model, initial latents (`x_initial`), and the sigma schedule (`sigmas_schedule`), and must assign the final denoised latents to a variable named `latents`. A default script implementing a basic Euler sampler is provided as a template.

## Inputs

- **`script`**: `STRING` (multiline)
  - **Description:** The Python script implementing the sampling loop.
  - **Default Value:** (See [Default Script](#default-script) below)

## Outputs

- **`SAMPLER`**: `SAMPLER`
  - **Description:** A ComfyUI compatible sampler object that can be directly used with KSampler nodes.

## Scripting Environment

When you provide a script to this node, it is executed within a specific environment where certain variables and modules are pre-defined and available for your use.

### Available Variables

Your Python script will have access to the following pre-defined local variables:

- `model`: The patched UNet model provided to the KSampler.
- `x_initial`: `torch.Tensor` - The initial latent tensor to be denoised.
- `sigmas_schedule`: `torch.Tensor` - A 1D tensor representing the sigma schedule (noise levels) for the sampling process.
- `extra_args`: `dict` - A dictionary containing extra arguments passed to the model, such as `cond`, `uncond`, `cond_scale`, etc.
- `callback`: `function` - A callback function that can be called during the sampling loop (e.g., for progress updates or previews). It typically expects a dictionary with keys like `'i'` (current step), `'denoised'` (model's prediction), `'x'` (current latents), and `'sigma'` (current sigmas).
- `disable`: `bool` - A flag indicating if the sampler execution should be disabled (e.g., during an error or interruption).
- `sampler_options`: `dict` - A dictionary containing additional options specific to the sampler.

### Required Output Variable

Your script **MUST** assign the final denoised latent tensor to a variable named `latents`. This variable will be returned by the sampling function.

```python
# ... your sampling logic ...
latents = current_latents # Example: assign your final result to 'latents'
```

### Available Globals & Modules

The script execution environment also includes:

- `torch`: The PyTorch library.
- `comfy`: The main ComfyUI library module.
- `nodes`: The `nodes` module from ComfyUI.
- `k_diffusion_sampling`: The `comfy.k_diffusion.sampling` module.
- Standard Python `__builtins__`.

## Default Script

The node provides a default script that implements a basic Euler sampler. This serves as a good starting point and example:

```python
# ComfyUI Wolf Simple Sampler Script Evaluator
#
# This script directly implements the sampling loop.
# The following variables are pre-defined and available for you to use:
#   model, x_initial, sigmas_schedule, extra_args, callback, disable, sampler_options
#
# Your script MUST assign the final denoised latents to a variable named 'latents'.

import torch

current_latents = x_initial.clone()
num_sampling_steps = len(sigmas_schedule) - 1

for i in range(num_sampling_steps):
    sigma_current = sigmas_schedule[i]
    sigma_next = sigmas_schedule[i+1]
    # latents_for_denoising = current_latents.clone() # Not needed without debug
    denoised_prediction = model(current_latents, sigma_current.unsqueeze(0), **extra_args)
    d = (current_latents - denoised_prediction) / sigma_current
    dt = sigma_next - sigma_current
    current_latents = current_latents + d * dt
    
    if callback is not None:
        callback({'i': i, 'denoised': denoised_prediction, 'x': current_latents, 'sigma': sigmas_schedule})

latents = current_latents
```

## Error Handling

- If your script contains syntax errors or runtime errors, the node will attempt to print a traceback to the console.
- If the script fails to compile or execute, the node will fall back to a standard Euler sampler (`k_diffusion_sampling_global.sample_euler`) and print an error message indicating this fallback.
- If even the fallback sampler fails, a critical error message will be logged, and the sampler output will raise a `RuntimeError` if used.

## Example Usage

1. Add the `Wolf Simple Sampler Script (🐺)` node to your workflow.
2. Modify the Python script in the `script` input text area. You can implement any custom sampling logic you need. Ensure you understand the available variables and the requirement to assign the result to the `latents` variable.
3. Connect the `SAMPLER` output of this node to the `sampler` input of a KSampler node (e.g., `KSampler`, `KSamplerAdvanced`).
4. Configure the KSampler node as usual with your model, latents, positive/negative conditioning, and connect the `sigmas` input if you are using custom sigmas (e.g., from another Wolf Sigma node). If no sigmas are explicitly passed to the KSampler that uses this custom sampler, the KSampler will generate them based on its own step count and scheduler settings, and these will be passed to your script as `sigmas_schedule`.

This node allows for powerful customization of the sampling process directly within ComfyUI, enabling rapid prototyping and experimentation with novel diffusion sampling techniques.

## Bonus Samplers

### Proportional Denoised Stepping (PDS) Sampler

The Proportional Denoised Stepping (PDS) sampler is an alternative, deterministic sampling method available for experimentation. Unlike methods that estimate derivatives (e.g., Euler), PDS directly utilizes the model's denoised prediction (often referred to as x0) at each step.

**Core Logic:**

1. At each sampling step `i`, given the `current_latents` at noise level `sigma_current`, the model is queried to predict the fully denoised latent, `denoised_prediction`.
2. A `step_proportion` is calculated based on the current and next sigma values: `(sigma_current - sigma_next) / (sigma_current + epsilon)`, where `epsilon` is a small value to prevent division by zero. This proportion represents the relative reduction in noise scheduled for the current step.
3. The `current_latents` are updated by linearly interpolating towards the `denoised_prediction` using this `step_proportion`. The update rule is: `current_latents = current_latents + step_proportion * (denoised_prediction - current_latents)`.

**Key Characteristics:**

- **Direct x0 Utilisation:** Leverages the model's direct estimate of the clean data at each step.
- **Schedule-Adaptive Stepping:** The magnitude of the update towards x0 is directly proportional to the scheduled decrease in sigma for that step. Larger sigma reductions lead to more substantial updates.
- **Deterministic:** Given the same inputs and schedule, the output will be consistent.
- **Conceptual Simplicity:** Offers a straightforward approach to iteratively reducing noise based on the model's prediction and the sigma schedule.

This method provides a distinct alternative to derivative-based samplers and may offer different convergence properties or visual characteristics, making it a candidate for user experimentation within the scriptable sampler framework.

```python
# Proportional Denoised Stepping (PDS) Sampler

current_latents = x_initial.clone()
num_sampling_steps = len(sigmas_schedule) - 1

# A small epsilon to add to sigma_current in the denominator to prevent division by zero
# if sigma_current itself is extremely small (though sigmas are usually > 0 until the last one).
epsilon = 1e-8 # You can adjust this if needed

for i in range(num_sampling_steps):
    sigma_current = sigmas_schedule[i]
    sigma_next = sigmas_schedule[i+1]

    # Get the model's prediction of the fully denoised image (x0) from the current noisy latents
    denoised_prediction = model(current_latents, sigma_current.unsqueeze(0), **extra_args)

    # Determine the proportion of the way to step towards the denoised_prediction.
    # This proportion is based on how much the noise level is reduced in this step,
    # relative to the current noise level.
    if sigma_current > epsilon: # Ensure sigma_current is meaningfully positive
        # Calculate the proportion of the "distance" (in terms of sigma reduction) to cover in this step.
        # If sigma_next is 0 (last step), step_proportion is 1 (or close to 1).
        # If sigma_current and sigma_next are close, step_proportion is small.
        step_proportion = (sigma_current - sigma_next) / (sigma_current + epsilon)
    else:
        # If sigma_current is effectively zero (or negative, which shouldn't happen in a valid schedule),
        # we should ideally already be at the denoised state or step fully to it.
        step_proportion = 1.0
    
    # Ensure step_proportion is within [0, 1] in case of unusual sigma schedules
    step_proportion = torch.clamp(step_proportion, 0.0, 1.0)

    # Linearly interpolate between the current latents and the denoised prediction.
    # new_latents = current_latents * (1 - step_proportion) + denoised_prediction * step_proportion
    # This can be rewritten as:
    current_latents = current_latents + step_proportion * (denoised_prediction - current_latents)
    
    if callback is not None:
        # The 'denoised' key in the callback usually expects the model's direct output for the current step
        callback({'i': i, 'denoised': denoised_prediction, 'x': current_latents, 'sigma': sigmas_schedule[i]}) # Pass current sigma

# The final latents after all steps
latents = current_latents
```

### Momentum-Guided Denoising (MGD) Sampler

The Momentum-Guided Denoising (MGD) sampler is an experimental method that introduces a momentum term into a Euler-like sampling framework. The core idea is to smooth the sampling trajectory by basing the current step's direction on an exponential moving average (EMA) of previous step directions.

**Core Logic:**

1. **Initialization:** A `momentum_d` tensor, representing the accumulated directional momentum, is initialized to zeros. A hyperparameter `beta` (typically between 0 and 1, e.g., 0.4) controls the influence of past directions on the current momentum.
2. **At each sampling step `i`:**
    a.  The model predicts the `denoised_prediction` from `current_latents` at `sigma_current`.
    b.  The "instantaneous" Euler-like direction, `d_current`, is calculated as `(current_latents - denoised_prediction) / sigma_current`.
    c.  The `momentum_d` is updated via an exponential moving average: `momentum_d = beta * momentum_d + (1 - beta) * d_current`.
    d.  The actual step is taken using this updated `momentum_d` as the direction: `current_latents = current_latents + momentum_d * (sigma_next - sigma_current)`.

**Key Characteristics:**

- **Trajectory Smoothing:** By incorporating an EMA of past directions, MGD aims to reduce oscillations and potentially lead to a smoother convergence path.
- **Hyperparameter `beta`:** This parameter allows control over the "memory" of the momentum.
  - A `beta` of 0 would make `momentum_d` equal to `d_current` each step, effectively reverting to a standard Euler-like step based on the instantaneous direction.
  - A `beta` closer to 1 gives more weight to past directions, resulting in stronger momentum effects.
- **Stateful Stepping:** Unlike simple Euler or PDS, MGD maintains state (`momentum_d`) across sampling steps.
- **Experimental Nature:** The impact of the momentum term can vary depending on the model, sigma schedule, and the choice of `beta`. It offers a way to explore different sampling dynamics.

This sampler provides a mechanism to investigate the effects of temporally smoothed guidance in the diffusion sampling process. The choice of `beta` is crucial and may require experimentation to find optimal values for specific use cases.

```python
# Momentum-Guided Denoising (MGD) Sampler
#
# This script implements a sampling loop that incorporates a momentum term.
# The direction of each step is influenced by an exponential moving average
# of past step directions, aiming to smooth the sampling trajectory.

current_latents = x_initial.clone()
num_sampling_steps = len(sigmas_schedule) - 1

# --- MGD Specific Parameters ---
# Beta controls the weight of past directions in the momentum.
# beta = 0 means no momentum (current d is used, similar to Euler).
# beta close to 1 means high momentum (past directions dominate).
beta = 0.4 # Hyperparameter: 0.0 <= beta < 1.0
momentum_d = torch.zeros_like(x_initial)
# -----------------------------

for i in range(num_sampling_steps):
    sigma_current = sigmas_schedule[i]
    sigma_next = sigmas_schedule[i+1]

    denoised_prediction = model(current_latents, sigma_current.unsqueeze(0), **extra_args)
    
    # Calculate the "ideal" Euler direction for the current state
    d_current = (current_latents - denoised_prediction) / sigma_current
    
    # Update the momentum with the current direction
    # This is an exponential moving average of d values
    momentum_d = beta * momentum_d + (1 - beta) * d_current
    
    # The step is taken using the momentum-influenced direction
    d_for_step = momentum_d
    
    dt = sigma_next - sigma_current
    current_latents = current_latents + d_for_step * dt
    
    if callback is not None:
        callback({'i': i, 'denoised': denoised_prediction, 'x': current_latents, 'sigma': sigmas_schedule[i]})

latents = current_latents
```
