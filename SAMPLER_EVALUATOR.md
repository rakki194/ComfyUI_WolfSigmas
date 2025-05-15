# Wolf Sampler Script Evaluator (🐺)

The `Wolf Sampler Script Evaluator` node in the ComfyUI Wolf Sigmas pack allows advanced users to define and execute custom sampling algorithms directly within ComfyUI using Python scripts. This provides maximum flexibility for experimenting with novel sampling techniques or fine-tuning existing ones without modifying the core ComfyUI or k-diffusion code.

## How It Works

The node takes a Python script as input. This script must define a specific structure to be compatible with the evaluator.

### The `wolf_sampler` Function

Your script must contain a top-level function named `wolf_sampler`. This function takes no arguments. Its sole purpose is to return another function: the *actual sampler function*.

```python
def wolf_sampler():
    # ... potentially some setup or helper functions ...
    
    def actual_sampler_function(model, x_initial, sigmas_schedule, *, extra_args, callback, disable, **sampler_options):
        # ... your custom sampling logic ...
        # This function could be a dispatcher that calls other specific samplers.
        return final_latents

    return actual_sampler_function
```

### The `actual_sampler_function`

This is where your custom sampling logic resides. It is called by ComfyUI's KSampler infrastructure with the following parameters:

* **`model`**: A `comfy.model_patcher.ModelPatcher` instance (the UNet model). Call as `model(latents, sigma_tensor, **extra_args)`.
* **`x_initial`**: The initial latent tensor.
* **`sigmas_schedule`**: A 1D PyTorch tensor of sigma values.
* **`extra_args`** (keyword-only dict): Contains `'cond'`, `'uncond'`, `'cond_scale'`, `'noise_seed'` (optional), `'image_cond'` (optional), etc.
* **`callback`** (keyword-only function): For previews. Call as `callback({'i': step, 'denoised': pred, 'x': latents, ...})`.
* **`disable`** (keyword-only bool): If `True`, suppress progress updates.
* **`**sampler_options`** (keyword-only dict): Additional options like `'eta'`, `'s_noise'`.

#### Return Value

The `actual_sampler_function` must return the final denoised latent tensor.

### Available Globals in Script

* `torch`: The PyTorch library.
* `comfy`: Main ComfyUI library modules (includes `comfy.model_sampling`).
* `nodes`: Access to other ComfyUI nodes.
* `k_diffusion_sampling`: The `comfy.k_diffusion.sampling` module, providing helpers like `default_noise_sampler`, `get_ancestral_step`, `to_d`.

## Example: Dispatching Ancestral Sampler for Different Model Types

This example demonstrates a sophisticated ancestral sampler that dispatches to different implementations based on the model type (specifically for flow-matching models like Stable Cascade vs. other diffusion models).

```python
def wolf_sampler():
    # This is the main function the evaluator calls.
    # It returns the actual sampler function KSampler will use.

    # --- RF-Style Euler Ancestral (for flow-matching models) ---
    def sample_euler_ancestral_RF(model, x, sigmas, *, extra_args, callback, disable=False, **sampler_options):
        # Fetch specific sampler options
        eta = sampler_options.get('eta', 1.0)
        s_noise = sampler_options.get('s_noise', 1.0)
        
        # The seed for the noise sampler is fetched from extra_args.
        seed = extra_args.get("seed", None)
        noise_sampler_func = k_diffusion_sampling.default_noise_sampler(x, seed=seed)
        
        s_in = x.new_ones([x.shape[0]])
        num_steps = len(sigmas) - 1

        if not disable:
            print(f"Wolf Euler Ancestral (RF-Style): Steps: {num_steps}, eta: {eta:.2f}, s_noise: {s_noise:.2f}")

        for i in range(num_steps):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

            if sigmas[i + 1] == 0:
                x = denoised
            else:
                downstep_ratio = 1 + (sigmas[i + 1] / sigmas[i] - 1) * eta
                sigma_down = sigmas[i + 1] * downstep_ratio
                
                alpha_ip1 = 1 - sigmas[i + 1] # Using 1-sigma as alpha proxy
                alpha_down = 1 - sigma_down   # Using 1-sigma as alpha proxy

                term_inside_sqrt = sigmas[i + 1]**2 - sigma_down**2 * (alpha_ip1**2 / alpha_down**2 if alpha_down != 0 else torch.zeros_like(alpha_down))
                
                if term_inside_sqrt.item() < 0:
                    if not disable:
                        print(f"  Warning: RF-Style term_inside_sqrt is negative ({term_inside_sqrt.item()}). Clamping renoise_coeff to 0.")
                    renoise_coeff = torch.zeros_like(term_inside_sqrt)
                else:
                    renoise_coeff = term_inside_sqrt**0.5
                
                sigma_down_i_ratio = sigma_down / sigmas[i]
                x = sigma_down_i_ratio * x + (1 - sigma_down_i_ratio) * denoised
                
                if eta > 0:
                    # Pass current and next sigma to the noise_sampler_func if that's what it expects
                    current_noise = noise_sampler_func(sigmas[i], sigmas[i + 1])
                    x = (alpha_ip1 / alpha_down if alpha_down != 0 else torch.ones_like(alpha_down)) * x + current_noise * s_noise * renoise_coeff
            
            if not disable and (i % max(1, num_steps // 10) == 0 or i == num_steps -1) : # Print progress
                 print(f"  RF-Style Step {i+1}/{num_steps}, sigma: {sigmas[i].item():.3f} -> {sigmas[i+1].item():.3f}")
        return x

    # --- ODE-Style Euler Ancestral (for typical diffusion models) ---
    def sample_euler_ancestral_ode_style(model, x, sigmas, *, extra_args, callback, disable=False, **sampler_options):
        # Fetch specific sampler options
        eta = sampler_options.get('eta', 1.0)
        s_noise = sampler_options.get('s_noise', 1.0)

        # The seed for the noise sampler is fetched from extra_args.
        seed = extra_args.get("seed", None)
        noise_sampler_func = k_diffusion_sampling.default_noise_sampler(x, seed=seed)
        
        s_in = x.new_ones([x.shape[0]])
        num_steps = len(sigmas) - 1

        if not disable:
            print(f"Wolf Euler Ancestral (ODE-Style): Steps: {num_steps}, eta: {eta:.2f}, s_noise: {s_noise:.2f}")

        for i in range(num_steps):
            denoised = model(x, sigmas[i] * s_in, **extra_args)
            sigma_down, sigma_up = k_diffusion_sampling.get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})

            if sigma_down == 0: # sigmas[i+1] == 0 implies sigma_down will be 0 from get_ancestral_step with eta=1
                x = denoised
            else:
                # Deterministic step using lerp towards sigma_down:
                # x_new = (1 - w) * x_old + w * denoised_target
                # weight w = (1 - sigma_down / sigmas[i])
                # This is equivalent to x_old + ((x_old - denoised) / sigmas[i]) * (sigma_down - sigmas[i])
                weight = (1. - (sigma_down / sigmas[i]))
                x = x.lerp(denoised, weight)
                
                if sigma_up > 0: # Add noise if there's a stochastic component
                    current_noise = noise_sampler_func(sigmas[i], sigmas[i + 1])
                    x = x + current_noise * s_noise * sigma_up # Add scaled noise
            
            if not disable and (i % max(1, num_steps // 10) == 0 or i == num_steps -1): # Print progress
                print(f"  ODE-Style Step {i+1}/{num_steps}, sigma: {sigmas[i].item():.3f} -> {sigmas[i+1].item():.3f}, down: {sigma_down.item():.3f}, up: {sigma_up.item():.3f}")

        return x

    # --- Dispatcher Sampler ---
    def dispatching_ancestral_sampler(model_wrapper, x_initial, sigmas_schedule, *, extra_args, callback, disable, **sampler_options):
        # Check model type for dispatching
        # Note: Accessing inner_model like this is specific to ComfyUI's structure
        is_flow_model = False
        try:
            # Check if the model_sampling attribute is CONST for flow models
            if isinstance(model_wrapper.inner_model.inner_model.model_sampling, comfy.model_sampling.CONST):
                is_flow_model = True
        except AttributeError:
            # If attributes don't exist, assume not a flow model or handle as needed
            if not disable:
                print("Wolf Dispatcher: Could not determine model_sampling type, defaulting to ODE-style ancestral.")
            is_flow_model = False # Default assumption
        
        if not disable:
            print(f"Wolf Dispatcher: Is flow model? {is_flow_model}. Calling appropriate ancestral sampler.")

        if is_flow_model:
            return sample_euler_ancestral_RF(model_wrapper, x_initial, sigmas_schedule, 
                                            extra_args=extra_args, callback=callback, disable=disable, **sampler_options)
        else:
            return sample_euler_ancestral_ode_style(model_wrapper, x_initial, sigmas_schedule, 
                                                  extra_args=extra_args, callback=callback, disable=disable, **sampler_options)

    return dispatching_ancestral_sampler # KSampler will use this dispatcher

```

Below, we break down the two ancestral samplers used by the dispatcher.

### Euler Ancestral Sampler (RF-Style for Flow Models)

This variant is specifically tailored for models that might have different conventions, such as some flow-matching models.

**Mathematical Formulation:**
(This formulation remains largely the same as previously documented for the "RF-Style")

Let $\\sigma_i$ be `sigmas[i]` and $\\sigma_{i+1}$ be `sigmas[i+1]`.

1. **Predict Denoised Sample ($\\hat{x_0}$):**
    The model $D$ predicts the clean image from the current noisy latent $x_i$ at noise level $\\sigma_i$:
    \\\\[ \\hat{x_0} = D(x_i, \\sigma_i) \\\\]

2. **Handle Final Step:**
    If $\\sigma_{i+1} = 0$, $x_{i+1} = \\hat{x_0}$.

3. **Calculate Step Ratios and Intermediate Sigma (`sigma_down`):**
    The `eta` parameter (typically 1.0 for ancestral-like behavior) controls the step.
    \\\\[ \\text{downstep_ratio} = 1 + (\\frac{\\sigma_{i+1}}{\\sigma_i} - 1) \\cdot \\eta \\\\]
    \\\\[ \\sigma_{down} = \\sigma_{i+1} \\cdot \\text{downstep_ratio} \\\\]

4. **Define Alpha Terms (Proxy):**
    This formulation uses $1-\\sigma$ as a proxy for alpha terms related to noise schedules.
    \\\\[ \\alpha_{proxy, i+1} = 1 - \\sigma_{i+1} \\\\]
    \\\\[ \\alpha_{proxy, down} = 1 - \\sigma_{down} \\\\]

5. **Calculate Renoise Coefficient (`renoise_coeff`):**
    \\\\[ \\text{renoise_coeff} = \\sqrt{\\max(0, \\sigma_{i+1}^2 - \\sigma_{down}^2 \\cdot \\frac{\\alpha_{proxy, i+1}^2}{\\alpha_{proxy, down}^2})} \\\\]
    (Ensuring non-negativity for the square root is important.)

6. **Perform Interpolation Step (Deterministic Part):**
    \\\\[ x\' = \\frac{\\sigma_{down}}{\\sigma_i} x_i + (1 - \\frac{\\sigma_{down}}{\\sigma_i}) \\hat{x_0} \\\\]

7. **Perform Renoise Step (Stochastic Part, if $\\eta > 0$):**
    If `eta > 0`:
    \\\\[ x_{i+1} = \\frac{\\alpha_{proxy, i+1}}{\\alpha_{proxy, down}} x\' + \\mathcal{N}(0,I) \\cdot s_{noise} \\cdot \\text{renoise_coeff} \\\\]
    (Division by $\\alpha_{proxy, down}$ requires it to be non-zero.)

**Python Implementation Notes:**
The Python code for `sample_euler_ancestral_RF` is shown within the main dispatcher example above. It includes checks for `alpha_down != 0` and term_inside_sqrt negativity.

### Euler Ancestral Sampler (ODE-Style for General Diffusion Models)

This variant is closer to the standard k-diffusion Euler Ancestral sampler and is generally suitable for typical latent diffusion models.

**Mathematical Formulation:**

Let $\\sigma_i$ be `sigmas[i]` and $\\sigma_{i+1}$ be `sigmas[i+1]`. The `eta` parameter (typically 1.0) and `s_noise` (typically 1.0) control the ancestral behavior and noise magnitude respectively.

1. **Predict Denoised Sample ($\\hat{x_0}$):**
    \\\\[ \\hat{x_0} = D(x_i, \\sigma_i) \\\\]

2. **Determine Step Components with `get_ancestral_step`:**
    The function $\\text{get_ancestral_step}(\\sigma_i, \\sigma_{i+1}, \\eta)$ calculates:
    * $\\sigma_{down}$: The target sigma for the deterministic part of the step.
    * $\\sigma_{up}$: The standard deviation of the noise to be added for the stochastic part.
    These are calculated such that the step effectively moves from $\\sigma_i$ to $\\sigma_{i+1}$ considering the ancestral noise. For $\\eta=1$, $\\sigma_{down}^2 + \\sigma_{up}^2 = \\sigma_{i+1}^2$ (if $\\sigma_i > \\sigma_{i+1}$).

3. **Handle Final/Zero Sigma Down Step:**
    If $\\sigma_{down} = 0$ (which occurs if $\\sigma_{i+1}=0$ and $\\eta=1$), then:
    \\\\[ x_{i+1} = \\hat{x_0} \\\\]

4. **Calculate Derivative ($d_i$) with `to_d` (Alternative Formulation Step):**
    The derivative (direction) can be calculated as:
    \\\\[ d_i = \\text{to_d}(x_i, \\sigma_i, \\hat{x_0}) = \\frac{x_i - \\hat{x_0}}{\\sigma_i} \\\\]
    This $d_i$ is used in the traditional Euler formulation: $x_i + d_i \\cdot (\\sigma_{down} - \\sigma_i)$.

5. **Perform Combined Euler Step (Deterministic + Stochastic):**
    The latent can be updated using `lerp` for the deterministic part, and then adding noise:
    * **Deterministic part (Euler step to $\\sigma_{down}$ using `lerp`):**
        Let $w = (1 - \\frac{\\sigma_{down}}{\\sigma_i})$.
        \\\\[ x\' = x_i \\cdot (1-w) + \\hat{x_0} \\cdot w = x_i \\cdot (\\frac{\\sigma_{down}}{\\sigma_i}) + \\hat{x_0} \\cdot (1 - \\frac{\\sigma_{down}}{\\sigma_i}) \\\\]
        This is equivalent to the standard Euler step: $x\' = x_i + \\frac{x_i - \\hat{x_0}}{\\sigma_i} (\\sigma_{down} - \\sigma_i)$.
    * **Stochastic part:**
        \\\\[ x_{i+1} = x\' + \\mathcal{N}(0,I) \\cdot s_{noise} \\cdot \\sigma_{up} \\\\]
    The term $\\mathcal{N}(0,I) \\cdot s_{noise} \\cdot \\sigma_{up}$ is the added ancestral noise. If $\\sigma_{up}$ is zero (e.g., if $\\eta=0$ or $\\sigma_{i+1} \\ge \\sigma_i$), no noise is added by this term.

**Python Implementation Notes:**
The Python code for `sample_euler_ancestral_ode_style` is shown within the main dispatcher example above. It uses `k_diffusion_sampling.get_ancestral_step` and `k_diffusion_sampling.to_d`.

---

## Other Example Sampler Implementations

### Custom Euler Sampler

(The existing Euler Sampler documentation can remain here or be slightly adjusted for context if needed)
The Euler method is a first-order numerical procedure... (rest of Euler explanation and code) ...

---

## Script Execution and Error Handling

The `WolfSamplerScriptEvaluator` node compiles and executes the provided script.

* If the script is syntactically incorrect or raises an exception during the definition phase (i.e., when `wolf_sampler()` is called), an error message will be displayed, and the node will attempt to fall back to a default Euler sampler.
* If `wolf_sampler` or the function it returns is not found or is not callable, it will also result in an error and fallback.
* Errors occurring *during* the execution of your `actual_sampler_function` (e.g., a tensor shape mismatch) will propagate up and may halt the ComfyUI queue. It's good practice to include `try-except` blocks within your sampler for robustness if you anticipate specific runtime issues.

The status message output of the node will indicate whether the script was evaluated successfully or if an error occurred (and if a fallback was used).

## Notes on `extra_args` and Model Interaction

The `model` object passed to your sampler is already patched by ComfyUI to handle CFG. This means you don't need to manually compute `cond` and `uncond` outputs and combine them. Simply pass all conditioning information via `**extra_args` when calling the model:

```python
denoised_prediction = model(current_latents, current_sigma_tensor, **extra_args)
```

The `ModelPatcher` will internally use `extra_args['cond']`, `extra_args['uncond']`, and `extra_args['cond_scale']` to perform the CFG-aware prediction.

## Interaction with Sigma Schedulers

The `Wolf Sampler Script Evaluator` expects a `SIGMAS` tensor as input to its KSampler execution path (though the node itself doesn't have a `sigmas` input jack; the KSampler using this custom sampler will). The `sigmas_schedule` parameter in your `actual_sampler_function` will be this tensor.

This node is designed to work seamlessly with the various sigma generation and transformation nodes provided in the `ComfyUI_WolfSigmas` pack, allowing you to pair custom samplers with highly tailored sigma schedules. However, it can operate with any valid `SIGMAS` tensor produced by other ComfyUI nodes.
