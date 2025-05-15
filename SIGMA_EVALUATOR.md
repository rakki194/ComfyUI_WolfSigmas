# Wolf Sigma Script Evaluator (🐺)

- **Class:** `WolfSigmaScriptEvaluator`
- **Display Name:** `Wolf Sigma Script Evaluator (🐺)`
- **Category:** `sampling/sigmas_wolf/script`
- **Description:** Evaluates a user-provided Python script to generate a custom sigma schedule. The script has access to several input parameters and must define a specific variable (`active_sigmas`) containing the desired sigma values. The node then post-processes these sigmas to ensure validity and appends a final `0.0` sigma. This node offers maximum flexibility for advanced users to define complex or experimental sigma schedules.

- **Inputs:**
  - `script_code`: `STRING` (multiline) - The Python script to execute.
    - **Default:** (see example script below)
    - The script can import `numpy` (as `np`), `math`, and `torch`.
    - It has access to the following pre-defined variables:
      - `num_steps` (int): The requested number of active sampling steps.
      - `sigma_max` (float): The maximum sigma value.
      - `sigma_min_positive` (float): The minimum positive sigma value (after internal clamping based on `min_epsilon_spacing`).
      - `min_epsilon_spacing` (float): The minimum allowed difference between consecutive sigmas during post-processing.
    - The script **must** assign a list or NumPy array of floats to a variable named `active_sigmas`. This list/array must have a length equal to `num_steps`.
  - `num_steps`: `INT` (default: 12, min: 1, max: 1000) - The desired number of active sigma values the script should generate. The final output will have `num_steps + 1` sigmas (with the last one being 0.0).
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001) - The target maximum sigma value. The script can use this as a guideline; the node will enforce this as the first sigma value after script execution.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - The target minimum positive sigma value for the active steps. The script can use this; the node will enforce this as the last *active* sigma value.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7, precision: 9) - A small value used during post-processing to ensure sigmas are strictly decreasing and have a minimum separation.

- **Outputs:**
  - `SIGMAS`: `SIGMAS` - A PyTorch tensor of shape `(num_steps + 1,)` containing the generated and post-processed sigma schedule. The last element is always 0.0.
  - `error_log`: `STRING` - A log of the script execution. Contains "Script executed successfully." on success, or an error message and traceback if the script fails or validation issues occur. If an error occurs, a fallback linear schedule is generated.

- **Scripting Details & Post-processing:**
    1. **Input Validation:** `num_steps` must be >= 1. `sigma_min_positive` is clamped to be at least `min_epsilon_spacing`. `sigma_max` is adjusted if it's not sufficiently larger than `sigma_min_positive`.
    2. **Script Execution:** The provided `script_code` is executed.
        - It *must* create a variable `active_sigmas` which is a list or NumPy array of `num_steps` floating-point numbers.
    3. **Validation:** The node checks if `active_sigmas` was set, is a list/array of numbers, and has the correct length (`num_steps`).
  - `script_code`: `STRING` (multiline) - The Python script to execute.
    - **Default:** (see example script below)
    - The script can import `numpy` (as `np`), `math`, and `torch`.
    - It has access to the following pre-defined variables:
      - `num_steps` (int): The requested number of active sampling steps.
      - `sigma_max` (float): The maximum sigma value.
      - `sigma_min_positive` (float): The minimum positive sigma value (after internal clamping based on `min_epsilon_spacing`).
      - `min_epsilon_spacing` (float): The minimum allowed difference between consecutive sigmas during post-processing.
    - The script **must** assign a list or NumPy array of floats to a variable named `active_sigmas`. This list/array must have a length equal to `num_steps`.
  - `num_steps`: `INT` (default: 12, min: 1, max: 1000) - The desired number of active sigma values the script should generate. The final output will have `num_steps + 1` sigmas (with the last one being 0.0).
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001) - The target maximum sigma value. The script can use this as a guideline; the node will enforce this as the first sigma value after script execution.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - The target minimum positive sigma value for the active steps. The script can use this; the node will enforce this as the last *active* sigma value.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7, precision: 9) - A small value used during post-processing to ensure sigmas are strictly decreasing and have a minimum separation.

- **Outputs:**
  - `SIGMAS`: `SIGMAS` - A PyTorch tensor of shape `(num_steps + 1,)` containing the generated and post-processed sigma schedule. The last element is always 0.0.
  - `error_log`: `STRING` - A log of the script execution. Contains "Script executed successfully." on success, or an error message and traceback if the script fails or validation issues occur. If an error occurs, a fallback linear schedule is generated.

- **Scripting Details & Post-processing:**
    1. **Input Validation:** `num_steps` must be >= 1. `sigma_min_positive` is clamped to be at least `min_epsilon_spacing`. `sigma_max` is adjusted if it's not sufficiently larger than `sigma_min_positive`.
    2. **Script Execution:** The provided `script_code` is executed.
        - It *must* create a variable `active_sigmas` which is a list or NumPy array of `num_steps` floating-point numbers.
    3. **Validation:** The node checks if `active_sigmas` was set, is a list/array of numbers, and has the correct length (`num_steps`).
    4. **Error Handling:** If any issues occur during script execution or validation, an error is logged, and a fallback linear schedule is generated (`linspace(sigma_max, sigma_min_positive, num_steps)`).
    5. **Post-processing `active_tensor` (derived from `active_sigmas`):**
        - If `num_steps == 1`: The single value from `active_sigmas` is clamped between `sigma_max` and `sigma_min_positive`.
        - If `num_steps > 1`:
            - `active_tensor[0]` is forced to `sigma_max`.
            - `active_tensor[num_steps - 1]` is forced to `sigma_min_positive`.
            - **Backward pass:** Iterating from `i = num_steps - 2` down to `0`:
                `active_tensor[i] = max(active_tensor[i], active_tensor[i+1] + min_epsilon_spacing)`
                `active_tensor[i] = min(active_tensor[i], sigma_max)`
            - `active_tensor[0]` is re-forced to `sigma_max`.
            - **Forward pass:** Iterating from `i = 0` up to `num_steps - 2`:
                `active_tensor[i+1] = min(active_tensor[i+1], active_tensor[i] - min_epsilon_spacing)`
                `active_tensor[i+1] = max(active_tensor[i+1], sigma_min_positive)`
            - `active_tensor[0]` is re-forced to `sigma_max`.
            - `active_tensor[num_steps - 1]` is re-adjusted to be `max(min(value, active_tensor[N-2] - epsilon), sigma_min_positive)`.
    6. **Final Sigmas:** The post-processed `active_tensor` (length `num_steps`) is copied to `final_sigmas`, and `final_sigmas[num_steps]` is set to `0.0`.

- **Default Script Example:**

  ```python
  # Example script for WolfSigmaScriptEvaluator
  # Available variables: num_steps, sigma_max, sigma_min_positive, min_epsilon_spacing
  # Must assign a list or numpy array of length `num_steps` to `active_sigmas`.

  import numpy as np

  if num_steps == 1:
      # For a single active step, this value will be used.
      # The node ensures it's clamped between sigma_max and sigma_min_positive.
      active_sigmas = [sigma_max]
  else:
      # Linear spacing example:
      active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()

  # Karras-like spacing example (comment out linear above if using this):
  # rho = 7.0
  # inv_rho_min = sigma_min_positive ** (1.0/rho)
  # inv_rho_max = sigma_max ** (1.0/rho)
  # t_values = np.linspace(0, 1, num_steps)
  # active_sigmas = ((inv_rho_max + t_values * (inv_rho_min - inv_rho_max)) ** rho).tolist()
  # if num_steps > 0:
  #    active_sigmas[0] = sigma_max # Ensure endpoints if script logic might miss
  #    active_sigmas[-1] = sigma_min_positive
  ```

- **Workflow Example:**
  This node can be connected directly to the `sigmas` input of a KSampler, allowing for highly customized sampling schedules.

  ```plaintext
  (Your Model) --> (CLIPTextEncode) --> (KSampler)
                                         ^
                                         |
  (WolfSigmaScriptEvaluator) -----------(sigmas)
  ```

## Additional Script Examples

### 1. Beta Distribution Schedule

This schedule utilizes the Percent Point Function (PPF), which is the inverse of the Cumulative Distribution Function (CDF), of a Beta distribution to shape the sigma values. The Beta distribution is defined by two positive shape parameters, $\alpha$ (`alpha_param`) and $\beta$ (`beta_param`).

**Mathematical Formulation:**

1. **Time Points ($t$):** Linearly spaced points are generated from approximately 0 to 1, representing probabilities for the PPF:
    $$ t_j = \text{linspace}(\epsilon, 1-\epsilon, \text{num\_steps}) \quad \text{for } j = 0, \dots, \text{num\_steps}-1 $$
    where $\epsilon$ is a small value (e.g., `1e-6`) to avoid issues at the exact boundaries 0 and 1 for the PPF.

2. **Beta PPF Values ($b$):** The PPF of the Beta distribution, $F^{-1}(p; \alpha, \beta)$, is evaluated at these time points:
    $$ b_j = \text{Beta.ppf}(t_j; \alpha, \beta) $$
    These $b_j$ values are in the range $[0, 1]$.

3. **Sigma Calculation ($\sigma$):** The $b_j$ values are then scaled and inverted to map them to the desired sigma range, from $\sigma_{\text{max}}$ down to $\sigma_{\text{min\_positive}}$:
    $$ \sigma_j = \sigma_{\text{min\_positive}} + (\sigma_{\text{max}} - \sigma_{\text{min\_positive}}) \times (1 - b_j) $$
    The term $(1 - b_j)$ is used because sigmas typically decrease as the sampling progresses (i.e., as $t_j$ increases).

The `scipy.stats.beta.ppf` function is used here, so the script requires `scipy` to be installed in the ComfyUI environment. If `scipy` is not available, the script will fall back to a linear schedule.
You can adjust `alpha_param` and `beta_param` to change the curve's shape:

- $\alpha > 1, \beta > 1$: Tends to concentrate steps in the middle of the sigma range.
- $\alpha < 1, \beta > 1$: Concentrates steps towards $\sigma_{\text{min\_positive}}$ (more steps at lower noise levels).
- $\alpha > 1, \beta < 1$: Concentrates steps towards $\sigma_{\text{max}}$ (more steps at higher noise levels).

```python
# Beta Distribution Schedule
# Requires scipy: pip install scipy
import numpy as np
try:
    from scipy.stats import beta as beta_dist
except ImportError:
    print("SciPy is not installed. Beta distribution schedule cannot be used.")
    # Fallback to linear if scipy is not available
    if num_steps == 1:
        active_sigmas = [sigma_max]
    else:
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()
else:
    if num_steps == 1:
        active_sigmas = [sigma_max]
    else:
        # Parameters for the Beta distribution
        alpha_param = 2.0  # Try values like 0.5, 1.0, 2.0, 5.0
        beta_param = 2.0   # Try values like 0.5, 1.0, 2.0, 5.0

        # Linearly spaced points in the range [0, 1] for the PPF
        # We use a small epsilon to avoid exact 0 and 1 for ppf if problematic
        t_points = np.linspace(1e-6, 1.0 - 1e-6, num_steps)
        
        # Get values from the inverse CDF (percent point function)
        # These values are in [0, 1], representing fractions of the total range
        beta_values = beta_dist.ppf(t_points, alpha_param, beta_param)
        
        # Scale these values to the [sigma_min_positive, sigma_max] range
        # We want sigmas to decrease, so we use (1 - beta_values)
        active_sigmas = sigma_min_positive + (sigma_max - sigma_min_positive) * (1 - beta_values)
        active_sigmas = active_sigmas.tolist()

        # Ensure endpoints are met by the script if desired, though post-processing handles this
        # active_sigmas[0] = sigma_max
        # active_sigmas[-1] = sigma_min_positive
```

### 2. Power Law Schedule (Decreasing)

This schedule spaces sigmas according to $t^{\text{power}}$, where $t$ goes from 1 down to 0.

- A `power > 1.0` will make sigmas decrease faster initially (more steps at lower sigmas).
- A `power < 1.0` will make sigmas decrease slower initially (more steps at higher sigmas).

```python
# Power Law Schedule
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    power = 2.0  # Try values like 0.5 (slower start), 1.0 (linear), 2.0 (faster start), 3.0

    # Normalized time from 0 to 1 (for mapping to sigma_max down to sigma_min_positive)
    t = np.linspace(0, 1, num_steps)
    
    # Apply power to (1-t) to make it decrease from sigma_max
    # (1-t) goes from 1 down to 0.
    # (1-t)^power also goes from 1 down to 0, but with a curve.
    curved_time = (1 - t) ** power 
    
    active_sigmas = sigma_min_positive + (sigma_max - sigma_min_positive) * curved_time
    active_sigmas = active_sigmas.tolist()
```

### 3. Exponential Decay Schedule

This schedule spaces sigmas using an exponential decay function.
The `decay_rate` controls how quickly the sigmas decrease.
A higher `decay_rate` means faster decay towards `sigma_min_positive`.

```python
# Exponential Decay Schedule
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    decay_rate = 5.0  # Try values like 1.0 (gentle) to 10.0 (steep)

    t = np.linspace(0, 1, num_steps)
    # Exponential decay: exp(-decay_rate * t) ranges from 1 down to exp(-decay_rate)
    # We want sigmas to go from sigma_max to sigma_min_positive
    
    # Normalized decay factor (from 1 down towards 0)
    decay_factor = np.exp(-decay_rate * t)
    
    # Rescale to map [exp(-decay_rate), 1] to [sigma_min_positive, sigma_max]
    # To ensure it covers the full range, we can scale it from its own min/max
    # Or, more simply, map it so that t=0 is sigma_max and t=1 is sigma_min_positive
    
    # Simpler approach:
    # active_sigmas = sigma_min_positive + (sigma_max - sigma_min_positive) * np.exp(-decay_rate * t) 
    # This makes it decay from sigma_max + sigma_min_positive down to sigma_min_positive + (sigma_max - sigma_min_positive)*exp(-decay_rate)
    # A better way for full range:
    
    # Calculate sigmas based on log-space interpolation with an exponential bias
    log_sigma_max = np.log(sigma_max)
    log_sigma_min_positive = np.log(sigma_min_positive)
    
    # Apply exponential decay to the interpolation factor
    # (1 - exp(-decay_rate * t)) / (1 - exp(-decay_rate)) normalizes t to be exponentially spaced from 0 to 1
    # Let's use a simpler direct exponential spacing from sigma_max to sigma_min_positive
    # Similar to k-diffusion's exp schedule: sigma(t) = sigma_max * (sigma_min_positive/sigma_max)**t
    
    active_sigmas = sigma_max * (sigma_min_positive / sigma_max) ** t
    active_sigmas = active_sigmas.tolist()
```

### 4. Cosine Annealing Schedule (Log Domain)

This schedule applies cosine annealing to the log of the sigmas.
This can create a smooth transition with slower changes at the extremes (near `sigma_max` and `sigma_min_positive`).

```python
# Cosine Annealing Schedule (Log Domain)
import numpy as np
import math

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    log_max = np.log(sigma_max)
    log_min = np.log(sigma_min_positive)
    
    t = np.linspace(0, 1, num_steps)
    
    # Cosine annealing factor: 0.5 * (1 + cos(pi * t)) goes from 1 to 0
    # We want it to go from 0 to 1 to map log_max to log_min, so use (1 - cos(pi*t))
    cosine_factor = 0.5 * (1 - np.cos(math.pi * t)) # Goes from 0 to 1
    
    log_sigmas = log_max + cosine_factor * (log_min - log_max)
    active_sigmas = np.exp(log_sigmas).tolist()
```

### 5. Karras Schedule (Explicit)

This script explicitly implements the Karras scheduling formula, often used for its good performance across various step counts.
The `rho` parameter controls the curvature. Typical values are around 7.0.

```python
# Karras Schedule (Explicit)
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    rho = 7.0 # Typical Karras rho value

    # Ensure sigma_min_positive is truly positive for the formula
    s_min = max(sigma_min_positive, 1e-9) 
    s_max = sigma_max

    # Create time steps from 0 to 1
    t_steps = np.linspace(0, 1, num_steps)

    # Karras formula
    inv_rho = 1.0 / rho
    sigmas = (s_max**inv_rho + t_steps * (s_min**inv_rho - s_max**inv_rho))**rho
    active_sigmas = sigmas.tolist()
```

### 6. Log-Linear Interpolation Schedule

This schedule interpolates linearly in log-space between `log(sigma_max)` and `log(sigma_min_positive)`.
This is useful when sigmas span several orders of magnitude, ensuring more proportional spacing.

```python
# Log-Linear Interpolation Schedule
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    # Ensure inputs are positive for log
    s_min_pos_safe = max(sigma_min_positive, 1e-9) # Avoid log(0)
    s_max_safe = max(sigma_max, s_min_pos_safe + 1e-9) # Ensure max > min for log

    log_start = np.log(s_max_safe)
    log_end = np.log(s_min_pos_safe)

    # Linearly interpolate in log space
    log_spaced_sigmas = np.linspace(log_start, log_end, num_steps)
    
    active_sigmas = np.exp(log_spaced_sigmas).tolist()
```

### 7. Simple Sigmoid Schedule

This schedule uses a sigmoid function (logistic function) to map linearly spaced time steps to sigma values.
The `steepness` parameter controls how sharply the sigmas transition from `sigma_max` to `sigma_min_positive`.

```python
# Simple Sigmoid Schedule
import numpy as np
import math

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    steepness = 5.0 # Controls the curve. Higher values make the transition sharper in the middle.
    
    t = np.linspace(0, 1, num_steps) # Normalized time from 0 to 1

    # Transform t to span a range for the sigmoid input, e.g., [-steepness/2, steepness/2] or similar
    # To make sigmas decrease: as t goes 0..1, sigmoid_input should go from positive to negative.
    # Example: x spans from `steepness` down to `-steepness`
    x = steepness * (1 - 2 * t) 
    
    sigmoid_values = 1 / (1 + np.exp(-x)) # Output ranges from high (near 1) to low (near 0)
    
    # Scale sigmoid_values (0..1 range) to (sigma_min_positive..sigma_max range)
    active_sigmas = sigma_min_positive + (sigma_max - sigma_min_positive) * sigmoid_values
    active_sigmas = active_sigmas.tolist()
```

### 8. Two-Phase Linear Schedule

This script divides the schedule into two linear phases. You can control:

- `phase1_step_proportion`: How many of the total steps are allocated to the first phase.
- `sigma_drop_frac_at_transition`: The fraction of the total sigma range (`sigma_max` - `sigma_min_positive`) that is covered by the end of the first phase. For example, 0.7 means 70% of the noise reduction happens in the first phase.

This allows creating schedules that might, for instance, decrease sigmas rapidly initially and then more slowly, or vice-versa.

```python
# Two-Phase Linear Schedule
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
elif num_steps < 1: # Should not happen if node's num_steps min is 1
    active_sigmas = []
else: # num_steps >= 2
    # --- Configuration ---
    # Proportion of total steps dedicated to the first phase (0.0 to 1.0).
    phase1_step_proportion = 0.5 
    
    # Fraction of the total sigma range (sigma_max - sigma_min_positive) 
    # that is covered/dropped by the end of the first phase.
    # 0.0: transition point is at sigma_max.
    # 1.0: transition point is at sigma_min_positive.
    sigma_drop_frac_at_transition = 0.7
    # --- End Configuration ---

    # Calculate the number of steps for each phase
    num_p1_steps = int(round(num_steps * phase1_step_proportion))
    # Ensure num_p1_steps is within valid bounds [0, num_steps]
    num_p1_steps = max(0, min(num_steps, num_p1_steps))
    num_p2_steps = num_steps - num_p1_steps
    
    # Calculate the sigma value at the transition point
    s_intermediate = sigma_max - (sigma_max - sigma_min_positive) * sigma_drop_frac_at_transition
    # Clamp s_intermediate to be strictly between sigma_max and sigma_min_positive if possible,
    # or at the boundaries if sigma_drop_frac_at_transition is 0.0 or 1.0.
    s_intermediate = max(sigma_min_positive, min(sigma_max, s_intermediate))

    if num_p1_steps == num_steps: 
        # All steps are in phase 1 (effectively a single linear schedule)
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps, endpoint=True).tolist()
    elif num_p2_steps == num_steps: 
        # All steps are in phase 2 (num_p1_steps is 0; effectively a single linear schedule)
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps, endpoint=True).tolist()
    elif num_p1_steps > 0 and num_p2_steps > 0: 
        # Both phases are active and have steps
        # Phase 1: from sigma_max to s_intermediate.
        # endpoint=False to generate num_p1_steps *not* including s_intermediate.
        sigmas_phase1 = np.linspace(sigma_max, s_intermediate, num_p1_steps, endpoint=False).tolist()
        
        # Phase 2: from s_intermediate to sigma_min_positive.
        # endpoint=True to generate num_p2_steps *including* s_intermediate as the first point
        # and sigma_min_positive as the last point.
        sigmas_phase2 = np.linspace(s_intermediate, sigma_min_positive, num_p2_steps, endpoint=True).tolist()
        
        active_sigmas = sigmas_phase1 + sigmas_phase2
    else: 
        # This case implies one or both of num_p1_steps/num_p2_steps is zero, but not covering the whole range.
        # This should ideally be caught by the num_p1_steps == num_steps or num_p2_steps == num_steps.
        # As a robust fallback for any unhandled configuration leading to empty phases when num_steps >= 2.
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps, endpoint=True).tolist()

    # Final check on length. The logic above aims to produce the correct length.
    # Node post-processing will also enforce monotonicity.
    if len(active_sigmas) != num_steps and num_steps > 0:
        # This indicates an issue in the phase calculation for a specific combination
        # of num_steps and phase1_step_proportion that wasn't perfectly handled.
        # print(f"Warning: Two-phase schedule length {len(active_sigmas)}, expected {num_steps}. Defaulting to linear.")
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps, endpoint=True).tolist()
    elif num_steps == 0 and not active_sigmas: # Ensure empty list for num_steps=0
        active_sigmas = []
```

### 9. Polynomial Schedule (Quadratic Example)

This schedule uses a quadratic polynomial $at^2 + bt + c$ to define the sigma values, where $t$ is normalized time from 0 to 1.
The coefficients are calculated to ensure the curve passes through $(0, \sigma_{\text{max}})$ and $(1, \sigma_{\text{min\_positive}})$.
A third control point, `mid_point_t` and `mid_point_sigma_fraction`, allows shaping the curve (e.g., making it bow inwards or outwards).
A `mid_point_sigma_fraction` of 0.5 at `mid_point_t` of 0.5 would aim for a symmetric curve if other points were linear.

```python
# Polynomial Schedule (Quadratic Example)
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    t = np.linspace(0, 1, num_steps) # Normalized time from 0 to 1

    # Define control points for the quadratic:
    # p0 = (0, sigma_max)
    # p1 = (1, sigma_min_positive)
    # We need a third point to define a unique quadratic. Let's define it mid-way.
    # mid_point_t: 0.0 to 1.0, where the curve should pass through a specific sigma fraction.
    # mid_point_sigma_fraction: 0.0 to 1.0, fraction of (sigma_max - sigma_min_positive).
    # 0.0 means it passes through sigma_min_positive at mid_point_t.
    # 1.0 means it passes through sigma_max at mid_point_t.
    
    mid_point_t = 0.5  # Example: control point at t=0.5
    # Desired sigma at mid_point_t: e.g. 25% of the way from sigma_min_positive to sigma_max
    mid_point_sigma_fraction = 0.75 # Higher fraction means curve bows "up" (slower decrease initially)
                                    # Lower fraction means curve bows "down" (faster decrease initially)
    
    y0 = sigma_max
    y1 = sigma_min_positive
    # y_mid = sigma_min_positive + mid_point_sigma_fraction * (sigma_max - sigma_min_positive) # if fraction maps 0->min, 1->max
    # For sigmas decreasing, we want fraction to map time 0->max, time 1->min
    # So, if mid_point_sigma_fraction is "how much of sigma_max is left", higher is earlier.
    y_mid = sigma_min_positive + (1.0 - mid_point_sigma_fraction) * (sigma_max - sigma_min_positive)


    # Solve for ax^2 + bx + c given (0, y0), (1, y1), (mid_point_t, y_mid)
    # c = y0
    # a + b + c = y1  => a + b = y1 - y0
    # a*mid_t^2 + b*mid_t + c = y_mid => a*mid_t^2 + b*mid_t = y_mid - y0
    
    # Simpler: use Lagrange interpolation or solve system.
    # Forcing y = A(1-t)^2 + B*2t(1-t) + C*t^2 for (0,y0), (mid_t, y_mid), (1,y1) is Bézier like.
    # Let's use a simpler quadratic form: At^2 + Bt + C
    # C = sigma_max
    # A + B + C = sigma_min_positive => A + B = sigma_min_positive - sigma_max
    # A*mid_point_t^2 + B*mid_point_t + C = y_mid  => A*mid_point_t^2 + B*mid_point_t = y_mid - sigma_max

    # Matrix form:
    # [ 1  1 ] [A] = [y1 - y0]
    # [m^2 m] [B] = [ym - y0]
    # where m = mid_point_t

    if mid_point_t == 0 or mid_point_t == 1: # Avoid division by zero / singular matrix, fallback
        # Fallback to linear if mid_point_t is at an endpoint, making quadratic underdetermined or linear.
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()
    else:
        # denom = mid_point_t**2 - mid_point_t
        denom = mid_point_t * (mid_point_t - 1.0)
        if abs(denom) < 1e-9: # Denominator is too small
             active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()
        else:
            val_a_plus_b = sigma_min_positive - sigma_max
            val_b_term = (y_mid - sigma_max - (sigma_min_positive - sigma_max) * mid_point_t**2) / (mid_point_t - mid_point_t**2)
            
            B = ( (y_mid - sigma_max) - (sigma_min_positive - sigma_max) * mid_point_t**2 ) / (mid_point_t * (1.0 - mid_point_t))
            A = sigma_min_positive - sigma_max - B
            C = sigma_max
            
            active_sigmas = A * t**2 + B * t + C
            active_sigmas = active_sigmas.tolist()
            
            # Clip to ensure bounds, as quadratic can overshoot with extreme midpoints
            active_sigmas = [max(sigma_min_positive, min(sigma_max, s)) for s in active_sigmas]

    # Ensure endpoints if script logic might miss, though post-processing handles this too.
    # active_sigmas[0] = sigma_max
    # active_sigmas[-1] = sigma_min_positive
```

### 10. Piecewise Linear Schedule (Multi-Segment)

This schedule allows defining specific sigma values at several "anchor" points in normalized time.
The script then interpolates linearly between these anchors.
The `anchors` list should contain `(time_fraction, sigma_value)` tuples.
Time fractions must be in ascending order from 0.0 to 1.0.
Sigma values should generally be decreasing.

```python
# Piecewise Linear Schedule (Multi-Segment)
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    # Define anchor points: (normalized_time, sigma_value)
    # Time must be 0.0 to 1.0 and increasing.
    # Sigmas should be decreasing.
    # The first anchor should ideally be (0.0, sigma_max)
    # The last anchor should ideally be (1.0, sigma_min_positive)
    anchors = [
        (0.0, sigma_max),
        (0.3, sigma_max * 0.5 + sigma_min_positive * 0.5), # Mid-point example
        (0.7, sigma_max * 0.2 + sigma_min_positive * 0.8), # Closer to min
        (1.0, sigma_min_positive)
    ]

    # Validate anchors (simple checks)
    valid_anchors = True
    if not anchors or anchors[0][0] != 0.0 or anchors[-1][0] != 1.0:
        valid_anchors = False
        print("Piecewise anchors must start at t=0.0 and end at t=1.0.")
    for i in range(len(anchors) - 1):
        if anchors[i+1][0] <= anchors[i][0] or anchors[i+1][1] > anchors[i][1] + 1e-6: # allow small tolerance for same sigma
            valid_anchors = False
            print(f"Piecewise anchors not valid: time must increase, sigmas must decrease. Error at index {i}.")
            break
    
    if not valid_anchors: # Fallback to linear
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()
    else:
        # Create the full schedule by interpolating between anchors
        t_points = np.linspace(0, 1, num_steps)
        anchor_times = np.array([a[0] for a in anchors])
        anchor_sigmas = np.array([a[1] for a in anchors])
        
        # np.interp requires xp (anchor_times) to be increasing.
        active_sigmas = np.interp(t_points, anchor_times, anchor_sigmas).tolist()
```

### 11. Square Root Interpolation Schedule

This schedule interpolates the *square* of the sigmas linearly from $\sigma_{\text{max}}^2$ down to $\sigma_{\text{min\_positive}}^2$,
and then takes the square root. This type of schedule is sometimes referred to in diffusion literature (e.g., related to variance preserving schedules if $\sigma^2 + 1$ is considered).

```python
# Square Root Interpolation Schedule
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    # Ensure inputs are non-negative for sqrt
    s_min_safe = max(sigma_min_positive, 0.0)
    s_max_safe = max(sigma_max, 0.0)

    sigma_max_sq = s_max_safe**2
    sigma_min_sq = s_min_safe**2
    
    t = np.linspace(0, 1, num_steps) # Normalized time from 0 to 1
    
    # Linearly interpolate the squared sigmas from max_sq down to min_sq
    interpolated_sq_sigmas = sigma_max_sq * (1 - t) + sigma_min_sq * t
    
    # Take the square root. Result must be positive.
    active_sigmas = np.sqrt(np.maximum(0, interpolated_sq_sigmas)).tolist()
```

### 12. FlowMatch PingPong Schedule

This schedule is inspired by the sigma generation and shifting logic from the `FlowMatchPingPongScheduler` found in the HuggingFace Diffusers library. The core idea is to start with a base set of sigma values and then apply a non-linear transformation (shift) to alter their distribution. This allows for finer control over how noise levels change during the sampling process.

The script first generates a base set of $N$ active sigma values, denoted as $\sigma_{text\:base,i}$, which are linearly interpolated between $\sigma_{text\:max}$ (the maximum sigma, typically the first value) and $\sigma_{text\:min\_positive}$ (the smallest positive sigma, typically the $N$-th value before the final 0.0).

$$\sigma_{text\:base,i} = \sigma_{text\:max} - i \cdot \frac{\sigma_{text\:max} - \sigma_{text\:min\_positive}}{N - 1} \quad \text{for} \; i = 0, \ldots, N-1$$

where $N$ is `num_steps` from the evaluator node's input.

Then, one of two shifting mechanisms can be applied to these base sigmas:

1. **Static Shift**:

This mechanism applies a non-linear transformation directly to each base sigma value. The formula for a shifted sigma, $\sigma'$, given a base sigma, $\sigma$, and a static shift value, $S$ (`static_shift_value` in the script), is:

$$\sigma' = \frac{S \cdot \sigma}{1 + (S - 1) \cdot \sigma}$$

- $\\sigma$: A sigma value from the initial linear schedule ($\\sigma_{\\text{base}, i}$).
- $S$: The `static_shift_value`.
  - If $S = 1.0$, then $\\sigma' = \\sigma$, meaning no shift is applied, and the schedule remains linear.
  - If $S > 1.0$: Sigmas are shifted *higher* compared to the linear base for $\\sigma \\in (0,1)$. This means the initial steps from $\\sigma_{\\text{max}}$ are smaller (sigmas decrease less rapidly at the start), concentrating more steps at higher noise levels.
  - If $0 < S < 1.0$: Sigmas are shifted *lower* compared to the linear base. This means the initial steps from $\\sigma_{\\text{max}}$ are larger (sigmas decrease more rapidly at the start), concentrating more steps at lower noise levels.
  - The script includes safeguards against potential division by zero if the denominator becomes too small.

**2. Dynamic Time Shift:**

This mechanism is inspired by a time-shifting function that alters the perceived "time" at which a certain sigma level is reached. Given a base sigma, $\\sigma$, and a dynamic mu value, $\\mu$ (`dynamic_mu_value` in the script), the shifted sigma, $\\sigma'$, is calculated as:

$$ \sigma' = \frac{\sigma \cdot e^\mu}{\sigma \cdot (e^\mu - 1) + 1} $$

- $\sigma$: A sigma value from the initial linear schedule ($\sigma_{\text{base}, i}$).
- $\mu$: The `dynamic_mu_value`.
- If $\mu = 0$, then $e^\mu = 1$, and the formula simplifies to $\sigma' = \sigma / (\sigma \cdot 0 + 1) = \sigma$. No shift is applied.
- If $\mu > 0$: Sigmas are shifted "earlier" in the diffusion process relative to a linear schedule. This means the schedule becomes denser (smaller steps) towards $\sigma_{\text{max}}$, resulting in a slower decrease in sigma at the beginning of the sampling process.
- If $\mu < 0$: Sigmas are shifted "later". This means the schedule becomes denser towards $\sigma_{\text{min\_positive}}$, resulting in a faster decrease in sigma initially, followed by finer steps at lower noise levels.
- This formula can be interpreted as remapping the sigma values (which can be thought of as normalized time $t$) through a function $f(t; \mu) = \frac{t \cdot e^\mu}{t \cdot (e^\mu - 1) + 1}$.
The script allows you to choose between these two mechanisms using the `use_dynamic_shifting` boolean flag and configure their respective parameters (`static_shift_value` or `dynamic_mu_value`). The `WolfSigmaScriptEvaluator` node then ensures the final output adheres to `sigma_max`, `sigma_min_positive`, and strict monotonicity.

```python
# FlowMatch PingPong Inspired Schedule
# Implements sigma shifting logic similar to FlowMatchPingPongScheduler.
import numpy as np
import math # math is imported by the evaluator, but np.exp is usually preferred for arrays

# --- Configuration ---
# Set these parameters to control the schedule generation.

# Choose the type of shifting to apply
use_dynamic_shifting = False # If True, uses dynamic_mu_value. If False, uses static_shift_value.

# Parameters for static shifting (if use_dynamic_shifting is False)
# Default is 1.0, which means no shift from the linear base.
# Values > 1.0 tend to push sigmas higher (more steps at higher noise levels initially, i.e. smaller initial steps from sigma_max).
# Values < 1.0 (but > 0) tend to push sigmas lower (more steps at lower noise levels initially, i.e. larger initial steps from sigma_max).
# Be cautious with values that might make the denominator zero or negative,
# e.g., if static_shift_value < 1 and some sigmas are large.
static_shift_value = 1.0

# Parameter for dynamic time shifting (if use_dynamic_shifting is True)
# mu > 0 shifts sigmas "earlier" (denser at the start of the range from sigma_max).
# mu < 0 shifts sigmas "later" (denser at the end of the range towards sigma_min_positive).
# mu = 0 results in no dynamic shift.
dynamic_mu_value = 0.5 # Example value, corresponds to 'mu' in the original scheduler's time_shift

# --- End Configuration ---

if num_steps == 1:
    active_sigmas = [sigma_max]
elif num_steps < 1: # Should not happen based on node constraints but good for script robustness
    active_sigmas = []
else:
    # 1. Generate a base linear schedule using the evaluator's provided min/max
    current_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps)

    # 2. Apply the chosen shifting mechanism
    if use_dynamic_shifting:
        # Apply dynamic time shift: result = (t * exp(mu)) / (t * (exp(mu) - 1) + 1)
        # where t is a sigma value, mu is dynamic_mu_value.
        exp_mu = np.exp(dynamic_mu_value)
        
        numerator = current_sigmas * exp_mu
        denominator = current_sigmas * (exp_mu - 1.0) + 1.0
        
        # Fallback to current_sigmas if denominator is too close to zero
        # (np.abs used for safety, though denominator should be positive with typical inputs)
        shifted_sigmas = np.where(
            np.abs(denominator) < 1e-9, # Check if denominator is close to zero
            current_sigmas, # Fallback to original sigma if denominator is too small
            numerator / denominator
        )
        active_sigmas = shifted_sigmas.tolist()
        
    else: # Apply static shift
        # Apply static shift: result = (S * x) / (1 + (S - 1) * x)
        # where S is static_shift_value, x is a sigma value.
        numerator = static_shift_value * current_sigmas
        denominator = 1.0 + (static_shift_value - 1.0) * current_sigmas
        
        # Fallback to current_sigmas if denominator is too close to zero
        shifted_sigmas = np.where(
            np.abs(denominator) < 1e-9, # Check if denominator is close to zero
            current_sigmas, # Fallback to original sigma if denominator is too small
            numerator / denominator
        )
        active_sigmas = shifted_sigmas.tolist()

    # The WolfSigmaScriptEvaluator node will handle final clamping to ensure:
    # active_sigmas[0] == sigma_max
    # active_sigmas[-1] == sigma_min_positive
    # and monotonicity.
    # So, the script focuses on generating the characteristic distribution.
```

---

### 13. Weibull Distribution Schedule

This schedule utilizes the Percent Point Function (PPF), the inverse of the Cumulative Distribution Function (CDF), of the Weibull distribution. The Weibull distribution is characterized by a shape parameter $k$ (`shape_k` in script) and a scale parameter $\lambda$ (implicitly handled by scaling to `sigma_max` and `sigma_min_positive`). It's often used in reliability analysis to model failure rates; here, we adapt it for sigma scheduling.

**Mathematical Formulation:**

1. **Time Points ($p$):** Linearly spaced points are generated from approximately 0 to 1, representing probabilities for the PPF:
    $p_j = \text{linspace}(\epsilon, 1-\epsilon, \text{num\_steps}) \quad \text{for } j = 0, \dots, \text{num\_steps}-1$
    where $\epsilon$ is a small value (e.g., `1e-6`) to avoid issues at the exact boundaries for the PPF.

2. **Weibull PPF Values ($w$):** The PPF of the Weibull distribution for a given probability $p$, shape $k$, and scale $\lambda=1$ (as we scale later) is:
    $w_j = (-\ln(1-p_j))^{1/k}$
    These $w_j$ values are then normalized by dividing by the PPF value at $1-\epsilon$ to ensure they roughly span $[0, 1]$ before scaling to the sigma range.

3. **Sigma Calculation ($\sigma$):** The normalized $w_j$ values are then scaled and inverted to map them to the desired sigma range, from $\sigma_{\text{max}}$ down to $\sigma_{\text{min\_positive}}$:
    $\sigma_j = \sigma_{\text{min\_positive}} + (\sigma_{\text{max}} - \sigma_{\text{min\_positive}}) \times (1 - w_{\text{norm},j})$
    The term $(1 - w_{\text{norm},j})$ is used because sigmas typically decrease.

The `scipy.stats.weibull_min.ppf` function is used, requiring `scipy`.
Adjust `shape_k`:

- $k < 1$: Indicates a decreasing failure rate (sigmas change rapidly initially, then slower).
- $k = 1$: Constant failure rate (exponential distribution, similar to exponential decay).
- $k > 1$: Indicates an increasing failure rate (sigmas change slowly initially, then more rapidly).

```python
# Weibull Distribution Schedule
# Requires scipy: pip install scipy
import numpy as np
try:
    from scipy.stats import weibull_min
except ImportError:
    print("SciPy is not installed. Weibull distribution schedule cannot be used.")
    if num_steps == 1:
        active_sigmas = [sigma_max]
    else:
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()
else:
    if num_steps == 1:
        active_sigmas = [sigma_max]
    else:
        # Parameter for the Weibull distribution
        shape_k = 0.8  # Try values like 0.5 (steep drop), 1.0 (exponential), 2.0 (S-curve like)

        # Linearly spaced points in the range (0, 1) for the PPF
        p_points = np.linspace(1e-6, 1.0 - 1e-6, num_steps)
        
        # Get values from the inverse CDF (percent point function)
        # Using scale=1 as we'll normalize and rescale manually
        weibull_values = weibull_min.ppf(p_points, shape_k, scale=1.0)
        
        # Normalize weibull_values to roughly [0, 1] range
        # by dividing by the value at near end of range
        if weibull_values[-1] > 1e-9: # Avoid division by zero if shape_k is extreme
            weibull_norm = weibull_values / weibull_min.ppf(1.0 - 1e-6, shape_k, scale=1.0)
        else:
            weibull_norm = np.linspace(0,1,num_steps) # fallback to linear if normalization fails

        # Clamp normalized values to be safe, though ppf should behave
        weibull_norm = np.clip(weibull_norm, 0, 1)

        # Scale these values to the [sigma_min_positive, sigma_max] range
        # We want sigmas to decrease, so we use (1 - weibull_norm)
        active_sigmas = sigma_min_positive + (sigma_max - sigma_min_positive) * (1 - weibull_norm)
        active_sigmas = active_sigmas.tolist()
```

---

### 14. Logit-Normal Schedule

This schedule generates values from a normal distribution, transforms them using the standard logistic function (inverse of the logit function) to map them to the range $(0, 1)$, and then scales them to the desired sigma range $[\sigma_{\text{min\_positive}}, \sigma_{\text{max}}]$.
The Logit-Normal distribution is flexible and can produce various shapes depending on the mean ($\mu$) and standard deviation ($\sigma_N$) of the underlying normal distribution.

**Mathematical Formulation:**

1. **Normal Distribution Points ($x$):** Generate `num_steps` points from a normal distribution. Instead of random sampling, we use the Percent Point Function (PPF) of the normal distribution with linearly spaced probabilities to get deterministic, ordered points. This provides stability.
    Let $p_j = \text{linspace}(\epsilon, 1-\epsilon, \text{num\_steps})$ for $j = 0, \dots, \text{num\_steps}-1$.
    $x_j = \text{Normal.ppf}(p_j, \text{loc}=\mu, \text{scale}=\sigma_N)$
    where $\mu$ is `norm_mean` and $\sigma_N$ is `norm_std_dev`.

2. **Logistic Transformation ($l$):** Apply the standard logistic function to these points:
    $l_j = \frac{1}{1 + e^{-x_j}}$
    The $l_j$ values are in the range $(0, 1)$.

3. **Sigma Calculation ($\sigma$):** Scale and invert the $l_j$ values to map them to the sigma range:
    $\sigma_j = \sigma_{\text{min\_positive}} + (\sigma_{\text{max}} - \sigma_{\text{min\_positive}}) \times (1 - l_j)$
    The $(1 - l_j)$ term ensures sigmas decrease.

Requires `scipy` for `scipy.stats.norm.ppf`.
Adjust `norm_mean` and `norm_std_dev`:

- `norm_mean`: Shifts the center of the distribution. A mean of 0 centers the logistic transformation around 0.5. Positive mean shifts density towards 1 (in $l_j$), negative towards 0.
- `norm_std_dev`: Controls the spread. Smaller values create a sharper transition in the logistic curve, concentrating sigmas more. Larger values spread them out.

```python
# Logit-Normal Schedule
# Requires scipy: pip install scipy
import numpy as np
try:
    from scipy.stats import norm as norm_dist
except ImportError:
    print("SciPy is not installed. Logit-Normal schedule cannot be used.")
    if num_steps == 1:
        active_sigmas = [sigma_max]
    else:
        active_sigmas = np.linspace(sigma_max, sigma_min_positive, num_steps).tolist()
else:
    if num_steps == 1:
        active_sigmas = [sigma_max]
    else:
        # Parameters for the underlying Normal distribution
        norm_mean = 0.0  # Mu: Center of the normal distribution.
        norm_std_dev = 1.5 # Sigma: Standard deviation of the normal distribution. Try 0.5 to 3.0

        # Probabilities for the PPF of the normal distribution
        p_points = np.linspace(1e-5, 1.0 - 1e-5, num_steps) # Avoid exact 0 and 1

        # Generate points from the normal distribution using its PPF
        # These points (x_j) will be used as input to the logistic function
        normal_values = norm_dist.ppf(p_points, loc=norm_mean, scale=norm_std_dev)

        # Apply the standard logistic function (sigmoid) to map normal_values to (0, 1)
        # logistic_values = 1 / (1 + np.exp(-normal_values))
        # scipy.special.expit is numerically stable for this
        try:
            from scipy.special import expit
            logistic_values = expit(normal_values)
        except ImportError:
            print("SciPy's expit not found, using numpy for logistic function.")
            logistic_values = 1 / (1 + np.exp(-normal_values))
            
        # Scale these (0,1) values to the [sigma_min_positive, sigma_max] range
        # We want sigmas to decrease, so we use (1 - logistic_values)
        active_sigmas = sigma_min_positive + (sigma_max - sigma_min_positive) * (1 - logistic_values)
        active_sigmas = active_sigmas.tolist()
```

---

### 15. Exponential Power Interpolation Schedule

This schedule interpolates between `sigma_max` and `sigma_min_positive` using a power function of normalized time $t$. The `exponent` parameter controls the curvature of the interpolation.

**Mathematical Formulation:**

1. **Normalized Time Points ($t$):** Linearly spaced points are generated from $0$ to $1$:
    \\[ t_j = \\frac{j}{\\text{num\\_steps} - 1} \\quad \\text{for } j = 0, \\dots, \\text{num\\_steps}-1 \\]
    For `num_steps = 1`, $t_0 = 0$ if we consider the formula, but the script handles this as a special case `active_sigmas = [sigma_max]`. For the general formula, we assume `num_steps > 1`.

2. **Sigma Calculation ($\\sigma$):** The sigmas are calculated using a weighted average based on $t_j$ raised to the power of `exponent` ($E$):
    \\[ \\sigma_j = \\sigma_{\\text{max}} \\cdot (1 - t_j^E) + \\sigma_{\\text{min\\_positive}} \\cdot t_j^E \\]
    This ensures that when $t_j=0$, $\\sigma_j = \\sigma_{\\text{max}}$, and when $t_j=1$, $\\sigma_j = \\sigma_{\\text{min\\_positive}}$.

**Behavior with `exponent` ($E$):**

- $E = 1$: Results in a linear interpolation between $\\sigma_{\\text{max}}$ and $\\sigma_{\\text{min\\_positive}}$.
- $E > 1$: Sigmas decrease slowly at the beginning (steps are smaller, more time spent at higher noise levels) and then more rapidly towards the end. The example uses $E=2.5$.
- $0 < E < 1$: Sigmas decrease rapidly at the beginning (steps are larger, less time spent at higher noise levels) and then more slowly towards the end.

The script explicitly sets the first and last active sigmas to `sigma_max` and `sigma_min_positive` respectively, to guarantee endpoint matching even with potential floating-point inaccuracies.

```python
# Exponential Power Interpolation Schedule
import numpy as np

if num_steps == 1:
    active_sigmas = [sigma_max]
else:
    t = np.linspace(0, 1, num_steps)
    
    # The exponent controls the curve shape
    # E.g., exponent = 2.5 gives a specific bend.
    # exponent = 1.0 would be linear.
    # exponent = 0.5 would bend the other way.
    exponent = 2.5 
    
    active_sigmas = sigma_max * (1 - t**exponent) + sigma_min_positive * t**exponent
    
    # Ensure endpoints match exactly, though the formula should achieve this for t=0 and t=1.
    # This handles any potential floating point inaccuracies for np.linspace endpoints.
    active_sigmas[0] = sigma_max
    active_sigmas[-1] = sigma_min_positive
    
    # Convert to list for the evaluator
    active_sigmas = active_sigmas.tolist()
```
