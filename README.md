# ComfyUI Wolf Sigmas

This custom node pack for ComfyUI provides a suite of tools for generating and manipulating sigma schedules for diffusion models. These nodes are particularly useful for fine-tuning the sampling process, experimenting with different step counts (especially low step counts), and adapting schedules for specific models like Chroma or those benefiting from AYS-style sigmas.

## Table of Contents

- [ComfyUI Wolf Sigmas](#comfyui-wolf-sigmas)
  - [Table of Contents](#table-of-contents)
  - [General Sigma Utilities](#general-sigma-utilities)
    - [Get Sigmas (🐺)](#get-sigmas-)
    - [Set Sigmas from JSON (🐺)](#set-sigmas-from-json-)
    - [Wolf Sigmas to JSON](#wolf-sigmas-to-json)
  - [1-Step Sigma Schedulers](#1-step-sigma-schedulers)
    - [1-Step Chroma Sigmas (Fixed, 🐺)](#1-step-chroma-sigmas-fixed-)
    - [1-Step Chroma Sigmas (Adjustable, 🐺)](#1-step-chroma-sigmas-adjustable-)
    - [1-Step Chroma Sigmas (Imbalanced, 🐺)](#1-step-chroma-sigmas-imbalanced-)
  - [3-Step Sigma Schedulers (4 Sigmas)](#3-step-sigma-schedulers-4-sigmas)
    - [3-Step Chroma Sigmas (Imbalanced, 🐺)](#3-step-chroma-sigmas-imbalanced-)
    - [3-Step Sigmas (Flux-Inspired, 🐺)](#3-step-sigmas-flux-inspired-)
  - [4-Step Sigma Schedulers (5 Sigmas)](#4-step-sigma-schedulers-5-sigmas)
    - [Wolf Sigma Chroma Karras (4-Step)](#wolf-sigma-chroma-karras-4-step)
    - [Wolf Sigma AYS Inspired (4-Step)](#wolf-sigma-ays-inspired-4-step)
    - [Wolf Sigma Log-Linear (4-Step)](#wolf-sigma-log-linear-4-step)
    - [Wolf Sigma Sigmoid (4-Step)](#wolf-sigma-sigmoid-4-step)
  - [8-Step Sigma Schedulers (9 Sigmas)](#8-step-sigma-schedulers-9-sigmas)
    - [Wolf Sigma Chroma Karras (8-Step)](#wolf-sigma-chroma-karras-8-step)
    - [Wolf Sigma AYS Inspired (8-Step)](#wolf-sigma-ays-inspired-8-step)
    - [Wolf Sigma Log-Linear (8-Step)](#wolf-sigma-log-linear-8-step)
    - [Wolf Sigma Sigmoid (8-Step)](#wolf-sigma-sigmoid-8-step)
    - [8-Step Chroma Sigmas (Imbalanced, 🐺)](#8-step-chroma-sigmas-imbalanced-)
    - [8-Step Sigmas (Flux-Inspired, 🐺)](#8-step-sigmas-flux-inspired-)
    - [Wolf Sigma Cosine LogSNR (8-Step)](#wolf-sigma-cosine-logsnr-8-step)
    - [Wolf Sigma Biased Karras (8-Step)](#wolf-sigma-biased-karras-8-step)
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
  - [N-Step Sigma Schedulers](#n-step-sigma-schedulers)
    - [Wolf Sigma Linear (N-Step)](#wolf-sigma-linear-n-step)
    - [Wolf Sigma Arctan (N-Step)](#wolf-sigma-arctan-n-step)
    - [Wolf Sigma Sigmoid Imbalanced (N-Step)](#wolf-sigma-sigmoid-imbalanced-n-step)
  - [N-Step Imbalanced Sigma Schedulers](#n-step-imbalanced-sigma-schedulers)
    - [Wolf Sigma Linear Imbalanced (N-Step)](#wolf-sigma-linear-imbalanced-n-step)
    - [Wolf Sigma Arctan Imbalanced (N-Step)](#wolf-sigma-arctan-imbalanced-n-step)
    - [Wolf Sigma Cosine LogSNR Imbalanced (N-Step)](#wolf-sigma-cosine-logsnr-imbalanced-n-step)
  - [N-Step Chroma Imbalanced Sigma Schedulers](#n-step-chroma-imbalanced-sigma-schedulers)
    - [Wolf Sigma Chroma Biased Karras (N-Step)](#wolf-sigma-chroma-biased-karras-n-step)
  - [AYS Paper Sigma Schedulers](#ays-paper-sigma-schedulers)
    - [Wolf Sigma AYS Paper Schedule](#wolf-sigma-ays-paper-schedule)
  - [General Purpose Sigma Generators](#general-purpose-sigma-generators)
    - [Wolf Sigma Geometric Progression](#wolf-sigma-geometric-progression)
    - [Wolf Sigma Polynomial](#wolf-sigma-polynomial)
    - [Sigma Tanh Generator (🐺)](#sigma-tanh-generator-)
  - [Advanced / Scriptable Sigma Generators](#advanced--scriptable-sigma-generators)
  - [12-Step Sigma Schedulers (13 Sigmas)](#12-step-sigma-schedulers-13-sigmas)
    - [Wolf Sigma Linear (12-Step)](#wolf-sigma-linear-12-step)
    - [Wolf Sigma Arctan (12-Step)](#wolf-sigma-arctan-12-step)
    - [Wolf Sigma Karras (12-Step)](#wolf-sigma-karras-12-step)
    - [Wolf Sigma Cosine LogSNR (12-Step)](#wolf-sigma-cosine-logsnr-12-step)
    - [Wolf Sigma Sigmoid (12-Step)](#wolf-sigma-sigmoid-12-step)
    - [Wolf Sigma Sigmoid Imbalanced (12-Step)](#wolf-sigma-sigmoid-imbalanced-12-step)
    - [Wolf Sigma AYS (12-Step)](#wolf-sigma-ays-12-step)
    - [Wolf Sigma AYS Imbalanced (12-Step)](#wolf-sigma-ays-imbalanced-12-step)
    - [12-Step Imbalanced Sigma Schedulers](#12-step-imbalanced-sigma-schedulers)
      - [Wolf Sigma Linear Imbalanced (12-Step)](#wolf-sigma-linear-imbalanced-12-step)
      - [Wolf Sigma Arctan Imbalanced (12-Step)](#wolf-sigma-arctan-imbalanced-12-step)
      - [Wolf Sigma Biased Karras (12-Step)](#wolf-sigma-biased-karras-12-step)
      - [Wolf Sigma Cosine LogSNR Imbalanced (12-Step)](#wolf-sigma-cosine-logsnr-imbalanced-12-step)
    - [12-Step Chroma Sigma Schedulers](#12-step-chroma-sigma-schedulers)
      - [Wolf Sigma Chroma Karras (12-Step)](#wolf-sigma-chroma-karras-12-step)
      - [Wolf Sigma Chroma Biased Karras (12-Step)](#wolf-sigma-chroma-biased-karras-12-step)

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

## 1-Step Sigma Schedulers

These nodes generate sigma schedules for a single sampling step (resulting in 2 sigma values: start and end).

### 1-Step Chroma Sigmas (Fixed, 🐺)

- **Class:** `WolfSigmaOneStepChromaFixed`
- **Display Name:** `1-Step Chroma Sigmas (Fixed, 🐺)`
- **Category:** `sampling/sigmas_wolf`
- **Description:** Provides a fixed 1-step sigma schedule: `[0.992, 0.0]`. The start sigma 0.992 is based on a typical initial noise level observed for the Chroma model.
- **Inputs:** None (fixed schedule).
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor `[0.992, 0.0]`.

### 1-Step Chroma Sigmas (Adjustable, 🐺)

- **Class:** `WolfSigmaOneStepChromaAdjustable`
- **Display Name:** `1-Step Chroma Sigmas (Adjustable, 🐺)`
- **Category:** `sampling/sigmas_wolf`
- **Description:** Provides an adjustable 1-step sigma schedule `[start_sigma, end_sigma]`. Useful for experimenting with 1-step inference. Ensures `start_sigma > end_sigma`.
- **Inputs:**
  - `start_sigma`: `FLOAT` (default: 0.992, min: 0.0, max: 100.0, step: 0.001) - The initial sigma value.
  - `end_sigma`: `FLOAT` (default: 0.0, min: 0.0, max: 100.0, step: 0.001) - The final sigma value.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor `[start_sigma, end_sigma]`.

### 1-Step Chroma Sigmas (Imbalanced, 🐺)

- **Class:** `WolfSigmaOneStepChromaImbalanced`
- **Display Name:** `1-Step Chroma Sigmas (Imbalanced, 🐺)`
- **Category:** `sampling/sigmas_wolf`
- **Description:** Provides an adjustable 1-step sigma schedule with an 'imbalance_factor'.
  - Positive imbalance: increases `start_sigma`. $s_{start} = s_{base\_start} + \text{imbalance\_factor} \times \text{scale\_factor}$
  - Negative imbalance: increases `end_sigma`. $s_{end} = s_{base\_end} + |\text{imbalance\_factor}| \times \text{scale\_factor}$
    Ensures $s_{start} > s_{end}$.
- **Inputs:**
  - `base_start_sigma`: `FLOAT` (default: 0.992, min: 0.0, max: 100.0, step: 0.001) - Base starting sigma.
  - `base_end_sigma`: `FLOAT` (default: 0.0, min: 0.0, max: 100.0, step: 0.001) - Base ending sigma.
  - `imbalance_factor`: `FLOAT` (default: 0.0, min: -1.0, max: 1.0, step: 0.01) - Controls the direction and magnitude of imbalance.
  - `scale_factor`: `FLOAT` (default: 0.1, min: 0.0, max: 0.5, step: 0.01) - Scales the effect of the imbalance.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor `[adjusted_start_sigma, adjusted_end_sigma]`.

---

## 3-Step Sigma Schedulers (4 Sigmas)

These nodes generate sigma schedules for a 3-step sampling process (resulting in 4 sigma values: $s_0, s_1, s_2, s_3$).

### 3-Step Chroma Sigmas (Imbalanced, 🐺)

- **Class:** `WolfSigmaFourStepChromaImbalanced`
- **Display Name:** `3-Step Chroma Sigmas (Imbalanced, 🐺)`
- **Category:** `sampling/sigmas_wolf/3_step`
- **Description:** Provides an imbalanced, Karras-style 4-sigma schedule for a 3-step process.
    $s_0$ (start sigma) is adjusted by positive imbalance. $s_3$ (end sigma) is adjusted by negative imbalance.
    Intermediate sigmas $s_1, s_2$ are Karras-interpolated between the adjusted $s_0$ and $s_3$.
    The Karras interpolation for $N$ points from $\sigma_{max}$ to $\sigma_{min}$ is:
    $$ \sigma_i = \left( \sigma_{max}^{1/\rho} + \frac{i}{N-1} (\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho}) \right)^\rho \quad \text{for } i = 0, \dots, N-1 $$
    Here, $N=4$.
- **Inputs:**
  - `base_start_sigma`: `FLOAT` (default: 0.992) - Base starting sigma.
  - `base_end_sigma`: `FLOAT` (default: 0.0) - Base ending sigma.
  - `imbalance_factor`: `FLOAT` (default: 0.0) - Controls imbalance.
  - `scale_factor`: `FLOAT` (default: 0.1) - Scales imbalance effect.
  - `rho`: `FLOAT` (default: 7.0) - Karras $\rho$ parameter.
  - `sigma_spacing_epsilon`: `FLOAT` (default: 0.0001) - Minimum difference between sigmas.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor `[s0, s1, s2, s3]`.

### 3-Step Sigmas (Flux-Inspired, 🐺)

- **Class:** `WolfSigmaFourStepChromaManual` (Note: "Chroma" in class name is historical)
- **Display Name:** `3-Step Sigmas (Flux-Inspired, 🐺)`
- **Category:** `sampling/sigmas_wolf/3_step`
- **Description:** Provides a 4-sigma schedule for 3 steps, inspired by the FLUX model's `sigma_fn`.
    The sigma at normalized time $t \in [0,1]$ is:
    $$ \sigma(t) = \frac{\sigma_{max}}{1 + (t(1-s_2)+s_2)^{s_1} \left( \frac{\sigma_{max}}{\sigma_{min_calc}} - 1 \right) } $$
    where $s_1$ is `flux_shift1`, $s_2$ is `flux_shift2`, and $\sigma_{min_calc} = \max(\sigma_{min}, 10^{-9})$. If $\sigma_{max} \le \sigma_{min}$, linear interpolation is used. Points are calculated at $t = 0, 1/3, 2/3, 1$.
- **Inputs:**
  - `start_sigma`: `FLOAT` (default: 0.992) - Corresponds to $\sigma_{max}$.
  - `end_sigma`: `FLOAT` (default: 0.0) - Corresponds to $\sigma_{min}$.
  - `flux_shift1`: `FLOAT` (default: 2.0) - Shift parameter $s_1$.
  - `flux_shift2`: `FLOAT` (default: 0.1) - Shift parameter $s_2$.
  - `enforce_endpoints`: `BOOLEAN` (default: True) - If true, uses input `start_sigma` and `end_sigma` directly for $s_0$ and $s_3$.
  - `min_step_epsilon`: `FLOAT` (default: 0.0001) - Minimum difference between sigmas.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor `[s0, s1, s2, s3]`.

---

## 4-Step Sigma Schedulers (5 Sigmas)

These nodes generate sigma schedules for a 4-step sampling process (resulting in 5 sigma values).

### Wolf Sigma Chroma Karras (4-Step)

- **Class:** `WolfSigmaChromaKarras4Step`
- **Display Name:** `Wolf Sigma Chroma Karras (4-Step)`
- **Category:** `sampling/sigmas_wolf/4_step`
- **Description:** Generates a Karras schedule for 4 sampling steps (5 sigmas).
    $$ \sigma_i = \left( \sigma_{max}^{1/\rho} + \frac{i}{N_{steps}} (\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho}) \right)^\rho $$
    The final sigma list includes $\sigma_0, \dots, \sigma_{N_{steps}}$, so $N_{steps}+1$ sigmas. Here $N_{steps}=4$.
- **Inputs:**
  - `sigma_min`: `FLOAT` (default: 0.002) - Minimum sigma value.
  - `sigma_max`: `FLOAT` (default: 1.0) - Maximum sigma value.
  - `rho`: `FLOAT` (default: 7.0) - Karras $\rho$ parameter.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 5 Karras-spaced sigmas.

### Wolf Sigma AYS Inspired (4-Step)

- **Class:** `WolfSigmaAYSInspired4Step`
- **Display Name:** `Wolf Sigma AYS Inspired (4-Step)`
- **Category:** `sampling/sigmas_wolf/4_step`
- **Description:** Generates a 4-step schedule (5 sigmas) by selecting points from a predefined AYS (Align Your Steps) schedule for SD1.5.
    The base AYS schedule is: `[14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029]` (11 points for 10 steps).
    Selected points: `AYS[0], AYS[3], AYS[6], AYS[9], 0.0`.
- **Inputs:** None.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 5 sigmas.

### Wolf Sigma Log-Linear (4-Step)

- **Class:** `WolfSigmaLogLinear4Step`
- **Display Name:** `Wolf Sigma Log-Linear (4-Step)`
- **Category:** `sampling/sigmas_wolf/4_step`
- **Description:** Generates 4 log-linearly spaced sigmas from `sigma_max` down to `sigma_min_positive`, followed by a final 0.0.
    The $k$ non-zero sigmas are calculated as: $\sigma_i = \exp(\text{linspace}(\log(\sigma_{max}), \log(\sigma_{min\_positive}), k))_i$. Here $k=4$.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002) - Smallest non-zero sigma.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 5 sigmas `[s0, s1, s2, s3, 0.0]`.

### Wolf Sigma Sigmoid (4-Step)

- **Class:** `WolfSigmaSigmoid4Step`
- **Display Name:** `Wolf Sigma Sigmoid (4-Step)`
- **Category:** `sampling/sigmas_wolf/4_step`
- **Description:** Generates 4 sigmas using a sigmoid curve, scaled between `sigma_max` and `sigma_min`, followed by a final 0.0.
    Normalized time $x_i = \text{linspace}(\text{steepness}, -\text{steepness}, k)$.
    Sigmoid values: $S_i = \frac{1}{1 + e^{-x_i}}$.
    Scaled sigmas: $\sigma_i = \sigma_{min} + (\sigma_{max} - \sigma_{min}) S_i$. Here $k=4$.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 14.615) - Maximum sigma.
  - `sigma_min`: `FLOAT` (default: 0.029) - Minimum sigma for scaling (can be 0).
  - `steepness_factor`: `FLOAT` (default: 5.0) - Controls the steepness of the sigmoid.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 5 sigmas `[s0, s1, s2, s3, 0.0]`.

---

## 8-Step Sigma Schedulers (9 Sigmas)

These nodes generate sigma schedules for an 8-step sampling process (resulting in 9 sigma values).

### Wolf Sigma Chroma Karras (8-Step)

- **Class:** `WolfSigmaChromaKarras8Step`
- **Display Name:** `Wolf Sigma Chroma Karras (8-Step)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Generates a Karras schedule for 8 sampling steps (9 sigmas). (See [Karras 4-Step](#wolf-sigma-chroma-karras-4-step) for formula with $N_{steps}=8$).
- **Inputs:**
  - `sigma_min`: `FLOAT` (default: 0.002)
  - `sigma_max`: `FLOAT` (default: 1.0)
  - `rho`: `FLOAT` (default: 7.0)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 9 Karras-spaced sigmas.

### Wolf Sigma AYS Inspired (8-Step)

- **Class:** `WolfSigmaAYSInspired8Step`
- **Display Name:** `Wolf Sigma AYS Inspired (8-Step)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Generates an 8-step schedule (9 sigmas) by selecting points from the predefined SD1.5 AYS schedule.
    Selected points: `AYS[0], AYS[1], AYS[2], AYS[3], AYS[4], AYS[6], AYS[8], AYS[10], 0.0`.
- **Inputs:** None.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 9 sigmas.

### Wolf Sigma Log-Linear (8-Step)

- **Class:** `WolfSigmaLogLinear8Step`
- **Display Name:** `Wolf Sigma Log-Linear (8-Step)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Generates 8 log-linearly spaced sigmas from `sigma_max` to `sigma_min_positive`, plus a final 0.0. (See [Log-Linear 4-Step](#wolf-sigma-log-linear-4-step) for formula with $k=8$).
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0)
  - `sigma_min_positive`: `FLOAT` (default: 0.002)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 9 sigmas `[s0, ..., s7, 0.0]`.

### Wolf Sigma Sigmoid (8-Step)

- **Class:** `WolfSigmaSigmoid8Step`
- **Display Name:** `Wolf Sigma Sigmoid (8-Step)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Generates 8 sigmas using a sigmoid curve, scaled between `sigma_max` and `sigma_min`, plus a final 0.0. (See [Sigmoid 4-Step](#wolf-sigma-sigmoid-4-step) for formula with $k=8$).
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 14.615)
  - `sigma_min`: `FLOAT` (default: 0.029)
  - `steepness_factor`: `FLOAT` (default: 5.0)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 9 sigmas `[s0, ..., s7, 0.0]`.

### 8-Step Chroma Sigmas (Imbalanced, 🐺)

- **Class:** `WolfSigmaEightStepChromaImbalanced`
- **Display Name:** `8-Step Chroma Sigmas (Imbalanced, 🐺)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Provides an imbalanced, Karras-style 9-sigma schedule for an 8-step process.
    Similar to the [3-Step Imbalanced](#3-step-chroma-sigmas-imbalanced-🐺) node, but for 8 intervals (9 sigmas).
- **Inputs:**
  - `base_start_sigma`: `FLOAT` (default: 0.992)
  - `base_end_sigma`: `FLOAT` (default: 0.0)
  - `imbalance_factor`: `FLOAT` (default: 0.0)
  - `scale_factor`: `FLOAT` (default: 0.1)
  - `rho`: `FLOAT` (default: 7.0)
  - `sigma_spacing_epsilon`: `FLOAT` (default: 0.0001)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor `[s0, ..., s8]`.

### 8-Step Sigmas (Flux-Inspired, 🐺)

- **Class:** `WolfSigmaEightStepChromaManual`
- **Display Name:** `8-Step Sigmas (Flux-Inspired, 🐺)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Provides a 9-sigma schedule for 8 steps, inspired by the FLUX model's `sigma_fn`.
    Similar to the [3-Step Flux-Inspired](#3-step-sigmas-flux-inspired-🐺) node, but points are calculated at $t = i/8$ for $i = 0, \dots, 8$.
- **Inputs:**
  - `start_sigma`: `FLOAT` (default: 0.992)
  - `end_sigma`: `FLOAT` (default: 0.0)
  - `flux_shift1`: `FLOAT` (default: 2.0)
  - `flux_shift2`: `FLOAT` (default: 0.1)
  - `enforce_endpoints`: `BOOLEAN` (default: True)
  - `min_step_epsilon`: `FLOAT` (default: 0.0001)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor `[s0, ..., s8]`.

### Wolf Sigma Cosine LogSNR (8-Step)

- **Class:** `WolfSigmaCosineLogSNR8Step`
- **Display Name:** `Wolf Sigma Cosine LogSNR (8-Step)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Generates an 8-step (9 sigmas) schedule where sigmas are spaced according to a cosine curve in the log-sigma domain. The final sigma is 0.0.
    Normalized time $t_i = \text{linspace}(0, 1, k)$.
    Cosine time $c_i = 0.5 \times (1 - \cos(t_i \times \pi))$.
    Log sigmas: $\log(\sigma_i) = \log(\sigma_{min\_positive}) + (1 - c_i) \times (\log(\sigma_{max}) - \log(\sigma_{min\_positive}))$.
    Sigmas $\sigma_i = \exp(\log(\sigma_i))$. Here $k=8$ non-zero sigmas.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002) - Smallest non-zero sigma.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 9 sigmas.

### Wolf Sigma Biased Karras (8-Step)

- **Class:** `WolfSigmaBiasedKarras8Step`
- **Display Name:** `Wolf Sigma Biased Karras (8-Step)`
- **Category:** `sampling/sigmas_wolf/8_step`
- **Description:** Generates an 8-step (9 sigmas) Karras schedule with a `bias_power`.
    `bias_power > 1.0` concentrates steps towards `sigma_max` (larger early steps).
    `bias_power < 1.0` concentrates steps towards `sigma_min` (smaller early steps).
    Normalized time $t_{norm} = i / (N_{steps})$. Biased time $t_{biased} = t_{norm}^{\text{bias\_power}}$.
    $$ \sigma_i = \left( \sigma_{max}^{1/\rho} + t_{biased,i} (\sigma_{min}^{1/\rho} - \sigma_{max}^{1/\rho}) \right)^\rho $$
    Here $N_{steps}=8$.
- **Inputs:**
  - `sigma_min`: `FLOAT` (default: 0.002)
  - `sigma_max`: `FLOAT` (default: 1.0)
  - `rho`: `FLOAT` (default: 7.0)
  - `bias_power`: `FLOAT` (default: 1.0, min: 0.2, max: 5.0) - Controls the bias.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 9 sigmas.

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

## N-Step Sigma Schedulers

These nodes generate sigma schedules for a user-defined number of sampling steps (N), resulting in N+1 sigma values (N active sigmas from `sigma_max` down to `sigma_min_positive`, plus a final 0.0).

### Wolf Sigma Linear (N-Step)

- **Class:** `WolfSigmaLinearNStep`
- **Display Name:** `Wolf Sigma Linear (N-Step)`
- **Category:** `sampling/sigmas_wolf/N_step`
- **Description:** Generates N_steps + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive) are linearly spaced. Last sigma is 0.0.
- **Inputs:**
  - `num_steps`: `INT` (default: 8, min: 1, max: 1000) - The number of sampling intervals (generates num_steps+1 sigmas).
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001) - Maximum sigma value.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - Smallest positive sigma value.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing for monotonicity.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 linearly spaced sigmas.

### Wolf Sigma Arctan (N-Step)

- **Class:** `WolfSigmaArctanNStep`
- **Display Name:** `Wolf Sigma Arctan (N-Step)`
- **Category:** `sampling/sigmas_wolf/N_step`
- **Description:** Generates N_steps + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive) are spaced such that `arctan(sigma / c_factor)` is linear. Last sigma is 0.0. Inspired by AYS paper Theorem 3.1.
- **Inputs:**
  - `num_steps`: `INT` (default: 8, min: 1, max: 1000) - Number of sampling intervals.
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - Smallest positive sigma.
  - `c_factor`: `FLOAT` (default: 1.0, min: 0.001, max: 100.0, step: 0.001) - The `c` factor in `arctan(sigma / c)`.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 arctan-spaced sigmas.

### Wolf Sigma Sigmoid Imbalanced (N-Step)

- **Class:** `WolfSigmaSigmoidImbalancedNStep`
- **Display Name:** `Wolf Sigma Sigmoid Imbalanced (N-Step)` (Note: Display name from code is `Wolf Sigma Sigmoid Imbalanced (N-Step)`, README previously had this as `Wolf Sigma Sigmoid (N-Step)`. Using code version.)
- **Category:** `sampling/sigmas_wolf/n_step` (Note: Category from code is `sampling/sigmas_wolf/n_step`, README previously had `N_step_imbalanced` for similar nodes. Standardizing to code version for now.)
- **Description:** Generates an N-step schedule using a sigmoid curve, with a bias factor to control step distribution. Produces N+1 sigma values.
- **Inputs:**
  - `num_steps`: `INT` (default: 20, min: 1, max: 1000) - Number of sampling intervals.
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1) - Maximum sigma.
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0, max: 10.0, step: 0.001) - Minimum sigma for scaling (can be 0, implies active_sigma_min is epsilon).
  - `skew_factor`: `FLOAT` (default: 3.0, min: 0.1, max: 10.0, step: 0.1) - Controls the steepness/range of the sigmoid input.
  - `bias_factor`: `FLOAT` (default: 1.0, min: 0.01, max: 100.0, step: 0.01) - Power for time normalization, affecting step concentration (1.0 is linear time).
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 sigmoid-spaced sigmas.

---

## N-Step Imbalanced Sigma Schedulers

These nodes generate N+1 sigmas with a bias factor to concentrate steps towards either `sigma_max` (early in diffusion) or `sigma_min_positive` (late in diffusion).

### Wolf Sigma Linear Imbalanced (N-Step)

- **Class:** `WolfSigmaLinearImbalancedNStep`
- **Display Name:** `Wolf Sigma Linear Imbalanced (N-Step)`
- **Category:** `sampling/sigmas_wolf/N_step_imbalanced`
- **Description:** Generates N_steps + 1 sigmas with a bias factor. Active sigmas (sigma_max to sigma_min_positive) are spaced non-linearly based on bias. Last sigma is 0.0.
- **Inputs:**
  - `num_steps`: `INT` (default: 8, min: 1, max: 1000) - Number of sampling intervals.
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - Smallest positive sigma.
  - `bias_factor`: `FLOAT` (default: 0.0, min: -0.9, max: 0.9, step: 0.05) - Controls step concentration. >0: more steps near sigma_max; <0: more steps near sigma_min_positive.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 imbalanced linearly spaced sigmas.

### Wolf Sigma Arctan Imbalanced (N-Step)

- **Class:** `WolfSigmaArctanImbalancedNStep`
- **Display Name:** `Wolf Sigma Arctan Imbalanced (N-Step)`
- **Category:** `sampling/sigmas_wolf/N_step_imbalanced`
- **Description:** Generates N_steps + 1 sigmas with a bias factor applied in the arctan domain. Active sigmas (sigma_max to sigma_min_positive) are spaced non-linearly. Last sigma is 0.0.
- **Inputs:**
  - `num_steps`: `INT` (default: 8, min: 1, max: 1000) - Number of sampling intervals.
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - Smallest positive sigma.
  - `c_factor`: `FLOAT` (default: 1.0, min: 0.001, max: 100.0, step: 0.001) - The `c` factor for arctan spacing.
  - `bias_factor`: `FLOAT` (default: 0.0, min: -0.9, max: 0.9, step: 0.05) - Controls step concentration in arctan space.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 imbalanced arctan-spaced sigmas.

### Wolf Sigma Cosine LogSNR Imbalanced (N-Step)

- **Class:** `WolfSigmaCosineLogSNRImbalancedNStep`
- **Display Name:** `Wolf Sigma Cosine LogSNR Imbalanced (N-Step)`
- **Category:** `sampling/sigmas_wolf/N_step_imbalanced`
- **Description:** Generates an N-step (N+1 sigmas) schedule with bias. Sigmas are spaced according to a biased cosine curve in log-sigma domain. Last sigma is 0.0.
- **Inputs:**
  - `num_steps`: `INT` (default: 12, min: 1, max: 1000) - Number of sampling intervals.
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1) - Maximum sigma.
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001) - Smallest positive sigma.
  - `bias_factor`: `FLOAT` (default: 0.0, min: -0.9, max: 0.9, step: 0.05) - Controls bias in cosine spacing of log-sigmas.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 imbalanced cosine log-SNR spaced sigmas.

---

## N-Step Chroma Imbalanced Sigma Schedulers

These nodes are variants of Karras schedulers, often with Chroma-inspired defaults, and include a bias parameter.

### Wolf Sigma Chroma Biased Karras (N-Step)

- **Class:** `WolfSigmaChromaBiasedKarrasNStep`
- **Display Name:** `Wolf Sigma Chroma Biased Karras (N-Step)`
- **Category:** `sampling/sigmas_wolf/N_step_chroma_imbalanced`
- **Description:** Generates an N-step Karras schedule with a `bias_power`, using Chroma defaults for sigma_min, sigma_max, and rho. `bias_power > 1.0` concentrates steps towards sigma_max; `< 1.0` concentrates towards sigma_min.
- **Inputs:**
  - `num_steps`: `INT` (default: 12, min: 1, max: 1000) - Number of sampling intervals.
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001) - Minimum sigma (Chroma default).
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1) - Maximum sigma (Chroma default).
  - `rho`: `FLOAT` (default: 7.0, min: 0.1, max: 100.0, step: 0.1) - Karras rho parameter (Chroma default).
  - `bias_power`: `FLOAT` (default: 1.0, min: 0.2, max: 5.0, step: 0.05) - Power for biasing step distribution.
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7) - Minimum spacing.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with N+1 biased Karras sigmas.

---

## AYS Paper Sigma Schedulers

### Wolf Sigma AYS Paper Schedule

- **Class:** `WolfSigmaAYSPaperSchedulePicker`
- **Display Name:** `Wolf Sigma AYS Paper Schedule`
- **Category:** `sampling/sigmas_wolf/AYS_paper`
- **Description:** Picks one of the pre-defined schedules from the AYS Paper (Table 2). Outputs N_steps + 1 sigmas, where the last sigma is 0.0. The number of steps is determined by the chosen schedule. Sigmas are scaled based on `target_sigma_max` relative to the paper's original max (e.g. 14.615 for SD1.5 based schedules).
- **Inputs:**
  - `schedule_name`: `COMBO[AYS_PAPER_SCHEDULES.keys()]` - Name of the AYS paper schedule to use.
  - `target_sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001) - The desired sigma_max for the output schedule. The chosen paper schedule will be scaled to this value.
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - The selected and scaled AYS paper sigma schedule.
  - `num_steps_out`: `INT` - The number of sampling steps in the chosen schedule.

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

## Advanced / Scriptable Sigma Generators

For advanced users who need maximum flexibility, the `Wolf Sigma Script Evaluator (🐺)` node allows for the creation of custom sigma schedules by executing a user-provided Python script. This provides a powerful way to define complex or experimental schedules.

For detailed information on how to use the script evaluator, its inputs, outputs, scripting capabilities, and example scripts, please see the [Wolf Sigma Script Evaluator Documentation](./SIGMA_EVALUATOR.md).

---

## 12-Step Sigma Schedulers (13 Sigmas)

These nodes generate sigma schedules for a 12-step sampling process, resulting in 13 sigma values (12 active sigmas from `sigma_max` down to `sigma_min_positive`, plus a final 0.0). Many are hardcoded versions of their N-Step counterparts for convenience.

### Wolf Sigma Linear (12-Step)

- **Class:** `WolfSigmaLinear12Step`
- **Display Name:** `Wolf Sigma Linear (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates 12 + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive) are linearly spaced. Last sigma is 0.0.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001)
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 linearly spaced sigmas.

### Wolf Sigma Arctan (12-Step)

- **Class:** `WolfSigmaArctan12Step`
- **Display Name:** `Wolf Sigma Arctan (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates 12 + 1 sigmas. Active sigmas (sigma_max to sigma_min_positive) are spaced such that `arctan(sigma / c_factor)` is linear. Last sigma is 0.0.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001)
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `c_factor`: `FLOAT` (default: 1.0, min: 0.001, max: 100.0, step: 0.001)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 arctan-spaced sigmas.

### Wolf Sigma Karras (12-Step)

- **Class:** `WolfSigmaKarras12Step`
- **Display Name:** `Wolf Sigma Karras (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates a 12-step (13 sigmas) Karras schedule. Last sigma is 0.0.
- **Inputs:**
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1)
  - `rho`: `FLOAT` (default: 7.0, min: 0.1, max: 100.0, step: 0.1)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 Karras-spaced sigmas.

### Wolf Sigma Cosine LogSNR (12-Step)

- **Class:** `WolfSigmaCosineLogSNR12Step`
- **Display Name:** `Wolf Sigma Cosine LogSNR (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates a 12-step (13 sigmas) schedule where sigmas are spaced according to a cosine curve in the log-sigma domain. Last sigma is 0.0.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1)
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 cosine log-SNR spaced sigmas.

### Wolf Sigma Sigmoid (12-Step)

- **Class:** `WolfSigmaSigmoid12Step`
- **Display Name:** `Wolf Sigma Sigmoid (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates a 12-step schedule using a sigmoid curve. Produces 13 sigma values, from sigma_max down to sigma_min_positive, then a final 0.0.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1)
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001)
  - `steepness_factor`: `FLOAT` (default: 5.0, min: 1.0, max: 10.0, step: 0.1)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 sigmoid-spaced sigmas.

### Wolf Sigma Sigmoid Imbalanced (12-Step)

- **Class:** `WolfSigmaSigmoidImbalanced12Step`
- **Display Name:** `Wolf Sigma Sigmoid Imbalanced (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates a 12-step schedule using a sigmoid curve, with a bias factor to control step distribution. Produces 13 sigma values.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1)
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0, max: 10.0, step: 0.001)
  - `skew_factor`: `FLOAT` (default: 3.0, min: 0.1, max: 10.0, step: 0.1)
  - `bias_factor`: `FLOAT` (default: 1.0, min: 0.01, max: 100.0, step: 0.01)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 imbalanced sigmoid-spaced sigmas.

### Wolf Sigma AYS (12-Step)

- **Class:** `WolfSigmaAYS12Step`
- **Display Name:** `Wolf Sigma AYS (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates a 12-step (13 sigmas) schedule for Chroma, inspired by Align Your Steps (AYS). It uses log-linear interpolation from a pre-defined high-resolution AYS base schedule, scaled to the target sigma_max and respecting sigma_min.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1)
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0, max: 10.0, step: 0.001)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 AYS-inspired sigmas for Chroma.

### Wolf Sigma AYS Imbalanced (12-Step)

- **Class:** `WolfSigmaAYSImbalanced12Step`
- **Display Name:** `Wolf Sigma AYS Imbalanced (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step`
- **Description:** Generates a 12-step schedule for Chroma, inspired by AYS, with a bias factor. Uses log-linear interpolation from a pre-defined AYS base schedule, scaled and biased.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.01, max: 1000.0, step: 0.1)
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0, max: 10.0, step: 0.001)
  - `bias_factor`: `FLOAT` (default: 1.0, min: 0.01, max: 100.0, step: 0.01)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-5, min: 1e-7, max: 1e-2, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 biased AYS-inspired sigmas for Chroma.

### 12-Step Imbalanced Sigma Schedulers

These are 12-step schedulers that include a bias/imbalance factor.

#### Wolf Sigma Linear Imbalanced (12-Step)

- **Class:** `WolfSigmaLinearImbalanced12Step`
- **Display Name:** `Wolf Sigma Linear Imbalanced (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step_imbalanced`
- **Description:** Generates 12 + 1 sigmas with a bias factor. Active sigmas are spaced non-linearly based on bias. Last sigma is 0.0.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001)
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `bias_factor`: `FLOAT` (default: 0.0, min: -0.9, max: 0.9, step: 0.05)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 imbalanced linearly spaced sigmas.

#### Wolf Sigma Arctan Imbalanced (12-Step)

- **Class:** `WolfSigmaArctanImbalanced12Step`
- **Display Name:** `Wolf Sigma Arctan Imbalanced (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step_imbalanced`
- **Description:** Generates 12 + 1 sigmas with a bias factor applied in the arctan domain. Active sigmas are spaced non-linearly. Last sigma is 0.0.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.001, max: 1000.0, step: 0.001)
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `c_factor`: `FLOAT` (default: 1.0, min: 0.001, max: 100.0, step: 0.001)
  - `bias_factor`: `FLOAT` (default: 0.0, min: -0.9, max: 0.9, step: 0.05)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 imbalanced arctan-spaced sigmas.

#### Wolf Sigma Biased Karras (12-Step)

- **Class:** `WolfSigmaBiasedKarras12Step`
- **Display Name:** `Wolf Sigma Biased Karras (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step_imbalanced`
- **Description:** Generates a 12-step (13 sigmas) Karras schedule with a `bias_power`.
- **Inputs:**
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1)
  - `rho`: `FLOAT` (default: 7.0, min: 0.1, max: 100.0, step: 0.1)
  - `bias_power`: `FLOAT` (default: 1.0, min: 0.2, max: 5.0, step: 0.05)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 biased Karras-spaced sigmas.

#### Wolf Sigma Cosine LogSNR Imbalanced (12-Step)

- **Class:** `WolfSigmaCosineLogSNRImbalanced12Step`
- **Display Name:** `Wolf Sigma Cosine LogSNR Imbalanced (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step_imbalanced`
- **Description:** Generates a 12-step (13 sigmas) schedule with bias. Sigmas are spaced according to a biased cosine curve in log-sigma domain. Last sigma is 0.0.
- **Inputs:**
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1)
  - `sigma_min_positive`: `FLOAT` (default: 0.002, min: 0.0001, max: 10.0, step: 0.0001)
  - `bias_factor`: `FLOAT` (default: 0.0, min: -0.9, max: 0.9, step: 0.05)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 imbalanced cosine log-SNR spaced sigmas.

### 12-Step Chroma Sigma Schedulers

These are 12-step schedulers specifically configured with Chroma-inspired defaults.

#### Wolf Sigma Chroma Karras (12-Step)

- **Class:** `WolfSigmaChromaKarras12Step`
- **Display Name:** `Wolf Sigma Chroma Karras (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step_chroma`
- **Description:** Generates a 12-step Karras schedule using Chroma-like sigma parameters. Produces 13 sigma values.
- **Inputs:**
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1)
  - `rho`: `FLOAT` (default: 7.0, min: 0.1, max: 100.0, step: 0.1)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 Chroma Karras-spaced sigmas.

#### Wolf Sigma Chroma Biased Karras (12-Step)

- **Class:** `WolfSigmaChromaBiasedKarras12Step`
- **Display Name:** `Wolf Sigma Chroma Biased Karras (12-Step)` (Implicit from class name)
- **Category:** `sampling/sigmas_wolf/12_step_chroma_imbalanced`
- **Description:** Generates a 12-step Karras schedule with a `bias_power`, using Chroma defaults.
- **Inputs:**
  - `sigma_min`: `FLOAT` (default: 0.002, min: 0.0001, max: 100.0, step: 0.0001)
  - `sigma_max`: `FLOAT` (default: 1.0, min: 0.1, max: 1000.0, step: 0.1)
  - `rho`: `FLOAT` (default: 7.0, min: 0.1, max: 100.0, step: 0.1)
  - `bias_power`: `FLOAT` (default: 1.0, min: 0.2, max: 5.0, step: 0.05)
  - `min_epsilon_spacing`: `FLOAT` (default: 1e-7, min: 1e-9, max: 0.1, step: 1e-7)
- **Outputs:**
  - `SIGMAS`: `SIGMAS` - Tensor with 13 Chroma biased Karras-spaced sigmas.

---

## Advanced / Scriptable Sampler Generators

For advanced users who need maximum flexibility in defining sampling logic, the `Wolf Sampler Script Evaluator (🐺)` node allows for the creation of custom samplers by executing a user-provided Python script. This provides a powerful way to define complex or experimental sampling routines that can be directly used with KSampler nodes.

### Wolf Sampler Script Evaluator (🐺)

- **Class:** `WolfSamplerScriptEvaluator`
- **Display Name:** `Wolf Sampler Script Evaluator (🐺)`
- **Category:** `sampling/ksampler_wolf`
- **Description:** Evaluates a Python script to define a custom sampler function. The script must define a function named `wolf_sampler()` which, when called, returns the actual sampler function. This actual sampler function will be wrapped in a ComfyUI `SAMPLER` object compatible with KSampler nodes.
- **Inputs**:
  - `script`: `STRING` (multiline) - The Python script defining the custom sampler. See the default script content for the required structure and an example.
  - `seed` (optional): `INT` - An optional seed value that can be made available to the script if needed (e.g., via `__script_seed__` global variable).
- **Outputs**:
  - `SAMPLER`: `SAMPLER` - The custom sampler object that can be connected to a KSampler's `sampler` input.
  - `status_message`: `STRING` - A message indicating the outcome of the script evaluation (e.g., success or error details).

**Scripting Details for `Wolf Sampler Script Evaluator`**:

The Python script provided to this node must define a top-level function: `wolf_sampler()`.
This function takes no arguments and must return *another function*, which is the actual sampler implementation.

The **actual sampler function** must have the following signature:
`def actual_sampler_function(model_wrap, sigmas, extra_args, callback, noise_tensor, latent_image, denoise_mask, disable_pbar, **kwargs):`

- `model_wrap`: The ComfyUI model wrapper (usually an instance of `comfy.model_patcher.ModelPatcher` or a compatible callable). You call this with `denoised_latents = model_wrap(current_latents, current_sigma, **model_call_extra_args)`. The `model_call_extra_args` is typically a dictionary like `{'cond': extra_args['cond'], 'uncond': extra_args['uncond'], 'cond_scale': extra_args['cond_scale']}`.
- `sigmas`: A 1D tensor containing the sigma values for each step in the schedule (e.g., `[sigma_max, ..., sigma_min, 0.0]`).
- `extra_args`: A dictionary passed from the KSampler, containing:
  - `'cond'`: Positive conditioning tensor.
  - `'uncond'`: Negative conditioning tensor.
  - `'cond_scale'`: The CFG (Classifier-Free Guidance) scale value (float).
  - `'noise_seed'` (optional): An integer seed, useful if your sampler needs to generate its own noise for specific techniques (distinct from `noise_tensor`).
  - `'s_churn'`, `'s_tmin'`, `'s_tmax'`, `'s_noise'`: Common parameters for some advanced samplers.
  - `'image_cond'` (optional): Image conditioning, e.g., for ControlNets.
  - `'cfg_scale_function'` (optional): A function that can modify `cond_scale` dynamically during sampling.
- `callback`: A function to call for previews, e.g., `callback(current_step, x0_prediction, current_latents, total_steps)`.
- `noise_tensor`: The initial noise tensor for the entire sampling process (e.g., for txt2img). This is pure, unprocessed noise.
- `latent_image`: The starting latent tensor. For txt2img, this is typically the same as `noise_tensor`. For img2img or inpainting, this will be the (potentially partially noised) input latent.
- `denoise_mask` (optional): A mask tensor used for inpainting. Values are typically 0 or 1.
- `disable_pbar`: A boolean; if `True`, the progress bar should be suppressed.
- `**kwargs`: For additional or future KSampler options.

The **actual sampler function** is responsible for the sampling loop (iterating through `sigmas`) and should return the final denoised latent tensor.

Refer to the default script in the node for a practical example that wraps the standard Euler sampler.

---
