from .nodes.wolf_sigmas_get import WolfSigmasGet
from .nodes.wolf_sigmas_set import WolfSigmasSet
from .nodes.wolf_sigma_one_step_chroma_fixed import WolfSigmaOneStepChromaFixed
from .nodes.wolf_sigma_one_step_chroma_adjustable import (
    WolfSigmaOneStepChromaAdjustable,
)
from .nodes.wolf_sigma_one_step_chroma_imbalanced import (
    WolfSigmaOneStepChromaImbalanced,
)
from .nodes.wolf_sigma_four_step_chroma_imbalanced import (
    WolfSigmaFourStepChromaImbalanced,
)
from .nodes.wolf_sigma_four_step_chroma_manual import WolfSigmaFourStepChromaManual
from .nodes.wolf_sigma_chroma_karras_4_step import WolfSigmaChromaKarras4Step
from .nodes.wolf_sigma_chroma_karras_8_step import WolfSigmaChromaKarras8Step
from .nodes.wolf_sigma_ays_inspired_4_step import WolfSigmaAYSInspired4Step
from .nodes.wolf_sigma_ays_inspired_8_step import WolfSigmaAYSInspired8Step
from .nodes.wolf_sigma_log_linear_4_step import WolfSigmaLogLinear4Step
from .nodes.wolf_sigma_log_linear_8_step import WolfSigmaLogLinear8Step
from .nodes.wolf_sigma_sigmoid_4_step import WolfSigmaSigmoid4Step
from .nodes.wolf_sigma_sigmoid_8_step import WolfSigmaSigmoid8Step
from .nodes.wolf_sigma_eight_step_chroma_imbalanced import (
    WolfSigmaEightStepChromaImbalanced,
)
from .nodes.wolf_sigma_eight_step_chroma_manual import WolfSigmaEightStepChromaManual
from .nodes.wolf_sigma_cosine_log_snr_8_step import WolfSigmaCosineLogSNR8Step
from .nodes.wolf_sigma_biased_karras_8_step import WolfSigmaBiasedKarras8Step
from .nodes.wolf_sigma_power_transform import WolfSigmaPowerTransform
from .nodes.wolf_sigma_clamp_t0 import WolfSigmaClampT0
from .nodes.wolf_sigma_shift_and_scale import WolfSigmaShiftAndScale
from .nodes.wolf_sigma_normalize_range import WolfSigmaNormalizeRange
from .nodes.wolf_sigma_quantize import WolfSigmaQuantize
from .nodes.wolf_sigmas_to_json import WolfSigmasToJSON
from .nodes.wolf_sigma_arctan_n_step import WolfSigmaArctanNStep
from .nodes.wolf_sigma_linear_n_step import WolfSigmaLinearNStep
from .nodes.wolf_sigma_ays_paper_schedule_picker import WolfSigmaAYSPaperSchedulePicker
from .nodes.wolf_sigma_respace_log_cosine import WolfSigmaRespaceLogCosine
from .nodes.wolf_sigma_linear_12_step import WolfSigmaLinear12Step
from .nodes.wolf_sigma_linear_imbalanced_n_step import WolfSigmaLinearImbalancedNStep
from .nodes.wolf_sigma_linear_imbalanced_12_step import WolfSigmaLinearImbalanced12Step
from .nodes.wolf_sigma_arctan_12_step import WolfSigmaArctan12Step
from .nodes.wolf_sigma_arctan_imbalanced_n_step import WolfSigmaArctanImbalancedNStep
from .nodes.wolf_sigma_arctan_imbalanced_12_step import WolfSigmaArctanImbalanced12Step
from .nodes.wolf_sigma_karras_12_step import WolfSigmaKarras12Step
from .nodes.wolf_sigma_biased_karras_12_step import WolfSigmaBiasedKarras12Step
from .nodes.wolf_sigma_chroma_karras_12_step import WolfSigmaChromaKarras12Step
from .nodes.wolf_sigma_chroma_biased_karras_n_step import (
    WolfSigmaChromaBiasedKarrasNStep,
)
from .nodes.wolf_sigma_chroma_biased_karras_12_step import (
    WolfSigmaChromaBiasedKarras12Step,
)
from .nodes.wolf_sigma_cosine_log_snr_12_step import WolfSigmaCosineLogSNR12Step
from .nodes.wolf_sigma_cosine_log_snr_imbalanced_n_step import (
    WolfSigmaCosineLogSNRImbalancedNStep,
)
from .nodes.wolf_sigma_cosine_log_snr_imbalanced_12_step import (
    WolfSigmaCosineLogSNRImbalanced12Step,
)
from .nodes.wolf_sigma_sigmoid_12_step import WolfSigmaSigmoid12Step
from .nodes.wolf_sigma_sigmoid_imbalanced_n_step import WolfSigmaSigmoidImbalancedNStep
from .nodes.wolf_sigma_sigmoid_imbalanced_12_step import (
    WolfSigmaSigmoidImbalanced12Step,
)
from .nodes.wolf_sigma_ays_12_step import WolfSigmaAYS12Step
from .nodes.wolf_sigma_ays_imbalanced_n_step import WolfSigmaAYSImbalancedNStep
from .nodes.wolf_sigma_ays_imbalanced_12_step import WolfSigmaAYSImbalanced12Step
from .nodes.wolf_sigma_geometric_progression import WolfSigmaGeometricProgression
from .nodes.wolf_sigma_polynomial import WolfSigmaPolynomial
from .nodes.wolf_sigma_reverse import WolfSigmaReverse
from .nodes.wolf_sigma_insert_value import WolfSigmaInsertValue
from .nodes.wolf_sigma_add_noise import WolfSigmaAddNoise
from .nodes.wolf_sigma_tanh_generator import WolfSigmaTanhGenerator
from .nodes.wolf_sigma_slice import WolfSigmaSlice
from .nodes.wolf_sigma_clip_values import WolfSigmaClipValues
from .nodes.wolf_sigma_reverse_and_rescale import WolfSigmaReverseAndRescale
from .wolf_sigma_constants import AYS_CHROMA_SIGMAS_BASE
from .nodes.wolf_sigma_script_evaluator import WolfSigmaScriptEvaluator
from .nodes.wolf_sampler_script_evaluator import WolfSamplerScriptEvaluator
from .nodes.wolf_simple_sampler_script_evaluator import WolfSimpleSamplerScriptEvaluator

NODE_CLASS_MAPPINGS = {
    "WolfSigmasGet": WolfSigmasGet,
    "WolfSigmasSet": WolfSigmasSet,
    "WolfSigmaOneStepChromaFixed": WolfSigmaOneStepChromaFixed,
    "WolfSigmaOneStepChromaAdjustable": WolfSigmaOneStepChromaAdjustable,
    "WolfSigmaOneStepChromaImbalanced": WolfSigmaOneStepChromaImbalanced,
    "WolfSigmaFourStepChromaImbalanced": WolfSigmaFourStepChromaImbalanced,
    "WolfSigmaFourStepChromaManual": WolfSigmaFourStepChromaManual,
    "WolfSigmaChromaKarras4Step": WolfSigmaChromaKarras4Step,
    "WolfSigmaAYSInspired4Step": WolfSigmaAYSInspired4Step,
    "WolfSigmaLogLinear4Step": WolfSigmaLogLinear4Step,
    "WolfSigmaSigmoid4Step": WolfSigmaSigmoid4Step,
    "WolfSigmaChromaKarras8Step": WolfSigmaChromaKarras8Step,
    "WolfSigmaAYSInspired8Step": WolfSigmaAYSInspired8Step,
    "WolfSigmaLogLinear8Step": WolfSigmaLogLinear8Step,
    "WolfSigmaSigmoid8Step": WolfSigmaSigmoid8Step,
    "WolfSigmaEightStepChromaImbalanced": WolfSigmaEightStepChromaImbalanced,
    "WolfSigmaEightStepChromaManual": WolfSigmaEightStepChromaManual,
    "WolfSigmaCosineLogSNR8Step": WolfSigmaCosineLogSNR8Step,
    "WolfSigmaBiasedKarras8Step": WolfSigmaBiasedKarras8Step,
    "WolfSigmaPowerTransform": WolfSigmaPowerTransform,
    "WolfSigmaClampT0": WolfSigmaClampT0,
    "WolfSigmaShiftAndScale": WolfSigmaShiftAndScale,
    "WolfSigmaNormalizeRange": WolfSigmaNormalizeRange,
    "WolfSigmaQuantize": WolfSigmaQuantize,
    "WolfSigmasToJSON": WolfSigmasToJSON,
    "WolfSigmaArctanNStep": WolfSigmaArctanNStep,
    "WolfSigmaLinearNStep": WolfSigmaLinearNStep,
    "WolfSigmaAYSPaperSchedulePicker": WolfSigmaAYSPaperSchedulePicker,
    "WolfSigmaRespaceLogCosine": WolfSigmaRespaceLogCosine,
    "WolfSigmaLinear12Step": WolfSigmaLinear12Step,
    "WolfSigmaLinearImbalancedNStep": WolfSigmaLinearImbalancedNStep,
    "WolfSigmaLinearImbalanced12Step": WolfSigmaLinearImbalanced12Step,
    "WolfSigmaArctan12Step": WolfSigmaArctan12Step,
    "WolfSigmaArctanImbalancedNStep": WolfSigmaArctanImbalancedNStep,
    "WolfSigmaArctanImbalanced12Step": WolfSigmaArctanImbalanced12Step,
    "WolfSigmaKarras12Step": WolfSigmaKarras12Step,
    "WolfSigmaBiasedKarras12Step": WolfSigmaBiasedKarras12Step,
    "WolfSigmaChromaKarras12Step": WolfSigmaChromaKarras12Step,
    "WolfSigmaChromaBiasedKarrasNStep": WolfSigmaChromaBiasedKarrasNStep,
    "WolfSigmaChromaBiasedKarras12Step": WolfSigmaChromaBiasedKarras12Step,
    "WolfSigmaCosineLogSNR12Step": WolfSigmaCosineLogSNR12Step,
    "WolfSigmaCosineLogSNRImbalancedNStep": WolfSigmaCosineLogSNRImbalancedNStep,
    "WolfSigmaCosineLogSNRImbalanced12Step": WolfSigmaCosineLogSNRImbalanced12Step,
    "WolfSigmaSigmoid12Step": WolfSigmaSigmoid12Step,
    "WolfSigmaSigmoidImbalancedNStep": WolfSigmaSigmoidImbalancedNStep,
    "WolfSigmaSigmoidImbalanced12Step": WolfSigmaSigmoidImbalanced12Step,
    "WolfSigmaAYS12Step": WolfSigmaAYS12Step,
    "WolfSigmaAYSImbalancedNStep": WolfSigmaAYSImbalancedNStep,
    "WolfSigmaAYSImbalanced12Step": WolfSigmaAYSImbalanced12Step,
    "WolfSigmaGeometricProgression": WolfSigmaGeometricProgression,
    "WolfSigmaPolynomial": WolfSigmaPolynomial,
    "WolfSigmaReverse": WolfSigmaReverse,
    "WolfSigmaInsertValue": WolfSigmaInsertValue,
    "WolfSigmaAddNoise": WolfSigmaAddNoise,
    "WolfSigmaTanhGenerator": WolfSigmaTanhGenerator,
    "WolfSigmaSlice": WolfSigmaSlice,
    "WolfSigmaClipValues": WolfSigmaClipValues,
    "WolfSigmaReverseAndRescale": WolfSigmaReverseAndRescale,
    "WolfSigmaScriptEvaluator": WolfSigmaScriptEvaluator,
    "WolfSamplerScriptEvaluator": WolfSamplerScriptEvaluator,
    "WolfSimpleSamplerScriptEvaluator": WolfSimpleSamplerScriptEvaluator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfSigmasGet": "Get Sigmas (🐺)",
    "WolfSigmasSet": "Set Sigmas from JSON (🐺)",
    "WolfSigmaOneStepChromaFixed": "1-Step Chroma Sigmas (Fixed, 🐺)",
    "WolfSigmaOneStepChromaAdjustable": "1-Step Chroma Sigmas (Adjustable, 🐺)",
    "WolfSigmaOneStepChromaImbalanced": "1-Step Chroma Sigmas (Imbalanced, 🐺)",
    "WolfSigmaFourStepChromaImbalanced": "3-Step Sigmas (Imbalanced Karras, 🐺)",
    "WolfSigmaFourStepChromaManual": "3-Step Sigmas (Flux-Inspired, 🐺)",
    "WolfSigmaChromaKarras4Step": "Wolf Sigma Karras (4-Step)",
    "WolfSigmaAYSInspired4Step": "Wolf Sigma AYS Inspired (4-Step)",
    "WolfSigmaLogLinear4Step": "Wolf Sigma Log-Linear (4-Step)",
    "WolfSigmaSigmoid4Step": "Wolf Sigma Sigmoid (4-Step)",
    "WolfSigmaChromaKarras8Step": "Wolf Sigma Karras (8-Step)",
    "WolfSigmaAYSInspired8Step": "Wolf Sigma AYS Inspired (8-Step)",
    "WolfSigmaLogLinear8Step": "Wolf Sigma Log-Linear (8-Step)",
    "WolfSigmaSigmoid8Step": "Wolf Sigma Sigmoid (8-Step)",
    "WolfSigmaEightStepChromaImbalanced": "8-Step Sigmas (Imbalanced Karras, 🐺)",
    "WolfSigmaEightStepChromaManual": "8-Step Sigmas (Flux-Inspired, 🐺)",
    "WolfSigmaCosineLogSNR8Step": "Wolf Sigma Cosine LogSNR (8-Step)",
    "WolfSigmaBiasedKarras8Step": "Wolf Sigma Biased Karras (8-Step)",
    "WolfSigmaPowerTransform": "Wolf Sigma Power Transform",
    "WolfSigmaClampT0": "Wolf Sigma Transform (Clamp T0)",
    "WolfSigmaShiftAndScale": "Wolf Sigma Transform (Shift & Scale)",
    "WolfSigmaNormalizeRange": "Wolf Sigma Transform (Normalize Range)",
    "WolfSigmaQuantize": "Wolf Sigma Transform (Quantize)",
    "WolfSigmasToJSON": "Wolf Sigmas to JSON",
    "WolfSigmaArctanNStep": "Wolf Sigma Arctan (N-Step)",
    "WolfSigmaLinearNStep": "Wolf Sigma Linear (N-Step)",
    "WolfSigmaAYSPaperSchedulePicker": "Wolf Sigma AYS Paper Schedule",
    "WolfSigmaRespaceLogCosine": "Wolf Sigma Transform (Respace Log-Cosine)",
    "WolfSigmaLinear12Step": "Wolf Sigma Linear (12-Step)",
    "WolfSigmaLinearImbalancedNStep": "Wolf Sigma Linear Imbalanced (N-Step)",
    "WolfSigmaLinearImbalanced12Step": "Wolf Sigma Linear Imbalanced (12-Step)",
    "WolfSigmaArctan12Step": "Wolf Sigma Arctan (12-Step)",
    "WolfSigmaArctanImbalancedNStep": "Wolf Sigma Arctan Imbalanced (N-Step)",
    "WolfSigmaArctanImbalanced12Step": "Wolf Sigma Arctan Imbalanced (12-Step)",
    "WolfSigmaKarras12Step": "Wolf Sigma Karras (12-Step)",
    "WolfSigmaBiasedKarras12Step": "Wolf Sigma Biased Karras (12-Step)",
    "WolfSigmaChromaKarras12Step": "Wolf Sigma Chroma Karras (12-Step)",
    "WolfSigmaChromaBiasedKarrasNStep": "Wolf Sigma Chroma Biased Karras (N-Step)",
    "WolfSigmaChromaBiasedKarras12Step": "Wolf Sigma Chroma Biased Karras (12-Step)",
    "WolfSigmaCosineLogSNR12Step": "Wolf Sigma Cosine LogSNR (12-Step)",
    "WolfSigmaCosineLogSNRImbalancedNStep": "Wolf Sigma Cosine LogSNR Imbalanced (N-Step)",
    "WolfSigmaCosineLogSNRImbalanced12Step": "Wolf Sigma Cosine LogSNR Imbalanced (12-Step)",
    "WolfSigmaSigmoid12Step": "Wolf Sigma Sigmoid (12-Step)",
    "WolfSigmaSigmoidImbalancedNStep": "Wolf Sigma Sigmoid Imbalanced (N-Step)",
    "WolfSigmaSigmoidImbalanced12Step": "Wolf Sigma Sigmoid Imbalanced (12-Step)",
    "WolfSigmaAYS12Step": "Wolf Sigma AYS Chroma (12-Step)",
    "WolfSigmaAYSImbalancedNStep": "Wolf Sigma AYS Chroma Imbalanced (N-Step)",
    "WolfSigmaAYSImbalanced12Step": "Wolf Sigma AYS Chroma Imbalanced (12-Step)",
    "WolfSigmaGeometricProgression": "Wolf Sigma Geometric Progression",
    "WolfSigmaPolynomial": "Wolf Sigma Polynomial",
    "WolfSigmaReverse": "Sigma Schedule Reverser (🐺)",
    "WolfSigmaInsertValue": "Sigma Insert Value (🐺)",
    "WolfSigmaAddNoise": "Sigma Add Noise (🐺)",
    "WolfSigmaTanhGenerator": "Sigma Tanh Generator (🐺)",
    "WolfSigmaSlice": "Sigma Slice (🐺)",
    "WolfSigmaClipValues": "Sigma Clip Values (🐺)",
    "WolfSigmaReverseAndRescale": "Wolf Sigma Reverse and Rescale",
    "WolfSigmaScriptEvaluator": "Wolf Sigma Script Evaluator (🐺)",
    "WolfSamplerScriptEvaluator": "Wolf Sampler Script Evaluator (🐺)",
    "WolfSimpleSamplerScriptEvaluator": "Wolf Simple Sampler Script (🐺)",
}
