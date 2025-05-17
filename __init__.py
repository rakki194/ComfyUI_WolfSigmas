from .nodes.wolf_sigmas_get import WolfSigmasGet
from .nodes.wolf_sigmas_set import WolfSigmasSet
from .nodes.wolf_sigma_power_transform import WolfSigmaPowerTransform
from .nodes.wolf_sigma_clamp_t0 import WolfSigmaClampT0
from .nodes.wolf_sigma_shift_and_scale import WolfSigmaShiftAndScale
from .nodes.wolf_sigma_normalize_range import WolfSigmaNormalizeRange
from .nodes.wolf_sigma_quantize import WolfSigmaQuantize
from .nodes.wolf_sigmas_to_json import WolfSigmasToJSON
from .nodes.wolf_sigma_respace_log_cosine import WolfSigmaRespaceLogCosine
from .nodes.wolf_sigma_geometric_progression import WolfSigmaGeometricProgression
from .nodes.wolf_sigma_polynomial import WolfSigmaPolynomial
from .nodes.wolf_sigma_reverse import WolfSigmaReverse
from .nodes.wolf_sigma_insert_value import WolfSigmaInsertValue
from .nodes.wolf_sigma_add_noise import WolfSigmaAddNoise
from .nodes.wolf_sigma_tanh_generator import WolfSigmaTanhGenerator
from .nodes.wolf_sigma_slice import WolfSigmaSlice
from .nodes.wolf_sigma_clip_values import WolfSigmaClipValues
from .nodes.wolf_sigma_reverse_and_rescale import WolfSigmaReverseAndRescale
from .nodes.wolf_sigma_script_evaluator import WolfSigmaScriptEvaluator
from .nodes.wolf_sampler_script_evaluator import WolfSamplerScriptEvaluator
from .nodes.wolf_simple_sampler_script_evaluator import WolfSimpleSamplerScriptEvaluator
from .nodes.wolf_scriptable_empty_latent import WolfScriptableEmptyLatent
from .nodes.wolf_simple_scriptable_empty_latent import WolfSimpleScriptableEmptyLatent
from .nodes.wolf_scriptable_noise import WolfScriptableNoise

NODE_CLASS_MAPPINGS = {
    "WolfSigmasGet": WolfSigmasGet,
    "WolfSigmasSet": WolfSigmasSet,
    "WolfSigmaPowerTransform": WolfSigmaPowerTransform,
    "WolfSigmaClampT0": WolfSigmaClampT0,
    "WolfSigmaShiftAndScale": WolfSigmaShiftAndScale,
    "WolfSigmaNormalizeRange": WolfSigmaNormalizeRange,
    "WolfSigmaQuantize": WolfSigmaQuantize,
    "WolfSigmasToJSON": WolfSigmasToJSON,
    "WolfSigmaRespaceLogCosine": WolfSigmaRespaceLogCosine,
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
    "WolfScriptableEmptyLatent": WolfScriptableEmptyLatent,
    "WolfSimpleScriptableEmptyLatent": WolfSimpleScriptableEmptyLatent,
    "WolfScriptableNoise": WolfScriptableNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WolfSigmasGet": "Get Sigmas (🐺)",
    "WolfSigmasSet": "Set Sigmas from JSON (🐺)",
    "WolfSigmaPowerTransform": "Wolf Sigma Power Transform",
    "WolfSigmaClampT0": "Wolf Sigma Transform (Clamp T0)",
    "WolfSigmaShiftAndScale": "Wolf Sigma Transform (Shift & Scale)",
    "WolfSigmaNormalizeRange": "Wolf Sigma Transform (Normalize Range)",
    "WolfSigmaQuantize": "Wolf Sigma Transform (Quantize)",
    "WolfSigmasToJSON": "Wolf Sigmas to JSON",
    "WolfSigmaRespaceLogCosine": "Wolf Sigma Transform (Respace Log-Cosine)",
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
    "WolfScriptableEmptyLatent": "Scriptable Empty Latent (Perlin) (🐺)",
    "WolfSimpleScriptableEmptyLatent": "Simple Scriptable Empty Latent (🐺)",
    "WolfScriptableNoise": "Scriptable Noise (🐺)",
}
