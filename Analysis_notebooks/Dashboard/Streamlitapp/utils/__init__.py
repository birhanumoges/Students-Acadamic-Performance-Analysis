"""
Utils package for Ethiopian Student Performance Dashboard
"""

from .data_processor import (
    DataProcessor,
    COLOR_SCHEME,
    load_and_preprocess_data,
    encode_categorical_features,
    prepare_target_encoders,
    process_raw_input_for_prediction
)

from .visualizations import (
    Visualizer,
    initialize_global_vars,
    set_global_data,
    set_prediction_result
)

from .predictions import (
    PredictionEngine,
    load_models,
    make_prediction_corrected,
    get_national_exam_model_performance,
    get_national_exam_feature_importance
)

__all__ = [
    'DataProcessor',
    'PredictionEngine',
    'Visualizer',
    'COLOR_SCHEME',
    'load_and_preprocess_data',
    'encode_categorical_features',
    'prepare_target_encoders',
    'process_raw_input_for_prediction',
    'initialize_global_vars',
    'set_global_data',
    'set_prediction_result',
    'load_models',
    'make_prediction_corrected',
    'get_national_exam_model_performance',
    'get_national_exam_feature_importance'
]