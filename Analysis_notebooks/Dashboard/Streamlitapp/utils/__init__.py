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
    set_prediction_result,
    create_datatype_bar_plot,
    create_feature_category_plot,
    create_correlation_heatmap,
    create_regression_comparison_plot,
    create_actual_vs_predicted_plot,
    create_feature_importance_plot,
    create_national_exam_model_comparison_plot,
    create_national_exam_feature_importance_plot,
    create_national_exam_performance_table,
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_cluster_distribution_plot,
    create_regional_risk_plot,
    create_score_distribution_plot,
    create_students_by_region_plot,
    create_feature_summary_table,
    create_risk_distribution_plot,
    create_shap_summary_plot,
    create_shap_global_plot_classification,
    create_regional_cluster_heatmap,
    create_regional_cluster_barchart,
    create_recommendations_summary_table
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
    'create_datatype_bar_plot',
    'create_feature_category_plot',
    'create_correlation_heatmap',
    'create_regression_comparison_plot',
    'create_actual_vs_predicted_plot',
    'create_feature_importance_plot',
    'create_national_exam_model_comparison_plot',
    'create_national_exam_feature_importance_plot',
    'create_national_exam_performance_table',
    'create_confusion_matrix_plot',
    'create_roc_curve_plot',
    'create_cluster_distribution_plot',
    'create_regional_risk_plot',
    'create_score_distribution_plot',
    'create_students_by_region_plot',
    'create_feature_summary_table',
    'create_risk_distribution_plot',
    'create_shap_summary_plot',
    'create_shap_global_plot_classification',
    'create_regional_cluster_heatmap',
    'create_regional_cluster_barchart',
    'create_recommendations_summary_table',
    'load_models',
    'make_prediction_corrected',
    'get_national_exam_model_performance',
    'get_national_exam_feature_importance'
]