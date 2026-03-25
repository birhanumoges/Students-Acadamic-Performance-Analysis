"""
Utils package for Ethiopian Student Performance Dashboard
r"C:/Users/DELL/Documents/project_data/ethiopian_students_dataset.csv"
"""

# utils/__init__.py
from .data_processor import DataProcessor
from .predictions import PredictionEngine
from .visualizations import Visualizer

__all__ = ['DataProcessor', 'PredictionEngine', 'Visualizer']