"""
Grouped Statistics Calculator

This module provides robust calculation of grouped statistics
(mean, standard deviation, percentiles) with comprehensive error
handling, performance optimization, and multiple output formats.
"""

from .models import ParamsData, ParamsMean, PercentileParams, ParamsPlot
from .api import (
    sta_eda_grouped_mean,
    sta_eda_grouped_std,
    sta_eda_grouped_percentiles
)

__all__ = [
    # models
    'ParamsData',
    'ParamsMean',
    'PercentileParams',
    'ParamsPlot',
    
    # api
    'sta_eda_grouped_mean',
    'sta_eda_grouped_std',
    'sta_eda_grouped_percentiles'
]