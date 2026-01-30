from numpy import around, float64
from numpy.typing import NDArray

from stemfard.stats.regression_linear.core_classes import (
    LRDescriptiveStatistics, InferenceCalculator, MatrixComputations,
    ModelFitter, ModelFormatter, LinearRegressionParameterParser,
    SquaresSumAndMean
)


class BaseLinearRegression:
    """
    Single entry-point class.
    All commonly used attributes are available as:
    self.x, self.y, self.n, self.beta, self.mse, ...
    """

    def __init__(self, **kwargs):

        self.params = LinearRegressionParameterParser.parse(**kwargs)

        self.dstats = LRDescriptiveStatistics(params=self.params)
        
        self.array_comp = MatrixComputations(
            params=self.params, dstats=self.dstats
        )

        self.variance = SquaresSumAndMean(
            array_comp=self.array_comp, dstats=self.dstats
        )
        
        self.inference = InferenceCalculator(
            params=self.params,
            dstats=self.dstats,
            array_comp=self.array_comp,
            variance_calc=self.variance
        )
        
        self.fitter = ModelFitter(
            params=self.params, array_comp=self.array_comp
        )
        
        self.formatter = ModelFormatter(
            params=self.params,
            array_comp=self.array_comp
        )

        # ---- Attribute flattening - enables e.g. self.x, self.y, etc
        
        self._bind_common_attributes()
    
    @property
    def is_simple_linear(self) -> bool:
        return (
            self.params.simple_linear_method == "normal" and
            (self.params.x.ndim == 1 or self.params.x.shape[1] == 1)
        )

    # ===============================
    # Attribute flattening
    # ===============================
    
    def _bind_common_attributes(self) -> None:
        """Expose frequently-used attributes at top level"""

        # Raw data
        self.decimals = self.params.decimals
        
        self.x = self.params.x
        self.x_rnd = around(self.x, self.decimals)
        
        self.y = self.params.y
        self.y_rnd = around(self.y, self.decimals)
        
        self.simple_linear_method = self.params.simple_linear_method
        self.conf_level = self.params.conf_level
        self.coefficients = self.params.coefficients
        self.standard_errors = self.params.standard_errors
        self.tstats = self.params.tstats
        self.pvalues = self.params.pvalues
        self.confidence_intervals = self.params.confidence_intervals
        self.anova = self.params.anova
        self.fitted_and_residuals = self.params.fitted_and_residuals
        self.others = self.params.others
        
        self.predict_at_x = self.params.predict_at_x
        if self.predict_at_x is not None:
            self.predict_at_x_rnd = around(self.predict_at_x, self.decimals)
        
        self.steps_compute = self.params.steps_compute
        self.steps_detailed = self.params.steps_detailed
        self.show_bg = self.params.show_bg
        
        # Descriptive statistics and related
        self.n = self.dstats.n
        self.k = self.dstats.k
        # ------------------------------------
        self.x_sq = self.dstats.x_sq
        self.x_sq_rnd = around(self.x_sq, self.decimals)
        
        self.xy = self.dstats.xy
        self.xy_rnd = around(self.xy, self.decimals)
        #-----------------------------
        self.sum_x = self.dstats.sum_x
        self.sum_x_rnd = around(self.sum_x, self.decimals)
        
        self.sum_y = self.dstats.sum_y
        self.sum_y_rnd = around(self.sum_y, self.decimals)
        
        self.sum_x_sq = self.dstats.sum_x_sq
        self.sum_x_sq_rnd = around(self.sum_x_sq, self.decimals)
        
        self.sum_xy = self.dstats.sum_xy
        self.sum_xy_rnd = around(self.sum_xy, self.decimals)
        # ------------------------------
        self.mean_x = self.dstats.mean_x
        self.mean_x_rnd = around(self.mean_x, self.decimals)
        
        self.mean_y = self.dstats.mean_y
        self.mean_y_rnd = around(self.mean_y, self.decimals)
        # ------------------------------
        
        self.beta0 = self.array_comp.beta0
        self.beta0_rnd = around(self.beta0, self.decimals)
        
        self.intercept = self.beta0  # Alias
        self.intercept_rnd = around(self.beta0, self.decimals)
        
        self.beta1 = self.array_comp.beta1
        self.beta1_rnd = around(self.beta1, self.decimals)
        
        self.slope = self.beta1  # Alias
        self.slope_rnd = around(self.slope, self.decimals)
        
        self.sb_denom = self.array_comp.sb_denom
        self.sb_denom_rnd = around(self.sb_denom, self.decimals)
        
        self.sb0 = self.array_comp.sb0
        self.sb0_rnd = around(self.sb0, self.decimals)
        
        self.sb1 = self.array_comp.sb1
        self.sb1_rnd = around(self.sb1, self.decimals)
        
        self.beta = self.array_comp.beta
        self.beta_rnd = around(self.beta, self.decimals)
        
        self.coefficients = self.beta  # Alias
        self.coefficients_rnd = around(self.coefficients, self.decimals)
        
        self.X = self.array_comp.X
        self.X_rnd = around(self.X, self.decimals)
        
        self.Y = self.array_comp.Y
        self.Y_rnd = around(self.Y, self.decimals)
        
        self.Q = self.array_comp.Q
        self.Q_rnd = around(self.Q, self.decimals)
        
        self.R = self.array_comp.R
        self.R_rnd = around(self.R, self.decimals)
        
        self.XT = self.array_comp.XT
        self.XT_rnd = around(self.XT, self.decimals)
        
        self.XTX = self.array_comp.XTX
        self.XTX_rnd = around(self.XTX, self.decimals)
        
        self.XTX_inv = self.array_comp.XTX_inv
        self.XTX_inv_rnd = around(self.XTX_inv, self.decimals)
        
        self.XTY = self.array_comp.XTY
        self.XTY_rnd = around(self.XTY, self.decimals)
        
        self.YT = self.array_comp.YT
        self.YT_rnd = around(self.YT, self.decimals)
        
        self.YTY = self.array_comp.YTY
        self.YTY_rnd = around(self.YTY, self.decimals)
        
        self.beta = self.array_comp.beta
        self.beta_rnd = around(self.beta, self.decimals)
        
        self.bTXTY = self.array_comp.bTXTY
        self.bTXTY_rnd = around(self.bTXTY, self.decimals)
        
        self.YT_N_Y = self.array_comp.YT_N_Y
        self.YT_N_Y_rnd = around(self.YT_N_Y, self.decimals)

        # Variance / ANOVA
        self.ssr = self.variance.ssr
        self.ssr_rnd = around(self.ssr, self.decimals)
        
        self.sse = self.variance.sse
        self.sse_rnd = around(self.sse, self.decimals)
        
        self.sst = self.variance.sst
        self.sst_rnd = around(self.sst, self.decimals)
        
        #---------------------------
        self.msr = self.variance.msr
        self.msr_rnd = around(self.msr, self.decimals)
        
        self.mse = self.variance.mse
        self.mse_rnd = around(self.mse, self.decimals)
        
        self.mst = self.variance.mst
        self.mst_rnd = around(self.mst, self.decimals)
        
        self.cov_beta = self.inference.cov_beta
        self.cov_beta_rnd = around(self.beta, self.decimals)
        
        # Inference
        self.alpha = self.inference.alpha
        
        self.standard_errors = self.inference.standard_errors
        self.standard_errors_rnd = around(self.standard_errors, self.decimals)
        
        self.t_statistics = self.inference.t_statistics
        self.t_statistics_rnd = around(self.t_statistics, self.decimals)
        
        self.p_values = self.inference.p_values
        self.p_values_rnd = around(self.p_values, self.decimals)
        
        self.confidence_intervals = self.inference.confidence_intervals
        self.confidence_intervals_rnd = around(self.confidence_intervals, self.decimals)
        
        self.f_statistic = self.inference.f_statistic
        self.f_statistic_rnd = around(self.f_statistic, self.decimals)
        
        self.f_pvalue = self.inference.f_pvalue
        self.f_pvalue_rnd = around(self.f_pvalue, self.decimals)
        
        self.t_critical = self.inference.t_critical
        self.t_critical_rnd = around(self.t_critical, self.decimals)
        
        self.f_critical = self.inference.f_critical
        self.f_critical_rnd = around(self.f_critical, self.decimals)
        
        # Fitted values
        self.fitted_values = self.fitter.fitted_values
        self.fitted_values_rnd = around(self.fitted_values, self.decimals)
        
        self.predictions = self.fitted_values  # Alias
        self.predictions_rnd = around(self.predictions, self.decimals)
        
        self.residuals = self.fitter.residuals
        self.residuals_rnd = around(self.residuals, self.decimals)
        
        self.residuals_squared = self.fitter.residuals_squared
        self.residuals_squared_rnd = around(self.residuals_squared, self.decimals)
        
        self.sum_residuals_squared = self.fitter.sum_residuals_squared
        self.sum_residuals_squared = around(
            self.sum_residuals_squared, self.decimals
        )
        
        # predict is added below as a function

        # Formatting
        self.model_string = self.formatter.model_string
        # self.model_latex = self.formatter.model_latex
        self.beta_symbols = self.formatter.beta_symbols
        self.xi_symbols = self.formatter.xi_symbols
    
    # ========================================
    # Computed Properties (for derived values)
    # ========================================

        self._counter = 0
        
    @property
    def counter(self) -> int:
        return self._counter
    
    @counter.setter
    def counter(self, increment: int) -> int:
        self._counter = increment
        
    # ===============================
    # Public API
    # ===============================
    
    def predict(self, x, decimals) -> NDArray[float64]:
        """Make predictions for new x values"""
        return self.fitter.predict(xvals=x, decimals=decimals)