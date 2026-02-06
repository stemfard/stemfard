from functools import cached_property

from numpy import around, float64
from numpy.typing import NDArray

from stemfard.stats.regression_linear._classes import (
    LinearRegressionStatistics, LinearRegressionInferences,
    LinearRegressionMatrices, LinearRegressionModelFitter,
    LinearRegressionModelFormatter, LinearRegressionParamsParser,
    LinearRegressionVariances
)


class BaseLinearRegression:
    """
    Single entry-point class.
    Allows calling attributes as:
        self.x, self.y, self.n, self.beta, self.mse, ...
    """

    def __init__(self, **kwargs):

        self.params = LinearRegressionParamsParser.parse(**kwargs)

        self.dstats = LinearRegressionStatistics(params=self.params)
        
        self.array_comp = LinearRegressionMatrices(
            params=self.params,
            dstats=self.dstats
        )

        self.variances = LinearRegressionVariances(
            array_comp=self.array_comp,
            dstats=self.dstats
        )
        
        self.inferences = LinearRegressionInferences(
            params=self.params,
            dstats=self.dstats,
            array_comp=self.array_comp,
            variance_calc=self.variances
        )
        
        self.fitter = LinearRegressionModelFitter(
            params=self.params,
            array_comp=self.array_comp
        )
        
        self.formatter = LinearRegressionModelFormatter(
            params=self.params,
            array_comp=self.array_comp
        )
        
        # used with `.setter` to allow modification
        self._counter = 0
        self._caption_num = 0
        self._appendix_counter = 0

        # ---- Attribute flattening - enables self.x, self.y, self.beta, etc
        
        self._bind_attributes()
    
    @property
    def is_simple_linear(self) -> bool:
        return (
            self.params.slinear_method == "normal" and
            (self.params.x.ndim == 1 or self.params.x.shape[1] == 1)
        )
        
    @property
    def counter(self) -> int:
        return self._counter
    
    @counter.setter
    def counter(self, increment: int) -> int:
        self._counter = increment
        
    @property
    def caption_num(self) -> int:
        return self._caption_num
    
    @caption_num.setter
    def caption_num(self, increment: int) -> int:
        self._caption_num = increment
    
    @property
    def appendix_counter(self) -> int:
        return self._counter
    
    @appendix_counter.setter
    def appendix_counter(self, increment: int) -> int:
        self._appendix_counter = increment
        
    @cached_property
    def sb_denom(self) -> float:
        return float(self.sum_x_squared - self.n * self.mean_x ** 2)
    
    @cached_property
    def sb_denom_rnd(self) -> float:
        return float(around(self.sb_denom, self.decimals))
    
    def predict(self, x, decimals) -> NDArray[float64]:
        """Make predictions for new x values"""
        return self.fitter.predict(xvals=x, decimals=decimals)

    # ===============================
    # Attribute flattening
    # ===============================
    
    def _bind_attributes(self) -> None:
        """Expose frequently-used attributes at top level"""

        # Raw data
        self.decimals = self.params.decimals
        
        self.x = self.params.x
        self.x_rnd = around(self.x, self.decimals)
        
        self.y = self.params.y
        self.y_rnd = around(self.y, self.decimals)
        
        self.slinear_method = self.params.slinear_method
        self.conf_level = self.params.conf_level
        self.statistics = self.params.statistics
        
        self.predict_at_x = self.params.predict_at_x
        if self.predict_at_x is not None:
            self.predict_at_x_rnd = around(self.predict_at_x, self.decimals)
        
        self.steps_compute = self.params.steps_compute
        self.steps_detailed = self.params.steps_detailed
        self.steps_bg = self.params.steps_bg
        
        # Descriptive statistics and related
        self.n = self.dstats.n
        self.k = self.dstats.k
        # ------------------------------------
        self.x_squared = self.dstats.x_squared
        self.x_squared_rnd = around(self.x_squared, self.decimals)
        
        self.xy = self.dstats.xy
        self.xy_rnd = around(self.xy, self.decimals)
        #-------------------------------------------
        self.sum_x = self.dstats.sum_x
        self.sum_x_rnd = around(self.sum_x, self.decimals)
        
        self.sum_y = self.dstats.sum_y
        self.sum_y_rnd = around(self.sum_y, self.decimals)
        
        self.sum_x_squared = self.dstats.sum_x_squared
        self.sum_x_squared_rnd = around(self.sum_x_squared, self.decimals)
        
        self.sum_xy = self.dstats.sum_xy
        self.sum_xy_rnd = around(self.sum_xy, self.decimals)
        # --------------------------------------------------
        self.mean_x = self.dstats.mean_x
        self.mean_x_rnd = around(self.mean_x, self.decimals)
        
        self.mean_y = self.dstats.mean_y
        self.mean_y_rnd = around(self.mean_y, self.decimals)
        
        # inferences
        self.beta = self.array_comp.beta
        self.beta_rnd = around(self.beta, self.decimals)
        
        self.beta0 = self.beta[0, 0]
        self.beta0_rnd = around(self.beta0, self.decimals)
        
        self.beta1 = self.beta[1, 0]
        self.beta1_rnd = around(self.beta1, self.decimals)
        
        self.X = self.array_comp.X
        self.X_rnd = around(self.X, self.decimals)
        
        self.Y = self.array_comp.Y
        self.Y_rnd = around(self.Y, self.decimals)
        
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
        
        self.bTXTY = self.array_comp.bTXTY
        self.bTXTY_rnd = around(self.bTXTY, self.decimals)
        
        self.YT_N_Y = self.array_comp.YT_N_Y
        self.YT_N_Y_rnd = around(self.YT_N_Y, self.decimals)

        # Variance / ANOVA
        self.ssr = self.variances.ssr
        self.ssr_rnd = around(self.ssr, self.decimals)
        
        self.sse = self.variances.sse
        self.sse_rnd = around(self.sse, self.decimals)
        
        self.sst = self.variances.sst
        self.sst_rnd = around(self.sst, self.decimals)
        #---------------------------------------------
        self.msr = self.variances.msr
        self.msr_rnd = around(self.msr, self.decimals)
        
        self.mse = self.variances.mse
        self.mse_rnd = around(self.mse, self.decimals)
        
        self.mst = self.variances.mst
        self.mst_rnd = around(self.mst, self.decimals)
        # --------------------------------------------
        self.df_r = self.variances.df_r
        self.df_e = self.variances.df_e
        self.df_t = self.variances.df_t
        # --------------------------------------
        self.r_squared = self.variances.r_squared
        self.r_squared_rnd = around(self.r_squared, self.decimals)
        
        self.adjusted_r_squared = self.variances.adjusted_r_squared
        self.adjusted_r_squared_rnd = around(
            self.adjusted_r_squared, self.decimals
        )
        
        self.root_mse = self.variances.root_mse
        self.root_mse_rnd = around(self.root_mse, self.decimals)
        # ------------------------------------------------------
        self.log_likelihood = self.variances.log_likelihood
        self.log_likelihood_rnd = around(self.log_likelihood, self.decimals)
        
        self.aic = self.variances.aic
        self.aic_rnd = around(self.aic, self.decimals)
        
        self.bic = self.variances.bic
        self.bic_rnd = around(self.bic, self.decimals)
        
        # Inference
        self.alpha = self.inferences.alpha
        self.alpha_rnd = around(self.alpha, self.decimals)
        
        self.cov_beta = self.inferences.cov_beta
        self.cov_beta_rnd = around(self.cov_beta, self.decimals)
        
        self.stderrors = self.inferences.stderrors
        self.stderrors_rnd = around(self.stderrors, self.decimals)
        # --------------------------------------------------------
        self.sb0 = self.stderrors[0, 0]
        self.sb0_rnd = around(self.sb0, self.decimals)
        
        self.sb1 = self.stderrors[1, 0]
        self.sb1_rnd = around(self.sb1, self.decimals)
        # --------------------------------------------
        self.t_statistics = self.inferences.t_statistics
        self.t_statistics_rnd = around(self.t_statistics, self.decimals)
        
        self.p_values = self.inferences.p_values
        self.p_values_rnd = around(self.p_values, self.decimals)
        
        self.confidence_intervals = self.inferences.confidence_intervals
        self.confidence_intervals_rnd = around(
            self.confidence_intervals, self.decimals
        )
        
        self.f_statistic_and_p_value = self.inferences.f_statistic_and_p_value
        self.f_statistic = self.f_statistic_and_p_value[0]
        self.f_statistic_rnd = around(self.f_statistic, self.decimals)
        
        self.f_pvalue = self.f_statistic_and_p_value[1]
        self.f_pvalue_rnd = around(self.f_pvalue, self.decimals)
        
        self.t_critical = self.inferences.t_critical
        self.t_critical_rnd = around(self.t_critical, self.decimals)
        
        self.f_critical = self.inferences.f_critical
        self.f_critical_rnd = around(self.f_critical, self.decimals)
        
        self.fitted_values = self.fitter.fitted_values
        self.fitted_values_rnd = around(self.fitted_values, self.decimals)
        
        self.sum_fitted_values = self.fitter.sum_fitted_values
        self.sum_fitted_values_rnd = around(
            self.sum_fitted_values, self.decimals
        )
        
        self.residuals = self.fitter.residuals
        self.residuals_rnd = around(self.residuals, self.decimals)
        
        self.residuals_squared = self.fitter.residuals_squared
        self.residuals_squared_rnd = around(
            self.residuals_squared, self.decimals
        )
        
        self.sum_residuals_squared = self.fitter.sum_residuals_squared
        self.sum_residuals_squared_rnd = around(
            self.sum_residuals_squared, self.decimals
        )
        
        self.diff_residuals = self.fitter.diff_residuals
        self.diff_residuals_rnd = around(self.diff_residuals, self.decimals)
        
        self.durbin_watson = self.fitter.durbin_watson
        self.durbin_watson_rnd = around(self.durbin_watson, self.decimals)
        
        self.omnibus = self.fitter.omnibus
        self.omnibus_k2 = self.omnibus[0]
        self.omnibus_k2_rnd = around(self.omnibus_k2, self.decimals)
        
        self.omnibus_k2_pvalue = self.omnibus[1]
        self.omnibus_k2_pvalue_rnd = around(
            self.omnibus_k2_pvalue, self.decimals
        )
        # ----------------------------------------------------
        self.y_minus_yhat_pow2 = self.fitter.y_minus_yhat_pow2
        self.y_minus_yhat_pow2_rnd = around(
            self.y_minus_yhat_pow2, self.decimals
        )
        self.y_minus_yhat_pow3 = self.fitter.y_minus_yhat_pow3
        self.y_minus_yhat_pow3_rnd = around(
            self.y_minus_yhat_pow3, self.decimals
        )
        self.y_minus_yhat_pow4 = self.fitter.y_minus_yhat_pow4
        self.y_minus_yhat_pow4_rnd = around(
            self.y_minus_yhat_pow4, self.decimals
        )
        # ------------------------------------------------------------
        self.sum_y_minus_yhat_pow2 = self.fitter.sum_y_minus_yhat_pow2
        self.sum_y_minus_yhat_pow2_rnd = around(
            self.sum_y_minus_yhat_pow2, self.decimals
        )
        self.sum_y_minus_yhat_pow3 = self.fitter.sum_y_minus_yhat_pow3
        self.sum_y_minus_yhat_pow3_rnd = around(
            self.sum_y_minus_yhat_pow3, self.decimals
        )
        self.sum_y_minus_yhat_pow4 = self.fitter.sum_y_minus_yhat_pow4
        self.sum_y_minus_yhat_pow4_rnd = around(
            self.sum_y_minus_yhat_pow4, self.decimals
        )
        # -------------------------------------------
        self.skew_kurt = self.fitter.skew_kurt
        self.skew = self.skew_kurt[0]
        self.skew_rnd = around(self.skew, self.decimals)
        
        self.kurt = self.skew_kurt[1]
        self.kurt_rnd = around(self.kurt, self.decimals)
        # ----------------------------------------------
        self.jarque_bera_pval = self.fitter.jarque_bera
        self.jarque_bera = self.jarque_bera_pval[0]
        self.jarque_bera_rnd = around(self.jarque_bera, self.decimals)
        
        self.jarque_bera_pvalue = self.jarque_bera_pval[1]
        self.jarque_bera_pvalue_rnd = around(
            self.jarque_bera_pvalue, self.decimals
        )
        # ----------------------------------------
        self.cond_number = self.fitter.cond_number
        self.cond_number_rnd = around(self.cond_number, self.decimals)

        # Formatting
        # ----------
        self.model_string = self.formatter.model_string
        self.model_latex = self.formatter.model_latex
        self.beta_symbols = self.formatter.beta_symbols
        self.xi_symbols_tuple = self.formatter.xi_symbols
        self.xi_symbols = self.xi_symbols_tuple[0]
        self.xi_symbols_str = self.xi_symbols_tuple[1]
        self.xi_latex = self.xi_symbols_tuple[2]
        self.xi_latex_str = self.xi_symbols_tuple[3]