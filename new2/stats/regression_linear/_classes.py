from functools import cached_property
from typing import Literal

from numpy import (
    around, asarray, diag, diff, exp, eye, float64, hstack, log, ones, pi, sqrt
)
from numpy.linalg import cond, solve, qr
from numpy.typing import NDArray
from scipy import stats
from sympy import latex, sympify

from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.stats.regression_linear._params_parser import (
    LinearRegressionParsedParams, linear_regression_parser
)

# ================ PARAMETER MANAGEMENT ================

class LinearRegressionParamsParser:
    """Handles input parsing and validation"""
    
    @staticmethod
    def parse(
        x: SequenceArrayLike,
        y: SequenceArrayLike,
        slinear_method: Literal["matrix", "normal"] = "normal",
        slinear_formula: Literal["raw", "expanded"] = "expanded",
        conf_level: float = 0.95,
        statistics: list[str] = ["beta"],
        predict_at_x: NDArray[float64] | None = None,
        steps_compute: bool = True,
        steps_detailed: bool = True,
        steps_bg: bool = True,
        decimals: int = 12,
    ) -> LinearRegressionParsedParams:
        """Parse and validate input parameters"""
        
        parsed_params = linear_regression_parser(
            x=x,
            y=y,
            slinear_method=slinear_method,
            slinear_formula=slinear_formula,
            conf_level=conf_level,
            statistics=statistics,
            predict_at_x=predict_at_x,
            steps_compute=steps_compute,
            steps_detailed=steps_detailed,
            steps_bg=steps_bg,
            decimals=decimals
        )
        
        return parsed_params
        

# ================ MATRIX COMPUTATIONS ================

class DesignMatrixBuilder:
    """Builds design matrices with intercept"""
    
    @staticmethod
    def with_intercept(x: NDArray[float64]) -> NDArray[float64]:
        """Add intercept column to design matrix"""
        return hstack((ones((x.shape[0], 1)), x))
    

class LinearRegressionStatistics:
    def __init__(self, params: LinearRegressionParsedParams):
        """Descriptive statistics: n, k, mean, sum"""
        self.params = params
    
    @property
    def n(self) -> int:
        return len(self.params.x)

    @property
    def k(self) -> int:
        ncols = self.params.x.shape[1]
        return ncols + 1 # i.e. independent + dependent
    
    @cached_property
    def x_squared(self) -> NDArray[float64]:
        return self.params.x ** 2
    
    @cached_property
    def sum_x(self) -> float:
        return float(self.params.x.sum())
    
    @cached_property
    def sum_x_squared(self) -> float:
        return float(self.x_squared.sum())
    
    @cached_property
    def mean_x(self) -> float:
        return float(self.params.x.mean())
    
    @cached_property
    def sum_y(self) -> float:
        return float(self.params.y.sum())
    
    @cached_property
    def mean_y(self) -> float:
        return float(self.params.y.mean())
    
    @cached_property
    def xy(self) -> NDArray[float64]:
        return self.params.x * self.params.y
    
    @cached_property
    def sum_xy(self) -> float:
        return float(self.xy.sum())


class LinearRegressionMatrices:
    """QR-based matrix computations"""
    
    def __init__(
        self, params: LinearRegressionParsedParams,
        dstats: LinearRegressionStatistics
    ):
        self.params = params
        self.dstats = dstats
        self.design_matrix = DesignMatrixBuilder.with_intercept(x=params.x)
        
    @cached_property
    def X(self) -> NDArray[float64]:
        """Design matrix"""
        return self.design_matrix
    
    @cached_property
    def Y(self) -> NDArray[float64]:
        """Response vector"""
        return self.params.y
    
    @cached_property
    def _qr_decomposition(self) -> tuple[NDArray[float64], NDArray[float64]]:
        """QR decomposition of design matrix"""
        return qr(self.X)
    
    @property
    def _Q(self) -> NDArray[float64]:
        return self._qr_decomposition[0]
    
    @property
    def _R(self) -> NDArray[float64]:
        return self._qr_decomposition[1]
    
    @cached_property
    def XT(self) -> NDArray[float64]:
        return self.X.T
    
    @cached_property
    def XTX(self) -> NDArray[float64]:
        return self.XT @ self.X
    
    @cached_property
    def XTY(self) -> NDArray[float64]:
        return self.XT @ self.Y
    
    @cached_property
    def YT(self) -> NDArray[float64]:
        return self.Y.T
    
    @cached_property
    def YTY(self) -> float:
        return float((self.YT @ self.Y)[0, 0])
    
    @cached_property
    def XTX_inv(self) -> NDArray[float64]:
        """
        Inverse of X.T @ X using QR stability
        `inv()` is not stable
        """
        R_inv = solve(self._R, eye(self._R.shape[0]))
        return R_inv @ R_inv.T
    
    @cached_property
    def beta(self) -> NDArray[float64]:
        """
        Coefficient estimates using QR solve
        b = inv(X.T @ X) @ (X.T @ Y)
        """
        return solve(self._R, self._Q.T @ self.Y)
    
    @cached_property
    def bT(self) -> NDArray[float64]:
        return self.beta.T
    
    @cached_property
    def bTXTY(self) -> float:
        """b.T @ X.T @ Y"""
        return float((self.bT @ self.XTY)[0, 0])
    
    @cached_property
    def YT_N_Y(self) -> NDArray[float64]:
        """
        Centered calculations
        Y.T @ ones_matrix @ Y
        """
        return self.Y.sum() ** 2 # beta than Y.T @ ones_matrix @ Y


# ================ STATISTICAL COMPUTATIONS ================

class LinearRegressionVariances:
    """Calculates variances and sum of squares"""
    
    def __init__(
        self,
        array_comp: LinearRegressionMatrices,
        dstats: LinearRegressionStatistics
        ):
        self.ndarrays = array_comp
        self.dstats = dstats
    
    @cached_property
    def sst(self) -> float:
        """
        Total sum of squares
        Y.T @ Y - (1 / n) * Y.T @ ones_nd_array @ Y
        """
        sst = self.ndarrays.YTY - (1 / self.dstats.n) * self.ndarrays.YT_N_Y
        return float(sst)
    
    @cached_property
    def ssr(self) -> float:
        """
        Regression sum of squares
        b.T @ X.T @ Y - (1 / n) * Y.T @ ones_nd_array @ Y
        """
        ssr = self.ndarrays.bTXTY - (1 / self.dstats.n) * self.ndarrays.YT_N_Y
        return float(ssr)
    
    @cached_property
    def sse(self) -> float:
        """
        Error sum of squares
        Y.T @ Y - b.T @ X.T @ Y
        """
        return float(self.ndarrays.YTY - self.ndarrays.bTXTY)
    
    @cached_property
    def mst(self) -> float:
        """Mean square total"""
        return self.sst / (self.dstats.n - 1)
    
    @cached_property
    def msr(self) -> float:
        """Mean square regression"""
        return self.ssr / (self.dstats.k - 1)
    
    @cached_property
    def mse(self) -> float:
        """Mean square error"""
        return self.sse / (self.dstats.n - self.dstats.k)
    
    @property
    def df_r(self) -> int:
        """Degrees of freedom for regression"""
        return self.dstats.k - 1
    
    @property
    def df_e(self) -> int:
        """Degrees of freedom for residuals"""
        return self.dstats.n - self.dstats.k
    
    @property
    def df_t(self) -> int:
        """Degrees of freedom for totals"""
        return self.dstats.n - 1
    
    # OTHERS
    # ------
    
    @property
    def r_squared(self) -> float:
        """Coefficient of determination"""
        return float(self.ssr / self.sst if self.sst != 0 else 0.0)
    
    @property
    def adjusted_r_squared(self) -> float:
        """Adjusted R-squared"""
        n = self.dstats.n
        k = self.dstats.k
        if n <= k:
            return 0.0
        return float(1 - (1 - self.r_squared) * (n - 1) / (n - k))
    
    @property
    def root_mse(self) -> float:
        """Residual standard error (sqrt of MSE)"""
        return float(self.mse ** 0.5)
    
    @cached_property
    def log_likelihood(self) -> float:
        n = self.dstats.n
        sse = self.sse
        sigma_squared = sse / self.dstats.n
        left_part = (1 / ((2 * pi * sigma_squared) ** (1/2))) ** n
        right_part = exp(-(sse / (2 * sigma_squared)))
        return float(log(left_part * right_part))
    
    @cached_property
    def aic(self) -> float:
        return float(2 * self.dstats.k - 2 * self.log_likelihood)
    
    @cached_property
    def bic(self) -> float:
        return float(
            self.dstats.k * log(self.dstats.n) - 2 * self.log_likelihood
        )
    

class LinearRegressionInferences:
    """Statistical inference calculations"""
    
    def __init__(
        self,
        params: LinearRegressionParsedParams,
        dstats: LinearRegressionStatistics,
        array_comp: LinearRegressionMatrices,
        variance_calc: LinearRegressionVariances
    ):
        self.params = params
        self.dstats = dstats
        self.ndarrays = array_comp
        self.variances = variance_calc
    
    @cached_property
    def cov_beta(self) -> NDArray[float64]:
        """
        Covariance matrix of coefficients
        MSE * inv(X.T @ X)
        """
        return self.variances.mse * self.ndarrays.XTX_inv
    
    @cached_property
    def stderrors(self) -> NDArray[float64]:
        """
        Standard errors of coefficients
        diag(sqrt(MSE * inv(X.T @ X)))
        """
        return sqrt(diag(self.cov_beta)).reshape(-1, 1)
    
    @cached_property
    def t_statistics(self) -> NDArray[float64]:
        """t-statistics for coefficients"""
        return self.ndarrays.beta / self.stderrors
    
    @cached_property
    def p_values(self) -> NDArray[float64]:
        """Two-sided p-values for coefficients"""
        df = self.dstats.n - self.dstats.k
        pvalues = 2 * (1 - stats.t.cdf(abs(self.t_statistics), df))
        return pvalues
    
    @cached_property
    def alpha(self) -> float:
        return float(around(1 - self.params.conf_level, 4))
    
    @cached_property
    def t_critical(self) -> float:
        """Critical t-values for confidence intervals"""
        t_crit = abs(
            stats.t.ppf(1 - self.alpha / 2, self.dstats.n - self.dstats.k)
        )
        return float(t_crit)
    
    @cached_property
    def f_critical(self) -> float:
        """Critical F-value for ANOVA"""
        f_crit = (
            abs(stats.f.ppf(1 - self.alpha / 2, self.dstats.k, self.dstats.n))
        )
        return float(f_crit)
    
    @cached_property
    def confidence_intervals(self) -> NDArray[float64]:
        """Confidence intervals for coefficients"""
        se_times_t_critical = self.stderrors * self.t_critical
        lower = self.ndarrays.beta - se_times_t_critical
        upper = self.ndarrays.beta + se_times_t_critical
        return hstack((lower, upper))
    
    @cached_property
    def f_statistic_and_p_value(self) -> float:
        """F-statistic for the model"""
        df_r = self.dstats.k - 1
        df_e = self.dstats.n - self.dstats.k
        f_statistic = float(self.variances.msr / self.variances.mse)
        p_value = float(stats.f.sf(f_statistic, df_r, df_e))
        return f_statistic, p_value 


# ================ MODEL FITTING ================

class LinearRegressionModelFitter:
    """Handles model fitting and predictions"""
    
    def __init__(self, 
        params: LinearRegressionParsedParams,
        array_comp: LinearRegressionMatrices
    ):
        self.params = params
        self.ndarrays = array_comp
        self.y = self.params.y
        self.n = len(self.y)
    
    @cached_property
    def fitted_values(self) -> NDArray[float64]:
        """Fitted values ŷ = Xβ"""
        return (self.ndarrays.X @ self.ndarrays.beta).reshape(-1, 1)
    
    @cached_property
    def sum_fitted_values(self) -> float:
        return float(self.fitted_values.sum())
    
    @cached_property
    def residuals(self) -> NDArray[float64]:
        """Residuals e = y - ŷ"""
        return self.params.y - self.fitted_values
    
    @cached_property
    def residuals_squared(self) -> NDArray[float64]:
        """Squared residuals e^2 = (y - ŷ)^2"""
        return self.residuals ** 2
    
    @cached_property
    def sum_residuals_squared(self) -> float:
        """Sum of squared residuals"""
        return float(self.residuals_squared.sum())
    
    def predict(
        self, xvals: NDArray[float64], decimals: 12
    ) -> NDArray[float64]:
        """Predict y values given x (xvals)"""
        
        xvals = asarray(xvals)
        if xvals.ndim == 1:
            xvals = xvals.reshape(-1, 1)
        
        # Add intercept column
        x_design = DesignMatrixBuilder.with_intercept(x=xvals)
        yvals = around(x_design @ self.ndarrays.beta, decimals)

        return hstack((xvals, yvals))
    
    @cached_property
    def diff_residuals(self) -> NDArray[float64]:
        return diff(self.residuals.flatten()).reshape(-1, 1)

    @cached_property
    def durbin_watson(self) -> float:
        numer = (self.diff_residuals ** 2).sum()
        return float(numer / self.sum_residuals_squared)
        
    @cached_property
    def omnibus(self) -> tuple[float, float]:
        if len(self.residuals) > 7:
            omnibus_k2, omnibus_k2_pvalue = stats.normaltest(self.residuals)
            return float(omnibus_k2), float(omnibus_k2_pvalue)
        return None, None
    
    @cached_property
    def y_minus_yhat_pow2(self) -> NDArray[float64]:
        return (self.y - self.fitted_values) ** 2
    
    @cached_property
    def sum_y_minus_yhat_pow2(self) -> float:
        return self.y_minus_yhat_pow2.sum()
    
    @cached_property
    def y_minus_yhat_pow3(self) -> NDArray[float64]:
        return (self.y - self.fitted_values) ** 3
    
    @cached_property
    def sum_y_minus_yhat_pow3(self) -> float:
        return self.y_minus_yhat_pow3.sum()
    
    @cached_property
    def y_minus_yhat_pow4(self) -> NDArray[float64]:
        return (self.y - self.fitted_values) ** 4
    
    @cached_property
    def sum_y_minus_yhat_pow4(self) -> float:
        return self.y_minus_yhat_pow4.sum()
    
    @cached_property
    def skew_kurt(self) -> tuple[float, float]:
        y2 = self.sum_y_minus_yhat_pow2
        y3 = self.sum_y_minus_yhat_pow3
        y4 = self.sum_y_minus_yhat_pow4
        n = self.n
        skew = float( (1 / n * y3) / (1 / n * y2) ** (3 / 2) )
        kurt = float(((1 / self.n) * y4) / (((1 / self.n) * y2) ** 2))
        return skew, kurt
    
    @cached_property
    def jarque_bera(self) -> tuple[float, float]:
        skew, kurt = self.skew_kurt
        jbera = float((self.n / 6) * (skew ** 2 + (1 / 4) * (kurt - 3) ** 2))
        jbera_pvalue = float(1.0 - stats.chi2(2).cdf(jbera))
        return jbera, jbera_pvalue
    
    @cached_property
    def cond_number(self) -> float:
        X = DesignMatrixBuilder.with_intercept(x=self.params.x)
        return cond(X)
    
# ================ MODEL FORMATTING ================

class LinearRegressionModelFormatter:
    """Formats model representations"""
    
    def __init__(
        self,
        params: LinearRegressionParsedParams,
        array_comp: LinearRegressionMatrices
    ):
        self.params = params
        self.ndarrays = array_comp
        self.n = len(self.params.x)
        self.k = self.params.x.shape[1] + 1
    
    @cached_property
    def model_string(self) -> str:
        """String representation of the model"""
        beta_arr = around(self.ndarrays.beta, self.params.decimals).flatten()
        terms = []
        for idx, beta in enumerate(beta_arr):
            if idx == 0:
                terms.append(f"{beta}")
            else:
                terms.append(f"{beta} * x{idx}")
        return " + ".join(terms)
    
    @cached_property
    def model_latex(self) -> str:
        """LaTeX representation of the model"""
        return latex(sympify(self.model_string))
    
    @cached_property
    def beta_symbols(self) -> list[str]:
        """
        LaTeX symbols for beta coefficients
        b0, b1, b2, ..., bk
        """
        return [f"\\beta_{{{idx}}}" for idx in range(self.k)]
    
    @cached_property
    def xi_symbols(self) -> tuple[list[str], str, list[str], str]:
        """
        xi symbols
        x0, x1, x2, ..., xn
        """
        xi = [f"x{idx}" for idx in range(self.k)]
        xi_joined = ", ".join(xi[1:])
        
        xi_latex = [f"x_{idx}" for idx in range(self.k)]
        xi_latex_joined = ", ".join(xi_latex[1:])
        
        return xi, xi_joined, xi_latex, xi_latex_joined