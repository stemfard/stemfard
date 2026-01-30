from functools import cached_property
from typing import Literal

from numpy import float64, sqrt
from numpy.typing import NDArray

from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.stats.regression_linear._parsed_params import LinearRegressionParsedParams
from stemfard.core.models import CoreParamsResult

# ================ PARAMETER MANAGEMENT ================

class LinearRegressionParameterParser:
    """Handles input parsing and validation"""
    
    @staticmethod
    def parse(
        x: SequenceArrayLike,
        y: SequenceArrayLike,
        simple_linear_method: Literal["matrix", "normal"] | None = "normal",
        conf_level: float = 0.95,
        coefficients: bool = True,
        standard_errors: bool = True,
        tstats: bool = True,
        pvalues: bool = True,
        confidence_intervals: bool = True,
        anova: bool = True,
        fitted_and_residuals: bool = False,
        others: bool = False,
        predict_at_x: NDArray[float64] | None = None,
        steps_compute: bool = True,
        steps_detailed: bool = True,
        show_bg: bool = True,
        decimals: int = 12,
        params: CoreParamsResult = ...
    ) -> LinearRegressionParsedParams:
        """Parse and validate input parameters"""
        from stemfard.stats.regression_linear._parsed_params import parse_linear_regression
        
        parsed_params = parse_linear_regression(
            x=x,
            y=y,
            simple_linear_method=simple_linear_method,
            conf_level=conf_level,
            coefficients=coefficients,
            standard_errors=standard_errors,
            tstats=tstats,
            pvalues=pvalues,
            confidence_intervals=confidence_intervals,
            anova=anova,
            fitted_and_residuals=fitted_and_residuals,
            others=others,
            predict_at_x=predict_at_x,
            steps_compute=steps_compute,
            steps_detailed=steps_detailed,
            show_bg=show_bg,
            decimals=decimals,
            params=params
        )
        
        return parsed_params
        

# ================ MATRIX COMPUTATIONS ================

class DesignMatrixBuilder:
    """Builds design matrices with intercept"""
    
    @staticmethod
    def with_intercept(x: NDArray[float64]) -> NDArray[float64]:
        """Add intercept column to design matrix"""
        from numpy import ones, hstack
        return hstack((ones((x.shape[0], 1)), x))
    

class LRDescriptiveStatistics:
    def __init__(self, params: LinearRegressionParsedParams):
        """Descriptive statistics: n, k, mean, sum"""
        self.params = params
    
    @property
    def n(self) -> int:
        return len(self.params.x)

    @property
    def k(self) -> int:
        ncols = self.params.x.shape[1]
        return ncols + 1
    
    @cached_property
    def x_sq(self) -> NDArray[float64]:
        return self.params.x ** 2
    
    @cached_property
    def sum_x(self) -> float:
        return float(self.params.x.sum())
    
    @cached_property
    def sum_x_sq(self) -> float:
        return float((self.params.x ** 2).sum())
    
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


class MatrixComputations:
    """QR-based matrix computations"""
    
    def __init__(self, params: LinearRegressionParsedParams, dstats: LRDescriptiveStatistics):
        self.params = params
        self.dstats = dstats
        self._design_matrix = DesignMatrixBuilder.with_intercept(x=params.x)
        self._mse = 7.999
    @cached_property
    def X(self) -> NDArray[float64]:
        """Design matrix"""
        return self._design_matrix
    
    @cached_property
    def Y(self) -> NDArray[float64]:
        """Response vector"""
        return self.params.y
    
    @cached_property
    def qr_decomposition(self) -> tuple[NDArray[float64], NDArray[float64]]:
        """QR decomposition of design matrix"""
        from numpy.linalg import qr
        return qr(self.X)
    
    @property
    def Q(self) -> NDArray[float64]:
        return self.qr_decomposition[0]
    
    @property
    def R(self) -> NDArray[float64]:
        return self.qr_decomposition[1]
    
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
    def YTY(self) -> NDArray[float64]:
        return self.YT @ self.Y
    
    @cached_property
    def XTX_inv(self) -> NDArray[float64]:
        """
        Inverse of X.T @ X using QR stability
        `inv()` is not stable
        """
        from numpy.linalg import solve
        from numpy import eye
        R_inv = solve(self.R, eye(self.R.shape[0]))
        return R_inv @ R_inv.T
    
    @cached_property
    def beta(self) -> NDArray[float64]:
        """
        Coefficient estimates using QR solve
        b = inv(X.T @ X) @ (X.T @ Y)
        """
        from numpy.linalg import solve
        return solve(self.R, self.Q.T @ self.Y)
    
    @cached_property
    def bT(self) -> NDArray[float64]:
        return self.beta.T
    
    @cached_property
    def bTXTY(self) -> NDArray[float64]:
        """b.T @ X.T @ Y"""
        return self.bT @ self.XTY
    
    @cached_property
    def YT_N_Y(self) -> NDArray[float64]:
        """
        Centered calculations
        Y.T @ ones_matrix @ Y
        """
        from numpy import ones
        n = self.dstats.n
        N = ones((n, n))
        return self.YT @ N @ self.Y
    
    @cached_property
    def sb_denom(self) -> float:
        """Denominator for slope standard error"""
        return float(
            self.dstats.sum_x_sq - self.dstats.n * self.dstats.mean_x ** 2
        )
        
    @cached_property
    def sb0(self) -> float:
        sb0 = sqrt(
            self._mse * 
            (1 / self.dstats.n + self.dstats.mean_x ** 2 / self.sb_denom)
        )
        return float(sb0)
    
    @cached_property
    def sb1(self) -> float:
        return float(sqrt(self._mse / self.sb_denom))
    
    @cached_property
    def beta1(self) -> float:
        """Slope estimate using normal equations"""
        n_mean_x_mean_y = (
            self.dstats.n * self.dstats.mean_x * self.dstats.mean_y
        )
        numer = self.dstats.sum_xy - n_mean_x_mean_y
        return float(numer / self.sb_denom)
    
    @cached_property
    def beta0(self) -> float:
        """Intercept estimate using normal equations"""
        return float(self.dstats.mean_y - self.beta1 * self.dstats.mean_x)


# ================ STATISTICAL COMPUTATIONS ================

class SquaresSumAndMean:
    """Calculates variances and sum of squares"""
    
    def __init__(
        self,
        array_comp: MatrixComputations,
        dstats: LRDescriptiveStatistics
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
    
    @property
    def r_squared(self) -> float:
        """Coefficient of determination"""
        return float(self.ssr / self.sst if self.sst != 0 else 0.0)
    
    @property
    def adjusted_r_squared(self) -> float:
        """Adjusted R-squared"""
        n = self.dstats.n
        k = self.dstats.k
        if self.n <= self.k:
            return 0.0
        return float(1 - (1 - self.r_squared) * (n - 1) / (n - k))
    
    @property
    def root_mse(self) -> float:
        """Residual standard error (sqrt of MSE)"""
        return float(self.mse ** 0.5)


class InferenceCalculator:
    """Statistical inference calculations"""
    
    def __init__(
        self,
        params: LinearRegressionParsedParams,
        dstats: LRDescriptiveStatistics,
        array_comp: MatrixComputations,
        variance_calc: SquaresSumAndMean
    ):
        self.params = params
        self.dstats = dstats
        self.ndarrays = array_comp
        self.variance = variance_calc
    
    @cached_property
    def cov_beta(self) -> NDArray[float64]:
        """
        Covariance matrix of coefficients
        MSE * inv(X.T @ X)
        """
        return self.variance.mse * self.ndarrays.XTX_inv
    
    @cached_property
    def standard_errors(self) -> NDArray[float64]:
        """
        Standard errors of coefficients
        diag(sqrt(MSE * inv(X.T @ X)))
        """
        from numpy import sqrt, diag
        return sqrt(diag(self.cov_beta)).reshape(-1, 1)
    
    @cached_property
    def t_statistics(self) -> NDArray[float64]:
        """t-statistics for coefficients"""
        return self.ndarrays.beta / self.standard_errors
    
    @cached_property
    def p_values(self) -> NDArray[float64]:
        import statsmodels.api as sm
        x = self.params.x
        y = self.params.y
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        return model.pvalues.reshape(-1, 1)
    
    @cached_property
    def alpha(self) -> float:
        return 1 - self.params.conf_level
    
    @cached_property
    def t_critical(self) -> float:
        """Critical t-values for confidence intervals"""
        from scipy import stats
        tcrit = abs(
            stats.t.ppf(1 - self.alpha / 2, self.dstats.n - self.dstats.k)
        )
        return float(tcrit)
    
    @cached_property
    def f_critical(self) -> float:
        """Critical F-value for ANOVA"""
        from scipy import stats
        fcrit = (
            abs(stats.f.ppf(1 - self.alpha / 2, self.dstats.k, self.dstats.n))
        )
        return float(fcrit)
    
    @cached_property
    def confidence_intervals(self) -> NDArray[float64]:
        """Confidence intervals for coefficients"""
        from numpy import vstack
        se_times_tcrit = self.standard_errors * self.t_critical
        lower = self.ndarrays.beta - se_times_tcrit
        upper = self.ndarrays.beta + se_times_tcrit
        return vstack((lower, upper)).T
    
    @cached_property
    def f_statistic(self) -> float:
        """F-statistic for the model"""
        return self.variance.msr / self.variance.mse
    
    @cached_property
    def f_pvalue(self) -> float:
        """p-value for F-statistic"""
        from scipy import stats
        df_r = self.dstats.k - 1
        df_e = self.dstats.n - 2
        return stats.f.sf(self.f_statistic, df_r, df_e)


# ================ MODEL FITTING ================

class ModelFitter:
    """Handles model fitting and predictions"""
    
    def __init__(self, 
        params: LinearRegressionParsedParams,
        array_comp: MatrixComputations
    ):
        self.params = params
        self.ndarrays = array_comp
    
    @cached_property
    def fitted_values(self) -> NDArray[float64]:
        """Fitted values ŷ = Xβ"""
        return (self.ndarrays.X @ self.ndarrays.beta).reshape(-1, 1)
    
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
        from numpy import asarray
        
        xvals = asarray(xvals)
        if xvals.ndim == 1:
            xvals = xvals.reshape(-1, 1)
        
        # Add intercept column
        x_design = DesignMatrixBuilder.with_intercept(x=xvals)

        return around(x_design @ self.ndarrays.beta)

# ================ MODEL FORMATTING ================

class ModelFormatter:
    """Formats model representations"""
    
    def __init__(
        self,
        params: LinearRegressionParsedParams,
        array_comp: MatrixComputations
    ):
        self.params = params
        self.ndarrays = array_comp
        self.n = len(self.params.x)
    
    @cached_property
    def model_string(self) -> str:
        """String representation of the model"""
        from numpy import around
        beta = around(self.ndarrays.beta, self.params.decimals).flatten()
        terms = []
        for i, b in enumerate(beta):
            if i == 0:
                terms.append(f"{b}")
            else:
                terms.append(f"{b}*x{i}")
        return " + ".join(terms)
    
    @cached_property
    def model_latex(self) -> str:
        """LaTeX representation of the model"""
        from sympy import latex, sympify
        return latex(sympify(self.model_string))
    
    @cached_property
    def beta_symbols(self) -> list[str]:
        """LaTeX symbols for beta coefficients"""
        return [f"\\beta_{{{i}}}" for i in range(self.n)]
    
    @cached_property
    def xi_symbols(self) -> list[str]:
        """xi symbols i.e. x1, x2, ..., xn"""
        return [f"x{{{i}}}" for i in range(self.n)]