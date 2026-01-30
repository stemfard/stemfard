from numpy import float64
from numpy.typing import NDArray

from stemfard.maths.test import (
    LRDescriptiveStatistics, InferenceCalculator, MatrixComputations,
    ModelFitter, ModelFormatter, NormalEquationsComputation,
    LinearRegressionParameterParser, SquaresSumAndMean
)


class BaseLinearRegression:
    """
    Public façade for linear regression results.

    All commonly used attributes are exposed as typed properties:
    x, y, n, k, beta, mse, residuals, fitted_values, etc.
    """

    def __init__(self, **kwargs):

        self.params = LinearRegressionParameterParser.parse(**kwargs)

        self.dstats = LRDescriptiveStatistics(params=self.params)

        if self.params.simple_linear_method == "normal":
            self.matrix = NormalEquationsComputation(
                params=self.params,
                dstats=self.dstats
            )
        else:
            self.matrix = MatrixComputations(params=self.params)

        self.variance = SquaresSumAndMean(
            array_comp=self.matrix,
            dstats=self.dstats
        )
        self.inference = InferenceCalculator(
            params=self.params,
            dstats=self.dstats,
            array_comp=self.matrix,
            variance_calc=self.variance
        )
        self.fitter = ModelFitter(params=self.params, array_comp=self.matrix)
        self.formatter = ModelFormatter(
            params=self.params,
            array_comp=self.matrix
        )

    # ===============================
    # Raw data (AUTOCOMPLETE)
    # ===============================
    
    @property
    def x(self) -> NDArray[float64]:
        return self.params.x

    @property
    def y(self) -> NDArray[float64]:
        return self.params.y

    # ===============================
    # Descriptives
    # ===============================
    
    @property
    def n(self) -> int:
        return self.dstats.n

    @property
    def k(self) -> int:
        return self.dstats.k

    @property
    def mean_x(self) -> float:
        return self.dstats.mean_x

    @property
    def mean_y(self) -> float:
        return self.dstats.mean_y

    # ===============================
    # Coefficients
    # ===============================
    
    @property
    def beta(self) -> NDArray[float64]:
        return self.matrix.beta

    @property
    def beta0(self) -> float:
        return getattr(self.matrix, "beta0", self.beta[0, 0])

    @property
    def beta1(self) -> float:
        return getattr(self.matrix, "beta1", self.beta[1, 0])

    # ===============================
    # Variance / ANOVA
    # ===============================
    
    @property
    def sst(self) -> float:
        return self.variance.sst

    @property
    def ssr(self) -> float:
        return self.variance.ssr

    @property
    def sse(self) -> float:
        return self.variance.sse

    @property
    def mse(self) -> float:
        return self.variance.mse

    @property
    def msr(self) -> float:
        return self.variance.msr

    # ===============================
    # Inference
    # ===============================
    
    @property
    def standard_errors(self) -> NDArray[float64]:
        return self.inference.standard_errors

    @property
    def t_statistics(self) -> NDArray[float64]:
        return self.inference.t_statistics

    @property
    def p_values(self) -> NDArray[float64]:
        return self.inference.p_values

    @property
    def confidence_intervals(self) -> NDArray[float64]:
        return self.inference.confidence_intervals

    @property
    def f_statistic(self) -> float:
        return self.inference.f_statistic

    @property
    def f_pvalue(self) -> float:
        return self.inference.f_pvalue

    # ===============================
    # Fitted values
    # ===============================
    
    @property
    def fitted_values(self) -> NDArray[float64]:
        return self.fitter.fitted_values

    @property
    def residuals(self) -> NDArray[float64]:
        return self.fitter.residuals

    # ===============================
    # Formatting
    # ===============================
    
    @property
    def model_string(self) -> str:
        return self.formatter.model_string

    @property
    def model_latex(self) -> str:
        return self.formatter.model_latex

    # ===============================
    # Public API
    # ===============================
    
    def predict(self, x: NDArray[float64]) -> NDArray[float64]:
        return self.fitter.predict(x)