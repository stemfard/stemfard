from functools import cached_property
from typing import Literal

from numpy import around, asarray, diag, eye, float64, hstack, ones, sqrt, vstack
from numpy.linalg import solve, qr
from numpy.typing import NDArray
from scipy import stats
import statsmodels.api as sm

from stemfard.stats.regression_linear._parsed_params import parse_linear_regression
from stemfard.core._type_aliases import SequenceArrayLike
from sympy import latex, sympify


class BaseLinearRegression:
    """Base class for Linear regression."""

    def __init__(
        self,
        x: SequenceArrayLike,
        y: SequenceArrayLike,
        simple_linear_method: Literal["matrix", "normal"] | None = "normal",
        conf_level: float = 0.95,
        coefficients: bool = True,
        stderrors: bool = True,
        tstats: bool = True,
        pvalues: bool = True,
        conf_intervals: bool = True,
        anova: bool = True,
        fitted_and_residuals: bool = False,
        others: bool = False,
        prm_predict_at_x: SequenceArrayLike = ...,
        steps_compute: bool = True,
        steps_detailed: bool = True,
        show_bg: bool = True,
        decimals: int = 12
    ):
        parsed_params = parse_linear_regression(
            x=x,
            y=y,
            simple_linear_method=simple_linear_method,
            conf_level=conf_level,
            coefficients=coefficients,
            stderrors=stderrors,
            tstats=tstats,
            pvalues=pvalues,
            conf_intervals=conf_intervals,
            anova=anova,
            fitted_and_residuals=fitted_and_residuals,
            others=others,
            prm_predict_at_x=prm_predict_at_x,
            steps_compute=steps_compute,
            steps_detailed=steps_detailed,
            show_bg=show_bg,
            decimals=decimals
        )
        
        self.prm_x = parsed_params.x
        self.prm_y = parsed_params.y
        self.prm_simple_linear_method = parsed_params.simple_linear_method
        self.prm_conf_level = parsed_params.conf_level
        self.prm_coefficients = parsed_params.coefficients
        self.prm_stderrors = parsed_params.stderrors
        self.prm_tstats = parsed_params.tstats
        self.prm_param_pvalues = parsed_params.pvalues
        self.prm_conf_intervals = parsed_params.conf_intervals
        self.prm_anova = parsed_params.anova
        self.bse_fitted_and_residuals = parsed_params.fitted_and_residuals
        self.prm_others = parsed_params.others
        self.prm_predict_at_x = parsed_params.prm_predict_at_x
        self.prm_steps_compute = parsed_params.steps_compute
        self.prm_steps_detailed = parsed_params.steps_detailed
        self.prm_show_bg = parsed_params.show_bg
        self.prm_decimals = parsed_params.decimals
        
        self.msc_sig_level = round(1 - self.prm_conf_level, self.prm_decimals)
        self.msc_params = parsed_params.params
        self._bse_counter = 0
        
        
    @property
    def bse_counter(self) -> int:
        return self._bse_counter
    
    @bse_counter.setter
    def bse_counter(self, new_counter: int) -> int:
        self._bse_counter = new_counter
        
    @property
    def bse_n(self) -> int:
        return len(self.prm_x)
    
    @property
    def bse_k(self) -> int:
        return 1 + self.prm_x.shape[1]
    
    @property
    def bse_x_rnd(self) -> NDArray[float64]:
        return around(self.prm_x, self.prm_decimals)
    
    @property
    def bse_y_rnd(self) -> NDArray[float64]:
        return around(self.prm_y, self.prm_decimals)
    
    @cached_property
    def bse_x_squared(self) -> NDArray[float64]:
        return self.prm_x ** 2
    
    @property
    def bse_x_squared_rnd(self) -> NDArray[float64]:
        return around(self.bse_x_squared, self.prm_decimals)
    
    @cached_property
    def bse_total_x_squared(self) -> float:
        return float(self.bse_x_squared.sum())
    
    @property
    def bse_total_x_squared_rnd(self) -> float:
        return float(around(self.bse_total_x_squared, self.prm_decimals))
    
    @cached_property
    def bse_total_x(self) -> float:
        return float(self.prm_x.sum())
    
    @property
    def bse_total_x_rnd(self) -> float:
        return float(around(self.bse_total_x, self.prm_decimals))
    
    @cached_property
    def bse_total_y(self) -> float:
        return float(self.prm_y.sum())
    
    @property
    def bse_total_y_rnd(self) -> float:
        return float(around(self.bse_total_y, self.prm_decimals))
    
    @cached_property
    def bse_xy(self) -> NDArray[float64]:
        return self.prm_x * self.prm_y
    
    @cached_property
    def bse_xy_rnd(self) -> NDArray[float64]:
        return around(self.bse_xy, self.prm_decimals)
    
    @cached_property
    def bse_total_xy(self) -> float:
        return float(self.bse_xy.sum())
    
    @property
    def bse_total_xy_rnd(self) -> float:
        return float(around(self.bse_total_xy, self.prm_decimals))
    
    @cached_property
    def bse_mean_x(self) -> float:
        return float(self.prm_x.mean())
    
    @property
    def bse_mean_x_rnd(self) -> float:
        return float(around(self.bse_mean_x, self.prm_decimals))
    
    @cached_property
    def bse_mean_y(self) -> float:
        return float(self.prm_y.mean())
    
    @property
    def bse_mean_y_rnd(self) -> float:
        return float(around(self.bse_mean_y, self.prm_decimals))
    
    @cached_property
    def bse_beta1_normal(self) -> float:
        _numer = (
            self.bse_total_xy - self.prm_n * self.bse_mean_x * self.bse_mean_y
        )
        return float(_numer / self.bse_sb_denom)
    
    @property
    def bse_beta1_normal_rnd(self) -> float:
        return float(around(self.bse_beta1_normal, self.prm_decimals))
    
    @cached_property
    def bse_beta0_normal(self) -> float:
        return float(self.bse_mean_y - self.bse_beta1_normal * self.bse_mean_x)
    
    @property
    def bse_beta0_normal_rnd(self) -> float:
        return float(around(self.bse_beta0_normal, self.prm_decimals))
    
    # matrices required for sum of squares
    
    @property
    def bse_Y(self) -> NDArray[float64]:
        return self.prm_y
    
    @property
    def bse_Y_rnd(self) -> NDArray[float64]:
        return self.bse_y_rnd
    
    @cached_property
    def bse_X(self) -> NDArray[float64]:
        """
        Append a column of 1's to the left of the independent variable(s)
        """
        return hstack((ones((self.prm_n, 1)), self.prm_x))
    
    @property
    def bse_X_rnd(self) -> NDArray[float64]:
        return around(self.bse_X, self.prm_decimals)
    
    @cached_property
    def bse_YT(self) -> NDArray[float64]:
        return self.bse_Y.T
    
    @property
    def bse_YT_rnd(self) -> NDArray[float64]:
        return around(self.bse_YT, self.prm_decimals)
    
    @cached_property
    def bse_YTY(self) -> NDArray[float64]:
        return self.bse_YT @ self.bse_Y
    
    @cached_property
    def bse_YTY_rnd(self) -> NDArray[float64]:
        return around(self.bse_YTY, self.prm_decimals)
    
    @cached_property
    def bse_XT(self) -> NDArray[float64]:
        return self.bse_X.T
    
    @property
    def bse_XT_rnd(self) -> NDArray[float64]:
        return around(self.bse_XT, self.prm_decimals)
    
    @cached_property
    def bse_XTX(self) -> NDArray[float64]:
        return self.bse_XT @ self.bse_X
    
    @property
    def bse_XTX_rnd(self) -> NDArray[float64]:
        return around(self.bse_XTX, self.prm_decimals)
    
    @cached_property
    def bse_XTY(self) -> NDArray[float64]:
        return self.bse_XT @ self.bse_Y
    
    @property
    def bse_XTY_rnd(self) -> NDArray[float64]:
        return around(self.bse_XTY, self.prm_decimals)
    
    @cached_property
    def bse_qr(self) -> NDArray[float64]:
        Q, R = qr(self.bse_X)
        return Q, R
    
    @cached_property
    def bse_XTX_inv(self) -> NDArray[float64]:
        """
        Approximate X.T @ X inverse using QR for stability.
        """
        R = self.bse_qr["R"]
        R_inv = solve(R, eye(R.shape[0]))  # equivalent to inv(R)
        return R_inv @ R_inv.T  # approximates inv(X.T @ X)
    
    @property
    def bse_XTX_inv_rnd(self) -> NDArray[float64]:
        return around(self.bse_XTX_inv, self.prm_decimals)
    
    @cached_property
    def bse_beta(self) -> NDArray[float64]:
        """b = inv(X.T @ X) @ (X.T @ Y)"""
        Q, R = self.bse_qr
        beta = solve(R, Q.T @ self.bse_Y) # b = inv(X.T @ X) @ (X.T @ Y)
        return beta.flatten()
    
    @property
    def bse_beta_rnd(self) -> NDArray[float64]:
        return around(self.bse_beta, self.prm_decimals)
    
    @cached_property
    def bse_model_str(self) -> str:
        b = self.bse_beta_rnd # use rounded value
        xi = [f"x{idx}" for idx in range(len(b))]
        return " + ".join(f"{b}*{xi}" for b, xi in zip(b, xi))
        
    @cached_property
    def bse_model_latex(self) -> str:
        return latex(sympify(self.prm_model_str)).replace(f"x_{{0}}", "0")
    
    @cached_property
    def bse_calc_predict_at_x(self) -> NDArray[float64]:
        predicted_y = []
        pred_at_x = self.prm_predict_at_x
        for idx in range(len(pred_at_x)):
            eqtn_rep = self.prm_model_str.replace("*x0", "")
            xis = [f"x{idx}" for idx in range(1, self.prm_k)]
            for xi in xis:
                xpredic_val = pred_at_x[idx]
                eqtn_rep_sympy = eqtn_rep.replace(xi, str(xpredic_val))
                
            predicted_y.append(float(sympify(eqtn_rep_sympy)))
            
        pred_at_x = asarray(pred_at_x).reshape(-1, 1)
        predicted_y = asarray(predicted_y).reshape(-1, 1)
        
        return hstack(tup=(pred_at_x, predicted_y))
    
    @property
    def bse_predict_at_x_rnd(self) -> NDArray[float64]:
        return around(self.prm_calc_predict_at_x, self.prm_decimals)
    
    @property
    def bse_beta_str(self) -> list[str]:
        return [f"\\beta_{{{idx}}}" for idx in range(self.prm_k)]
    
    @cached_property
    def bse_stderrors(self) -> NDArray[float64]:
        # MSE * inv(X.T @ X)
        cov_beta = self.bse_mse * self.bse_XTX_inv
        return sqrt(diag(cov_beta)).reshape(-1, 1)

    @cached_property
    def bse_stderrors_rnd(self) -> NDArray[float64]:
        return around(self.bse_stderrors, self.prm_decimals)
    
    @cached_property
    def bse_t_stats(self) -> NDArray[float64]:
        return self.bse_beta / self.bse_stderrors
    
    @property
    def bse_t_stats_rnd(self) -> NDArray[float64]:
        return around(self.prm_t_stats, self.prm_decimals)
    
    @property
    def bse_tcrit(self) -> float:
        return float(abs(stats.t.ppf(1 - self.prm_sig_level / 2, self.prm_n - self.prm_k)))
    
    @property
    def bse_tcrit_rnd(self) -> float:
        return float(around(self.prm_tcrit, self.prm_decimals))
    
    @property
    def bse_fcrit(self) -> float:
        fcrit = stats.f.ppf(1 - self.prm_sig_level / 2, self.prm_k, self.prm_n)
        return float(abs(fcrit))
    
    @property
    def bse_fcrit_rnd(self) -> float:
        return float(around(self.prm_fcrit, self.prm_decimals))
    
    @cached_property
    def bse_ci(self) -> NDArray[float64]:
        se_times_tcrit = self.bse_stderrors * self.prm_tcrit
        lower = self.bse_beta - se_times_tcrit
        upper = self.bse_beta + se_times_tcrit
        return vstack(tup=(lower, upper)).T
    
    @property
    def bse_ci_rnd(self) -> NDArray[float64]:
        return around(self.prm_ci, self.prm_decimals)
    
    @cached_property
    def bse_bTXTY(self) -> NDArray[float64]:
        return self.bse_beta.T @ self.bse_XTY
    
    @property
    def bse_bTXTY_rnd(self) -> NDArray[float64]:
        return around(self.bse_bTXTY, self.prm_decimals)
    
    @cached_property
    def bse_YT_1sND_Y(self) -> NDArray[float64]:
        N = ones((self.prm_n, self.prm_n))
        return self.bse_YT @ N @ self.bse_Y
    
    # SSR / MSR
    
    @cached_property
    def bse_ssr(self) -> float:
        """SSR = b.T @ X.T @ Y - (1 / n) * Y.T @ ones_nd_array @ Y"""
        _ssr = self.bse_bTXTY - (1 / self.prm_n) * self.bse_YT_1sND_Y
        return _ssr[0, 0]
    
    @property
    def bse_ssr_rnd(self) -> float:
        return float(around(self.bse_ssr, self.prm_decimals))
    
    @cached_property
    def bse_msr(self) -> float:
        return self.bse_ssr / (self.prm_k - 1)
    
    @property
    def bse_msr_rnd(self) -> float:
        return float(around(self.bse_msr, self.prm_decimals))
    
    # SSE / MSE
    
    @cached_property
    def bse_sse(self) -> float:
        """SSE = Y.T @ Y - b.T @ X.T @ Y"""
        _sse = self.bse_YTY - self.bse_bTXTY
        return _sse[0, 0]
    
    @property
    def bse_sse_rnd(self) -> float:
        return float(around(self.bse_sse, self.prm_decimals))
    
    @cached_property
    def bse_mse(self) -> float:
        return self.bse_sse / (self.prm_n - self.prm_k)
    
    @property
    def bse_mse_rnd(self) -> float:
        return float(around(self.bse_mse, self.prm_decimals))
    
    # SST / MST
    
    @cached_property
    def bse_sst(self) -> float:
        """SST = Y.T @ Y - (1 / n) * Y.T @ ones_nd_array @ Y"""
        _sst = self.bse_YTY - (1 / self.prm_n) * self.bse_YT_1sND_Y
        return _sst[0, 0]
    
    @property
    def bse_sst_rnd(self) -> float:
        return float(around(self.bse_sst, self.prm_decimals))
    
    @cached_property
    def bse_mst(self) -> float:
        return self.bse_sst / (self.prm_n - 1)
    
    @property
    def bse_mst_rnd(self) -> float:
        return float(around(self.bse_mst, self.prm_decimals))
    
    # standar errors
    
    @cached_property
    def bse_sb_denom(self) -> float:
        return self.bse_total_x_squared - self.prm_n * self.bse_mean_x ** 2
    
    @property
    def bse_sb_denom_rnd(self) -> float:
        return around(self.bse_sb_denom, self.prm_decimals)
    
    @cached_property
    def bse_sb0(self) -> float:
        sb0 = sqrt(
            self.bse_mse * 
            (1 / self.prm_n + self.bse_mean_x ** 2 / self.bse_sb_denom)
        )
        return sb0
    @property
    def bse_sb0_rnd(self) -> float:
        return around(self.bse_sb0, self.prm_decimals)
    
    @cached_property
    def bse_sb1(self) -> float:
        return sqrt(self.bse_mse / self.bse_sb_denom)
    
    @property
    def bse_sb1_rnd(self) -> float:
        return around(self.bse_sb1, self.prm_decimals)
    
    @cached_property
    def bse_model(self):
        X = sm.add_constant(self.prm_x)
        return sm.OLS(self.prm_y, X).fit()

    @property
    def bse_fitted_residuals(self):
        model = self.bse_model
        return {
            "fitted": model.fittedvalues,
            "residuals": model.resid,
            "pvalues": model.pvalues,
        }
    
    @cached_property
    def bse_fitted_yhat(self) -> NDArray[float64]:
        return self.bse_fitted_residuals.get("fitted").reshape(-1, 1)
    
    @cached_property
    def bse_fitted_yhat_rnd(self) -> NDArray[float64]:
        return around(self.bse_fitted_yhat, self.prm_decimals)
    
    @cached_property
    def bse_total_fitted_yhat(self) -> float:
        return float(self.bse_fitted_yhat.sum())
    
    @cached_property
    def bse_total_fitted_yhat_rnd(self) -> float:
        return float(around(self.bse_total_fitted_yhat, self.prm_decimals))
    
    @cached_property
    def bse_residuals(self) -> NDArray[float64]:
        return self.bse_fitted_residuals.get("residuals").reshape(-1, 1)

    @cached_property
    def bse_residuals_rnd(self) -> NDArray[float64]:
        return around(self.bse_residuals, self.prm_decimals)
    
    @cached_property
    def bse_residuals_squared(self) -> NDArray[float64]:
        return self.bse_residuals ** 2
    
    @cached_property
    def bse_residuals_squared_rnd(self) -> NDArray[float64]:
        return around(self.bse_residuals_squared, self.prm_decimals)
    
    @cached_property
    def bse_total_residuals_squared(self) -> float:
        return float(self.bse_residuals_squared.sum())
    
    @cached_property
    def bse_total_residuals_squared_rnd(self) -> float:
        return float(around(self.bse_total_residuals_squared, self.prm_decimals))
    
    @property
    def bse_pvalues(self) -> NDArray[float64]:
        return self.bse_fitted_residuals.get("pvalues").reshape(-1, 1)
    
    @cached_property
    def bse_pvalues_rnd(self) -> NDArray[float64]:
        return around(self.bse_pvalues, self.prm_decimals)
    
    @cached_property
    def bse_fstat(self) -> float:
        return self.bse_msr / self.bse_mse
    
    @property
    def bse_fstat_rnd(self) -> float:
        return float(around(self.bse_fstat, self.prm_decimals))
    
    @cached_property
    def bse_pvalue_fstat(self) -> float:
        df_r = self.prm_k - 1
        df_e = self.prm_n - 2
        return stats.f.sf(self.bse_fstat, df_r, df_e)
    
    @property
    def bse_pvalue_fstat_rnd(self) -> float:
        return float(around(self.bse_pvalue_fstat, self.prm_decimals))