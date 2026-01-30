from dataclasses import dataclass
from functools import cached_property

from numpy import around, array, asarray, hstack, nan, vstack

from stemfard.core._html import html_bg_level2
from stemfard.core.constants import StemConstants
from stemfard.core.arrays_highlight import highlight_array_vals_arr
from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import str_color_values
from stemfard.core._enumerate import ColorCSS
from stemfard.stats.regression_linear._base import BaseLinearRegression


@dataclass(slots=True, frozen=True)
class HtmlStrings:
    coeffs: str
    standard_errors: str
    tstats: str
    remark_nk: str


class LinearRegressionCalculations(BaseLinearRegression):
    """Linear regression calculations"""
    
    def html_str(self) -> HtmlStrings:
        if self.coefficients:
            coeffs = (
                f" calculated in \\( \\text{{Regression coefficients}} \\) "
                "section."
            )
        else:
            coeffs = (
                f". Set \\( {str_color_values('coefficients=True')} \\) "
                "to perform step-by-step calculations of regression "
                "coefficients"
            )
        
        if self.standard_errors:
            standard_errors = (
                f" calculated in \\( \\text{{Standard erors}} \\) section."
            )
        else:
            standard_errors = (
                f". Set \\( {str_color_values('standard_errors=True')} \\) "
                "to perform step-by-step calculations of standard errors"
            )
            
        if self.tstats:
            tstats = (
                f"The above \\( t \\) statistics were calculated in the "
                f"\\( \\text{{t Statistics}} \\) section."
            )
        else:
            tstats = (
                f"Set \\( \\text{{tstats=True}} \\) to see how the \\( t \\) "
                "statistics are computed."
            )
            
        remark_nk = (
            f"\\( {str_color_values('Remark:', color='green')} \\) "
            "\\( n \\) and \\( k \\) is the sample size and total number of "
            "variables (dependent and independent) respectively."
        )
            
        return HtmlStrings(
            coeffs=coeffs,
            standard_errors=standard_errors,
            tstats=tstats,
            remark_nk=remark_nk
        )
            
    
    @cached_property
    def table_normal_coefs_latex(self) -> str:
        data = (
            self.x_rnd, self.y_rnd, self.xy_rnd, self.x_sq_rnd
        )
        column_sums = (
            f"\\sum x_{{i}} = {self.sum_x_rnd}",
            f"\\sum y_{{i}} = {self.sum_y}",
            f"\\sum x_{{i}}y_{{i}} = {self.sum_xy_rnd}",
            f"\\sum x_{{i}}^{{2}} = {self.sum_x_sq_rnd}"
        )
        column_sums = asarray(column_sums).reshape(1, -1)
        arr = vstack(tup=(hstack(tup=data), column_sums))
        nrows = arr.shape[0]
        index = array(["i"] + list(range(1, nrows)) + [""]).reshape(-1, 1)
        
        arr_latex = highlight_array_vals_arr(
            arr=arr,
            index=index,
            brackets=None,
            col_names=[f"x", "y", "xy", f"x^{{2}}"],
            color_rows=nrows
        ).replace("\\ {", "\\ \\hline {")
        
        return arr_latex
    
    @cached_property
    def table_normal_standard_errors_latex(self) -> str:
        
        data = (
            self.x_rnd,
            self.y_rnd,
            self.x_sq_rnd,
            self.fitted_yhat_rnd,
            self.residuals_squared_rnd
        )
        
        column_sums = (
            f"\\sum x_{{i}} = {self.sum_x_rnd}", 
            f"\\sum y_{{i}} = {self.sum_y_rnd}", 
            f"\\sum x_{{i}}^{{2}} = {self.sum_x_sq_rnd}", 
            f"\\sum \\hat{{y_{{i}}}} = {self.sum_fitted_yhat_rnd}", 
            (
                f"\\sum \\left(y_{{i}} - \\hat{{y_{{i}}}}\\right)^{{2}} "
                f"= {self.sum_residuals_squared_rnd}"
            )
        )
        
        column_sums = asarray(column_sums).reshape(1, -1)
        arr = vstack(tup=(hstack(tup=data), column_sums))
        nrows = arr.shape[0]
        index = ["i"] + list(range(1, nrows)) + [""]
        
        arr_latex = highlight_array_vals_arr(
            arr=arr,
            index=index,
            brackets=None,
            col_names=[
                f"x_{{i}}",
                f"y_{{i}}",
                f"x_{{i}}^{{2}}",
                f"\\hat{{y_{{i}}}}",
                f"\\left(y_{{i}} - \\hat{{y_{{i}}}}\\right)^{{2}}"
            ],
            color_rows=nrows
        ).replace("\\ {", "\\ \\hline {")
        
        return arr_latex
    
    
    @cached_property
    def table_fitted_and_residuals_latex(self) -> str:
        
        data = (
            self.x_rnd,
            self.y_rnd,
            self.fitted_yhat_rnd,
            self.residuals_rnd,
            self.residuals_squared_rnd
        )
        
        arr = hstack(tup=data)
        
        nrows = arr.shape[0]
        index = ["i"] + list(range(1, nrows + 1))
        
        n = self.n
        xi_color = ColorCSS.COLORDEFAULT.value
        fitted_color = "blue"
        residual_color = "orange"
        
        if n <= 5:
            xi_color_map = {
                (1, 1): xi_color,
                (2, 1): xi_color,
                (3, 1): xi_color,
                (4, 1): xi_color,
                (5, 1): xi_color
            }
            fitted_color_map = {
                (1, 3): fitted_color,
                (2, 3): fitted_color,
                (3, 3): fitted_color,
                (4, 3): fitted_color,
                (5, 3): fitted_color
            }
            residuals_color_map = {
                (1, 4): residual_color,
                (2, 4): residual_color,
                (3, 4): residual_color,
                (4, 4): residual_color,
                (5, 4): residual_color
            }
        else:
            xi_color_map = {
                (1, 1): xi_color,
                (2, 1): xi_color,
                (3, 1): xi_color,
                (n, 1): xi_color
            }
            fitted_color_map = {
                (1, 3): fitted_color,
                (2, 3): fitted_color,
                (3, 3): fitted_color,
                (n, 3): fitted_color
            }
            residuals_color_map = {
                (1, 4): residual_color,
                (2, 4): residual_color,
                (3, 4): residual_color,
                (n, 4): residual_color
            }
            
        color_idx = xi_color_map | fitted_color_map | residuals_color_map
        
        arr_latex = highlight_array_vals_arr(
            arr=arr,
            index=index,
            brackets=None,
            col_names=[
                f"x_{{i}}",
                f"y_{{i}}",
                f"\\hat{{y_{{i}}}}",
                f"y_{{i}} - \\hat{{y_{{i}}}}",
                f"\\left(y_{{i}} - \\hat{{y_{{i}}}}\\right)^{{2}}"
            ],
            color_map_indices=color_idx
        )
        
        return arr_latex
    
    
    @cached_property
    def predict_latex(self) -> str:
        
        steps_mathjax: list[str] = []
        self._counter += 1 # must update `self.counter` even though not used
        
        predict_xy_arr = self.calc_predict_at_x_rnd
        nrows = predict_xy_arr.shape[0]
        
        xis = [f"x_{{{idx}}}" for idx in range(1, self.k)]
        xis_str = (f"\\( {', '.join(xis)} \\) and \\( x_{{{self.k}}} \\)")
        fvars = (
            f"variable \\( x_{{1}} \\)"
            if self.k == 2 else
            f"variables \\( {xis_str} \\)"
        )
        
        ith = "" if nrows == 1 else f" \\( \\text{{ith}} \\)"
        
        steps_mathjax.append(
            f"To calculate the{ith} predicted value, "
            f"substitute for the independent {fvars} in the model below."
        )
        steps_mathjax.append(f"\\( y = {self.model} \\)")
        
        steps_mathjax.append(StemConstants.BORDER_HTML)

        for idx in range(nrows):
            
            eqtn_rep = self.model_str.replace("*x0", "")
            xis = [f"x{idx}" for idx in range(1, self.k)]
            for xi in xis:
                xpredic_val = predict_xy_arr[idx, 0]
                eqtn_rep = (
                    eqtn_rep
                        .replace("*", "\\:")
                        .replace(xi, str_color_values(f"({xpredic_val})"))
                )
            
            steps_mathjax.append(
                f"\\( \\hat{{y}}_{{{idx + 1}}} = {eqtn_rep} \\)")
            steps_mathjax.append(f"\\( \\quad = {predict_xy_arr[idx, 1]} \\)")
            
        if nrows > 1:
            steps_mathjax.append(StemConstants.BORDER_HTML)

            steps_mathjax.append(
                "These predicted values are presented in the table below."
            )
            
            index = [""] + list(range(1, nrows + 1))
            col_names = [f"\\quad x", f"\\hat{{y}}"]

            arr_latex = highlight_array_vals_arr(
                arr=predict_xy_arr,
                index=index,
                col_names=col_names,
                brackets=None
            )

            steps_mathjax.append(f"\\( {arr_latex} \\)")
        
        return steps_mathjax
        
    @cached_property
    def table_anova(self) -> list[str]:
        "ANOVA"
        
        steps_mathjax: list[str] = []
        
        arr = [
            [
                self.ssr,
                self.k - 1,
                self.msr,
                self.fstat_rnd,
                self.fcrit_rnd,
                self.pvalue_fstat_rnd
            ],
            [self.sse, self.n - self.k, self.mse, nan, nan, nan],
            [self.sst, self.n - 1, self.mst, nan, nan, nan]
        ]
        
        index = asarray([
            f"\\text{{Source}}",
            f"\\text{{Model}}",
            f"\\text{{Residual}}",
            f"\\hline \\text{{Total}}"
        ]).reshape(-1, 1)
        col_names = asarray([
            f"\\quad\\text{{SS}}",
            f"\\quad\\text{{DF}}",
            f"\\text{{Mean SS}}",
            f"\\quad\\text{{F}}_\\text{{{{calc}}}}",
            f"\\text{{F}}_{{\\alpha}}",
            f"\\text{{P value}}"
        ])
        
        arr_latex = highlight_array_vals_arr(
            arr=around(arr, self.decimals),
            brackets=None,
            index=index,
            col_names=col_names
        )
        
        steps_mathjax.append(f"\\( {arr_latex} \\)")
        
        return steps_mathjax
    
    
    @property
    def model_latex(self):

        steps_mathjax: list[str] = []
        
        steps_mathjax.append(
            "The regression coefficients as calculated above are presented "
            "below."
        )
        
        for idx, coeff in enumerate(self.beta_rnd):
            steps_mathjax.append(f"\\( \\quad b_{{{idx}}} = {coeff} \\)")
        
        steps_mathjax.append(
            "The regression equation obtained from the above coefficients is:"
        )
        steps_mathjax.append(f"\\( \\quad y = {self.model_latex} \\)")

        return steps_mathjax
    
    
    @cached_property
    def matrix_coeffs_latex(self) -> list[str]:
        """Regression coefficients"""
        steps_mathjax: list[str] = []
        
        return steps_mathjax
    
    
    @cached_property
    def normal_coeffs_formula_expanded_latex(self) -> list[str]:
        """Regression coefficients"""
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        
        # ---------------------------
        #   b1 - Slope / Gradient
        # ---------------------------
        
        steps_mathjax.append(
            "Below are the calculations for the regression coefficients."
        )
        
        title = (
            f"\\( {counter}.1) \\: \\beta_{{1}} "
            f"- \\text{{slope / gradient}} \\)"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The regression coefficient \\(\\beta_{{1}} \\) "
            f"(i.e. slope / gradient) is given by Equation "
            f"\\( ({{{counter}}}.1) \\) below."
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad\\beta_{{1}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"x_{{i}} \\: y_{{i}} - n \\: \\bar{{x}} \\: "
            f"\\bar{{y}}}}{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"x_{{i}}^{{2}} - n \\: \\bar{{x}}^{{2}}}} "
            f"\\qquad \\cdots \\qquad ({{{counter}}}.1) \\)"
        )
        steps_mathjax.append(
            "Create the following table for ease of calculations of the "
            f"parameters that appear in Equation \\(({{{counter}}}.1)\\) "
            "above."
        )
        
        steps_mathjax.append(f"\\( {self.table_normal_coefs_latex} \\)")
        
        steps_mathjax.append(
            "From the above table, we get the following summations (last row)."
        )
        steps_mathjax.append(
            f"\\( \\quad\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}} \\: "
            f"y_{{i}} = {self.sum_xy_rnd} \\:, \\quad "
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}}^{{2}} "
            f"= {self.sum_x_sq_rnd} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad"
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}} \\quad"
            f"\\longrightarrow\\quad\\bar{{x}} = {self.sum_x_rnd} "
            f"= \\frac{{1}}{{{self.n}}} "
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}} "
            f"= \\frac{{{self.sum_x_rnd}}}{{{self.n}}} "
            f"= {self.mean_x_rnd} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad"
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: y_{{i}} \\quad"
            f"\\longrightarrow\\quad\\bar{{y}} = {self.sum_y_rnd} "
            f"= \\frac{{1}}{{{self.n}}} "
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: y_{{i}}"
            f"= \\frac{{{self.sum_y_rnd}}}{{{self.n}}} "
            f"= {self.mean_y_rnd} \\)"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(
            f"Substitute the above values into Equation \\(({counter}.1)\\) "
            f"and work through to find the value of \\( \\beta_{{1}} \\)."
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\beta_{{1}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"x_{{i}} \\: y_{{i}} - n \\: \\bar{{x}} \\: "
            f"\\bar{{y}}}}{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"x_{{i}}^{{2}} - n \\: \\bar{{x}}^{{2}}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = \\displaystyle\\frac{{{self.sum_xy_rnd} "
            f"- {self.n} \\left({self.mean_x_rnd}\\right) "
            f"\\left({self.mean_y_rnd}\\right)}}{{{self.sum_x_sq_rnd} "
            f"- {self.n} \\: \\left({self.mean_x_rnd}\\right)^{{2}}}} \\)"
        )
        
        n_mean_x_mean_y = self.n * self.mean_x * self.mean_y
        n_mean_x_mean_y_rnd = around(n_mean_x_mean_y, self.decimals)
        n_mean_x_sq = self.n * self.mean_x ** 2
        n_mean_x_sq_rnd = around(n_mean_x_sq, self.decimals)
        
        steps_mathjax.append(
            f"\\( \\quad = \\displaystyle\\frac{{{self.sum_xy_rnd} "
            f"- {n_mean_x_mean_y_rnd}}}{{{self.sum_x_sq_rnd} "
            f"- {n_mean_x_sq_rnd}}} \\)"
        )
        
        _numer = around(self.sum_xy - n_mean_x_mean_y, self.decimals)
        
        steps_mathjax.append(
            f"\\( \\quad "
            f"= \\displaystyle\\frac{{{_numer}}}{{{self.sb_denom_rnd}}} \\)"
        )
        steps_mathjax.append(f"\\(\\quad = {self.beta1_rnd} \\)")
        
        # ---------------------------
        #   b0 - Intercept
        # ---------------------------
        
        title = f"\\( {counter}.2) \\: \\beta_{{0}} - \\text{{intercept}} \\)"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The regression coefficient \\( \\beta_{{0}} \\) "
            f"(i.e. intercept) is given by Equation \\(({counter}.2) \\) "
            "below."
        )
        steps_mathjax.append(
            f"\\( \\quad\\beta_{{0}} = \\bar{{y}} "
            f"- \\beta_{{1}} \\: \\bar{{x}} "
            f"\\qquad \\cdots \\qquad ({counter}.2) \\)"
        )
        
        steps_mathjax.append("where,")
        steps_mathjax.append(
            f"\\(\\quad\\bar{{y}} = {self.mean_y_rnd}\\:,"
            f"\\quad\\beta_{{1}} = {self.beta1_rnd}\\:,"
            f"\\quad\\bar{{x}} = {self.mean_x_rnd}\\)"
        )
        steps_mathjax.append("are from preceding calculations.")
        steps_mathjax.append(
            "Now substitute the above values into Equation "
            f"\\( ({{{counter}}}.2) \\) and work through to get the value of "
            f"\\( \\beta_{{0}} \\) as follows."
        )
        
        steps_mathjax.append(
            f"\\( \\beta_{{0}} = \\bar{{y}} - \\beta_{{1}} \\: \\bar{{x}} \\)"
        )
        
        self_b1 = self.beta1_rnd
        b1 = f"\\left({self_b1}\\right)" if self_b1 < 0 else self_b1
        
        steps_mathjax.append(
            f"\\( \\quad = {self.mean_y_rnd} "
            f"- {b1} \\: \\left({self.mean_x_rnd}\\right) \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.beta0_rnd} \\)")
        
        # ---------------------------
        #   b0 - Intercept
        # ---------------------------
        
        title = f"{counter}.3) Regression equation"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.extend(self.model_latex)
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_standard_errors_latex(self) -> list[str]:
        """Standard errors"""
        
        steps_mathjax: list[str] = []
        
        return steps_mathjax
    
    
    @cached_property
    def normal_standard_errors_latex(self) -> list[str]:
        """Standard errors"""
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        
        # Standard error for the intercept
        # --------------------------------
        
        steps_mathjax.append(
            "Below are the calculations for the standard errors."
        )
        
        title = (
            f"{counter}.1) Standard error for the intercept "
            f"\\( (\\hat{{\\beta_{{0}}}}) \\)"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The standard error for the intercept \\(\\beta_{{0}}\\) is "
            f"given by Equation \\( ({counter}.1) \\) below."
        )
        
        mse_formula = (
            f"\\displaystyle s(\\hat{{\\beta_{{0}}}}) "
            f"= \\sqrt{{MSE \\: \\left(\\frac{{1}}{{n}} "
            f"+ \\frac{{\\bar{{x}}^{{2}}}}{{\\sum\\limits_{{i=1}}^{{n}} \\: "
            f"x_{{i}}^{{2}} - n \\: \\bar{{x}}^{{2}}}}\\right)}}"
        )
        steps_mathjax.append(
            f"\\( \\qquad {mse_formula} \\qquad \\cdots \\qquad "
            f"({{{counter}}}.1) \\)"
        )
        steps_mathjax.append(
            "Prepare the following table for ease of calculations."
        )
        
        steps_mathjax.append(f"\\( {self.table_normal_standard_errors_latex} \\)")
        
        if self.coefficients:
            model_str = (
                f"obtained under \\( \\text{{Regression coefficients}} \\)"
            )
            coefs_str = ""
        else:
            model_str = ""
            coefs_str = (
                f" (set \\( \\text{{coefficients = True}} \\) for "
                "regression coefficients calculations)"
            )
            
        steps_mathjax.append(
            f"The quantity \\( (y_{{i}} - \hat{{y}}_{{i}}) \\) in the above "
            f"calculations represents the difference between the values "
            f"of dependent variable \\( y \\) and the fitted_and_residuals / predicted "
            f"values \\( \\hat{{y}} \\) which are found by substituting for "
            f"\\( x \\) in the regression model{model_str} with the values "
            f"of the independent variable \\( x \\){coefs_str}."
        )
        
        steps_mathjax.append(
            "With the summations in the above table, the computations are "
            "performed as follows."
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(
            f"\\( \\displaystyle\\bar{{x}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"x_{{i}}}}{{{self.n}}} "
            f"= \\frac{{{self.sum_x_rnd}}}{{{self.n}}} "
            f"= {self.mean_x_rnd} \\)"
        )
        
        steps_mathjax.append(
            f"\\( \\sum\\limits_{{i=1}}^{{n}} \\: x_{{i}}^{{2}} "
            f"- n \\: \\bar{{x}}^{{2}}"
            f"= {self.sum_x_sq_rnd} "
            f"- {self.n} \\: \\left({self.mean_x_rnd}\\right)^{{2}} "
            f"= {self.sb_denom_rnd} \\)"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(
            f"\\( \\displaystyle MSE = \\frac{{SSE}}{{n - k}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: (y_{{i}} "
            f"- \\hat{{y}}_{{i}})^{{2}}}}{{n - k}} "
            f"= \\frac{{{self.sse_rnd}}}{{{self.n} - {self.k}}} "
            f"= {self.mse_rnd} \\)"
        )
        steps_mathjax.append(self.html_str().remark_nk)
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(
            f"Replace the above values into Equation \\({counter}.1\\) and "
            f"work through as follows."
        )
        steps_mathjax.append(f"\\( {mse_formula} \\)")
        steps_mathjax.append(
            f"\\( \\quad = "
            f"\\sqrt{{{self.mse_rnd} \\: \\left(\\dfrac{{1}}{{{self.n}}} "
            f"+ \\dfrac{{\\left({self.mean_x_rnd}"
            f"\\right)^{{2}}}}{{{self.sb_denom_rnd}}}\\right)}} \\)"
        )
        
        n_rnd = around(1 / self.n, self.decimals)
        term2 = self.mean_x ** 2 / self.sb_denom_rnd
        term2_rnd = around(term2, self.decimals)
        
        steps_mathjax.append(
            f"\\( \\quad = "
            f"\\sqrt{{{self.mse_rnd} \\: \\left({n_rnd} "
            f"+ {term2_rnd} \\right)}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.sb0_rnd} \\)")
        
        # Standard error for the regression coefficient
        # ---------------------------------------------
        
        title = (
            f"{counter}.2) Standard error for the regression coefficient "
            f"\\( (\\hat{{\\beta_{{1}}}}) \\)"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The Standard error for the slope / gradient "
            f"\\( \\beta_{{1}} \\) is given by Equation "
            f"\\( ({counter}.2) \\) below"
        )
        
        steps_mathjax.append(
            f"\\( \\displaystyle s(\\hat{{\\beta_{{1}}}})"
            f"= \\sqrt{{\\dfrac{{MSE}}{{\\sum\\limits_{{i=1}}^{{{self.n}}} "
            f"\\: x_{{i}}^{{2}} - n \\: \\bar{{x}}^{{2}}}}}} "
            f"\\qquad \\cdots \\qquad ({counter}.2) \\)"
        )

        term = self.sum_x_sq - self.n * self.mean_x ** 2
        term_rnd = around(term, self.decimals)
        steps_mathjax.append(
            f"\\( \\quad "
            f"= \\sqrt{{\\dfrac{{{self.mse_rnd}}}{{{term_rnd}}}}} \\)"
        )     
        steps_mathjax.append(
            f"\\( \\quad "
            f"= \\sqrt{{{around(self.mse / term, self.decimals)}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.sb1_rnd} \\)")
        
        return steps_mathjax
            
    
    @cached_property
    def tstats_latex(self) -> list[str]:
        "t statistics"
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        
        steps_mathjax.append(
            f"The \\( t \\) statistics are calculated using Equation "
            f"\\( ({counter}.1) \\) below."
        )
        steps_mathjax.append(
            f"\\( \\quad\\displaystyle t "
            f"= \\frac{{\\hat{{\\beta}}}}{{s(\\hat{{\\beta}})}} "
            f"\\qquad \\cdots \\qquad ({counter}.1) \\)"
        )
        steps_mathjax.append("where,")
        steps_mathjax.append(
            f"\\( \\quad\\hat{{\\beta}} = {tex_to_latex(self.b_rnd)} - \\: \\) "
            f"Regression coefficients{self.html_str().coeffs}"
        )   
        steps_mathjax.append(
            f"\\( \\quad s(\\hat{{\\beta}}) = {tex_to_latex(self.std_errors_rnd)} "
            f"- \\: \\) Standard errors{self.html_str().standard_errors}"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(
            f"Substitute the above values of \\( \\hat{{\\beta}} \\) and "
            f"\\( s(\\hat{{\\beta}}) \\) into Equation \\( ({counter}.1) \\) "
            f"to calculate the \\( t \\) statistics."
        )
        
        for idx in range(len(self.b_rnd)):
            steps_mathjax.append(
                f"\\( \\qquad\\displaystyle t_{{{idx}}} "
                f"= \\frac{{\\hat{{\\beta_{{{idx}}}}}}}"
                f"{{s(\\hat{{\\beta_{{{idx}}}}})}} "
                f"= \\frac{{{self.b_rnd[idx]}}}{{{self.std_errors_rnd[idx]}}} "
                f"= {self.t_statistics_rnd[idx]} \\)"
            )
        
        return steps_mathjax
    
    
    @cached_property
    def pvalues_latex(self) -> list[str]:
        """p-values"""
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        n, k, conf_level = self.n, self.k, self.conf_level
        
        steps_mathjax.append(
            "The \\( p \\) values associated with the \\( t \\) statistics "
            f"with \\( (n - k - 1) = ({n} - {k} - 1) = {n - k - 1} \\) "
            f"degrees of freedom at the \\( {conf_level * 100}\\% \\) "
            "confidence level are given by:"
        )
        
        dfn = self.n - self.k - 1
        pvalues = self.pvalues_rnd
        
        p_eqtn_str = (
            f"\\( p_{{j}} "
            f"= \\text{{P}}\\big[t_{{\\alpha}} (n - k - 1) > t_{{j}}\\big] \\)"
        ) # do not use `i` because of the use of `\\big[\\big]`
        
        for idx in range(len(pvalues)):
            title = (
                f"{counter}.{idx + 1}) P value associated with "
                f"\\( \\beta_{{{idx}}} \\)"
            )
            steps_mathjax.append(html_bg_level2(title=title))
            
            steps_mathjax.append(p_eqtn_str.replace("j", str(idx)))
            steps_mathjax.append(
                f"\\( \\quad = P\\big[t_{{{conf_level}}}({dfn}) > "
                f"{self.t_statistics_rnd[idx]}\\big] \\)"
            )
            steps_mathjax.append(f"\\( \\quad = {pvalues[idx, 0]} \\)")
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(self.html_str().tstats)
        
        steps_mathjax.append(
            "The above \\( p \\) values are read from the Student\'s "
            "\\( t \\) tables by taking the upper tail of the given "
            "\\( t \\) statistic (any other statistical / mathematical "
            "software could also be used)."
        )
        
        return steps_mathjax
    
    
    @cached_property
    def confidence_intervals_latex(self) -> list[str]:
        """Confidence intervals"""
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        
        steps_mathjax.append(
            f"The \\( {self.conf_level * 100} \\)% confidence intervals are "
            f"calculated using Equation \\( ({counter}.1) \\) below."
        )
        
        eqtn_number = f"\\qquad \\cdots \\qquad ({counter}.1)"
        bi = f"\\beta_{{i}}"
        eqtn_str = (
            f"\\quad\\hat{{\\beta_{{i}}}} - s(\\hat{{\\beta_{{i}}}}) \\: "
            f"t_{{1 - \\alpha \\: / \\: 2}} (n - k) \\leq "
            f"{str_color_values(value=bi)} \\leq "
            f"\\hat{{\\beta_{{i}}}} + s(\\hat{{\\beta_{{i}}}}) \\: "
            f"t_{{1 - \\alpha \\: / \\: 2}} (n - k)"
        )
        
        steps_mathjax.append(f"\\( {eqtn_str} {eqtn_number} \\)")
        steps_mathjax.append("where")
        steps_mathjax.append(
            f"\\( \\quad\\hat{{\\beta}} = {tex_to_latex(self.b_rnd)} - \\: \\)"
            f"Regresssion coefficients{self.html_str().coeffs}"
        )
        steps_mathjax.append(
            f"\\( \\quad s(\\hat{{\\beta}}) "
            f"= {tex_to_latex(self.std_errors_rnd)} - \\: \\) "
            f"Standard errors{self.html_str().standard_errors}"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        n, k, sig_level = self.n, self.k, self.sig_level
         
        steps_mathjax.append(f"\\(\\qquad \\alpha = {sig_level}\\)")
        steps_mathjax.append(f"\\(\\qquad n - k = {n} - {k} = {n - k}\\)")
        steps_mathjax.append(
            f"\\(\\qquad\\displaystyle t_{{1 - \\alpha \\: / \\: 2}} (n - k) "
            f"= t_{{{round(1 - sig_level / 2, 4)}}}({n - k}) "
            f"= {self.tcrit_rnd} "
            f"\\leftarrow \\text{{Student's }} t \\text{{ table value}}\\)"
        )
        steps_mathjax.append(self.html_str().remark_nk)
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(
            f"Now substitute the above values of \\( \\hat{{\\beta}} \\), "
            f"\\( s(\\hat{{\\beta}}) \\) and "
            f"\\( t_{{1 - \\alpha \\: / \\: 2}}(n - k) \\) into Equation "
            f"\\( ({counter}.1) \\) and work through to get the "
            f"confidence intervals for the \\( {k} \\) regression parameters."
        )
        
        for idx in range(len(self.b)):
            title = (
                f"{(1 - sig_level) * 100}% confidence interval "
                f"for \\( \\beta_{{{idx}}} \\)"
            )
            steps_mathjax.append(html_bg_level2(title=title))
            
            # re-write this, using the `eqtn_str` defined earlier will
            # replace `i` only the first time
            bi = f"\\beta_{{{idx}}}"
            eqtn_str = (
                f"\\hat{{\\beta_{{i}}}} - s(\\hat{{\\beta_{{i}}}}) \\: "
                f"t_{{1 - \\alpha \\: / \\: 2}} (n - k) \\leq "
                f"{str_color_values(value=bi)} \\leq "
                f"\\hat{{\\beta_{{i}}}} + s(\\hat{{\\beta_{{i}}}}) \\: "
                f"t_{{1 - \\alpha \\: / \\: 2}} (n - k)"
            )
            
            steps_mathjax.append(f"\\( {eqtn_str} \\)")
            
            b_rnd = self.b_rnd
            std_rnd = self.std_errors_rnd
            tcrit_rnd = self.tcrit_rnd
            
            beta_idx = str_color_values(f"\\beta_{{{idx}}}")
            
            steps_mathjax.append(
                f"\\( {b_rnd[idx]} - {std_rnd[idx]} \\: "
                f"\\left({tcrit_rnd}\\right) \\leq {beta_idx} "
                f"\\leq {b_rnd[idx]} "
                f"+ {std_rnd[idx]} \\left({tcrit_rnd}\\right)\\)"
            )
            
            se_times_t = self.std_errors[idx] * self.tcrit
            se_times_t_rnd = around(se_times_t, self.decimals)
            
            steps_mathjax.append(
                f"\\( {b_rnd[idx]} - {se_times_t_rnd} "
                f"\\leq {beta_idx} \\leq {b_rnd[idx]} + {se_times_t_rnd} \\)"
            )
            steps_mathjax.append(
                f"\\( {self.ci_rnd[idx, 0]} \\leq {beta_idx} "
                f"\\leq {self.ci_rnd[idx, 1]} \\)"
            )
            
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_mathjax.append(
            "These calculated confidence intervals are summarized in the "
            "table below."
        )
        
        p = self.conf_level * 100
        arr_latex = highlight_array_vals_arr(
            arr=self.ci_rnd,
            index=([""] + self.calc_beta_str),
            col_names=[f"\\text{{{p}% Lower CI}}", f"\\text{{{p}% Upper CI}}"],
            brackets=None
        )

        steps_mathjax.append(f"\\( {arr_latex} \\)")
        
        return steps_mathjax
    
    
    @cached_property
    def other_stats_latex(sellf) -> list[str]:
        """Other statistics"""

        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_anova_latex(self) -> list[str]:
        "Analysis of variance"
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.extend(self.table_anova)
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_anova_latex(self) -> list[str]:
        "Analysis of variance"
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.extend(self.table_anova)
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_y_transpose_y_latex(self) -> list[str]:
        """
        Y.T @ Y
        """
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property 
    def matrix__b_transpose_x_transpose_y_latex(self) -> list[str]:
        """
        B.T @ X.T @ Y
        """
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_y_transpose_ones_y_latex(self) -> list[str]:
        """
        Y.T @ ones_ndarray @ Y
        """
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_ssr_latex(self) -> list[str]:
        """SSR"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_ssr_latex(self) -> list[str]:
        """SSR"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_msr_latex(self) -> list[str]:
        """MSR"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_msr_latex(self) -> list[str]:
        """MSR"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_sse_latex(self) -> list[str]:
        """SSE"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_sse_latex(self) -> list[str]:
        """SSE"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_mse_latex(self) -> list[str]:
        """MSE"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_mse_latex(self) -> list[str]:
        """MSE"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_sst_latex(self) -> list[str]:
        """SST"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_sst_latex(self) -> list[str]:
        """SST"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_mst_latex(self) -> list[str]:
        """mst"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_mst_latex(self) -> list[str]:
        """mst"""
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def fitted_and_residuals_latex(self) -> list[str]:
        """Fitted and residuals values"""
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        max_show = 3
        
        ith = (
            f"(i = 1, 2, 3, 4, 5)" 
            if self.k <= 6 else f"(i = 1, 2, 3, ..., {self.k - 1})"
        )
        xi = f"x_{{1}}" if self.k == 2 else f"x_{{i}} \\: for {ith}"
        
        vars_str = (
            "values of the independent variable" if self.k == 2
            else "values of the respective independent variables"
        )
        
        calc_str = (
            "These calculations are presented below"
            if self.k <= 5 else 
            "The first 3 and last 1 calculations are presented below"
        )
        
        steps_mathjax.append(
            f"The fitted (or predicted) values, denoted by \\( \\hat{{y}} \\), "
            f"are found by substituting for \\( {xi} \\) in the calculated "
            f"regression equation with the {vars_str} from the dataset "
            f"{calc_str}."
        )
        
        title = f"{counter}.1) Fitted values"
        steps_mathjax.append(html_bg_level2(title=title))
        
        xis = [f"x_{{{idx}}}" for idx in range(1, self.k)]
        xis_str = (f"\\( {', '.join(xis)} \\) and \\( x_{{{self.k}}} \\)")
        fvars = (
            f"variable \\( x_{{1}} \\)"
            if self.k == 2 else
            f"variables \\( {xis_str} \\)"
        )
        
        steps_mathjax.append(
            f"To calculate the \\( \\text{{ith}} \\) fitted value, "
            f"substitute for the independent {fvars} in the model below."
        )
        steps_mathjax.append(f"\\( y = {self.model} \\)")
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
        
        for idx in range(self.n):
            
            if max_show <= idx <= self.n - 2:
                continue
            
            eqtn_rep = self.model_str.replace("*x0", "").replace("*", "\\:")
            xis = [f"x{idx}" for idx in range(1, self.k)]
            for xi in xis:
                xi_val = str_color_values(f"({self.x[idx, 0]})")
                eqtn_rep = (eqtn_rep.replace(xi, xi_val))
            
            steps_mathjax.append(
                f"\\( \\hat{{y}}_{{{idx + 1}}} = {eqtn_rep} \\)")
            steps_mathjax.append(
                f"\\( \\quad = "
                f"{str_color_values(self.fitted_yhat_rnd[idx, 0], 'blue')} \\)"
            )
            
            if idx == max_show - 1:
                
                steps_mathjax.append(StemConstants.BORDER_HTML)
                
                steps_mathjax.append("\\( \\qquad\\qquad\\vdots \\)")
                steps_mathjax.append(
                    f"\\( \\qquad "
                    f"\\textit{{{self.n - max_show} calculations omitted}} \\)"
                )
                steps_mathjax.append("\\( \\qquad\\qquad\\vdots \\)")
            
            steps_mathjax.append(StemConstants.BORDER_HTML)
            
        steps_mathjax.append(
            "These fitted values are presented in the "
            f"\\( \\text{{third}} \\) column of the table below (shown after "
            f"\\( \\text{{Residuals}} \\) section.)"
        )
        
        title = f"{counter}.2) Residuals"
        steps_mathjax.append(html_bg_level2(title=title))
        
        calc_str = (
            "These calculations are presented below"
            if self.k <= 5 else 
            "The first 3 and last 1 calculations are presented below"
        )
        steps_mathjax.append(
            f"Residuals, denoted by \\( \\hat{{e}} \\), subtracting the "
            f"predicted values \\( (\\hat{{y}}) \\) from the observed "
            f"values of the dependent variable \\( (y) \\). {calc_str}."
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML)
            
        for idx in range(self.n):
            
            if max_show <= idx <= self.n - 2:
                continue 
            
            steps_mathjax.append(
                f"\\( \\hat{{e}}_{{{idx + 1}}} "
                f"= y - \\hat{{y}}_{{{idx + 1}}} \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = {self.y_rnd[idx, 0]} "
                f"- {self.fitted_yhat_rnd[idx, 0]} \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = "
                f"{str_color_values(self.residuals_rnd[idx, 0], 'orange')} \\)"
            )
            
            if idx == max_show - 1:
                
                steps_mathjax.append(StemConstants.BORDER_HTML)
                
                steps_mathjax.append("\\( \\qquad\\qquad\\vdots \\)")
                steps_mathjax.append(
                    f"\\( \\qquad "
                    f"\\textit{{{self.n - max_show} calculations omitted}} \\)"
                )
                steps_mathjax.append("\\( \\qquad\\qquad\\vdots \\)")
            
            steps_mathjax.append(StemConstants.BORDER_HTML)  
        
        steps_mathjax.append(
            "These fitted values and residuals are presented in the "
            f"\\( \\text{{third}} \\) and \\( \\text{{fourth}} \\) columns "
            "of the table below. The last column contains values of "
            "residuals squared (it is usually the norm to square the "
            "residuals)."
        )
        
        steps_mathjax.append(
            f"\\( {self.table_fitted_and_residuals_latex} \\)"
        )
        
        if self.n > 5:
            steps_mathjax.append(
                "The colored values are those that were calculated above for "
                "demonstration purposes."
            )
        
        return steps_mathjax
    

class OtherStatistics(LinearRegressionCalculations):
    pass