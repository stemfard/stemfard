from dataclasses import dataclass
from functools import cached_property

from numpy import (
    around, asarray, diag, diff, hstack, isnan, nan, pi, sqrt, vstack
)
from scipy import stats

from stemfard.core._html import html_bg_level2, html_style_bg
from stemfard.core.constants import StemConstants
from stemfard.core.arrays_highlight import highlight_array_vals_arr
from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import (
    str_color_math, str_color_text, str_eqtn_number, str_omitted
)
from stemfard.core._enumerate import ColorCSS
from stemfard.stats.regression_linear._base import BaseLinearRegression
from stemfard.core.statistical_tables import tables_fisher


@dataclass(slots=True, frozen=True)
class SumMeanSquaresNew:
    sum_sq: str | list[str]
    mean_sq: str | list[str]
    

@dataclass(slots=True, frozen=True)
class FStatisticAndPvalue:
    statistic: str
    p_value: str
    

@dataclass(slots=True, frozen=True)
class HtmlStrings:
    coeffs: str
    std_errors: str
    tstats: str
    log_lk: str
    remark_nk: str
    remark_k: str
    
    
@dataclass(slots=True, frozen=True)
class TableAppendix:
    ncols: int
    latex: str
    
    
@dataclass(slots=True, frozen=True)
class MatricesLatex:
    X_latex: str
    XT_latex: str
    Y_latex: str
    XTX_latex: str
    XTY_latex: str
    XTX_inv_latex: str
    beta_latex: str
    
    YT_latex: str
    YTY_latex: str
    bT_latex: str
    bTXTY_latex: str
    

class LinearRegressionLatexSteps(BaseLinearRegression):
    """Linear regression calculations"""
    
    @property
    def matrices_latex(self) -> MatricesLatex:
        
        return MatricesLatex(
            X_latex = tex_to_latex(self.X_rnd),
            XT_latex = tex_to_latex(self.XT_rnd),
            Y_latex = tex_to_latex(self.Y_rnd),
            XTX_latex = tex_to_latex(self.XTX_rnd),
            XTY_latex = tex_to_latex(self.XTY_rnd),
            XTX_inv_latex = tex_to_latex(self.XTX_inv_rnd),
            beta_latex = tex_to_latex(self.beta_rnd),
            # MSE
            YT_latex = tex_to_latex(self.YT_rnd),
            YTY_latex = self.YTY_rnd,
            bT_latex = tex_to_latex(self.beta_rnd),
            bTXTY_latex = self.bTXTY_rnd
        )
    
    @property
    def html_str(self) -> HtmlStrings:
        if "beta" in self.statistics:
            coeffs = (
                f" calculated in \\( \\text{{Regression coefficients}} \\) "
                "section."
            )
        else:
            coeffs = (
                f". Specify {str_color_text('beta')} in "
                f"{str_color_text('statistics')} parameter to show "
                "step-by-step calculations of regression coefficients."
            )
        
        if "se" in self.statistics:
            std_errors = (
                f" calculated in \\( \\text{{Standard erors}} \\) section."
            )
        else:
            std_errors = (
                f". Specify {str_color_text('se')} in "
                f"{str_color_text('statistics')} parameter to show "
                "step-by-step calculations of standard errors."
            )
            
        if "t" in self.statistics:
            tstats = (
                f"The above \\( t \\) statistics were calculated in the "
                f"\\( \\text{{t Statistics}} \\) section."
            )
        else:
            tstats = (
                f"Specify {str_color_text('t')} in  "
                f"{str_color_text('statistics')} parameter to show "
                "step-by-step calculations of \\( t \\) statistics."
            )
            
        if "loglk" in self.statistics:
            log_lk = (
                f"\\( \\text{{Log-Likelihood}} \\) statistic was calculated "
                f"in the \\( \\text{{Log-Likelihood}} \\) section."
            )
        else:
            log_lk = (
                f"Specify {str_color_text('loglk')} in  "
                f"{str_color_text('statistics')} parameter to show "
                "step-by-step calculations of the "
                f"\\( \\text{{Log-Likelihood}} \\) statistic."
            )
            
        remark_nk = (
            f"{str_color_text('Remark:')} \\( n \\) and \\( k \\) is the "
            "sample size and total number of variables in the regression "
            "model (dependent and independent) respectively."
        )
        
        remark_k = (
            f"{str_color_text('Remark:')} \\( k \\) is the total number of "
            "variables in the regression model (dependent and independent) "
            "respectively."
        )
            
        return HtmlStrings(
            coeffs=coeffs,
            std_errors=std_errors,
            tstats=tstats,
            log_lk=log_lk,
            remark_nk=remark_nk,
            remark_k=remark_k
        )
        
        
    @cached_property
    def table_appendix(self) -> TableAppendix:
        
        include_appendix = False
        counter = 0
        steps_mathjax: list[str] = []
        arr = hstack((self.y, self.fitted_values))
        
        col_names = [f"y_{{i}}", f"\\hat{{y}}"]
        sum_1_to_n = f"\\sum\\limits_{{i = 1}}^{{{self.n}}}"
        col_sums = [f"\\hline{sum_1_to_n} y_{{i}}", f"{sum_1_to_n} \\hat{{y}}"]
        
        if "dw" in self.statistics:
            
            include_appendix = True
            counter += 1
            
            ei = self.residuals
            ei2 = self.residuals_squared
            ei_diff2 = vstack((nan, self.diff_residuals ** 2))

            arr_dw = hstack((arr, ei, ei2, ei_diff2))
            
            col_names_dw = [
                f"e_{{i}} = (y_{{i}} - \\hat{{y_{{i}}}})",
                f"e_{{i}}^{{2}} = \\big(y_{{i}} - \\hat{{y_{{i}}}}\\big)^2",
                f"\\big(e_{{i}} - e_{{i - 1}}\\big)^{{2}}"
            ]
            
            col_sums.extend([
                f"{sum_1_to_n} e_{{i}}",
                f"{sum_1_to_n} e_{{i}}^{{2}}",
                f"{sum_1_to_n} \\big(e_{{i}} - e_{{i - 1}}\\big)^{{2}}"
            ])
            
            arr_with_zero = arr_dw.copy()
            arr_with_zero[isnan(arr_with_zero)] = 0
            col_sums_num = arr_with_zero.sum(axis=0).round(self.decimals)
            col_sums_str_num = [
                f"{str_} = {num}" for str_, num in zip(col_sums, col_sums_num)
            ]
            arr_dw = vstack((arr_dw.round(self.decimals), col_sums_str_num))
            ncols = arr_dw.shape[1]
            
            arr_latex = highlight_array_vals_arr(
                arr=arr_dw,
                index="auto",
                col_names=col_names + col_names_dw,
                heading="",
                last_row="",
                brackets=None,
                cap_label="Table",
                cap_number=f"A.{counter}",
                cap_title="Calculations for Durbin-Watson"
            )
            
            steps_mathjax.append(arr_latex)
            
        if "loglk" in self.statistics:
            
            include_appendix = True
            counter += 1
            
            arr_loglk = hstack((arr, self.residuals, self.residuals_squared))
            c1 = f"y_{{i}} - \\hat{{y_{{i}}}}"
            c2 = f"\\big(y_{{i}} - \\hat{{y_{{i}}}}\\big)^{{2}}"
            col_names_loglk = [c1, c2]
            
            col_sums_num = arr_loglk.sum(axis=0).round(self.decimals)
            col_sums.extend([f"{sum_1_to_n} {c1}", f"{sum_1_to_n} {c2}"])
            col_sums_str_num = [
                f"{str_} = {num}" for str_, num in zip(col_sums, col_sums_num)
            ]
            arr_loglk = vstack(
                (arr_loglk.round(self.decimals), col_sums_str_num)
            )
            
            ncols = arr_loglk.shape[1]
            
            arr_latex = highlight_array_vals_arr(
                arr=arr_loglk,
                index="auto",
                col_names=col_names + col_names_loglk,
                heading="",
                last_row="",
                brackets=None,
                cap_label="Table",
                cap_number=f"A.{counter}",
                cap_title="Calculations for Log-Likelihood"
            )
            
            steps_mathjax.append(arr_latex)
            
        if "skew" in self.statistics or "kurt" in self.statistics:
            
            include_appendix = True
            counter += 1
            
            c2 = f"\\big(y_{{i}} - \\hat{{y_{{i}}}}\\big)^{{2}}"
            c3 = f"\\big(y_{{i}} - \\hat{{y_{{i}}}}\\big)^{{3}}"
            c4 = f"\\big(y_{{i}} - \\hat{{y_{{i}}}}\\big)^{{4}}"
            
            if "skew" in self.statistics and not "kurt" in self.statistics:
                cap_title = "Calculations for Skewness"
                arr_sk = hstack(
                    (
                        arr,
                        self.y_minus_yhat_pow2,
                        self.y_minus_yhat_pow3
                    )
                )
            
                col_names_sk = [c2, c3]
                col_sums.extend([f"{sum_1_to_n} {c2}", f"{sum_1_to_n} {c3}"])
            
            elif "skew" not in self.statistics and "kurt" in self.statistics:
                cap_title = "Calculations for Kurtosis"
                arr_sk = hstack(
                    (
                        arr,
                        self.y_minus_yhat_pow2,
                        self.y_minus_yhat_pow4
                    )
                )

                col_names_sk = [c2, c4]
                col_sums.extend([f"{sum_1_to_n} {c2}", f"{sum_1_to_n} {c4}"])
                
            else: # both
                cap_title = "Calculations for Skewness and Kurtosis"
                arr_sk = hstack(
                    (
                        arr,
                        self.y_minus_yhat_pow2,
                        self.y_minus_yhat_pow3,
                        self.y_minus_yhat_pow4
                    )
                )
            
                col_names_sk = [c2, c3, c4]
                col_sums.extend(
                    [f"{sum_1_to_n} {c2}",
                     f"{sum_1_to_n} {c3}",
                     f"{sum_1_to_n} {c4}"]
                )
            
            col_sums_num = arr_sk.sum(axis=0).round(self.decimals)
            col_sums_str_num = [
                f"{str_} = {num}" for str_, num in zip(col_sums, col_sums_num)
            ]
            arr_sk = vstack((arr_sk.round(self.decimals), col_sums_str_num))
            ncols = arr_sk.shape[1]
            
            arr_latex = highlight_array_vals_arr(
                arr=arr_sk,
                index="auto",
                col_names=col_names + col_names_sk,
                brackets=None,
                heading="",
                last_row="",
                cap_label="Table",
                cap_number=f"A.{counter}",
                cap_title=cap_title
            )
            
            steps_mathjax.append(arr_latex)
            
        arr_latex_joined = " ".join(steps_mathjax)

        if include_appendix == True:
            return TableAppendix(ncols=ncols, latex=arr_latex_joined)
        return TableAppendix(ncols=0, latex="")
    
    def table_normal_coefs_latex(self, caption_num: int) -> str:
        data = (
            self.x_rnd, self.y_rnd, self.xy_rnd, self.x_squared_rnd
        )
        column_sums = [
            f"\\sum x_{{i}} = {self.sum_x_rnd}",
            f"\\sum y_{{i}} = {self.sum_y}",
            f"\\sum x_{{i}}y_{{i}} = {self.sum_xy_rnd}",
            f"\\sum x_{{i}}^{{2}} = {self.sum_x_squared_rnd}"
        ]
        column_sums = asarray(column_sums).reshape(1, -1)
        arr = vstack(tup=(hstack(tup=data), column_sums))
        nrows = arr.shape[0]
        
        arr_latex = highlight_array_vals_arr(
            arr=arr,
            index="auto",
            col_names=[f"x", "y", "xy", f"x^{{2}}"],
            heading="i",
            last_row="",
            cap_label="Table",
            cap_number=caption_num,
            cap_title=f"Values of x, y, xy and \\( x^{{2}} \\).",
            brackets=None,
            color_rows=nrows
        ).replace("\\ {", "\\ \\hline {")
        
        return arr_latex
    

    def table_normal_standard_errors_latex(self, caption_num: int) -> str:
        
        data = (
            self.x_rnd,
            self.y_rnd,
            self.x_squared_rnd,
            self.fitted_values_rnd,
            self.residuals_squared_rnd
        )
        
        column_sums = (
            f"\\sum x_{{i}} = {self.sum_x_rnd}", 
            f"\\sum y_{{i}} = {self.sum_y_rnd}", 
            f"\\sum x_{{i}}^{{2}} = {self.sum_x_squared_rnd}", 
            f"\\sum \\hat{{y_{{i}}}} = {self.sum_fitted_values_rnd}", 
            (
                f"\\sum \\left(y_{{i}} - \\hat{{y_{{i}}}}\\right)^{{2}} "
                f"= {self.sum_residuals_squared_rnd}"
            )
        )
        
        column_sums = asarray(column_sums).reshape(1, -1)
        arr = vstack(tup=(hstack(tup=data), column_sums))
        nrows = arr.shape[0]
        
        col_names=[
            f"x_{{i}}",
            f"y_{{i}}",
            f"x_{{i}}^{{2}}",
            f"\\hat{{y_{{i}}}}",
            f"\\left(y_{{i}} - \\hat{{y_{{i}}}}\\right)^{{2}}"
        ]
        
        arr_latex = highlight_array_vals_arr(
            arr=arr,
            index="auto",
            col_names=col_names,
            heading="i",
            last_row="",
            cap_label="Table",
            cap_number=caption_num,
            cap_title="Tabulated Values for Calculation of Standard Errors.",
            brackets=None,
            color_rows=nrows
        ).replace("\\ {", "\\ \\hline {")
        
        return arr_latex
    
    
    def table_fitted_and_residuals_latex(self, caption_num: int) -> str:
        
        data = (
            self.x_rnd,
            self.y_rnd,
            self.fitted_values_rnd,
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
        
        col_names = [
            f"x_{{i}}",
            f"y_{{i}}",
            f"\\hat{{y_{{i}}}}",
            f"y_{{i}} - \\hat{{y_{{i}}}}",
            f"\\left(y_{{i}} - \\hat{{y_{{i}}}}\\right)^{{2}}"
        ]
        
        arr_latex = highlight_array_vals_arr(
            arr=arr,
            index=index,
            cap_number=caption_num,
            cap_title="Fitted Values and Residuals",
            brackets=None,
            col_names=col_names,
            color_map_indices=color_idx
        )
        
        return arr_latex
    
    
    @cached_property
    def predict_latex(self) -> str:
        
        steps_mathjax: list[str] = []
        self._counter += 1
        
        predicted_arr = self.predict(self.predict_at_x, self.decimals)
        nrows = predicted_arr.shape[0]
        
        xis_str = (
            f"\\( {', '.join(self.xi_symbols[1:])} \\) and "
            f"\\( x_{{{self.k}}} \\)"
        )
        fvars = (
            f"variable \\( x_{{1}} \\)"
            if self.k == 2 else
            f"variables \\( {xis_str} \\)"
        )
        
        ith = "" if nrows == 1 else f" \\( \\text{{ith}} \\)"
        
        steps_mathjax.append(
            f"To calculate the{ith} predicted value, "
            f"substitute for the independent {fvars} in the regression "
            f"equation obtained in \\( \\text{{Regression coefficients}} \\) "
            "section (also given below)."
        )
        steps_mathjax.append(f"\\( y = {self.model_latex} \\)")
        
        steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)

        for idx in range(nrows):
            xpredict_val = predicted_arr[idx, 0]
            steps_mathjax.append(f"When \\( x_{{1}} = {xpredict_val}, \\)")
            eqtn_rep = self.model_string
            for xi in self.xi_symbols[1:]:
                eqtn_rep = (
                    eqtn_rep
                        .replace("*", "\\:")
                        .replace(xi, str_color_math(f"({xpredict_val})"))
                )
            
            steps_mathjax.append(
                f"\\( \\hat{{y}}_{{{idx + 1}}} = {eqtn_rep} \\)"
            )
            idx_str = f"\\quad = {predicted_arr[idx, 1]}"
            steps_mathjax.append(f"\\( {str_color_math(idx_str, 'blue')} \\)")
            
            if idx != nrows - 1:
                steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
            
        if nrows > 1:
            steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)

            steps_mathjax.append(
                "These predicted values calculated above are presented in "
                f"{str_color_text(f'Table {self.caption_num + 1}')} below."
            )
            
            index = [""] + list(range(1, nrows + 1))
            col_names = self.xi_latex[1:] + [f"\\hat{{y}}"]

            self.caption_num += 1
            caption = (
                "Predicted Values of The Dependent Variable Given "
                f"\\( {self.xi_latex_str} \\)."
            )
            
            arr_latex = highlight_array_vals_arr(
                arr=predicted_arr,
                index=index,
                col_names=col_names,
                heading="",
                last_row="",
                cap_label="Table",
                cap_number=self.caption_num,
                cap_title=caption,
                brackets=None,
                color_map_cols={
                    1: ColorCSS.COLORDEFAULT.value,
                    2: "blue"
                }
            )
            
            steps_mathjax.append(arr_latex)
        
        return steps_mathjax
    
    
    def anova_table_general(self, caption_num: int) -> str:
        
        anova_str = [
            [
                f"\\text{{SSR}}",
                f"\\text{{k - 1}}",
                f"\\text{{MSR}} = \\frac{{\\text{{SSR}}}}{{\\text{{k-1}}}}",
                f"\\frac{{\\text{{SSR}}}}{{\\text{{SSE}}}}",
                (
                    f"\\text{{P}}\\left(\\text{{F}} > "
                    f"\\frac{{\\text{{SSR}}}}{{\\text{{SSE}}}}\\right)"
                ),
                ""
            ],
            [
                f"\\text{{SSE}}",
                f"\\text{{n - k}}",
                f"\\text{{MSE}} = \\frac{{\\text{{SSE}}}}{{\\text{{n - k}}}}",
                "",
                "",
                ""
            ],
            [
                f"\\text{{SST}}",
                f"\\text{{n - 1}}",
                f"\\text{{MST}} = \\frac{{\\text{{SST}}}}{{\\text{{n - 1}}}}",
                "",
                "",
                ""
            ]
        ]
        index = [
            f"\\text{{Source}}",
            f"\\text{{Model}}",
            f"\\hline\\text{{Residual}}",
            f"\\hline\\text{{Total}}"
        ]
        col_names = [
            f"\\text{{Sum of Squares}}",
            f"\\quad\\text{{DF}}\\quad",
            f"\\text{{Mean SS}}",
            f"\\quad\\text{{F}}_\\text{{calc}}",
            f"\\text{{F}}_{{\\alpha\\:/\\:2}}",
            f"\\text{{P value}}"
        ]
        
        arr_latex = highlight_array_vals_arr(
            arr=anova_str,
            index=index,
            col_names=col_names,
            cap_number=caption_num,
            cap_title="Analysis of Variance Table (General Form).",
            brackets=None
        ).replace("r|rrrrrr", "|l|c|c|c|c|c|c|")
        
        return arr_latex
        

    def table_anova_statistics(self, caption_num: int) -> list[str]:
        "ANOVA table with statistics"
        
        steps_mathjax: list[str] = []
        
        arr = [
            [
                self.ssr,
                self.df_r,
                self.msr,
                self.f_statistic_rnd,
                self.f_critical_rnd,
                self.f_pvalue_rnd
            ],
            [self.sse, self.df_e, self.mse, nan, nan, nan],
            [self.sst, self.df_t, self.mst, nan, nan, nan]
        ]
        
        index = asarray([
            f"\\text{{Source}}",
            f"\\text{{Model}}",
            f"\\text{{Residual}}",
            f"\\hline \\text{{Total}}"
        ])
        col_names = asarray([
            f"\\text{{Sum of Squares}}",
            f"\\quad\\text{{DF}}",
            f"\\text{{Mean SS}}",
            f"\\quad\\text{{F}}_\\text{{calc}}",
            f"\\text{{F}}_{{{self.alpha / 2 }}}",
            f"\\text{{P value}}"
        ])
        
        arr_latex = highlight_array_vals_arr(
            arr=around(arr, self.decimals),
            index=index,
            col_names=col_names,
            cap_number=caption_num,
            cap_title="Analysis of Variance",
            brackets=None
        )
        
        steps_mathjax.append(arr_latex)
        
        return steps_mathjax
    
    
    def regression_model_latex(self, counter: str):

        steps_mathjax: list[str] = []
        
        if self.is_simple_linear and self.slinear_method == "normal":
            steps_mathjax.append(
                f"The regression coefficients as calculated in "
                f"{str_color_text(f'Section {counter}.1')} and "
                f"{str_color_text(f'{counter}.2')} are presented below."
            )
        else:
            steps_mathjax.append(
                "The regression coefficients as calculated above are presented "
                "below."
            )
        
        beta_rnd_flat = self.beta_rnd.flatten()
        for idx, beta in enumerate(beta_rnd_flat):
            steps_mathjax.append(f"\\( \\quad b_{{{idx}}} = {beta} \\)")
        
        steps_mathjax.append(
            "The regression equation is obtained by substituting the above "
            "regression coefficients into "
            f"{str_color_text(f'Equation {counter}.1')}. This gives the following "
            "equation."
        )
        steps_mathjax.append(f"\\( \\quad y = {self.model_latex} \\)")

        return steps_mathjax
    
    
    @cached_property
    def matrix_beta_latex(self) -> list[str]:
        """Regression coefficients"""
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        
        steps_mathjax.append(
            "The regression parameters are found by evaluating "
            f"{str_color_text(f'Equation {counter}.1')} below."
        )
        steps_mathjax.append(
            f"\\( \\quad\\hat{{\\beta}} "
            f"= \\left(X^{{T}} X\\right)^{{-1}} X^{{T}}Y"
            f"{str_eqtn_number(f'{counter}.1')} \\)"
        )
        steps_mathjax.append("where,")
        
        s = "" if self.k == 2 else "s"
        
        steps_mathjax.append(
            f"\\( X - \\) Matrix formed by appending a columns of "
            f"\\( 1 \\)'s to the left of the independent variable{s}."
        )
        steps_mathjax.append(
            f"\\( \\quad\\implies X = {self.matrices_latex.X_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( X^{{T}} - \\) Transpose of matrix \\( X \\) formed above."
        )
        steps_mathjax.append(
            f"\\( \\quad\\implies X^{{T}} = {self.matrices_latex.XT_latex} \\)"
        )
        steps_mathjax.append("\\( Y - \\) Values of the dependent variable.")
        steps_mathjax.append(
            f"\\( \\quad\\implies Y {self.matrices_latex.Y_latex} \\)"
        )
        
        title = f"{counter}.1) STEP 1: Evaluate \\( X^{{T}} X \\)"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"\\( X^{{T}} X = {self.matrices_latex.XT_latex} "
            f"{self.matrices_latex.X_latex} \\)")
        steps_mathjax.append(
            f"\\( \\quad = {self.matrices_latex.XTX_latex} \\)"
        )
        
        title = f"{counter}.2) STEP 2: Find the inverse of \\( X^{{T}} X \\)"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"\\( \\quad\\left(X^{{T}} X\\right)^{{-1}} "
            f"= {self.matrices_latex.XTX_inv_latex} \\)"
        )
        
        title = f"{counter}.3) STEP 3: Evaluate \\( X^{{T}} Y \\)"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"\\( X^{{T}} Y = {self.matrices_latex.XT_latex} "
            f"{self.matrices_latex.Y_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.matrices_latex.XTY_latex} \\)"
        )
        
        title = (
            f"{counter}.4) STEP 3: Evaluate "
            f"\\( \\beta = \\left(X^{{T}} X\\right)^{{-1}} X^{{T}}Y \\)"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"\\( \\beta = \\left(X^{{T}} X\\right)^{{-1}} X^{{T}}Y \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.matrices_latex.XTX_inv_latex} "
            f"{self.matrices_latex.XTY_latex} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.matrices_latex.beta_latex} \\)"
        )
        
        steps_mathjax.append(
            "The result above gives the required regression coefficients. "
            "That is,"
        )
        for symbol, beta in zip(self.beta_symbols, self.beta_rnd.flatten()):
            steps_mathjax.append(f"\\( {symbol} = {beta} \\)")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_beta_formula_expanded_latex(self) -> list[str]:
        """Regression coefficients"""
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        
        # ---------------------------
        #   b1 - Slope / Gradient
        # ---------------------------
        
        steps_mathjax.append(
            f"The equation representing a simple linear regression model is "
            f"given by {str_color_text(f'Equation {counter}.1')} below."
        )
        steps_mathjax.append(
            f"\\( \\quad y = \\beta_{{0}} + \\beta_{{1}} x_{{1}} "
            f"{str_eqtn_number(f'{counter}.1')} \\)"
        )
        steps_mathjax.append(
            f"The coefficients \\( \\beta_{{0}} \\) and \\( \\beta_{{1}} \\) "
            "are calculated as follows."
        )
        
        title = (
            f"\\( {counter}.1) \\: \\beta_{{1}} "
            f"- \\text{{slope / gradient}} \\)"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The regression coefficient \\(\\beta_{{1}} \\) is given by "
            f"{str_color_text(f'Equation {counter}.2')} below."
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad\\beta_{{1}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"x_{{i}} \\: y_{{i}} - n \\: \\bar{{x}} \\: "
            f"\\bar{{y}}}}{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"x_{{i}}^{{2}} - n \\: \\bar{{x}}^{{2}}}} "
            f"{str_eqtn_number(f'{counter}.2')} \\)"
        )
        steps_mathjax.append(
            f"Create {str_color_text(f'Table {self.caption_num + 1}')} for "
            "ease of calculations of the parameters that appear in "
            f"{str_color_text(f'Equation {counter}.2')} above."
        )
        
        self.caption_num += 1
        steps_mathjax.append(self.table_normal_coefs_latex(self.caption_num))
        
        steps_mathjax.append(
            f"From {str_color_text(f'Table {self.caption_num}')}, we get the "
            "following summations (last row)."
        )
        steps_mathjax.append(
            f"\\( \\quad\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}} \\: "
            f"y_{{i}} = {self.sum_xy_rnd} \\:, \\quad "
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}}^{{2}} "
            f"= {self.sum_x_squared_rnd} \\)"
        )
        steps_mathjax.append(
            "\\( \\displaystyle\\quad"
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}} \\quad"
            f"\\longrightarrow\\quad\\bar{{x}} = {self.sum_x_rnd} "
            f"= \\frac{{1}}{{{self.n}}} "
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: x_{{i}} "
            f"= \\frac{{{self.sum_x_rnd}}}{{{self.n}}} "
            f"= {self.mean_x_rnd} \\)"
        )
        steps_mathjax.append(
            "\\( \\displaystyle\\quad"
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: y_{{i}} \\quad"
            f"\\longrightarrow\\quad\\bar{{y}} = {self.sum_y_rnd} "
            f"= \\frac{{1}}{{{self.n}}} "
            f"\\sum\\limits_{{i=1}}^{{{self.n}}} \\: y_{{i}}"
            f"= \\frac{{{self.sum_y_rnd}}}{{{self.n}}} "
            f"= {self.mean_y_rnd} \\)"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(
            f"Substitute these values into "
            f"{str_color_text(f'Equation {counter}.2')} above and work "
            f"through to find the value of \\( \\beta_{{1}} \\)."
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
            f"\\left({self.mean_y_rnd}\\right)}}{{{self.sum_x_squared_rnd} "
            f"- {self.n} \\: \\left({self.mean_x_rnd}\\right)^{{2}}}} \\)"
        )
        
        n_mean_x_mean_y = self.n * self.mean_x * self.mean_y
        n_mean_x_mean_y_rnd = around(n_mean_x_mean_y, self.decimals)
        n_mean_x_squared = self.n * self.mean_x ** 2
        n_mean_x_squared_rnd = around(n_mean_x_squared, self.decimals)
        
        steps_mathjax.append(
            f"\\( \\quad = \\displaystyle\\frac{{{self.sum_xy_rnd} "
            f"- {n_mean_x_mean_y_rnd}}}{{{self.sum_x_squared_rnd} "
            f"- {n_mean_x_squared_rnd}}} \\)"
        )
        
        _numer = around(self.sum_xy - n_mean_x_mean_y, self.decimals)
        
        steps_mathjax.append(
            "\\( \\quad "
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
            f"(i.e. intercept) is given by "
            f"{str_color_text(f'Equation {counter}.3')} below."
        )
        steps_mathjax.append(
            f"\\( \\quad\\beta_{{0}} = \\bar{{y}} "
            f"- \\beta_{{1}} \\: \\bar{{x}} "
            f"{str_eqtn_number(f'{counter}.3')} \\)"
        )
        
        steps_mathjax.append("where,")
        steps_mathjax.append(
            f"\\(\\quad\\bar{{y}} = {self.mean_y_rnd}\\:,"
            f"\\quad\\beta_{{1}} = {self.beta1_rnd}\\:,"
            f"\\quad\\bar{{x}} = {self.mean_x_rnd}\\)"
        )
        
        steps_mathjax.append(
            "are from calculations presented in "
            f"{str_color_text(f'Section {counter}.1')} above."
        )
        steps_mathjax.append(
            "Substitute the above values into "
            f"{str_color_text(f'Equation {counter}.3')} above and work "
            f"through to get the value of \\( \\beta_{{0}} \\)."
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
        #   Regression model
        # ---------------------------
        
        title = f"{counter}.3) Regression equation"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.extend(self.regression_model_latex(counter))
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_standard_errors_latex(self) -> list[str]:
        """Standard errors"""
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        
        MSE = self.mse_rnd
        XTX_inv_latex = tex_to_latex(self.XTX_inv_rnd)
        MSE_XTX_inv_latex = tex_to_latex(self.cov_beta_rnd)
        diag_cova_beta = tex_to_latex(diag(self.cov_beta_rnd))
        standard_errors_latex = tex_to_latex(self.stderrors_rnd)
        
        steps_mathjax.append(
            f"The standard errors are found by evaluating the square root of "
            f"the major diagonal of the matrix found from "
            f"{str_color_text(f'Equation {counter}.1')} below."
        )
        
        sb_eqtn = (
            f"s^{{2}}(\\hat{{\\beta}})^{{\\color{{red}}{{\\star}}}} "
            f"= \\text{{MSE}} \\: (X^{{T}}X)^{{-1}} "
        )
        steps_mathjax.append(
            f"\\( {sb_eqtn} {str_eqtn_number(f'{counter}.1')} \\)"
        )
        
        title = f"STEP 1: Evaluate MSE"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{MSE}} "
            f"= \\frac{{\\text{{SSE}}}}{{n - k}} {str_eqtn_number('a')} \\)"
        )
        steps_mathjax.append("where,")
        steps_mathjax.append(
            f"\\( n = {self.n} \\:, k = {self.k} \\) "
            "(sample size and total number of variables in the model)."
        )
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        steps_mathjax.extend(self.matrix_sse_mse.sum_sq)
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        steps_mathjax.extend(self.matrix_sse_mse.mean_sq)
        
        title = f"STEP 2: Get \\( (X^{{T}}X)^{{-1}} \\)"
        steps_mathjax.append(html_bg_level2(title=title))
        
        if "beta" in self.statistics:
            beta_str = (
                f"This quantity was calculated in "
                f"\\( \\text{{Regression Coefficients}} \\) section above."
            )
        else:
            beta_str = (
                f"Specify {str_color_text('beta')} in "
                f"{str_color_text('statistics')} to show step by step "
                f"calculations of the above quantity."
            )
        
        steps_mathjax.append(beta_str)
        steps_mathjax.append(f"\\( (X^{{T}}X)^{{-1}} = {XTX_inv_latex} \\)")
        
        title = f"STEP 3: Compute the standard errors"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"Subtitute the result of \\( \\text{{STEP 1}} \\) and "
            f"\\( \\text{{STEP 2}} \\) into "
            f"{str_color_text(f'Equation {counter}.1')} and work though to "
            "get the required standard errors."
        )
        steps_mathjax.append(f"\\( {sb_eqtn} \\)")
        steps_mathjax.append(f"\\( \\quad = {MSE} {XTX_inv_latex} \\)")
        steps_mathjax.append(f"\\( \\quad = {MSE_XTX_inv_latex} \\)")
        steps_mathjax.append(
            "The standard errors are found by evaluating the square root of "
            "the values on the major diagonal of the matrix above. That is,"
        )
        steps_mathjax.append(
            f"\\( s(\\hat{{\\beta}}) = \\sqrt{{{diag_cova_beta}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {standard_errors_latex} \\)")
        
        steps_mathjax.append(f"The standard errors are thus given as:")
        
        for idx in range(len(self.stderrors)):
            steps_mathjax.append(
                f"\\( s(\\hat{{\\beta_{{{idx}}}}}) "
                f"= {self.stderrors_rnd[idx, 0]} \\)"
            )
        
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
            f"given by {str_color_text(f'Equation {counter}.1')} below."
        )
        
        mse_formula = (
            f"\\displaystyle s(\\hat{{\\beta_{{0}}}}) "
            f"= \\sqrt{{\\text{{MSE}} \\: \\left(\\frac{{1}}{{n}} "
            f"+ \\frac{{\\bar{{x}}^{{2}}}}{{\\sum\\limits_{{i=1}}^{{n}} \\: "
            f"x_{{i}}^{{2}} - n \\: \\bar{{x}}^{{2}}}}\\right)}}"
        )
        steps_mathjax.append(
            f"\\( \\quad {mse_formula} {str_eqtn_number(f'{counter}.1')} \\)"
        )
        steps_mathjax.append(
            f"Prepare {str_color_text(f'Table {self.caption_num + 1}')} "
            "below for ease of calculations."
        )
         
        self.caption_num += 1
        steps_mathjax.append(
            self.table_normal_standard_errors_latex(self.caption_num)
        )
        
        if "fit" in self.statistics:
            fitted_str = (
                "which are calculated in "
                f"\\( \\text{{Fitted Values and Residuals}} \\) section"
            )
        else:
            fitted_str = (
                f". (Set {str_color_text('fit')} in "
                f"{str_color_text('statistics')} to show step-by-step "
                "calculations of fitted values"
            )
            
        steps_mathjax.append(
            f"The values in column \\( \\hat{{y_{{i}}}} \\) are the fitted "
            f"values {fitted_str}."
        )
        
        steps_mathjax.append(
            f"With the summations in  "
            f"{str_color_text(f'Table {self.caption_num}')} above, the "
            "computations are performed as follows."
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)
        
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
            f"= {self.sum_x_squared_rnd} "
            f"- {self.n} \\: \\left({self.mean_x_rnd}\\right)^{{2}} "
            f"= {self.sb_denom_rnd} \\)"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{MSE}} "
            f"= \\frac{{\\text{{SSE}}}}{{n - k}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{self.n}}} \\: (y_{{i}} "
            f"- \\hat{{y}}_{{i}})^{{2}}}}{{n - k}} "
            f"= \\frac{{{self.sse_rnd}}}{{{self.n} - {self.k}}} "
            f"= {self.mse_rnd} \\)"
        )
        steps_mathjax.append(self.html_str.remark_nk)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)
        
        steps_mathjax.append(
            "Replace the above values into "
            f"{str_color_text(f'Equation {counter}.1')} and work through as "
            "follows."
        )
        steps_mathjax.append(f"\\( {mse_formula} \\)")
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad = "
            f"\\sqrt{{{self.mse_rnd} \\: \\left(\\frac{{1}}{{{self.n}}} "
            f"+ \\frac{{\\left({self.mean_x_rnd}"
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
            f"\\( \\beta_{{1}} \\) is given by "
            f"{str_color_text(f'Equation {counter}.2')} below."
        )
        
        steps_mathjax.append(
            f"\\( \\displaystyle s(\\hat{{\\beta_{{1}}}})"
            f"= \\sqrt{{\\frac{{\\text{{MSE}}}}{{\\sum\\limits_{{i=1}}^{{{self.n}}} "
            f"\\: x_{{i}}^{{2}} - n \\: \\bar{{x}}^{{2}}}}}} "
            f"{str_eqtn_number(f'{counter}.2')} \\)"
        )

        term = self.sum_x_squared - self.n * self.mean_x ** 2
        term_rnd = around(term, self.decimals)
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= \\sqrt{{\\frac{{{self.mse_rnd}}}{{{term_rnd}}}}} \\)"
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
            "The \\( t \\) statistics are calculated using "
            f"{str_color_text(f'Equation {counter}.1')} below."
        )
        steps_mathjax.append(
            f"\\( \\quad\\displaystyle t_{{i}} "
            f"= \\frac{{\\hat{{\\beta_{{i}}}}}}{{s(\\hat{{\\beta_{{i}}}})}} "
            f"{str_eqtn_number(f'{counter}.1')} \\)"
        )
        steps_mathjax.append("where,")
        steps_mathjax.append(
            f"\\( \\quad\\hat{{\\beta}} = {tex_to_latex(self.beta_rnd)} - \\: "
            f"\\) Regression coefficients{self.html_str.coeffs}"
        )   
        steps_mathjax.append(
            f"\\( \\quad s(\\hat{{\\beta}}) "
            f"= {tex_to_latex(self.stderrors_rnd)} - \\: \\) "
            f"Standard errors{self.html_str.std_errors}"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(
            f"Substitute these values of \\( \\hat{{\\beta}} \\) and "
            f"\\( s(\\hat{{\\beta}}) \\) into "
            f"{str_color_text(f'Equation {counter}.1')} above to calculate "
            "the \\( t \\) statistics."
        )
        
        for idx in range(len(self.beta_rnd)):
            steps_mathjax.append(
                f"\\( \\displaystyle t_{{{idx}}} "
                f"= \\frac{{\\hat{{\\beta_{{{idx}}}}}}}"
                f"{{s(\\hat{{\\beta_{{{idx}}}}})}} "
                f"= \\frac{{{self.beta_rnd[idx, 0]}}}"
                f"{{{self.stderrors_rnd[idx, 0]}}} "
                f"= {self.t_statistics_rnd[idx, 0]} \\)"
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
        steps_mathjax.append(self.html_str.tstats)
        
        dfn = self.n - self.k - 1
        pvalues = self.p_values_rnd
        
        p_eqtn_str = (
            f"\\( p_{{j}} "
            f"= \\text{{P}}\\big[t_{{\\alpha}} (n - k - 1) > t_{{j}}\\big] \\)"
        ) # do not use `p_{{i}}` because of the use of `\\big[\\big]`
        
        for idx in range(len(pvalues)):
            title = (
                f"{counter}.{idx + 1}) P value associated with "
                f"\\( \\beta_{{{idx}}} \\)"
            )
            steps_mathjax.append(html_bg_level2(title=title))
            
            steps_mathjax.append(p_eqtn_str.replace("j", str(idx)))
            steps_mathjax.append(
                f"\\( \\quad = P\\big[t_{{{conf_level}}}({dfn}) > "
                f"{self.t_statistics_rnd[idx, 0]}\\big] \\)"
            )
            steps_mathjax.append(f"\\( \\quad = {pvalues[idx, 0]} \\)")
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(
            "The above \\( p \\) values are read from the Student\'s "
            "\\( t \\) tables by taking the upper tail of the given "
            "\\( t \\) statistic. Alternatively, you can also use any "
            "statistical or mathematical software."
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
            f"calculated using {str_color_text(f'Equation {counter}.1')} "
            "below."
        )
        
        eqtn_number = f"{str_eqtn_number(f'{counter}.1')}"
        bi = f"\\beta_{{i}}"
        eqtn_str = (
            f"\\quad\\hat{{\\beta_{{i}}}} - s(\\hat{{\\beta_{{i}}}}) \\: "
            f"t_{{1 - \\alpha \\: / \\: 2}} (n - k) \\leq "
            f"{str_color_math(value=bi)} \\leq "
            f"\\hat{{\\beta_{{i}}}} + s(\\hat{{\\beta_{{i}}}}) \\: "
            f"t_{{1 - \\alpha \\: / \\: 2}} (n - k)"
        )
        
        steps_mathjax.append(f"\\( {eqtn_str} {eqtn_number} \\)")
        steps_mathjax.append("where")
        steps_mathjax.append(
            f"\\( \\quad\\hat{{\\beta}} = {tex_to_latex(self.beta_rnd)} - \\: "
            f"\\) Regresssion coefficients{self.html_str.coeffs}"
        )
        steps_mathjax.append(
            f"\\( \\quad s(\\hat{{\\beta}}) "
            f"= {tex_to_latex(self.stderrors_rnd)} - \\: \\) "
            f"Standard errors{self.html_str.std_errors}"
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        n, k, alpha = self.n, self.k, self.alpha
         
        steps_mathjax.append(f"\\(\\quad \\alpha = {alpha}\\)")
        steps_mathjax.append(f"\\(\\quad n - k = {n} - {k} = {n - k}\\)")
        steps_mathjax.append(
            f"\\(\\quad\\displaystyle t_{{1 - \\alpha \\: / \\: 2}} (n - k) "
            f"= t_{{{round(1 - alpha / 2, 4)}}}({n - k}) "
            f"= {self.t_critical_rnd} "
            f"\\leftarrow \\text{{Student's }} t \\text{{ table value}}\\)"
        )
        steps_mathjax.append(self.html_str.remark_nk)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(
            f"Substitute these values of \\( \\hat{{\\beta}} \\), "
            f"\\( s(\\hat{{\\beta}}) \\) and "
            f"\\( t_{{1 - \\alpha \\: / \\: 2}}(n - k) \\) into "
            f"{str_color_text(f'Equation {counter}.1')} and work through to "
            f"get the confidence intervals for the \\( {k} \\) regression "
            "parameters."
        )
        
        for idx in range(len(self.beta)):
            title = (
                f"{counter}.{idx + 1}) {(self.conf_level) * 100}% confidence "
                f"interval for \\( \\beta_{{{idx}}} \\)"
            )
            steps_mathjax.append(html_bg_level2(title=title))
            
            # re-write this, using the `eqtn_str` defined earlier will
            # replace `i` only the first time
            bi = f"\\beta_{{{idx}}}"
            eqtn_str = (
                f"\\hat{{\\beta_{{i}}}} - s(\\hat{{\\beta_{{i}}}}) \\: "
                f"t_{{1 - \\alpha \\: / \\: 2}} (n - k) \\leq "
                f"{str_color_math(value=bi)} \\leq "
                f"\\hat{{\\beta_{{i}}}} + s(\\hat{{\\beta_{{i}}}}) \\: "
                f"t_{{1 - \\alpha \\: / \\: 2}} (n - k)"
            )
            
            steps_mathjax.append(f"\\( {eqtn_str} \\)")
            
            beta_rnd = self.beta_rnd[:, 0]
            std_rnd = self.stderrors_rnd[:, 0]
            t_critical_rnd = self.t_critical_rnd
            
            beta_idx = str_color_math(f"\\beta_{{{idx}}}")
            
            steps_mathjax.append(
                f"\\( {beta_rnd[idx]} - {std_rnd[idx]} \\: "
                f"\\left({t_critical_rnd}\\right) \\leq {beta_idx} "
                f"\\leq {beta_rnd[idx]} "
                f"+ {std_rnd[idx]} \\left({t_critical_rnd}\\right)\\)"
            )
            
            se_times_t = self.stderrors[idx, 0] * self.t_critical
            se_times_t_rnd = around(se_times_t, self.decimals)
            
            steps_mathjax.append(
                f"\\( {beta_rnd[idx]} - {se_times_t_rnd} "
                f"\\leq {beta_idx} \\leq {beta_rnd[idx]} + {se_times_t_rnd} "
                "\\)"
            )
            
            steps_mathjax.append(
                f"\\( {self.confidence_intervals_rnd[idx, 0]} "
                f"\\leq {beta_idx} \\leq "
                f"{self.confidence_intervals_rnd[idx, 1]} \\)"
            )
            
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(
            "These calculated confidence intervals are summarized in "
            f"{str_color_text(f'{self.caption_num + 1}')} below."
        )
        
        p = self.conf_level * 100
        col_names = [f"\\text{{{p}% Lower CI}}", f"\\text{{{p}% Upper CI}}"]
        
        self.caption_num += 1
        arr_latex = highlight_array_vals_arr(
            arr=self.confidence_intervals_rnd,
            index=([""] + self.beta_symbols),
            col_names=col_names,
            cap_number=self.caption_num,
            cap_title=(
                f"{p}% Confidence Intervals for Regression Coefficients."
            ),
            brackets=None
        )

        steps_mathjax.append(arr_latex)
        
        return steps_mathjax
    
    
    @cached_property
    def normal_anova_latex(self) -> list[str]:
        "Analysis of variance"
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        
        steps_mathjax.append(
            "The general form of the analysis of variance (ANOVA) table is as "
            f"presented in {str_color_text(f'Table {self.caption_num + 1}')} "
            "below."
        )
        
        self.caption_num += 1
        steps_mathjax.append(self.anova_table_general(self.caption_num))
        
        steps_mathjax.append(
            "The degrees of freedom (DF column) are found as follows."
        )
        steps_mathjax.append(
            f"\\( \\quad k - 1 = {self.k} - 1 = {self.k - 1} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad n - 1 = {self.n} - 1 = {self.n - 1} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad n - k = {self.n} - {self.k} = {self.n - self.k} \\)"
        )
        steps_mathjax.append(self.html_str.remark_nk)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_BLUE_WIDTH_2)
        
        steps_mathjax.append(
            f"The calculations of the other values in "
            f"{str_color_text(f'Table {self.caption_num}')} above are "
            "presented below. Begin by calculating the mean of \\( y \\) as "
            "follows."
        )
        
        n = self.n
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad\\bar{{y}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{n}}} \\: y_{{i}}}}{{{n}}} "
            f"= \\frac{{{self.sum_y_rnd}}}{{{n}}} "
            f"= {self.mean_y_rnd}\\)"
        )
        steps_mathjax.append(
            f"Then generate {str_color_text(f'Table {self.caption_num + 1}')} "
            "below for ease of calculations. The summations in this table "
            "will be required in the calculation of "
            f"\\( \\text{{SSR, SSE}} \\) and \\( \\text{{SST}} \\)."
        )
        
        yhat = self.fitted_values
        y_minus_ybar_sq = (self.y - self.mean_y) ** 2
        yhat_minus_ybar_sq = (yhat - self.mean_y) ** 2
        y_minus_yhat_sq = (self.y - yhat) ** 2
        
        arr = hstack(
            (self.y, yhat, y_minus_ybar_sq, yhat_minus_ybar_sq, y_minus_yhat_sq)
        )
        arr_sums = arr.sum(axis=0).round(self.decimals)
        n = self.n
        col_sums = (
            f"\\sum\\limits_{{i=1}}^{{{n}}} y_{{i}} = {arr_sums[0]}",
            f"\\sum\\limits_{{i=1}}^{{{n}}} \\hat{{y_{{i}}}} = {arr_sums[1]}",
            (
                f"\\sum\\limits_{{i=1}}^{{{n}}} \\left(y "
                f"- \\bar{{y}}\\right)^{{2}} = {arr_sums[2]}"
            ),
            (
                f"\\sum\\limits_{{i=1}}^{{{n}}} \\left(\\hat{{y_{{i}}}} "
                f"- \\bar{{y}}\\right)^{{2}} = {arr_sums[3]}"
            ),
            (
                f"\\sum\\limits_{{i=1}}^{{{n}}} \\left(y_{{i}} "
                f"- \\hat{{y_{{i}}}}\\right)^{{2}} = {arr_sums[4]}"
            )
        )
        arr = vstack((arr.round(self.decimals), col_sums))
        
        index = ["i"] + list(range(1, self.n + 1)) + [""]
        col_names = (
            f"y_{{i}}",
            f"\\hat{{y_{{i}}}}",
            f"\\left(y - \\bar{{y}}\\right)^{{2}}",
            f"\\left(\\hat{{y_{{i}}}} - \\bar{{y}}\\right)^{{2}}",
            f"\\left(y_{{i}} - \\hat{{y_{{i}}}}\\right)^{{2}}"
        )
        
        self.caption_num += 1
        caption_title=(
            "Tabulated Values for Calculation of Sum and Mean of Squares."
        )
        arr_latex = highlight_array_vals_arr(
            arr=arr,
            index=index,
            col_names=col_names,
            cap_number=self.caption_num,
            cap_title=caption_title,
            brackets=None,
            color_rows=arr.shape[0]
        ).replace("\\ {", "\\ \\hline {")
        
        steps_mathjax.append(arr_latex)
        
        steps_mathjax.append(
            f"With the summations in "
            f"{str_color_text(f'Table {self.caption_num}')} above, we can "
            "proceed with the calculations as follows."
        )
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.1) SSR, MSR"))
        
        steps_mathjax.append(self.normal_ssr_msr.sum_sq)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(self.normal_ssr_msr.mean_sq)
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.2) SSE, MSE"))
        
        steps_mathjax.append(self.normal_sse_mse.sum_sq)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(self.normal_sse_mse.mean_sq)
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.3) SST, MST"))
        
        steps_mathjax.append(self.normal_sst_mst.sum_sq)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(self.normal_sst_mst.mean_sq)
        
        title = f"\\( {counter}.4) \\: \\text{{F}}_\\text{{calc}} \\)"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.extend(self.f_stat_and_pvalue.statistic)
        
        title = (
            f"\\( {counter}.5) \\: \\text{{F}}_{{\\alpha\\:/\\:2}} "
            f"= \\text{{F}}_{{{self.alpha / 2}}} \\)"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The critical value of \\( \\text{{F}} \\) is found from the "
            f"\\( \\text{{F}} \\) distribution tables by reading row "
            f"\\( {self.df_e} \\) (i.e. degrees of freedom for regression "
            f"sum of squares) and column \\( {self.df_r} \\) (i.e. degrees "
            "of freedom for error sum of squares). This is highlighted in "
            f"{str_color_text(f'Table {self.caption_num + 1}')} below."
        )
        
        self.caption_num += 1
        fcrit_latex = tables_fisher(
            statistic=self.f_critical_rnd,
            df_r=self.k - 1,
            df_e=self.n - self.k,
            conf_level=self.conf_level,
            cap_number=self.caption_num
        )
        
        steps_mathjax.extend(fcrit_latex)
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.6) P value"))
        
        steps_mathjax.extend(self.f_stat_and_pvalue.p_value)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        self.caption_num += 1
        steps_mathjax.append(
            "The values from the above calculations are summarized in "
            f"{str_color_text(f'Table {self.caption_num}')} below."
        )

        steps_mathjax.extend(self.table_anova_statistics(self.caption_num))
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_anova_latex(self) -> list[str]:
        "Analysis of variance"
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.1) SSR, MSR"))
        
        steps_mathjax.extend(self.matrix_ssr_msr.sum_sq)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.extend(self.matrix_ssr_msr.mean_sq)
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.2) SSE, MSE"))
        
        steps_mathjax.extend(self.matrix_sse_mse.sum_sq)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.extend(self.matrix_sse_mse.mean_sq)
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.3) SST, MST"))
        
        steps_mathjax.extend(self.matrix_sst_mst.sum_sq)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.extend(self.matrix_sst_mst.mean_sq)
        
        title = f"\\( {counter}.4) \\: \\text{{F}}_\\text{{calc}} \\)"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.extend(self.f_stat_and_pvalue.statistic)
        
        title = (
            f"\\( {counter}.5) \\: \\text{{F}}_{{\\alpha\\:/\\:2}} "
            f"= \\text{{F}}_{{{self.alpha / 2}}} \\)"
        )
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"The critical value of \\( \\text{{F}} \\) is found from the "
            f"\\( \\text{{F}} \\) distribution tables by reading row "
            f"\\( {self.df_e} \\) (i.e. degrees of freedom for regression "
            f"sum of squares) and column \\( {self.df_r} \\) (i.e. degrees "
            "of freedom for error sum of squares). This is highlighted in "
            f"{str_color_text(f'Table {self.caption_num + 1}')} below."
        )
        
        self.caption_num += 1
        fcrit_latex = tables_fisher(
            statistic=self.f_critical_rnd,
            df_r=self.k - 1,
            df_e=self.n - self.k,
            conf_level=self.conf_level,
            cap_number=self.caption_num
        )
        
        steps_mathjax.extend(fcrit_latex)
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.6) P value"))
        
        steps_mathjax.extend(self.f_stat_and_pvalue.p_value)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        self.caption_num += 1
        steps_mathjax.append(
            "The values from the above calculations are summarized in "
            f"{str_color_text(f'Table {self.caption_num}')} below."
        )
        steps_mathjax.extend(self.table_anova_statistics(self.caption_num))
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_YTY_latex(self) -> list[str]:
        """
        Y.T @ Y
        """
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property 
    def matrix_bTXTY_latex(self) -> list[str]:
        """
        B.T @ X.T @ Y
        """
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def matrix_YT_ND_Y_latex(self) -> list[str]:
        """
        Y.T @ ones_ndarray @ Y -> sum(Y ^ 2)
        """
        
        steps_mathjax: list[str] = []
        
        steps_mathjax.append("To update...")
        
        return steps_mathjax
    
    
    @cached_property
    def normal_ssr_msr(self) -> SumMeanSquaresNew:
        """
        Calculate Regression sum and mean of squares
        SSR, MSR
        """
        # Regression sum of squares
        # -------------------------
        ssr = (
            f"\\( \\displaystyle \\text{{SSR}} "
            f"= \\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"\\big(\\hat{{y}}_{{i}} - \\bar{{y}}\\big)^{{2}} "
            f"= {self.ssr_rnd} \\)"
        )
        
        # Regression mean of squares
        # --------------------------
        msr = (
            f"\\( \\displaystyle \\text{{MSR}} "
            f"= \\frac{{\\text{{SSR}}}}{{k - 1}} "
            f"= \\frac{{{self.ssr_rnd}}}{{{self.k} - 1}}"
            f"= {self.msr_rnd} \\)"
        )
        
        return SumMeanSquaresNew(
            sum_sq=ssr,
            mean_sq=msr
        )
    
    
    @cached_property
    def normal_sse_mse(self) -> SumMeanSquaresNew:
        """
        Calculate Error sum and mean of squares
        SSE, MSE
        """
        # Error sum of squares
        # --------------------
        sse = (
            f"\\( \\displaystyle \\text{{SSE}} "
            f"= \\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"\\big(y_{{i}} - \\hat{{y}}_{{i}}\\big)^{{2}} "
            f"= {self.sse_rnd} \\)"
        )
        
        # Error mean of squares
        # ---------------------
        mse = (
            f"\\( \\displaystyle\\text{{MSE}}"
            f"= \\frac{{\\text{{SSE}}}}{{n - k}} "
            f"= \\frac{{{self.ssr_rnd}}}{{{self.n} - {self.k}}}"
            f"= {self.mse_rnd} \\)"
        )
        
        return SumMeanSquaresNew(
            sum_sq=sse,
            mean_sq=mse
        )
        
        
    @cached_property
    def normal_sst_mst(self) -> SumMeanSquaresNew:
        """
        Calculate Total sum and mean of squares
        SST, MST
        """
        # Total sum of squares
        # --------------------
        sst = (
            f"\\( \\displaystyle \\text{{SST}} "
            f"= \\sum\\limits_{{i=1}}^{{{self.n}}} \\: "
            f"\\big(y_{{i}} - \\bar{{y}}\\big)^{{2}} "
            f"= {self.sst_rnd} \\)"
        )
        
        # Regression mean of squares
        # -------------------------
        mst = (
            f"\\( \\displaystyle \\text{{MST}} "
            f"= \\frac{{\\text{{SST}}}}{{n - 1}} "
            f"= \\frac{{{self.sst_rnd}}}{{{self.n} - 1}}"
            f"= {self.mst_rnd} \\)"
        )
        
        return SumMeanSquaresNew(
            sum_sq=sst,
            mean_sq=mst
        )
        
    
    @cached_property
    def matrix_ssr_msr(self) -> SumMeanSquaresNew:
        """
        Calculate the Error and Mean sum of squares
        SSR, MSR
        """
        sse: list[str] = []
        
        # Regression sum of squares
        # -------------------------
        sse.append(
            f"\\( \\text{{SSR}} = Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y "
            f"{str_eqtn_number('b')} \\)")
        sse.append(
            "For organized calculations, evaluate the terms in "
            f"{str_color_text('Equation (b)')} above separately as follows."
        )
        
        title = f"\\( \\text{{Evaluate}} \\: Y^{{T}}Y \\)"
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( Y^{{T}}Y "
            f"= {self.matrices_latex.YT_latex} "
            f"{self.matrices_latex.Y_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.matrices_latex.YTY_latex} \\)")
        
        title = f"\\( \\text{{Evaluate}} \\: \\beta^{{T}} X^{{T}}Y \\)"
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( \\beta^{{T}} X^{{T}}Y "
            f"= {self.matrices_latex.bT_latex} "
            f"{self.matrices_latex.XT_latex} "
            f"{self.matrices_latex.Y_latex} \\)"
        )
        sse.append(
            f"\\( \\quad = {self.matrices_latex.bT_latex} "
            f"{self.matrices_latex.XTY_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.matrices_latex.bTXTY_latex} \\)")
        
        title = (
            f"\\( \\text{{Evaluate }} \\: "
            f"\\text{{SSR}} = Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y \\)"
        )
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( \\text{{SSR}} "
            f"= Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y \\)"
        )
        sse.append(
            f"\\( \\quad = {self.matrices_latex.YTY_latex} "
            f"- {self.matrices_latex.bTXTY_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.sse_rnd} \\)")
        
        # Regression mean of squares
        # --------------------------
        mse: list[str] = []
        mse.append(
            "Substitute the result above into "
            f"{str_color_text('Equation (a)')} together with "
            f"\\( n = {self.n} \\) and \\( k = {self.k} \\) to find "
            f"\\( \\text{{MSR}} \\)."
        )
        mse.append(
            f"\\( \\displaystyle\\text{{MSR}}"
            f"= \\frac{{\\text{{SSR}}}}{{n - k}} "
            f"= \\frac{{{self.sse_rnd}}}{{{self.n} - {self.k}}}"
            f"= {self.mse_rnd} \\)"
        )
        
        return SumMeanSquaresNew(
            sum_sq=sse,
            mean_sq=mse
        )


    @cached_property
    def matrix_sse_mse(self) -> SumMeanSquaresNew:
        """
        Calculate the Error and Mean sum of squares
        SSE, MSE
        """
        sse: list[str] = []
        
        # Error sum of squares
        # --------------------
        sse.append(
            f"\\( \\text{{SSE}} = Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y "
            f"{str_eqtn_number('b')} \\)")
        sse.append(
            "For organized calculations, evaluate the terms in "
            f"{str_color_text('Equation (b)')} above separately as follows."
        )
        
        title = f"\\( \\text{{Evaluate}} \\: Y^{{T}}Y \\)"
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( Y^{{T}}Y "
            f"= {self.matrices_latex.YT_latex} "
            f"{self.matrices_latex.Y_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.matrices_latex.YTY_latex} \\)")
        
        title = f"\\( \\text{{Evaluate}} \\: \\beta^{{T}} X^{{T}}Y \\)"
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( \\beta^{{T}} X^{{T}}Y "
            f"= {self.matrices_latex.bT_latex} "
            f"{self.matrices_latex.XT_latex} "
            f"{self.matrices_latex.Y_latex} \\)"
        )
        sse.append(
            f"\\( \\quad = {self.matrices_latex.bT_latex} "
            f"{self.matrices_latex.XTY_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.matrices_latex.bTXTY_latex} \\)")
        
        title = (
            f"\\( \\text{{Evaluate }} \\: "
            f"\\text{{SSE}} = Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y \\)"
        )
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( \\text{{SSE}} "
            f"= Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y \\)"
        )
        sse.append(
            f"\\( \\quad = {self.matrices_latex.YTY_latex} "
            f"- {self.matrices_latex.bTXTY_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.sse_rnd} \\)")
        
        # Error mean of squares
        # ---------------------
        mse: list[str] = []
        mse.append(
            "Substitute the result above into "
            f"{str_color_text('Equation (a)')} together with "
            f"\\( n = {self.n} \\) and \\( k = {self.k} \\) to find "
            f"\\( \\text{{MSE}} \\)."
        )
        mse.append(
            f"\\( \\displaystyle\\text{{MSE}}"
            f"= \\frac{{\\text{{SSE}}}}{{n - k}} "
            f"= \\frac{{{self.sse_rnd}}}{{{self.n} - {self.k}}}"
            f"= {self.mse_rnd} \\)"
        )
        
        return SumMeanSquaresNew(
            sum_sq=sse,
            mean_sq=mse
        )
        
        
    @cached_property
    def matrix_sst_mst(self) -> SumMeanSquaresNew:
        """
        Calculate the Error and Mean sum of squares
        SST, MST
        """
        sse: list[str] = []
        
        # Total sum of squares
        # --------------------
        sse.append(
            f"\\( \\text{{SST}} = Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y "
            f"{str_eqtn_number('b')} \\)")
        sse.append(
            "For organized calculations, evaluate the terms in "
            f"{str_color_text('Equation (b)')} above separately as follows."
        )
        
        title = f"\\( \\text{{Evaluate}} \\: Y^{{T}}Y \\)"
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( Y^{{T}}Y "
            f"= {self.matrices_latex.YT_latex} "
            f"{self.matrices_latex.Y_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.matrices_latex.YTY_latex} \\)")
        
        title = f"\\( \\text{{Evaluate}} \\: \\beta^{{T}} X^{{T}}Y \\)"
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( \\beta^{{T}} X^{{T}}Y "
            f"= {self.matrices_latex.bT_latex} "
            f"{self.matrices_latex.XT_latex} "
            f"{self.matrices_latex.Y_latex} \\)"
        )
        sse.append(
            f"\\( \\quad = {self.matrices_latex.bT_latex} "
            f"{self.matrices_latex.XTY_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.matrices_latex.bTXTY_latex} \\)")
        
        title = (
            f"\\( \\text{{Evaluate }} \\: "
            f"\\text{{SST}} = Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y \\)"
        )
        sse.append(html_style_bg(title=title, bg=ColorCSS.COLORF5F5F5))
        sse.append(
            f"\\( \\text{{SST}} "
            f"= Y^{{T}}Y - \\beta^{{T}} X^{{T}}Y \\)"
        )
        sse.append(
            f"\\( \\quad = {self.matrices_latex.YTY_latex} "
            f"- {self.matrices_latex.bTXTY_latex} \\)"
        )
        sse.append(f"\\( \\quad = {self.sse_rnd} \\)")
        
        # Total mean of squares
        # ---------------------
        mse: list[str] = []
        mse.append(
            "Substitute the result above into "
            f"{str_color_text('Equation (a)')} together with "
            f"\\( n = {self.n} \\) and \\( k = {self.k} \\) to find "
            f"\\( \\text{{MST}} \\)."
        )
        mse.append(
            f"\\( \\displaystyle\\text{{MST}}"
            f"= \\frac{{\\text{{SST}}}}{{n - k}} "
            f"= \\frac{{{self.sse_rnd}}}{{{self.n} - {self.k}}}"
            f"= {self.mse_rnd} \\)"
        )
        
        return SumMeanSquaresNew(
            sum_sq=sse,
            mean_sq=mse
        )
        
    
    @cached_property
    def fitted_and_residuals_latex(self) -> list[str]:
        """Fitted and residuals values"""
        
        steps_mathjax: list[str] = []
        self._counter += 1
        counter = self._counter
        max_show = 3
        
        ith = (
            f"(i = 1, 2, 3, 4, 5)" 
            if self.k <= 6 else 
            f"(i = 1, 2, 3, ..., {self.k - 1})"
        )
        xi = f"x_{{1}}" if self.k == 2 else f"x_{{i}} \\: for {ith}"
        
        vars_str = (
            "values of the independent variable"
            if self.k == 2 else
            "values of the respective independent variables"
        )
        
        calc_str = (
            "These calculations are presented below"
            if self.k <= 5 else 
            "The first 3 and last 1 calculations are presented below"
        )
        
        steps_mathjax.append(
            f"The fitted (or predicted) values, denoted by "
            f"\\( \\hat{{y}} \\), are found by substituting for \\( {xi} \\) "
            f"in the calculated regression equation with the {vars_str} from "
            f"the dataset. {calc_str}."
        )
        
        title = f"{counter}.1) Fitted values"
        steps_mathjax.append(html_bg_level2(title=title))
        
        xis_str = (
            f"\\( {', '.join(self.xi_symbols[1:])} \\) and "
            f"\\( x_{{{self.k}}} \\)"
        )
        fvars = (
            f"variable \\( x_{{1}} \\)"
            if self.k == 2 else
            f"variables \\( {xis_str} \\)"
        )
        
        steps_mathjax.append(
            f"To calculate the \\( \\text{{ith}} \\) fitted value, "
            f"substitute for the independent {fvars} in the model below."
        )
        steps_mathjax.append(f"\\( y = {self.model_latex} \\)")
        
        steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)
        
        for idx in range(self.n):
            
            if max_show <= idx <= self.n - 2:
                continue
            
            xi_val = self.x[idx, 0]
            steps_mathjax.append(f"When \\( x = {xi_val} \\)")
            eqtn_rep = self.model_string.replace("*", "\\:")
            
            for xi in self.xi_symbols[1:]:
                eqtn_rep = eqtn_rep.replace(xi, str_color_math(f'({xi_val})'))
            
            steps_mathjax.append(
                f"\\( \\hat{{y}}_{{{idx + 1}}} = {eqtn_rep} \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = "
                f"{str_color_math(self.fitted_values_rnd[idx, 0], 'blue')} \\)"
            )
            
            if idx == max_show - 1:
                steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
                steps_mathjax.extend(
                    str_omitted(msg=f"{self.n - max_show} calculations")
                )
            
            if idx != self.n - 1:
                steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
            
        steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)
            
        steps_mathjax.append(
            f"The above fitted values are presented in the "
            f"\\( \\text{{third}} \\) column of the "
            f"{str_color_text(f'Table {self.caption_num + 1}')} "
            f"(shown after {str_color_text(f'Section {counter}.2')} below."
        )
        
        title = f"{counter}.2) Residuals"
        steps_mathjax.append(html_bg_level2(title=title))
        
        calc_str = (
            "These calculations are presented below"
            if self.k <= 5 else 
            "The first 3 and last 1 calculations are presented below"
        )
        
        steps_mathjax.append(
            f"Residuals, denoted by \\( \\hat{{e}} \\), are found by "
            f"subtracting the predicted values \\( (\\hat{{y}}) \\) from the "
            "observed values of the dependent variable \\( (y) \\). "
            f"{calc_str}."
        )
        
        steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)
            
        for idx in range(self.n):
            
            if max_show <= idx <= self.n - 2:
                continue 
            
            steps_mathjax.append(
                f"\\( \\hat{{e}}_{{{idx + 1}}} "
                f"= y - \\hat{{y}}_{{{idx + 1}}} \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = {self.y_rnd[idx, 0]} "
                f"- {self.fitted_values_rnd[idx, 0]} \\)"
            )
            steps_mathjax.append(
                f"\\( \\quad = "
                f"{str_color_math(self.residuals_rnd[idx, 0], 'orange')} \\)"
            )
            
            if idx == max_show - 1:
                steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
                steps_mathjax.extend(
                    str_omitted(msg=f"{self.n - max_show} calculations")
                )
            
            if idx != self.n - 1:
                steps_mathjax.append(StemConstants.BORDER_HTML_DASHED)
        
        steps_mathjax.append(StemConstants.BORDER_HTML_SOLID)
        
        steps_mathjax.append(
            "The above fitted values and residuals are presented in the "
            f"\\( \\text{{third}} \\) and \\( \\text{{fourth}} \\) columns "
            f"of {str_color_text(f'Table {self.caption_num + 1}')} below. "
            "The last column contains values of residuals squared (it is "
            "usually the norm to square the residuals)."
        )
        
        self.caption_num += 1
        steps_mathjax.append(
            self.table_fitted_and_residuals_latex(self.caption_num)
        )
        
        if self.n > 5:
            steps_mathjax.append(
                "The colored values are those that were calculated above."
            )
        
        return steps_mathjax
    
    @property
    def f_stat_and_pvalue(self) -> FStatisticAndPvalue:

        f_statistic: list[str] = []
        
        f_statistic.append(
            f"The \\( \\text{{F}} \\) statistic is calculated by dividing "
            f"the mean error of squares \\( (\\text{{MSE}}) \\) by the mean "
            f"regression of squares \\( (\\text{{MSR}}) \\). That is,"
        )
        f_statistic.append(
            f"\\( \\displaystyle \\text{{F}}_\\text{{calc}} "
            f"= \\frac{{\\text{{MSR}}}}{{\\text{{MSE}}}} "
            f"= \\frac{{{self.msr_rnd}}}{{{self.mse_rnd}}} "
            f"= {self.f_statistic_rnd} \\)"
        )

        f_pvalue: list[str] = []
        
        f_pvalue.append(
            f"The \\( \\text{{p}} \\)-value associated with the "
            f"\\( \\text{{F}} \\) statistic is calculated as follows."
        )
        f_pvalue.append(
            f"\\( p = \\text{{P}}\\big[\\text{{F}}_{{{1 - self.alpha}}}"
            f"(k - 1, n - k) > F\\big] \\)"
        )
        f_pvalue.append(
            f"\\( \\quad = \\text{{P}}\\big[\\text{{F}}_{{{1 - self.alpha}}} "
            f"({self.k - 1}, {self.n - self.k}) > {self.f_statistic_rnd} "
            "\\big] \\)"
        )
        f_pvalue.append(f"\\( \\quad = {self.f_pvalue_rnd} \\)")
        
        return FStatisticAndPvalue(
            statistic=f_statistic,
            p_value=f_pvalue
        )

    
    # =========================================================================
    # SECTION: COMMON FUNCTIONS
    # =========================================================================
    
    # =========================================================================
    # SECTION: NORMAL EQUATIONS COMPUTATIONS
    # =========================================================================

    
    # =========================================================================
    # SECTION: MATRIX COMPUTATIONS
    # =========================================================================


    # =========================================================================
    # SECTION: OTHER STATISTICS
    # =========================================================================


    @property
    def root_mse_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        
        if "anova" in self.statistics:
            anova_str = (
                ", which was calculated in "
                f"\\( \\text{{Analysis of Variance}} \\) section."
            )
        else:
            anova_str = (
                f". Set {str_color_text('anova')} in "
                f"{str_color_text('statistics')} to show step-by-step "
                f"calculations of \\( \\text{{MSE}} \\)"
            )
        
        steps_mathjax.append(
            f"This is the square root of the mean square of errors{anova_str}"
        )
        steps_mathjax.append(
            f"\\( \\text{{Root MSE}} = \\sqrt{{\\text{{MSE}}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = \\sqrt{{{self.mse_rnd}}} \\)")
        steps_mathjax.append(f"\\( \\quad = {self.root_mse_rnd} \\)")
        
        return steps_mathjax
    
    
    @property
    def r_squared_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        
        if "anova" in self.statistics:
            steps_mathjax.append(
                f"\\( \\text{{SSR, SSE}} \\), and \\( \\text{{SST}} \\) are "
                f"calculated in \\( \\text{{Analysis of variance}} \\) "
                "section."
            )
        else:
            steps_mathjax.append(
                f"Specify {str_color_text('anova')} in "
                f"{str_color_text('statistics')} to show step-by-step "
                f"calculations of \\( \\text{{SSR, SSE}} \\) and "
                f"\\( \\text{{SST}} \\)."
            )
        
        title = f"{counter}.1) Formula 1"
        steps_mathjax.append(html_bg_level2(title=title))
        steps_mathjax.append(
            f"\\( \\displaystyle R^{{2}} "
            f"= \\frac{{\\text{{SSR}}}}{{\\text{{SST}}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= \\frac{{{self.ssr_rnd}}}{{{self.sst_rnd}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad = {self.r_squared_rnd} \\)"
        )
        
        title = f"{counter}.2) Formula 2"
        steps_mathjax.append(html_bg_level2(title=title))
        steps_mathjax.append(
            f"\\( \\displaystyle R^{{2}} "
            f"= 1 - \\frac{{\\text{{SSE}}}}{{\\text{{SST}}}} \\)"
        )
        steps_mathjax.append(
            "\\( \\displaystyle\\quad "
            f"= 1 - \\frac{{{self.sse_rnd}}}{{{self.sst_rnd}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.r_squared_rnd} \\)")
        
        return steps_mathjax
    
        
    @property
    def adj_r_squared_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        
        if "anova" in self.statistics:
            steps_mathjax.append(
                f"\\( \\text{{MSR}} \\), and \\( \\text{{MST}} \\) are "
                f"calculated in \\( \\text{{Analysis of variance}} \\) "
                "section."
            )
        else:
            steps_mathjax.append(
                f"Specify {str_color_text('anova')} in "
                f"{str_color_text('statistics')} to show step-by-step "
                f"calculations of \\( \\text{{SSR}} \\) and "
                f"\\( \\text{{SST}} \\)."
            )
        
        title = f"{counter}.1) Formula 1"
        steps_mathjax.append(html_bg_level2(title=title))
        
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Adjusted}} \\: R^{{2}} "
            f"= 1 - \\frac{{\\text{{MSE}}}}{{\\text{{MST}}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad = "
            f"1 - \\frac{{{self.mse_rnd}}}{{{self.mst_rnd}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad = {self.adjusted_r_squared_rnd} \\)"
        )
        
        title = f"{counter}.2) Formula 2"
        steps_mathjax.append(html_bg_level2(title=title))
        
        if "r2" in self.statistics:
            steps_mathjax.append(
                f"\\( R^{{2}} \\) is calculated in "
                f"\\( \\text{{R Squared}} \\) section above."
            )
        else:
            steps_mathjax.append(
                f"Specify {str_color_text('r2')} in "
                f"{str_color_text('statistics')} to show step-by-step "
                f"calculations of \\( R^{{2}} \\)."
            )
            
        steps_mathjax.append(
            f"\\( \\displaystyle \\text{{Adjusted}} \\: R^{{2}} "
            f"= 1 - \\frac{{\\big(1 - R^{{2}}\\big) \\: (n - 1)}}{{n - k}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= 1 - \\frac{{\\big(1 - {self.r_squared_rnd}\\big) \\: "
            f"({self.n} - 1)}}{{{self.n} - {self.k}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad "
            f"= 1 - \\frac{{{around((1 - self.r_squared) * (self.n - 1), self.decimals)}}} "
            f"{{{self.n - self.k}}} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.adjusted_r_squared_rnd} \\)")
        
        return steps_mathjax
    
    
    @property
    def dw_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        self.appendix_counter += 1
        
        steps_mathjax.append(
            "The Durbin-Watson statistic is calculated using the "
            f"{str_color_text(f'Equation {counter}.1')} below."
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Durbin-Watson (DW)}} "
            f"= \\frac{{\\sum\\limits_{{i=2}}^{{n}} \\: "
            f"(e_{{i}} - e_{{i - 1}})^{2}}}"
            f"{{\\sum\\limits_{{i=1}}^{{n}} \\: e_{{i}}^{{2}}}} "
            f"{str_eqtn_number(f'{counter}.1')} \\)"
        )
        steps_mathjax.append(f"\\( \\text{{where,}} \\)")
        steps_mathjax.append(
            f"\\( \\quad \\) \\( e_{{i}} \\) are the residuals while "
            f"\\( e_{{i}} - e_{{i-1}} \\) are the differences between two "
            "consecutive residuals."
        )
        
        steps_mathjax.append(
            f"Refer to {str_color_text(f'Table A.{self.appendix_counter}')} "
            "in the Appendix section for the calculation of these values. "
            f"Substitute these values in "
            f"{str_color_text(f'Equation {counter}.1')} and work through as"
            "shown below."
        )
        
        numer = (diff(self.residuals.flatten()) ** 2).sum()
        numer_rnd = around(numer, self.decimals)
        residuals_sq = self.residuals_squared.sum()
        residuals_sq_rnd = around(residuals_sq, self.decimals)
    
        steps_mathjax.append(
            f"\\( \\displaystyle "
            f"\\text{{Durbin-Watson (DW)}} "
            f"= \\frac{{{numer_rnd}}}{{{residuals_sq_rnd}}} "
            f"= {self.durbin_watson_rnd} \\)"
        )
        
        return steps_mathjax
    
        
    @property
    def log_likelihood_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        self.appendix_counter += 1
        
        sigma_squared = self.sse / self.n
        sigma_squared_rnd = around(sigma_squared, self.decimals)
        
        steps_mathjax.append(
            f"The Log-likelihood statistic is given by "
            f"{str_color_text(f'Equation {counter}.1')} below. Refer to "
            f"{str_color_text(f'Table A.{self.appendix_counter}')} "
            f"in {str_color_text('Appendix A')} for the calculation of the "
            "summation that appears in this formula."
        )
        steps_mathjax.append(
            "\\( \\displaystyle\\text{Log-likelihood} "
            f"= \\ln \\Bigg[\\prod\\limits_{{i=1}}^{{n}} "
            f"\\frac{{1}}{{\\sqrt{{2 \\pi \\sigma^{{2}}}}}} "
            f"\\times \\exp\\left(-\\frac{{\\sum\\limits_{{i=1}}^{{n}}"
            f"\\big(y_{{i}} - \\hat{{y_{{i}}}}\\big)^{{2}}}}{{2 \\sigma^2}}"
            f"\\right)\\Bigg] {str_eqtn_number(f'{counter}.1')} \\)"
        )
        steps_mathjax.append(f"\\( \\text{{where,}} \\)")
        steps_mathjax.append(f"\\( \\quad n = {self.n} \\)")
        steps_mathjax.append(
            f"\\( \\quad\\sum\\limits_{{i=1}}^{{n}}\\big(y_{{i}} "
            f"- \\hat{{y_{{i}}}}\\big)^2 "
            f"= \\sum\\limits_{{i=1}}^{{{self.n}}}\\big(y_{{i}} "
            f"- \\hat{{y_{{i}}}}\\big)^2 = {self.sse_rnd} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad\\sigma^{{2}} "
            f"= \\frac{{\\sum\\limits_{{i=1}}^{{{self.n}}}\\big(y_{{i}} "
            f"- \\hat{{y_{{i}}}}\\big)^2}}{{{self.n}}} "
            f"= \\frac{{{self.sse_rnd}}}{{{self.n}}} "
            f"= {sigma_squared_rnd} \\)"
        )
        steps_mathjax.append(
            f"Substitute the above values into "
            f"{str_color_text(f'Equation {counter}.1')} to get the following."
        )
        steps_mathjax.append(
            "\\( \\displaystyle\\text{Log-likelihood} "
            f"= \\ln \\Bigg[\\prod\\limits_{{i=1}}^{{{self.n}}} "
            f"\\frac{{1}}{{\\sqrt{{2 \\: ({around(pi, self.decimals)}) \\: "
            f"({sigma_squared_rnd})}}}} "
            f"\\times \\exp\\left(-\\frac{{{self.sse_rnd}}} "
            f"{{2 \\: ({sigma_squared_rnd})}}\\right)\\Bigg] \\)"
        )
        
        after_prod = 1 / sqrt(2 * pi * sigma_squared)
        after_prod_rnd = around(after_prod, self.decimals)
        in_exp = -self.sse / (2 * sigma_squared)
        in_exp_rnd = around(in_exp, self.decimals)
        
        steps_mathjax.append(
            "\\( \\displaystyle\\quad "
            f"= \\ln \\Bigg[\\prod\\limits_{{i=1}}^{{{self.n}}} "
            f"{after_prod_rnd} \\times \\exp\\left({in_exp_rnd}\\right)"
            "\\Bigg] \\)"
        )
        steps_mathjax.append(
            "\\( \\displaystyle\\quad "
            f"= \\ln \\big[\\left( {after_prod_rnd} \\right)^{{{self.n}}}"
            f"\\times \\exp\\left({in_exp_rnd} \\right)\\big] \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.log_likelihood_rnd} \\)"
        )
        
        return steps_mathjax
    
        
    @property
    def aic_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        
        steps_mathjax.append(self.html_str.log_lk)
        steps_mathjax.append(
            f"\\( \\text{{AIC}} = 2 \\: k "
            f"- 2 \\: \\times \\text{{Log Likelihood}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = 2 \\: ({self.k}) "
            f"- 2 \\: ({self.log_likelihood_rnd}) \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.aic_rnd} \\)")
        steps_mathjax.append(self.html_str.remark_k)
        
        return steps_mathjax
    
        
    @property
    def bic_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        
        steps_mathjax.append(self.html_str.log_lk)
        steps_mathjax.append(
            f"\\( \\text{{BIC}} = k \\ln (n) "
            f"- 2 \\times \\text{{Log Likelihood}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {self.k} \\ln ({self.n}) "
            f"- 2 \\times \\left({self.log_likelihood_rnd} \\right) \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.bic_rnd} \\)")
        steps_mathjax.append(self.html_str.remark_nk)
        
        return steps_mathjax
    
    
    @property
    def omnibus_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        
        if self.n > 7:
            steps_mathjax.append(
                "The Omnibus statistic and its associated p-value are given "
                "below."
            )
            
            steps_mathjax.append(html_bg_level2(title=f"{counter}.1) Omnibus"))
            steps_mathjax.append(
                f"\\( \\text{{Omnibus}} = {self.omnibus_k2_rnd} \\)"
            )
            
            steps_mathjax.append(html_bg_level2(title=f"{counter}.2) P-value"))
            steps_mathjax.append(
                f"\\( \\text{{P-value}} = {self.omnibus_k2_pvalue_rnd} \\)"
            )
        else:
            steps_mathjax.append(
                "Omnibus can only be calculated if \\( n > 7 \\). The given "
                f"data has \\( n = {self.n} \\)"
            )
        
        return steps_mathjax
    
    
    @property
    def skew_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        self.appendix_counter += 1
        
        steps_mathjax.append(
            "The summations used in this section are given in "
            f"{str_color_text(f'Table A.{self.appendix_counter}')} "
            f"presented in {str_color_text('Appendix A')}."
        )
        
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Skewness}} "
            f"= \\frac{{\\hat{{\\mu}}_{{3}}}}{{\\hat{{\\sigma}}^{{3}}}} "
            f"{str_eqtn_number(f'{counter}.1')} \\)"
        )
        steps_mathjax.append(f"\\( \\text{{where}}, \\)")
        
        n = self.n
        
        steps_mathjax.append(
            f"\\( \\quad\\displaystyle\\hat{{\\mu}}_{{3}} "
            f"= \\frac{{1}}{{n}}\\sum\\limits_{{i=1}}^{{n}} "
            f"\\Big(y_{{i}} - \\hat{{y_{{i}}}}\\Big)^{{3}} "
            f"= \\frac{{1}}{{{n}}}\\Big({self.sum_y_minus_yhat_pow3_rnd}\\Big)"
            f"= {around((1 / n) * self.sum_y_minus_yhat_pow3, self.decimals)} "
            "\\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad\\hat{{\\sigma}}^{{3}} "
            f"= \\bigg[\\frac{{1}}{{n}} \\sum\\limits_{{i=1}}^{{n}} "
            f"\\Big(y_{{i}} - \\hat{{y_{{i}}}}\\Big)^{{2}} "
            f"\\bigg]^{{3\\:/\\:2}} "
            f"= \\bigg[\\frac{{1}}{{{n}}} "
            f"\\Big({self.sum_y_minus_yhat_pow2_rnd}\\Big)"
            f"\\bigg]^{{3\\:/\\:2}} "
            f"= {around(((1 / n) * self.sum_y_minus_yhat_pow2) ** (3 / 2), self.decimals)} "
            "\\)"
        )
        steps_mathjax.append(
            "Substituting the above values into "
            f"{str_color_text(f'Equation {counter}.1')} gives,"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Skewness}} = "
            f"\\frac{{{around((1 / n) * self.sum_y_minus_yhat_pow3, self.decimals)}}}{{"
            f"{around(((1 / n) * self.sum_y_minus_yhat_pow2) ** (3 / 2), self.decimals)}}} "
            f"= {self.skew_rnd} \\)"
        )
        
        return steps_mathjax
    
    
    @property
    def kurt_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        if "skew" in self.statistics:
            kurt_counter = self.appendix_counter - 1
        else:
            kurt_counter = self.appendix_counter

        steps_mathjax.append(
            "The summations used in this section are given in "
            f"{str_color_text(f'Table A.{kurt_counter}')} "
            f"presented in {str_color_text('Appendix A')}."
        )
        
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Kurtosis}} "
            f"= \\frac{{\\hat{{\\mu}}_{{4}}}}{{\\hat{{\\sigma}}^{{4}}}} "
            f"{str_eqtn_number(f'{counter}.1')} \\)"
        )
        
        steps_mathjax.append(f"\\( \\text{{where}}, \\)")
        
        n = self.n
        
        steps_mathjax.append(
            f"\\( \\quad\\displaystyle\\hat{{\\mu}}_{{4}} "
            f"= \\frac{{1}}{{n}}\\sum\\limits_{{i=1}}^{{n}} "
            f"\\Big(y_{{i}} - \\hat{{y_{{i}}}}\\Big)^{{4}} "
            f"= \\frac{{1}}{{{n}}}\\Big({self.sum_y_minus_yhat_pow4_rnd}\\Big)"
            f"= {around((1 / n) * self.sum_y_minus_yhat_pow4, self.decimals)} "
            "\\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad\\hat{{\\sigma}}^{{4}} "
            f"= \\bigg[\\frac{{1}}{{n}} \\sum\\limits_{{i=1}}^{{n}} "
            f"\\Big(y_{{i}} - \\hat{{y_{{i}}}}\\Big)^{{2}} \\bigg]^{{2}} "
            f"= \\bigg[\\frac{{1}}{{{n}}} "
            f"\\Big({self.sum_y_minus_yhat_pow2_rnd}\\Big)\\bigg]^{{2}} "
            f"= {around(((1 / n) * self.sum_y_minus_yhat_pow2) ** 2, self.decimals)} "
            "\\)"
        )
        steps_mathjax.append(
            "Substituting the above values into "
            f"{str_color_text(f'Equation {counter}.1')} gives,"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Kurtosis}} = "
            f"\\frac{{{around((1 / n) * self.sum_y_minus_yhat_pow4, self.decimals)}}}{{"
            f"{around(((1 / n) * self.sum_y_minus_yhat_pow2) ** 2, self.decimals)}}} "
            f"= {self.kurt_rnd} \\)"
        )
        
        return steps_mathjax
    
    
    @property
    def jbera_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        counter = self.counter
        
        steps_mathjax.append(
            f"The \\( \\text{{Jarque-Bera}} \\) statistic and its associated "
            f"\\( \\text{{p-value}} \\) are calculated as follows."
        )
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.1) Jarque-Bera"))
        
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Jarque-Bera}} "
            f"= \\frac{{n}}{{6}} \\left[S^{{2}} "
            f"+ \\frac{{1}}{{4}}\\Big(K - 3\\Big)^{{2}}\\right] "
            f"{str_eqtn_number(f'{counter}.1')} \\)"
        )
        
        skew_tex = f"\\( \\text{{Skewness}} \\)"
        kurt_tex = f"\\( \\text{{Kurtosis}} \\)"
        
        common_str = (
            f"where \\( n, S \\) and \\( K \\) are respectively the sample "
            f"size, {skew_tex} and {kurt_tex}"
        )
        
        if "skew" in self.statistics and "kurt" in self.statistics:
            steps_mathjax.append(f"{common_str}. (Calculated above).")
        
        elif "skew" in self.statistics and "kurt" not in self.statistics:
            steps_mathjax.append(
                f"{common_str}. (Specify  {str_color_text('kurt')} in "
                f"{str_color_text('statistics')} to show step-by-step "
                f"calculations of {kurt_tex})."
            )
            
        elif "skew" not in self.statistics and "kurt" in self.statistics:
            steps_mathjax.append(
                f"{common_str}. (Specify {str_color_text('skew')} in "
                f"{str_color_text('statistics')} to show step-by-step "
                f"calculations of {skew_tex})."
            )
            
        else: #not ("skew" in self.statistics and "kurt" in self.statistics):
            steps_mathjax.append(
                f"{common_str}. (Specify {str_color_text('skew')} and "
                f"{str_color_text('kurt')} in {str_color_text('statistics')} "
                f"to show step-by-step calculations of {skew_tex} and "
                f"{kurt_tex})."
            )
        
        steps_mathjax.append(
            "Substituting the mentioned values into "
            f"{str_color_text(f'Equation {counter}.1')} gives,"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\text{{Jarque-Bera}} "
            f"= \\frac{{{self.n}}}{{6}} \\left[{self.skew_rnd}^{{2}} "
            f"+ \\frac{{1}}{{4}}\\Big({self.kurt_rnd} - 3\\Big)^{{2}}"
            f"\\right] \\)"
        )
        in_brackets_rnd = around(
            self.skew_rnd ** 2 + (1 / 4) * (self.kurt_rnd - 3) ** 2,
            self.decimals
        )
        steps_mathjax.append(
            f"\\( \\quad = {around(self.n / 6, self.decimals)} "
            f"\\left({in_brackets_rnd}\\right) \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.jarque_bera_rnd} \\)")
        
        steps_mathjax.append(html_bg_level2(title=f"{counter}.2) P-value"))
        steps_mathjax.append(
            f"The \\( \\text{{p-value}} \\) of the above "
            f"\\( \\text{{Jarque-Bera}} \\) statistic is calculated as "
            "follows."
        )
        steps_mathjax.append(
            f"\\( \\text{{Prob. Jarque-Bera}} "
            f"= \\text{{P}}\\big[\\chi_{{{self.conf_level}}}(2) "
            f"> {self.jarque_bera_rnd}\\big] \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = 1 - \\text{{P}}\\big[\\chi_{{{self.conf_level}}}(2) "
            f"\\leq {self.jarque_bera_rnd}\\big] \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = "
            f"1 - {around(stats.chi2(2).cdf(self.jarque_bera), self.decimals)} "
            "\\)"
        )
        steps_mathjax.append(f"\\( \\quad = {self.jarque_bera_pvalue_rnd} \\)")
        
        return steps_mathjax
    
    
    @property
    def cond_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        self.counter += 1
        
        steps_mathjax.append(
            "The condition number of a square matrix \\( A \\) denoted by "
            "\\( \\kappa(A) \\) is calculated as follows,"
        )
        steps_mathjax.append(
            f"\\( \\quad\\kappa(A) = \\left|\\left|A\\right|\\right|\\cdot"
            f"\\left|\\left|A^{{-1}}\\right|\\right| \\)"
        )
        s = "s" if self.k == 2 else "s"
        steps_mathjax.append(
            "Where \\( A \\) is the matrix formed by appending a column of "
            f"\\( 1 \\)'s before the independent variable{s}, then performing the "
            f"matrix operation \\( X^{{T}} X \\). Doing so gives the "
            "conditional number as,"
        )
        steps_mathjax.append(f"\\( \\kappa(A) = {self.cond_number_rnd} \\)")
        
        return steps_mathjax
    
    
    @property
    def appendix_latex(self) -> list[str]:
        
        steps_mathjax: list[str] = []
        
        appendix = self.table_appendix
        
        if appendix.ncols == 0:
            return None

        steps_mathjax.append(appendix.latex)
        
        return steps_mathjax