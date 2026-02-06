from typing import Literal

from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.stats.regression_linear.core import LinearRegressionLatexSteps
from stemfard.core._html import html_bg_level1
from stemfard.core._strings import str_remove_tzeros
from stemfard.stats.regression_linear._params_parser import VALID_STATISTICS
from stemfard.core._verify_param_operations import verify_param_operations


def sta_regress(
    x: SequenceArrayLike,
    y: SequenceArrayLike,
    slinear_method: Literal["matrix", "normal"] = "normal",
    conf_level: float = 0.95,
    statistics: tuple[str, ...] = "beta",
    predict_at_x: SequenceArrayLike | None = None,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 12
):
    """
    >>> import stemfard as stm
    >>> 
    >>> x = [2930, 3350, 2640, 3250, 4080, 3670, 2230, 3280, 3880, 3400, 4330, 3900]
    >>> y = [456,  17,  22,  20,  15,  18,  26,  20,  16,  19,  14,  14]
    >>> stm.sta_regress(x, y)
    
    >>> df = pd.read_stata("D:/data_dta.dta").iloc[:12, :]
    >>> y = df["mpg"]
    >>> x = df["weight"]
    >>> y[0] = 456
    >>> X = df[["mpg", "weight", "length", "price", "displacement"]]
    >>> start = time.perf_counter()
    >>> result = stm.sta_regress(
        x=x,
        y=y,
        slinear_method="matrix",
        conf_level=0.95,
        statistics="beta",
        predict_at_x=[2500, 3250, 4135, 2810],
        steps_compute=True,
        steps_detailed=True,
        steps_bg=True,
        decimals=5
    )
    >>> end = time.perf_counter()
    >>> end - start
    """
    params = {
        "x": x,
        "y": y,
        "slinear_method": slinear_method,
        "conf_level": conf_level,
        "statistics": statistics,
        "predict_at_x": predict_at_x,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "decimals": decimals
    }
    
    steps_mathjax: list[str] = []
    
    steps_list = LinearRegressionLatexSteps(**params)
    
    if not steps_compute:
        return "Return tables only, not steps"
    
    counter = 0
    
    statistics = verify_param_operations(param=statistics, valid=VALID_STATISTICS)
    
    if "beta" in statistics or "b" in statistics:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Regression Coefficients")
        )
        
        if slinear_method == "normal":
            steps_mathjax.extend(
                steps_list.normal_beta_formula_expanded_latex
            )
        else:
            steps_mathjax.extend(
                steps_list.matrix_beta_latex
            )
    
    if predict_at_x:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Predict values")
        )
        steps_mathjax.extend(steps_list.predict_latex)
            
    if "se" in statistics:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Standard Errors")
        )
        if slinear_method == "normal":
            steps_mathjax.extend(steps_list.normal_standard_errors_latex)
        else:
            steps_mathjax.extend(
                steps_list.matrix_standard_errors_latex
            )
        
    if "t" in statistics:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) t Statistics")
        )
        steps_mathjax.extend(steps_list.tstats_latex)
    
    if "p" in statistics:
        counter += 1
        steps_mathjax.append(html_bg_level1(title=f"{counter}) P Values"))
        steps_mathjax.extend(steps_list.pvalues_latex)
    
    if "ci" in statistics:
        counter += 1
        title = f"{counter}.) {conf_level * 100}% Confidence Intervals"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.confidence_intervals_latex)
    
    if "anova" in statistics:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Analysis of Variance (ANOVA)")
        )
        
        if slinear_method == "normal":
            steps_mathjax.extend(steps_list.normal_anova_latex)
        else:
            steps_mathjax.extend(steps_list.matrix_anova_latex)
    
    if "fit" in statistics:
        counter += 1
        title = f"{counter}) Fitted Values and Residuals"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.fitted_and_residuals_latex)
    
    if "rmse" in statistics:
        counter += 1
        title = f"{counter}) Root Mean Square of Errors (RMSE)"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.root_mse_latex)
    
    if "r2" in statistics:
        counter += 1
        title = f"{counter}) R Squared"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.r_squared_latex)
    
    if "adj_r2" in statistics:
        counter += 1
        title = f"{counter}) Adjusted R Squared"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.adj_r_squared_latex)
    
    if "dw" in statistics:
        counter += 1
        title = f"{counter}) Durbin-Watson"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.dw_latex)
    
    if "loglk" in statistics:
        counter += 1
        title = f"{counter}) Log-likelihood"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.log_likelihood_latex)
    
    if "aic" in statistics:
        counter += 1
        title = f"{counter}) Akaike's information criterion (AIC)"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.aic_latex)
    
    if "bic" in statistics:
        counter += 1
        title = f"{counter}) Bayesion information criterion (BIC)"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.bic_latex)
        
    if "omnibus" in statistics:
        counter += 1
        title = f"{counter}) Omnibus"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.omnibus_latex)
        
    if "skew" in statistics:
        counter += 1
        title = f"{counter}) Skewness"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.skew_latex)
        
    if "kurt" in statistics:
        counter += 1
        title = f"{counter}) Kurtosis"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.kurt_latex)
        
    if "jb" in statistics:
        counter += 1
        title = f"{counter}) Jarque-Bera"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.jbera_latex)
        
    if "cond" in statistics:
        counter += 1
        title = f"{counter}) Conditional number"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(steps_list.cond_latex)
        
    if steps_list.appendix_latex is not None:
        steps_mathjax.append(html_bg_level1(title="Appendix A: Tables"))
        steps_mathjax.extend(steps_list.appendix_latex)
    
    return [str_remove_tzeros(step) for step in steps_mathjax]    