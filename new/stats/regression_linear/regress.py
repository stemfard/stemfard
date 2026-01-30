from typing import Literal

from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.stats.regression_linear.core import LinearRegressionCalculations
from stemfard.core._html import html_bg_level1
from stemfard.core._strings import str_remove_tzeros


def sta_regress(
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
    # predict: SequenceArrayLike = ...,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 12
):

    params = {
        "x": x,
        "y": y,
        "simple_linear_method": simple_linear_method,
        "conf_level": conf_level,
        "coefficients": coefficients,
        "stderrors": stderrors,
        "tstats": tstats,
        "pvalues": pvalues,
        "conf_intervals": conf_intervals,
        "anova": anova,
        "fitted_and_residuals": fitted_and_residuals,
        "others": others,
        # "predict": predict,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "decimals": decimals
    }
    
    steps_mathjax: list[str] = []
    
    results = LinearRegressionCalculations(**params)
    
    if not steps_compute:
        return "Just retrn the the tables only, not steps"
    
    counter = 0
    
    if coefficients:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Regression Coefficients")
        )
        
        if simple_linear_method == "normal":
            steps_mathjax.extend(
                results.normal_coeffs_formula_expanded_latex
            )
        else:
            steps_mathjax.extend(
                results.normal_coeffs_formula_expanded_latex
            )
    predict = [3, 4, 8]       
    if predict:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Predict values")
        )
        steps_mathjax.extend(results.predict_latex)
            
    if stderrors:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Standard Errors")
        )
        if simple_linear_method == "normal":
            steps_mathjax.extend(results.normal_stderrors_latex)
        else:
            steps_mathjax.extend(
                results.normal_stderrors_latex
            )
        
    if tstats:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) t Statistics")
        )
        steps_mathjax.extend(results.tstats_latex)
    
    if pvalues:
        counter += 1
        steps_mathjax.append(html_bg_level1(title=f"{counter}) P Values"))
        steps_mathjax.extend(results.pvalues_latex)
    
    if conf_intervals:
        counter += 1
        title = f"{counter}.) {conf_level * 100}% Confidence Intervals"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(results.conf_intervals_latex)
    
    if anova:
        counter += 1
        steps_mathjax.append(
            html_bg_level1(title=f"{counter}) Analysis of Variance (ANOVA)")
        )
        
        if simple_linear_method == "normal":
            steps_mathjax.extend(results.normal_anova_latex)
        else:
            steps_mathjax.extend(results.matrix_anova_latex)
    
    if others:
        counter += 1
        title = title=f"{counter}) Other Statistics"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(results.other_stats_latex)
    
    if fitted_and_residuals:
        counter += 1
        title = f"{counter}) Fitted Values and Residuals"
        steps_mathjax.append(html_bg_level1(title=title))
        steps_mathjax.extend(results.fitted_and_residuals_latex)
    
    return [str_remove_tzeros(step) for step in steps_mathjax]    