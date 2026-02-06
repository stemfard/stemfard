from numpy import around

from pandas import DataFrame
from stemfard.core._html import html_style_bg
from stemfard.core.dframes import df_add_rows
from stemfard.core.utils_classes import ResultDict
from stemfard.stats.stats_grouped_data.formulas import (
    FORMULAS_ASSUMED_MEAN_TYPES, TABLE_CALC_INTRO
)
from stemfard.stats.stats_grouped_data.models import EDAGroupedDataFacade
from stemfard.stats.stats_grouped_data.syntax import _syntax_mean
from stemfard.stats.stats_grouped_data.tables import table_grouped_mean, table_grouped_qtn


def _step_i_formula(fparams: EDAGroupedDataFacade) -> list[str]:
    
    steps_mathjax = []
    
    steps_mathjax.append("Consider the grouped data in the table below.")
    steps_mathjax.append(
        f"\\[ {table_grouped_qtn().latex_rowise} \\]"
    )
    steps_mathjax.append("The mean is calculated as follows.")
    
    step_temp = html_style_bg(
        title="STEP 1: Write down the formula for the mean"
    )
    steps_mathjax.append(step_temp)
    steps_mathjax.append(
        "The formula for calculating the mean of grouped data given an "
        "assumed mean \\( A \\) is given as follows."
    )
    
    class_width_str = (
        ", which is calculated as the difference between any upper and "
        f"lower class limits. For example, using the first class limits, "
        f"the class width is calculated as, "
        f"\\[ w = ({around(fparams.upper[0], fparams.decimals)} - "
        f"{around(fparams.lower[0], fparams.decimals)}) + 1 "
        f"= {fparams.class_width_rounded} \\]"
    )
    
    steps_temp = FORMULAS_ASSUMED_MEAN_TYPES[fparams.formula]
    steps_temp[-1] = steps_temp[-1].replace("CLASS_WIDTH", class_width_str)
    steps_mathjax.extend(steps_temp)
    
    return steps_mathjax


def _step_ii_table(fparams: EDAGroupedDataFacade) -> list[str]:
    
    steps_mathjax = []
    
    n = len(fparams.lower)
    provided_or_calculated = (
        "calculated" if fparams.is_calculate_assumed_mean else "provided"
    )
    
    step_temp = html_style_bg(
        title="STEP 2: Create a table for calculations"
    )
    steps_mathjax.append(step_temp)
    
    if fparams.is_calculate_assumed_mean:
        steps_mathjax.append(
            "Begin by calculating the assumed mean from the class limits "
            "as follows:"
        )
        steps_mathjax.append(
            f"\\[ \\displaystyle A = \\frac{{\\mathrm{{LCL}}_{{1}} "
            f"+ \\mathrm{{UCL}}_{{{n}}}}}{{2}} "
            f"= \\frac{{{around(fparams.lower[0], fparams.decimals)} "
            f"+ {around(fparams.upper[-1], fparams.decimals)}}}{{2}} "
            f"= {fparams.assumed_mean_calculated_rounded} \\]"
        )
        
        if fparams.formula == "x-a":
            class_width_str = (
                "This assumed mean and the class width appear in the third "
                "column header of the table below."
            )
        else:
            class_width_str = (
                "This assumed mean and the class width (calculated in "
                f"\\( \\textbf{{STEP 2}} \\)) appear in the third column "
                "header of the table below."
            )
            
        steps_mathjax.append(
            f"where \\( \\mathrm{{LCL}}_{{1}} \\) and "
            f"\\( \\mathrm{{UCL}}_{{{n}}} \\) is the "
            f"\\( \\textbf{{first}} \\) lower class limit and "
            f"\\( \\textbf{{last}} \\) uppper class limit respectively. "
            f"{class_width_str}"
        )
    else:
        if fparams.formula == "x-a":
            steps_mathjax.append(
                f"{TABLE_CALC_INTRO}. Note that the "
                f"\\( {fparams.assumed_mean_rounded} \\) that appears in "
                f"the third column header is the {provided_or_calculated} "
                f"\\( \\textbf{{assumed mean}} \\)."
            )
        elif fparams.formula == "x/w-a":
            steps_mathjax.append(
                f"{TABLE_CALC_INTRO}. Note that the "
                f"\\( {fparams.class_width_rounded} \\) and "
                f"\\( {fparams.assumed_mean_asteriks_rounded} \\) that "
                "appear in the third column header is the calculated "
                f"\\( \\textbf{{class width}} \\) and the "
                f"\\( \\textbf{{new assumed mean}} \\) respectively. "
                "This new assumed mean is found by dividing the "
                f"{provided_or_calculated} assumed mean by the class "
                "width. That is, "
            )
            steps_mathjax.append(
                f"\\[ \\displaystyle A^{{*}} "
                f"= \\frac{{{fparams.assumed_mean}}}{{{fparams.class_width_rounded}}} "
                f"= {fparams.assumed_mean_rounded} \\]"
            )
        else:
            steps_mathjax.append(
                f"{TABLE_CALC_INTRO}. Note that the "
                f"\\( {fparams.assumed_mean_rounded} \\) and "
                f"\\( {fparams.class_width_rounded} \\) that appear in the "
                f"third  column header is the {provided_or_calculated} "
                "\\( \\textbf{{assumed mean}} \\) and the calculated "
                f"\\( \\textbf{{class width}} \\) respectively."
            )
            
    steps_mathjax.append(f"\\[ {table_grouped_mean().latex} \\]")
    
    return steps_mathjax


def _step_iii_mean_of_t(fparams: EDAGroupedDataFacade) -> list[str]:
    
    steps_mathjax = []
    
    step_temp = html_style_bg(
        title="STEP 3: Calculate the mean of \\( t \\)"
    )
    steps_mathjax.append(step_temp)
    steps_mathjax.append(
        "The mean of \\( t \\) is found by dividing the sum in column "
        "\\( ft \\) by the sum in column \\( f \\) of the table above as "
        "follows."
    )
    steps_mathjax.append(
        f"\\( \\displaystyle\\bar{{t}} "
        f"= \\frac{{\\sum \\mathrm{{ft}}}}{{\\sum \\mathrm{{f}}}} \\)"
    )
    steps_mathjax.append(
        "\\( \\displaystyle\\quad "
        f"= \\frac{{{fparams.total_ft_rounded}}}{{{fparams.total_freq}}} \\)"
    )
    steps_mathjax.append(f"\\( \\quad = {fparams.mean_of_t_rounded} \\)")
    
    return steps_mathjax


def _step_iv_mean_of_x(fparams: EDAGroupedDataFacade) -> list[str]:
    
    steps_mathjax = []
    
    step_temp = html_style_bg(
        title="STEP 4: Calculate the mean of \\( x \\)"
    )
    steps_mathjax.append(step_temp)
    
    if fparams.formula == "x-a":
        steps_mathjax.append(
            "The mean of \\( x \\) is then found by adding the mean of "
            f"\\( t \\) calculated in \\( \\textbf{{STEP 3}} \\) to the "
            "assumed mean \\( A \\) as shown below."
        )
        steps_mathjax.append(f"\\( \\bar{{x}} = A + \\bar{{t}} \\)")
        steps_mathjax.append(
            f"\\( \\quad = {fparams.assumed_mean} "
            f"+ {fparams.mean_of_t_rounded} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {fparams.mean_of_x_rounded} \\)"
        )
    elif fparams.formula == "x/w-a":
        steps_mathjax.append(
            "The mean of \\( x \\) is then found by adding the mean of "
            f"\\( t \\) calculated in \\( \\textbf{{STEP 3}} \\) to the "
            f"new assumed mean \\( A^{{*}} \\) then multiplying the "
            "result by the class width. This is shown below."
        )
        steps_mathjax.append(
            f"\\( \\bar{{x}} = (A^{{*}} + \\bar{{t}} \\:) \\times w \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = ({fparams.assumed_mean_asteriks_rounded} "
            f"+ {fparams.mean_of_t_rounded}) "
            f"\\times {fparams.class_width_rounded} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = "
            f"{around(fparams.assumed_mean + fparams.mean_of_t, fparams.decimals)} "
            f"\\times {fparams.class_width_rounded} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {fparams.mean_of_x_rounded} \\)")
    else:
        steps_mathjax.append(
            "The mean of \\( x \\) is found by first multiplying the "
            f"mean of \\( t \\) calculated in \\( \\textbf{{STEP 3}} \\) "
            f"by the \\( \\textbf{{class width}} \\), then adding the "
            "result to the assumed mean. This is done below."
        )
        steps_mathjax.append(
            f"\\( \\bar{{x}} = A + \\bar{{t}} \\times w \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {fparams.assumed_mean_rounded} "
            f"+ {fparams.mean_of_t_rounded} "
            f"\\times {fparams.class_width_rounded} \\)"
        )
        steps_mathjax.append(f"\\( \\quad = {fparams.mean_of_x_rounded}\\)")
        
    return steps_mathjax


def _mean_assumed_mean_none(fparams: EDAGroupedDataFacade) -> list[str]:
    """Mean with no assumed mean provided"""
    
    steps_mathjax = []
    steps_mathjax.append("Consider the grouped data in the table below.")
    steps_mathjax.append(
        f"\\[ {table_grouped_qtn().latex_rowise} \\]"
    )
    steps_mathjax.append("The mean is calculated as follows.")
    
    step_temp = html_style_bg(
        title="STEP 1: Write down the formula for the mean"
    )
    steps_mathjax.append(step_temp)
    steps_mathjax.append(
        "The formula for calculating the mean of grouped data "
        "(without using an assumed mean) is as given below."
    )
    
    steps_mathjax.append(
        f"\\( \\displaystyle\\quad \\bar{{x}} "
        f"= \\frac{{\\sum \\mathrm{{f}}x}}{{\\sum \\mathrm{{f}}}} \\)"
    )
    steps_mathjax.append(
        "Where \\( f \\) is the frequency and \\( x \\) are the midpoint "
        "values which are calculated from the class limits. "
        f"These calculations are presented in \\( \\textbf{{STEP 2}} \\) "
        "below."
    )
    
    step_temp = html_style_bg(
        title="STEP 2: Create a table for calculations"
    )
    steps_mathjax.append(step_temp)
    
    steps_mathjax.append(
        "First calculate the midpoints \\( (x) \\) by adding each lower "
        "and upper class limit then dividing the result by \\( 2 \\). "
        "For example, using the first class limits, the "
        f"\\( \\textbf{{midpoint}} \\) is calculated as, "
    )
    steps_mathjax.append(
        f"\\[ \\displaystyle x_{{1}} "
        f"= \\frac{{{around(fparams.lower[0], fparams.decimals)} "
        f"+ {around(fparams.upper[0], fparams.decimals)}}}{{2}} "
        f"= {around((fparams.lower[0] + fparams.upper[0]) / 2, fparams.decimals)} \\]"
    )
    steps_mathjax.append(
        "All the other midpoints are calculated in a similar manner and "
        "their values presented in the second column of the table below."
    )
    
    steps_mathjax.append(f"\\[ {table_grouped_mean().latex} \\]")
    
    step_temp = html_style_bg(
        title="STEP 3: Calculate the mean of \\( x \\)"
    )
    steps_mathjax.append(step_temp)
    steps_mathjax.append(
        "The mean of \\( x \\) is then found by dividing the sum in "
        "column \\( fx \\) by the sum in column \\( f \\) of the table "
        "above as shown below."
    )
    
    steps_mathjax.append(
        f"\\( \\displaystyle\\bar{{x}} "
        f"= \\frac{{\\sum \\mathrm{{f}}x}}{{\\sum \\mathrm{{f}}}} \\)"
    )
    steps_mathjax.append(
        "\\( \\displaystyle\\quad "
        f"= \\frac{{{fparams.total_fx_rounded}}}{{{fparams.total_freq}}} "
        "\\)"
    )
    steps_mathjax.append(f"\\( \\quad = {fparams.mean_of_x_rounded} \\)")
    
    return steps_mathjax


def _mean_given_assumed_mean(fparams: EDAGroupedDataFacade) -> list[str]:
    
    """x-a, x/w-a and (x-a)/w"""
    
    steps_mathjax = []
    
    # ---------------
    # STEP 1: FORMULA
    # ---------------
    
    steps_temp = _step_i_formula(fparams=fparams)
    steps_mathjax.extend(steps_temp)
    
    # --------------
    # STEP 2: TABLE
    # --------------
    
    steps_temp = _step_ii_table(fparams=fparams)
    steps_mathjax.extend(steps_temp)
    
    # -----------------------------
    # STEP 3: CALCULATE MEAN of t
    # -----------------------------
    
    steps_temp = _step_iii_mean_of_t(fparams=fparams)
    steps_mathjax.extend(steps_temp)
    
    # ----------------------------
    # STEP 4: CALCULATE MEAN of x
    # ----------------------------
    
    steps_temp = _step_iv_mean_of_x(fparams=fparams)
    steps_mathjax.extend(steps_temp)
    
    return steps_mathjax


def stats_grouped_mean_steps(fparams: EDAGroupedDataFacade) -> ResultDict:
    """Calculate mean for grouped data."""
    if fparams.assumed_mean is not None:            
        df_series = {
            "Class": fparams.class_labels,
            "Midpoint (x)": fparams.midpoints,
            fparams.tname: fparams.t_values_rounded,
            "Frequency (f)": fparams.freq,
            "ft": fparams.fx_or_ft_rounded
        }
        row_totals = [
            ["", "", "", fparams.total_freq, fparams.total_fx_or_ft_rounded]
        ]
    else:
        df_series = {
            "Class": fparams.class_labels,
            "Midpoint (x)": fparams.midpoints_rounded,
            "Frequency (f)": fparams.freq,
            "fx": fparams.fx_or_ft_rounded
        }
        row_totals = [
            ["", "", fparams.total_freq, fparams.total_fx_or_ft_rounded]
        ]
        
    dframe = DataFrame(data=df_series)
    dframe = df_add_rows(
        df=dframe,
        rows=row_totals,
        row_names=["Total"]
    )
    
    nrows, ncols = dframe.shape
    
    tables_qtn = table_grouped_qtn(fparams=fparams)
    tables_res = table_grouped_mean(fparams=fparams)
    
    if fparams.assumed_mean is None:
        steps =  _mean_assumed_mean_none(fparams=fparams)
    else:
        steps = _mean_given_assumed_mean(fparams=fparams)
        
    syntax = _syntax_mean(fparams=fparams)
    
    return ResultDict(
        params=fparams.params,
        params_parsed=fparams.params_parsed,
        answer=fparams.mean_x_rounded,
        tables_qtn=ResultDict(
            raw=tables_qtn.raw,
            latex=tables_qtn.latex,
            latex_rowise=tables_qtn.latex_rowise,
            csv=tables_qtn.csv
        ),
        tables_result=ResultDict(
            df=dframe,
            raw=tables_res.raw,
            latex=tables_res.latex,
            csv=tables_res.csv
        ),
        stats=ResultDict(
            nrows=nrows,
            ncols=ncols, 
            mean=fparams.mean_x_rounded,
            total_freq=fparams.total_freq,
            total_weighted=fparams.total_fx_or_ft_rounded
        ),
        syntax=ResultDict(
            matlab=syntax.matlab,
            numpy=syntax.numpy,
            scipy=syntax.scipy,
            statsmodels=syntax.statsmodels,
            sympy=syntax.sympy,
            r=syntax.r,
            stata=syntax.stata
        ),
        steps=steps
    )