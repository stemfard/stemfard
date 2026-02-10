from typing import Literal
from numpy import around, cumsum, float64, int64, nan
from numpy.typing import NDArray
from pandas import DataFrame
from stemcore import numeric_format

from stemfard.core._strings import str_remove_tzeros
from stemfard.core.convert import dframe_to_array, result_to_csv
from stemfard.core.dframes import df_add_rows
from stemfard.core.utils_classes import ResultDict


def table_grouped_qtn(
    class_labels: list[str],
    freq: NDArray[float64]    
) -> ResultDict:
    
    df_series = {
        "Class": class_labels,
        "Frequency": freq
    }
    
    table_qtn_df = DataFrame(data=df_series)
    table_qtn_df_rowise = table_qtn_df.T
    table_qtn_df_rowise.index = ["Class", "Frequency"]
    table_qtn_df_rowise.columns = range(1, table_qtn_df_rowise.shape[1] + 1)
    
    # ComputedDataModel for `data`
    obj_values_data = str_remove_tzeros(table_qtn_df.values.tolist())
    latex_data = dframe_to_array(
        df=table_qtn_df,
        include_index=False,
        outer_border=True,
        inner_vlines=True
    )

    latex_data_rowise = dframe_to_array(
        df=table_qtn_df_rowise,
        include_index=True,
        outer_border=True,
        inner_vlines=True
    )
    latex_data_rowise = latex_data_rowise\
        .replace("l|", "c|")\
        .replace("c|", "l|", 1)\
        .replace(f"\\mathrm{{Frequency}}", f"\\hline\\mathrm{{Frequency}}")\
        .replace("  & 1", "\\qquad i  & 1", 1)
    
    csv_data = result_to_csv(obj=table_qtn_df)
    
    return ResultDict(
        raw=obj_values_data,
        latex=latex_data,
        latex_rowise=latex_data_rowise,
        csv=csv_data
    )
    
    
def table_grouped_mean(
    class_labels: list[str],
    class_width: int | float,
    class_width_rounded: int | float,
    midpoints: NDArray[float64],
    midpoints_rounded: NDArray[float64],
    freq: NDArray[int64],
    total_freq: int,
    fx_or_ft_rounded: NDArray[float64],
    total_fx_or_ft_rounded: int | float,
    assumed_mean: int | float,
    assumed_mean_rounded: int | float,
    assumed_mean_asteriks_rounded: int | float,
    formula: Literal["x-a", "x/w-a", "(x-a)/w"],
    tvalues_rounded: NDArray[float64],
    decimals: int
) -> ResultDict:
    
    series_dct = {
        "Class": class_labels,
        "Midpoint (x)": midpoints_rounded
    }
    
    def formulas_def(value=Literal["t", "csv", "latex"]) -> str:
        
        if formula == "x-a":
            result = {
                "t": tvalues_rounded,
                "csv": f", t = x - {assumed_mean_rounded}",
                "latex": f"& \\mathrm{{t = x - {assumed_mean_rounded}}}"
            }
        elif formula == "x/w-a":
            result = {
                "t": tvalues_rounded,
                "csv": (
                    f", t = x / {class_width_rounded} "
                    f"- {assumed_mean_asteriks_rounded}"
                ),
                "latex": (
                    f"& \\mathrm{{t = \\frac{{x}}{{{class_width_rounded}}} "
                    f"- {assumed_mean_asteriks_rounded}"
                )
            }
        elif formula == "(x-a)/w":
            result = {
                "t": tvalues_rounded,
                "csv": f", t = (x - {assumed_mean_rounded}) / {class_width_rounded}",
                "latex": (
                    f"& \\mathrm{{t = \\frac{{x - {assumed_mean_rounded}}}{{{class_width_rounded}}}}}"
                )
            }
            
        return result.get(value)
    
    if assumed_mean:            
        tvalues = formulas_def(value="t")
        if formula == "x/w-a":
            series_dct.update(
                {
                    "x": around(midpoints / class_width, decimals),
                    "t": tvalues,
                    "Frequency (f)": freq,
                    "ft": fx_or_ft_rounded
                }
            )
            rows=[[nan, nan, nan, nan, total_freq, total_fx_or_ft_rounded]]
        else:
            series_dct.update(
                {
                    "t": tvalues,
                    "Frequency (f)": freq,
                    "ft": fx_or_ft_rounded
                }
            )
            rows=[[nan, nan, nan, total_freq, total_fx_or_ft_rounded]]
        
        table_df = DataFrame(data=series_dct)
        
        table_df = df_add_rows(
            df=table_df,
            rows=rows,
            row_names=["Total"]
        )
        
        t_csv_replaced = formulas_def(value="csv")
        t_latex_replaced = formulas_def(value="latex")
    else:
        series_dct.update(
            {
                "Frequency (f)": freq,
                "fx": fx_or_ft_rounded
            }
        )
        table_df = DataFrame(data=series_dct)
        
        table_df = df_add_rows(
            df=table_df,
            rows=[[nan, nan, total_freq, total_fx_or_ft_rounded]],
            row_names=["Total"]
        )

        t_csv_replaced = ""
        t_latex_replaced = ""
    
    # ComputedDataModel for `calculations`
    obj_values = str_remove_tzeros(table_df.values.tolist())
    latex = dframe_to_array(
        df=table_df,
        include_index=False,
        outer_border=True,
        inner_vlines=True
    )
    
    latex = latex\
        .replace("|l|r|r|r|", "|l|c|r|r|")\
        .replace("\\\\\n\t &", "\\\\\n\t\\hline\n\t &")\
        .replace(f"& \\mathrm{{t}} &", t_latex_replaced)\
        .replace("& \\mathrm{f", "& \\qquad\\mathrm{f")
    
    split_str = "\\\\\n\t\\hline\n\t &"
    latex_left_right = latex.split(split_str)
    
    if len(latex_left_right) == 2:
        latex_left, latex_right = latex_left_right
        latex_right = latex_right.split("&")

        if len(latex_right) == 3:
            latex_right[1] = (
                f"{{\\color{{blue}}{{\\sum \\mathrm{{f}} "
                f"= {latex_right[1].strip()}}}}}"
            )
            latex_right_1, latex_right_2 = latex_right[2].strip().split(" ")
            latex_right[2] = (
                f"{{\\color{{blue}}{{\\sum \\mathrm{{fx}} "
                f"= {latex_right_1}}}}} {latex_right_2}"
            )
        elif len(latex_right) == 4:
            latex_right[2] = (
                f"{{\\color{{blue}}{{\\sum \\mathrm{{f}} "
                f"= {latex_right[2].strip()}}}}}"
            )
            latex_right_1, latex_right_2 = latex_right[3].strip().split(" ")
            latex_right[3] = (
                f"{{\\color{{blue}}{{\\sum \\mathrm{{ft}} "
                f"= {latex_right_1}}}}} {latex_right_2}"
            )
        elif len(latex_right) == 5:
            latex_right[3] = (
                f"{{\\color{{blue}}{{\\sum \\mathrm{{f}} "
                f"= {latex_right[3].strip()}}}}}"
            )

            latex_right_1, latex_right_2 = latex_right[4].strip().split(" ")
            
            latex_right[4] = (
                f"{{\\color{{blue}}{{\\sum \\mathrm{{ft}} "
                f"= {latex_right_1}}}}} {latex_right_2}"
            )
        
        latex = latex_left + split_str + " & ".join(latex_right)
    
    csv = result_to_csv(obj=table_df).replace(", t,", t_csv_replaced)
    
    return ResultDict(raw=obj_values, latex=latex, csv=csv)


def table_grouped_cumfreq(
    class_boundaries: list[str],
    class_limits: list[str],
    freq: NDArray[int64]    
) -> tuple[str, str, str]:
    
    series_dct = {
        "Class boundaries": class_boundaries,
        "Class limits": class_limits,
        "Frequency": freq,
        "Cumulative frequency": cumsum(freq)
    }
    
    table_df = DataFrame(data=series_dct)
    
    # ComputedDataModel for `calculations`
    obj_values = str_remove_tzeros(table_df.values.tolist())
    latex = dframe_to_array(
        df=table_df,
        include_index=False,
        outer_border=True,
        inner_vlines=True,
        inner_hlines=False
    )
    
    csv = result_to_csv(obj=table_df)
    
    return ResultDict(raw=obj_values, latex=latex, csv=csv)


def percentiles_calc(
    lower_bnds: NDArray[float64],
    upper_bnds: NDArray[float64],
    class_limits: NDArray[float64],
    class_boundaries: list[str],
    class_width: int | float,
    percentiles: NDArray[float64],
    freq: NDArray[float64]
) -> tuple:
    
    L_list = []
    nth_list = []
    C_list = []
    f_list = []
    result_list = []
    
    n = len(freq)
    
    if isinstance(percentiles, (int, float)):
        percentiles = [percentiles]
    
    for percentile in percentiles:
        nth = (percentile / 100) * n
        nth = numeric_format(nth)
        nth_list.append(nth)
        cum_freq = cumsum(freq).astype(int)
        ith_cum_freq = cum_freq[cum_freq >= nth][0]
        index = list(cum_freq).index(ith_cum_freq)
        L = numeric_format(lower_bnds[index])
        L_list.append(L)
        C = cum_freq[index - 1]
        C_list.append(C)
        median_class = class_limits[index]
        f = freq[index]
        f_list.append(f)
        
        index = list(freq).index(max(freq))
        modal_class = class_limits[index]
        
        result = numeric_format(L + ( (nth - C) * class_width ) / f)
        result_list.append(result)
    
    return ResultDict(
        lower_bnds=lower_bnds,
        upper_bnds=upper_bnds,
        class_boundaries=class_boundaries,
        cum_freq=cum_freq,
        modal_class=modal_class,
        median_class=median_class,
        L=L_list,
        n=n,
        nth=nth_list,
        C=C_list,
        class_width=class_width,
        f=f_list,
        result=result_list
    )