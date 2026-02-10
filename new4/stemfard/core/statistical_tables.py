from numpy import around, linspace

from stemfard.core.arrays_highlight import highlight_array_vals_arr


def tables_normal() -> str:
    arr_latex = "xxx"
    
    return arr_latex


def tables_chisquare() -> str:
    arr_latex = "xxx"
    
    return arr_latex


def tables_students_t() -> str:
    arr_latex = "xxx"
    
    return arr_latex


def tables_fisher(
    statistic: float,
    df_r: int,
    df_e: int,
    conf_level: float = 0.95,
    cap_label: str = "Table",
    cap_number: int = ...,
    cap_title: str = "Fisher's Distribution Table"
) -> str:
    
    steps_mathjax: list[str] = []
    alpha = around(1 - conf_level, 4)

    arr = linspace(10, 90, 60).reshape(12, 5).round(4)
    
    array_latex = highlight_array_vals_arr(
        arr=arr,
        index=None,
        col_names=None,
        cap_label=cap_label,
        cap_number=cap_number,
        cap_title=cap_title,
        brackets=None,
        color_rows=[df_r],
        color_cols=[df_e]
    )
    
    steps_mathjax.append(array_latex)
    
    steps_mathjax.append("This value is read as,")
    steps_mathjax.append(
        f"\\( \\text{{F}}_{{\\alpha\\:/\\:2}} "
        f"= \\text{{F}}_{{{alpha / 2}}} "
        f"= {statistic} \\)"
    )
    
    return steps_mathjax