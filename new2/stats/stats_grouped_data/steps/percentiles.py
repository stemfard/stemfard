import os
from typing import Any, Literal
from uuid import uuid4

from numpy import around, asarray, float64, linspace, ndarray
from numpy.typing import NDArray
from pandas import DataFrame
from matplotlib.pyplot import (
    close, figure, grid, minorticks_on, plot, savefig, scatter, xlabel, xlim,
    xticks, ylabel, ylim
)
from scipy.interpolate import PchipInterpolator

from stemfard.core._strings import str_ordinal
from stemfard.core.decimals import round_to_nearest_half_up

from stemfard.core.constants import StemConstants

from stemfard.core.utils_classes import ResultDict
from stemfard.stats.stats_grouped_data.models import (
    ParamsData, ParamsPercentiles, ParamsPlot
)

colors = [
    "red", "green", "blue", "purple", "teal", "magenta", "yellow",
    "brown", "pink", "lime", "orange", "cyan", "navy", "maroon",
    "olive", "coral", "gold", "darkviolet", "darkgreen", "skyblue"
]

from stemcore import numeric_format
from stemfard.core._html import html_style_bg
from stemfard.core.convert import dframe_to_array
from stemfard.stats.stats_grouped_data.tables import (
    percentiles_calc, table_grouped_cumfreq, table_grouped_qtn
)


def percentiles_steps(
    data: ParamsData,
    percentile_params: ParamsPercentiles,
    plot_params: ParamsPlot
) -> dict[str, list[str]]:
    
    steps_mathjax = []
    remarks = []
    warnings = []
    
    lower = data.lower
    upper = data.upper
    freq = data.freq
    decimals = data.decimals
    
    percentiles_y = percentile_params.percentiles
    cumfreq_curve = percentile_params.cumfreq_curve
    
    table_qtn_latex = table_grouped_qtn(data=data).latex_rowise
    
    steps_mathjax.append("Consider the grouped data in the table below.")
    steps_mathjax.append(f"\\[ {table_qtn_latex} \\]")
    splural = "percentile is" if len(freq) == 1 else "percentiles are"
    steps_mathjax.append(f"The requested {splural} calculated as follows.")
    
    step_temp = html_style_bg(
        title="STEP 1: Create a table of cumulative frequencies"
    )
    steps_mathjax.append(step_temp)
    steps_mathjax.append(
        "Create a table with class boundaries and cumulative frequencies. "
        f"The ith cumulative frequency \\( \\textbf{{last column}} \\) is "
        "found by summing all the preceding frequencies."
    )
    
    table_latex = table_grouped_cumfreq(data=data).latex
    
    steps_mathjax.append(f"\\[ {table_latex} \\]")
    
    result_dct = percentiles_calc(
        percentiles=percentiles_y,
        lower=lower,
        upper=upper,
        freq=freq
    )

    steps_temp_calculations, result_list = _calculations(
        percentiles=percentiles_y,
        result_dct=result_dct,
        decimals=decimals
    )
    
    if cumfreq_curve:
        steps_temp = _step_II_cumfreq_curve(
            percentile_params=percentile_params,
            plot_params=plot_params,
            result_dct=result_dct,
            result_list=result_list,
            decimals=1
        )
        steps_mathjax.extend(steps_temp)
    else:
        steps_temp = _step_II_formula(percentiles=percentiles_y)
        steps_mathjax.extend(steps_temp)

        steps_mathjax.extend(steps_temp_calculations)
    
    return {
        "steps_mathjax": steps_mathjax,
        "others": {
            "remarks": remarks,
            "warnings": warnings
        }
    }
    
    
def _step_II_cumfreq_curve(
    percentile_params: ParamsPercentiles,
    plot_params: ParamsPlot,
    result_dct: dict[str, Any],
    result_list: list[int | float] | NDArray[float64],
    decimals: int = 1
) -> tuple[tuple[Any, ...], str]:
    
    x_axis = [result_dct["lc_boundaries"][0]] + list(result_dct["uc_boundaries"])
    cum_freq = [0] + list(result_dct["cum_freq"])
    sum_cum_freq = cum_freq[-1]
    series_df = {
        "Upper class boundaies (x)": x_axis,
        "Cumulative frequencies (y)": cum_freq
    }
    
    dframe = DataFrame(data=series_df).T
    dframe.columns = range(1, dframe.shape[1] + 1)
    dframe_latex = dframe_to_array(
        df=dframe,
        outer_border=True,
        inner_hlines=True,
        inner_vlines=True
    )
    
    steps_mathjax = []
    
    percentiles_y = percentile_params.percentiles
    x_values = percentile_params.x_values
    
    steps_temp = html_style_bg(
        title=f"STEP 2: Plot the cumulative frequency curve (ogive)"
    )
    steps_mathjax.append(steps_temp)
    steps_mathjax.append(
        "The cumulative frequency curve is created by plotting the "
        "cumulative frequencies against the upper class boundaries. "
        "These are obtained from the table above and presented below."
    )
    steps_mathjax.append(f"\\[ {dframe_latex} \\]")
    steps_mathjax.append(
        "The cumulative frequency curve from the above data is as given below."
    )
    
    # plot_url, at_percentiles_y, at_x_values = plot_cumfreq_curve(
    #     plot_params=plot_params,
    #     x_axis=x_axis,
    #     cum_freq=cum_freq
    # )
    plot_url, at_percentiles_y = plot_cumfreq_percentiles(
        plot_params=plot_params,
        x_axis=x_axis,
        cum_freq=cum_freq
    )
    
    if plot_url.startswith("/"):
        plot_url = plot_url[1:]
        
    steps_mathjax.append(
        f'<img src="{StemConstants.FASTAPI_HOST_URL}{plot_url}" />'
    )
    
    m = len(percentiles_y)
    s = "" if m == 1 else "s"
    
    steps_temp = html_style_bg(
        title=f"STEP 3: Read the percentiles{s} from the graph"
    )
    steps_mathjax.append(steps_temp)
    
    for index in range(m):
        percentile = numeric_format(percentiles_y[index])
        steps_mathjax.append(
            f"Calculate the point on the \\( \\textbf{{y-axis}} \\) for "
            f"\\( P = {percentile} \\)."
        )
        y = numeric_format(around(percentile/100 * sum_cum_freq, 8))
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad y_{{{index + 1}}} "
            f"= \\frac{{{percentile}}}{{100}} \\times {sum_cum_freq} = {y} \\)"
        )
        steps_mathjax.append(
            f"Find the point \\( {y} \\) on the \\( \\textbf{{y-axis}} \\) "
            "and draw a horizontal line until it touches the curve. From "
            "this point on the curve, draw a vertical line to the "
            f"\\( \\textbf{{x-axis}} \\), and read the value of \\( x \\) "
            "at this point of intersection. From the above graph, this "
            "value is as given below,"
        )
        steps_mathjax.append(
            f"\\( \\quad x_{{{index + 1}}} "
            f"= {round_to_nearest_half_up(at_percentiles_y[index], is_delta=True)} \\)"
        )
        if index != m - 1:
            steps_mathjax.append(StemConstants.BORDER_HTML)

    steps_temp = html_style_bg(
        title=f"\\( \\textbf{{REMARK}} \\)", bg="#F5F5F5"
    )
    steps_mathjax.append(steps_temp)
    steps_mathjax.append(
        f"The value{s} you get may vary from the value{s} found above. "
        "This depends on the accuracy of the curve you have drawn and "
        "how accurate you are in your reading. Below are the calculated "
        "values for your comparison."
    )
    
    for index in range(len(result_list)):
        steps_mathjax.append(
            f"\\( \\quad P_{{{numeric_format(percentiles_y[index])}}} "
            f"= {around(result_list[index], decimals)} \\)"
        )
    
    if x_values:
        plot_url, at_x_values = plot_cumfreq_x_values(
            percentile_params=percentile_params,
            plot_params=plot_params,
            x_axis=x_axis,
            cum_freq=cum_freq
        )
        n = len(x_values)
        s = "" if n == 1 else "s"
        steps_temp = html_style_bg(
            title=f"STEP 4: Read the given \\( x \\) value{s} from the graph"
        )
        steps_mathjax.append(steps_temp)
        
        steps_mathjax.append(
            "Below is the cumulative curve that we had in "
            f"\\( \\textbf{{STEP 3}} \\). We have drawn it separately for "
            "instructional clarity, but as a student you will have it on the "
            "same graph with the previous one."
        )
        
        if plot_url.startswith("/"):
            plot_url = plot_url[1:]
            
        steps_mathjax.append(
            f'<img src="{StemConstants.FASTAPI_HOST_URL}{plot_url}" />'
        )
        
        for index in range(n):
            x = numeric_format(x_values[index])
            steps_mathjax.append(
                f"Find the point \\( {x} \\) on the \\( \\textbf{{x-axis}} \\) "
                "and draw a vertical line until it touches the curve. From "
                "this point on the curve, draw a horizontal line to the "
                f"\\( \\textbf{{y-axis}} \\), and read the value of \\( y \\) "
                "at this point of intersection. From the above graph, this "
                f"value is as given below,"
            )
            steps_mathjax.append(
                f"\\( \\quad "
                f"x_{{{round_to_nearest_half_up(at_x_values['x'][index])}}} "
                f"= {round_to_nearest_half_up(at_x_values['y'][index], is_delta=True)} \\)"
            )
            
            if index != n - 1:
                steps_mathjax.append(StemConstants.BORDER_HTML)
        
        steps_temp = html_style_bg(
            title=f"\\( \\textbf{{REMARK}} \\)", bg="#F5F5F5"
        )
        steps_mathjax.append(steps_temp)
        steps_mathjax.append(
            f"The value{s} you get may vary from the value{s} found above. "
            "This depends on the accuracy of the curve you have drawn and "
            "how accurate you are in your reading."
        )
    
    return steps_mathjax


def _step_II_formula(
    percentiles: list[int, float] | int | float
) -> list[str]:
    
    steps_mathjax = []
    
    where_list_quartiles = [
        "where",
        "\\( \\quad p \\) is the required percentile",
        "\\( \\quad n \\) is the total frequency",
        f"\\( \\quad C \\) is the cumulative frequency above the median class",
        "\\( \\quad i \\) is the class interval",
        "\\( \\quad f \\) is the frequency of the median class"
    ]
    
    where_list_general = [
        "where",
        "\\( \\quad p \\) is the required percentile",
        "\\( \\quad n \\) is the total frequency",
        f"\\( \\quad C \\) is the cumulative frequency above the median class",
        "\\( \\quad i \\) is the class interval",
        "\\( \\quad f \\) is the frequency of the median class"
    ]
    
    if len(percentiles) == 1:
        if percentiles[0] == 25:
            steps_temp = html_style_bg(
                title=(
                    "STEP 2: Write the formula for the lower quartile "
                    f"(\\( Q_{{1}} \\))"
                )
            )
            steps_mathjax.append(steps_temp)
            steps_mathjax.append("The formula is given as follows,")
            steps_mathjax.append(
                f"\\[ Q_{{1}} = \\displaystyle L "
                f"+ \\frac{{ \\left(\\frac{{n}}{{4}} - C\\right) "
                f"\\times i}}{{f}} \\]"
            )
        elif percentiles[0] == 50:
            steps_temp = html_style_bg(
                title=(
                    "STEP 2: Write the formula for the median "
                    f"(\\( Q_{{2}} \\))"
                )
            )
            steps_mathjax.append(steps_temp)
            steps_mathjax.append("The formula is given as follows,")
            steps_mathjax.append(
                f"\\[ \\mathrm{{Median}}~(Q_{{2}}) = \\displaystyle L "
                f"+ \\frac{{ \\left(\\frac{{n}}{{2}} - C\\right) "
                f"\\times i}}{{f}} \\]"
            )
        elif percentiles[0] == 75:
            steps_temp = html_style_bg(
                title=(
                    f"STEP 2: Write the formula for the upper quartile "
                    f"(\\( Q_{{3}} \\))"
                )
            )
            steps_mathjax.append(steps_temp)
            steps_mathjax.append("The formula is given as follows,")
            steps_mathjax.append(
                f"\\[ Q_{{3}} = \\displaystyle L "
                f"+ \\frac{{ \\left(\\frac{{3}}{{4}} n - C\\right) "
                f"\\times i}}{{f}} \\]"
            )
        else:
            steps_temp = html_style_bg(
                title=f"STEP 2: Write the formula for calculating percentiles"
            )
            steps_mathjax.append(steps_temp)
            steps_mathjax.append("The formula is given as follows,")
            steps_mathjax.append(
                f"\\[ \\mathrm{{Pth~Percentile}} = \\displaystyle L "
                f"+ \\frac{{ \\left(\\frac{{p}}{{100}} n - C\\right) "
                f"\\times i}}{{f}} \\]"
            )
            
        steps_mathjax.extend(where_list_quartiles)
        
        return steps_mathjax
    
    steps_temp = html_style_bg(
        title="STEP 2: Write the formula for calculating percentiles"
    )
    steps_mathjax.append(steps_temp)
    steps_mathjax.append("The formula is given as follows,")
    steps_mathjax.append(
        f"\\[ \\mathrm{{Pth~Percentile}} = \\displaystyle L "
        f"+ \\frac{{ \\left(\\frac{{p}}{{100}} n - C\\right) "
        f"\\times i}}{{f}} \\]"
    )
    steps_mathjax.extend(where_list_general)
    steps_mathjax.append(
        "Note that for lower quartile, median and upper quartile, it is "
        f"common to express the quantity \\( \\frac{{p}}{{100}} n \\) in the "
        f"formula above as \\( \\frac{{n}}{{4}}, \\frac{{n}}{{2}} \\) and "
        f"\\( \\frac{{3}}{{4}} n \\) respectively."
    )
    
    return steps_mathjax
    
def _calculations(
    percentiles: list[int | float] | int | float,
    result_dct: dict[str, int | float],
    decimals: int
) -> list[str]:
    
    steps_mathjax = []
    result_list = []
    
    steps_temp = html_style_bg(title=f"STEP 3: Perform the calculations")
    steps_mathjax.append(steps_temp)
    steps_mathjax.append(
        "The calculations are performed using the formula given in "
        f"\\( \\textbf{{STEP 2}} \\) as presented below."
    )
    
    for index in range(len(percentiles)):
        kth_percentile = numeric_format(around(percentiles[index], decimals))
        L = result_dct["L"][index]
        n = result_dct["n"]
        nth = result_dct["nth"][index]
        C = result_dct["C"][index]
        i = result_dct["class_width"]
        f = result_dct["f"][index]
        result = result_dct["result"][index]
        
        result_list.append(result)
        
        if kth_percentile == 25:
            percentile_info = (
                f"Q_{{1}}",
                f"\\frac{{n}}{{4}}",
                f"\\frac{{{n}}}{{4}}",
                f"\\mathrm{{Lower~quartile}} ~ (Q_{{1}})",
            )
        elif kth_percentile == 50:
            percentile_info = (
                f"\\mathrm{{Median}}",
                f"\\frac{{n}}{{2}}",
                f"\\frac{{{n}}}{{2}}",
                f"\\mathrm{{Median}} ~ (Q_{{2}})",
            )
        elif kth_percentile == 75:
            percentile_info = (
                f"Q_{{3}}",
                f"\\frac{{3}}{{4}} n",
                f"\\frac{{3}}{{4}} \\times {n}",
                f"\\mathrm{{Upper~quartile}} ~ (Q_{{3}})",
            )
        else:
            percentile_info = (
                f"P_{{{kth_percentile}}}",
                f"\\frac{{{kth_percentile}}}{{100}} n",
                f"\\frac{{{kth_percentile}}}{{100}}  \\times {n}",
                f"\\mathrm{{{str_ordinal(kth_percentile)} ~ percentile}} ~ "
                f"(P_{{{kth_percentile}}})"
            )
        
        steps_temp = html_style_bg(
            title=f"\\( {percentile_info[-1]} \\)",
            bg="#F5F5F5"
        )
        steps_mathjax.append(steps_temp)

        steps_mathjax.append(
            f"\\( \\displaystyle {percentile_info[0]} = L "
            f"+ \\frac{{ \\left({percentile_info[1]} - C\\right) "
            f"\\times i }}{{f}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad = {L} "
            f"+ \\frac{{ \\left({percentile_info[2]} - {C}\\right) "
            f"\\times {i} }}{{{f}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\displaystyle\\quad = {L} "
            f"+ \\frac{{ \\left({around(nth, decimals)} - {C}\\right) "
            f"\\times {i} }}{{{f}}} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {L} + "
            f"{numeric_format(around(( (nth - C) * i ) / f, decimals))} \\)"
        )
        steps_mathjax.append(
            f"\\( \\quad = {numeric_format(around(result, decimals))} \\)"
        )
    
    return steps_mathjax, result_list


def _get_render_static_dir() -> str:
    """
    Get the appropriate static directory based on environment.
    On Render: use /tmp/static (ephemeral storage)
    Local development: use ./static
    """
    if os.getenv('RENDER'):
        # On Render, use ephemeral storage in /tmp
        static_dir = "/tmp/static"
    else:
        # Local development
        static_dir = "static"
    
    # Ensure directory exists
    os.makedirs(static_dir, exist_ok=True)
    
    return static_dir


def find_y_for_x(
    x_value: float | list[float] | NDArray[float64],
    x_axis: NDArray[float64],
    cum_freq: NDArray[float64],
    method: str = "default"    
) -> float | NDArray[float64]:
    """
    Find the cumulative frequency (y) for given x value(s) using interpolation.
    
    Parameters
    ----------
    x_value : float or array_like
        The x value(s) to find the corresponding y for (single value or array)
    x_axis : array_like
        Array of x values
    cum_freq : array_like
        Array of cumulative frequencies (y values)
    method : {"default", "pchip", "cubic"}, optional
        Interpolation method (default: "default")
    
    Returns
    -------
    y_value : float or NDArray[float64]
        The interpolated y value(s) for the given x
    
    Notes
    -----
    - "default" and "pchip" use PchipInterpolator (monotone cubic interpolation)
    - "cubic" uses CubicSpline with natural boundary conditions
    """
    # Ensure inputs are numpy arrays
    if not isinstance(x_axis, ndarray):
        x = asarray(x_axis, dtype=float)
    else:
        x = x_axis
    
    if not isinstance(cum_freq, ndarray):
        y = asarray(cum_freq, dtype=float)
    else:
        y = cum_freq
    
    # Choose interpolation method
    if method in ("default", "pchip"):
        interpolator = PchipInterpolator(x, y)
    else:  # cubic
        from scipy.interpolate import CubicSpline
        interpolator = CubicSpline(x, y, bc_type="natural")
    
    # Get y value(s) for the given x value(s)
    y_value = interpolator(x_value)
    
    return y_value


def _create_base_plot(
    x: NDArray[float64],
    y: NDArray[float64],
    x_smooth: NDArray[float64],
    y_smooth: NDArray[float64],
    line_color: str,
    label: str,
    xaxis_orientation: int,
    x_label: str,
    y_label: str
) -> None:
    """
    Create the base cumulative frequency curve plot.
    
    Parameters
    ----------
    x : NDArray[float64]
        Original x values
    y : NDArray[float64]
        Original y values (cumulative frequencies)
    x_smooth : NDArray[float64]
        Smoothed x values for curve
    y_smooth : NDArray[float64]
        Smoothed y values for curve
    line_color : str
        Color for the curve and data points
    label : str
        Label for the interpolation method
    xaxis_orientation : int
        Rotation angle for x-axis labels
    x_label : str
        Title for x-axis
    """
    # Plot smooth curve
    plot(x_smooth, y_smooth, color=line_color, label=label, linewidth=0.75)
    # Plot data points (skip first point i.e. 0 frequency)
    scatter(
        x[1:],
        y[1:],
        zorder=3,
        marker="D",
        s=20,
        label="Data points",
        color=line_color
    )
    
    # X-axis labels
    xticks(x, labels=[str(val) for val in x], rotation=xaxis_orientation)
    xlim(left=x[0] - (x[1] - x[0]))
    ylim(bottom=-0.25)
    
    # Graph-paper style
    minorticks_on()
    grid(True, which="major", linewidth=0.5)
    grid(True, which="minor", linestyle=":", linewidth=0.5)
    
    # Axes labels
    xlabel(x_label)
    ylabel(y_label)


def plot_cumfreq_percentiles(
    plot_params: ParamsPlot,
    x_axis: list[float],
    cum_freq: list[float],
    static_dir: str | None = None,
    method: Literal["default", "pchip", "cubic"] = "default",
) -> tuple[str, list[float]]:
    """
    Plot cumulative frequency curve with percentiles marked.
    
    Parameters
    ----------
    plot_params : ParamsPlot
        Parameters for the plot including percentiles to mark
    x_axis : list[float]
        X-axis values (class boundaries or midpoints)
    cum_freq : list[float]
        Cumulative frequency values
    static_dir : str, optional
        Directory to save the plot image (default: "static")
    method : {"default", "pchip", "cubic"}, optional
        Interpolation method (default: "default")
    
    Returns
    -------
    plot_url : str
        URL/path to the saved plot
    at_percentiles_y : list[float]
        X-values corresponding to the requested percentiles
    
    Examples
    --------
    >>> plot_url, percentiles = plot_cumfreq_percentiles(
    ...     plot_params, x_data, y_data
    ... )
    """
    # Convert inputs to arrays
    if not isinstance(x_axis, ndarray):
        x = asarray(x_axis, dtype=float)
    else:
        x = x_axis
    
    if not isinstance(cum_freq, ndarray):
        y = asarray(cum_freq, dtype=float)
    else:
        y = cum_freq
    
    # Extract parameters
    percentiles_y = plot.percentiles
    fig_width = ParamsPlot.fig_width
    fig_height = ParamsPlot.fig_height
    line_color = ParamsPlot.line_color
    xaxis_orientation = ParamsPlot.xaxis_orientation
    x_label = ParamsPlot.x_label
    y_label = ParamsPlot.y_label
    grid = ParamsPlot.grid
    
    line_color = "blue" if line_color == "default" else line_color
    
    # Calculate percentiles
    inv_pchip = PchipInterpolator(y, x)
    percentiles_cumfreq = [p / 100 * y[-1] for p in percentiles_y]
    at_percentiles_y = inv_pchip(percentiles_cumfreq)
    
    # Choose interpolation method
    if method in ("default", "pchip"):
        interpolator = PchipInterpolator(x, y)
        label = "Monotone cubic (PCHIP)"
    else:
        from scipy.interpolate import CubicSpline
        interpolator = CubicSpline(x, y, bc_type="natural")
        label = "Cubic spline"
    
    # Smooth curve
    x_smooth = linspace(x.min(), x.max(), 500)
    y_smooth = interpolator(x_smooth)
    
    # Create directory and filename
    os.makedirs(static_dir, exist_ok=True)
    filename = f"cumfreq_percentiles_{uuid4().hex}.png"
    filepath = os.path.join(static_dir, filename)
    
    # Create figure
    fig = figure(figsize=(fig_width, fig_height))
    
    # Create base plot
    _create_base_plot(
        x=x, y=y, x_smooth=x_smooth, y_smooth=y_smooth,
        line_color=line_color, label=label,
        xaxis_orientation=xaxis_orientation, x_label=x_label
    )
    
    # Horizontal and vertical dashed lines for percentiles
    for i, (px, py) in enumerate(zip(at_percentiles_y, percentiles_cumfreq)):
        color = colors[i % len(colors)]
        
        # Horizontal line from x_min to px
        plot(
            [x.min() - (x[1] - x[0]), px],
            [py, py],
            color=color, linestyle="--",
            linewidth=0.75,
            alpha=0.7
        )
        
        # Vertical line from y_min (0) to py
        plot(
            [px, px],
            [0, py],
            color=color,
            linestyle="--",
            linewidth=0.75,
            alpha=0.7
        )
        
        # Marker at intersection
        scatter(px, py, color=color, marker="+", s=35, zorder=4)
    
    # Save figure
    savefig(filepath, dpi=150, bbox_inches="tight")
    close()
    
    return (
        f"/static/{filename}",
        at_percentiles_y.tolist() if hasattr(
                at_percentiles_y, 'tolist'
            ) else at_percentiles_y
    )


def plot_cumfreq_x_values(
    plot_params: ParamsPlot,
    percentile_params: ParamsPercentiles,
    x_axis: list[float],
    cum_freq: list[float],
    static_dir: str | None = None,
    method: Literal["default", "pchip", "cubic"] = "default",
) -> tuple[str, dict | None]:
    """
    Plot cumulative frequency curve with specific x-values marked.
    
    Parameters
    ----------
    plot_params : ParamsPlot
        Parameters for the plot including x-values to mark
    x_axis : list[float]
        X-axis values (class boundaries or midpoints)
    cum_freq : list[float]
        Cumulative frequency values
    static_dir : str, optional
        Directory to save the plot image (default: "static")
    method : {"default", "pchip", "cubic"}, optional
        Interpolation method (default: "default")
    
    Returns
    -------
    plot_url : str
        URL/path to the saved plot
    y_for_x_info : dict or None
        Dictionary with x-values and their corresponding y-values,
        or None if no x-values were requested
        
        Keys:
        - 'x': list of input x-values
        - 'y': list of interpolated y-values
    
    Raises
    ------
    ValueError
        If parsed_params.x_values_parsed is None
    
    Examples
    --------
    >>> plot_url, values_info = plot_cumfreq_x_values(
    ...     parsed_params, x_data, y_data
    ... )
    >>> print(f"X values: {values_info['x']}")
    >>> print(f"Y values: {values_info['y']}")
    """
    # Convert inputs to arrays
    if not isinstance(x_axis, ndarray):
        x = asarray(x_axis, dtype=float)
    else:
        x = x_axis
    
    if not isinstance(cum_freq, ndarray):
        y = asarray(cum_freq, dtype=float)
    else:
        y = cum_freq
    
    # Extract parameters
    find_y_for_x_values = percentile_params.x_values_parsed
    if find_y_for_x_values is None:
        raise ValueError(
            "No x-values provided in parsed_params.x_values_parsed"
        )
    
    fig_width = plot_params.fig_width
    fig_height = plot_params.fig_height
    line_color = plot_params.line_color
    xaxis_orientation = plot_params.xaxis_orientation
    x_label = plot_params.x_label
    
    line_color = "blue" if line_color == "default" else line_color
    
    # Get y values for the requested x values
    # Convert to list if it's a single value
    if not isinstance(find_y_for_x_values, (list, tuple, ndarray)):
        find_y_for_x_values = [find_y_for_x_values]
    
    # Get y values for all x values
    y_values = find_y_for_x(find_y_for_x_values, x, y, method)
    
    # Create info dictionary
    y_for_x_info = {
        'x': find_y_for_x_values,
        'y': y_values.tolist() if hasattr(
                y_values, 'tolist'
            ) else [y_values]
    }
    
    # Choose interpolation method
    if method in ("default", "pchip"):
        interpolator = PchipInterpolator(x, y)
        label = "Monotone cubic (PCHIP)"
    else:
        from scipy.interpolate import CubicSpline
        interpolator = CubicSpline(x, y, bc_type="natural")
        label = "Cubic spline"
    
    # Smooth curve
    x_smooth = linspace(x.min(), x.max(), 500)
    y_smooth = interpolator(x_smooth)
    
    # Determine where to save (Render or local)
    if static_dir is None:
        static_dir = _get_render_static_dir()  # ← Auto-detect
    else:
        os.makedirs(static_dir, exist_ok=True)

    # Generate filename
    filename = f"cumfreq_percentiles_{uuid4().hex}.png"
    filepath = os.path.join(static_dir, filename)
    
    # Create figure
    fig = figure(figsize=(fig_width, fig_height))
    
    # Create base plot
    _create_base_plot(
        x=x, y=y, x_smooth=x_smooth, y_smooth=y_smooth,
        line_color=line_color, label=label,
        xaxis_orientation=xaxis_orientation, x_label=x_label
    )
    
    # Convert to arrays for easier handling
    x_values = asarray(find_y_for_x_values)
    y_values_array = asarray(y_for_x_info['y'])
    
    # Plot each requested x-value point
    for i, (x_val, y_val) in enumerate(zip(x_values, y_values_array)):
        scatter(
            [x_val],
            [y_val],
            zorder=5,
            marker="+",
            s=35,
            color=colors[i],
            linewidths=1
        )
        
        # Add dashed lines for each point
        plot(
            [x_val, x_val],
            [0, y_val], 
            color=colors[i],
            linestyle="--",
            linewidth=0.75,
            alpha=0.7
        )
        plot(
            [x.min() - (x[1] - x[0]), x_val],
            [y_val, y_val], 
            color=colors[i],
            linestyle="--",
            linewidth=0.75,
            alpha=0.7
        )
    
    # Save figure
    savefig(filepath, dpi=150, bbox_inches="tight")
    close()
    
    return f"/static/{filename}", y_for_x_info


def get_y_for_x(
    x_value: float | list[float] | NDArray[float64],
    x_axis: list[float] | NDArray[float64],
    cum_freq: list[float] | NDArray[float64],
    method: str = "default"
) -> dict:
    """
    Get the cumulative frequency (y) for given x value(s) without plotting.
    
    Parameters
    ----------
    x_value : float or array_like
        The x value(s) to find y for (single value or list/array)
    x_axis : array_like
        List or array of x values
    cum_freq : array_like
        List or array of cumulative frequencies
    method : {"default", "pchip", "cubic"}, optional
        Interpolation method (default: "default")
    
    Returns
    -------
    result : dict
        Dictionary containing:
        - 'x': input x value(s) as list
        - 'y': interpolated y value(s) as list
    
    Examples
    --------
    >>> result = get_y_for_x(25.5, x_data, y_data)
    >>> print(f"Y value at x={result['x'][0]}: {result['y'][0]}")
    
    >>> result = get_y_for_x([25.5, 30.2, 35.7], x_data, y_data)
    >>> for x_val, y_val in zip(result['x'], result['y']):
    ...     print(f"Y value at x={x_val}: {y_val}")
    """
    # Convert to arrays
    if not isinstance(x_axis, ndarray):
        x = asarray(x_axis, dtype=float)
    else:
        x = x_axis
    
    if not isinstance(cum_freq, ndarray):
        y = asarray(cum_freq, dtype=float)
    else:
        y = cum_freq
    
    # Get y value(s)
    y_values = find_y_for_x(x_value, x, y, method)
    
    return {
        'x': x_value if isinstance(
                x_value, (list, tuple, ndarray)
            ) else [x_value],
        'y': y_values.tolist() if hasattr(
                y_values, 'tolist'
            ) else [y_values],
    }
    
    
def stats_grouped_percentiles_steps(
    data: ParamsData,
    percentile_params: ParamsPercentiles,
    plot_params: ParamsPlot
) -> dict[str, list[str]]:
    
    steps_mathjax = []
    
    lower = data.lower
    upper = data.upper
    class_labels = data.class_labels
    class_boundaries = data.class_labels
    class_limits = data.class_limits
    freq = data.freq
    decimals = data.decimals
    
    percentiles = percentile_params.percentiles
    cumfreq_curve = percentile_params.cumfreq_curve
    
    table_qtn_latex = table_grouped_qtn(
        class_labels=class_labels,
        freq=freq
    )
    
    steps_mathjax.append("Consider the grouped data in the table below.")
    steps_mathjax.append(f"\\[ {table_qtn_latex.rowise} \\]")
    splural = "percentile is" if len(freq) == 1 else "percentiles are"
    steps_mathjax.append(f"The requested {splural} calculated as follows.")
    
    step_temp = html_style_bg(
        title="STEP 1: Create a table of cumulative frequencies"
    )
    steps_mathjax.append(step_temp)
    steps_mathjax.append(
        "Create a table with class boundaries and cumulative frequencies. "
        f"The ith cumulative frequency \\( \\textbf{{last column}} \\) is "
        "found by summing all the preceding frequencies."
    )
    
    table_latex = table_grouped_cumfreq(
        class_boundaries=class_boundaries,
        class_limits=class_limits,
        freq=freq    
    )
    
    steps_mathjax.append(f"\\[ {table_latex.latex} \\]")
    
    result_dct = percentiles_calc(
        percentiles=percentiles,
        lower=lower,
        upper=upper,
        freq=freq
    )

    steps_temp_calculations, result_list = _calculations(
        percentiles=percentiles,
        result_dct=result_dct,
        decimals=decimals
    )
    
    if cumfreq_curve:
        steps_temp = _step_II_cumfreq_curve(
            percentile_params=percentile_params,
            plot_params=plot_params,
            result_dct=result_dct,
            result_list=result_list,
            decimals=1
        )
        steps_mathjax.extend(steps_temp)
    else:
        steps_temp = _step_II_formula(percentiles=percentiles)
        steps_mathjax.extend(steps_temp)

        steps_mathjax.extend(steps_temp_calculations)
    
    return steps_mathjax


def stats_grouped_percentiles_steps() -> ResultDict:
    return None