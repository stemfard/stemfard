from numpy import array, float64, floor, percentile, sort
from numpy.typing import NDArray
from sympy import flatten

from stemfard.core.models import AnswerStepsResult
from stemfard.stats.descriptives._base import BaseDescriptives
from stemfard.core.models import AnswerStepsResult
from stemfard.stats.descriptives._base import BaseDescriptives
from stemfard.core._enumerate import ColorCSS
from stemfard.core._html import html_style_bg
from stemfard.core._type_aliases import (
    ScalarSequenceArrayLike, SequenceArrayLike
)
from stemfard.core.arrays_highlight import one_d_array_stack
from stemfard.core.constants import StemConstants
from stemfard.core.convert import dframe_to_array
from stemfard.core._strings import str_ordinal


class StatsPosition(BaseDescriptives):

    def _stats_percentiles(
        self,
        steps_compute: bool = True,
        p: ScalarSequenceArrayLike | None = None,
        quartiles_use_linear: bool = True,
        decimals: int = 4
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            percentiles = p
            if percentiles is None:
                percentiles = [25, 50, 75]
            else:
                if isinstance(percentiles, (int, float)):
                    percentiles = [percentiles]
                
            data_percentiles = percentile(self._data, percentiles)
            answer = dict(zip([f"p{p}" for p in percentiles], data_percentiles))
            
            # Step 1: Sort the data
            step1_sort = []
            step1_sort.append(html_style_bg(title="STEP 1: Sort the Data"))
            latex_str = one_d_array_stack(data=self._data_sorted)
            step1_sort.append(
                "Sort the given data from the smallest to the largest value "
                "(ascending order). The sorted data is given below."
            )
            step1_sort.append(f"\\[ {latex_str} \\]")
            step1_sort.append(
                f"\\( \\textit{{The data could also have been sorted from the "
                f"largest to the smallest value (descending order).}} \\)"
            )
            
            p_list = []
            n = len(percentiles)
            for index, p in enumerate(percentiles):
                if n > 1:
                    steps_temp = html_style_bg(
                        title=(
                            f"{index + 1}.) {str_ordinal(p)} Percentile "
                            f"\\( \\mathrm{{(P_{{{p}}})}} \\)"
                        ),
                        bg=ColorCSS.COLORD8F0F8.value,
                        lw=2
                    )
                    p_list.append(steps_temp)
                    
                if index == 0:
                    p_list.append(step1_sort)
                    
                if n > 1 and index != 0:
                    p_list.append(html_style_bg(title="STEP 1: Sort the Data"))
                    p_list.append(
                        f"(This was already done in the first iteration)."
                    )
                perc_list = percentile_latex(
                    data=self._data,
                    p=p,
                    quartiles_use_linear=quartiles_use_linear,
                    decimals=decimals
                )
                p_list.extend(perc_list)

            steps_mathjax.extend(array(flatten(p_list), dtype=object))
        
        return AnswerStepsResult(
            answer=answer,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    

    def _stats_p25(
        self,
        steps_compute: bool = True,
        use_linear: bool = True,
        decimals: int = 4
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            p = 25
            answer = percentile(a=self._data, q=p)
            perc_list = percentile_latex(
                data=self._data,
                p=p,
                quartiles_use_linear=use_linear,
                decimals=decimals
            )
            steps_mathjax.extend(perc_list)
            
        return AnswerStepsResult(
            answer=answer,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_p50(
        self,
        steps_compute: bool = True,
        use_linear: bool = True,
        decimals: int = 4
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            p = 50
            answer = percentile(a=self._data, q=p)
            perc_list = percentile_latex(
                data=self._data,
                p=p,
                quartiles_use_linear=use_linear,
                decimals=decimals
            )
            steps_mathjax.extend(perc_list)
            
        return AnswerStepsResult(
            answer=answer,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_p75(
        self,
        steps_compute: bool = True,
        use_linear: bool = True,
        decimals: int = 4
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            p = 75
            answer = percentile(a=self._data, q=p)
            perc_list = percentile_latex(
                data=self._data,
                p=p,
                quartiles_use_linear=use_linear,
                decimals=decimals
            )
            steps_mathjax.extend(perc_list)
            
        return AnswerStepsResult(
            answer=answer,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_zscores(
        self,
        steps_compute: bool = True,
        ddof: int = 0,
        decimals: int = 4
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.extend([
                "The z scores are calculated by subtracting the sample "
                "arithmetic mean from each of the values of the dataset "
                "then dividing the result by the population standard "
                "deviation. That is, ",
                f"\\( \\quad z = \\frac{{x_{{i}} - \\bar{{x}}}}{{\\sigma}} \\)",
                "The values are calculated and presented in the second "
                "column of the table below."
            ])
            zscores = (self._data - self.mean) / self.std(ddof=ddof)
            zscores_latex = dframe_to_array(
                df=zscores,
                include_index=False,
                outer_border=True,
                inner_hlines=True,
                inner_vlines=True
            )
            steps_mathjax.append(zscores_latex)
            
        return AnswerStepsResult(
            answer=self.zscores,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    

def quartiles_latex(
    data: NDArray[float64], 
    p: float, 
    decimals: int = 4
) -> list[str]:
    """
    Calculate the 25th, 50th or 75th percentile of a dataset.
    
    Parameters
    ----------
    data : list of float
        Dataset of numbers.
    p : float
        Percentile (0-100)
    decimals: int, default=4
        Number of decimal points.
    
    Returns
    -------
    steps_mathjax : list[str]
        Step by step mathjax calculations.
    """
    steps_mathjax = []
    
    if not (0 <= p <= 100):
        raise ValueError(f"Expected 'p' to be between 0 and 100, got {p}.")
    
    # Step 1: Sort the data
    data_sorted = sort(data)
    
    # (steps_mathjax DONE ELSEWHERE)
    
    def data_split(data: SequenceArrayLike) -> int:
        n = data.size
        if n % 2 == 0:
            k = int(data.size / 2)
            data_lower, data_upper = data[:k], data[k:]
        else:
            k = int((data.size + 1) / 2) - 1 # ` - 1` is important
            data_lower, data_upper = data[:k], data[k + 1:]

        return data_lower, data_upper
    
    def _median(data: SequenceArrayLike) -> list[str]:
        steps_mathjax = []
        n = data.size
        if n % 2 == 0:
            k = int(n / 2) - 1 # `1` since is 0-indexed
            x1 = round(data[k], decimals)
            x2 = round(data[k + 1], decimals)
            p50 = round((x1 + x2) / 2, decimals)
            steps_mathjax.append(
                html_style_bg(title="STEP 2: Find the Rank Positions")
            )
            
            # note the use of `k + 1` in the latex display
            steps_mathjax.extend([
                f"For the given data, \\( n = {n} \\) (even). The rank "
                "positions (i.e. the positions of the two values at the "
                "center of the sorted data) are evaluated as follows.",
                f"\\( \\displaystyle\\quad k_{{1}} = \\frac{{n}}{{2}} "
                f"= \\frac{{{n}}}{{2}} = {k + 1} \\)",
                f"\\( \\quad k_{{2}} = k_{{1}} + 1 = {k + 2} \\)"
            ])
            
            steps_mathjax.extend([
                f"Therefore, the median is the average of the "
                f"\\( \\text{{{str_ordinal(k + 1)}}} \\) and "
                f"\\( \\text{{{str_ordinal(k + 2)}}} \\) values in the "
                "sorted data. These values are: ",
            ])
            
            steps_mathjax.extend([
                f"\\( \\quad x_{{{k + 1}}} = {x1} \\)",
                f"\\( \\quad x_{{{k + 2}}} = {x2} \\)",
                "The above values are also highlighted in the sorted data "
                " below."
            ])
            
            array_colored = one_d_array_stack(data, color_idx=[k, k + 1])
            steps_mathjax.append(f"\\[ {array_colored} \\]")
            
            steps_mathjax.append(
                html_style_bg(title="STEP 3: Calculate the Median")
            )
            steps_mathjax.append(
                "The median is found by calculating the average of the two "
                f"values identified in \\( \\text{{Step 2}} \\) above."
            )
            steps_mathjax.append(
                f"\\( \\displaystyle\\text{{P}}_{{50}} "
                f"= \\frac{{{x1} + {x2}}}{{2}} \\)"
            )
            steps_mathjax.append(f"\\( \\quad = {p50} \\)")
        else:
            k = int((n + 1) / 2) - 1 # `1` since is 0-indexed
            p50 = round(data[k], decimals)
            steps_mathjax.append(
                html_style_bg(title="STEP 2: Find the Rank Position")
            )
            
            # note the use of `k + 1` in the latex display
            steps_mathjax.extend([
                f"For the given data, \\( n = {n} \\) (odd). The rank "
                "position (i.e. the position of the value at the center "
                "of the sorted data) is evaluated as follows.",
                f"\\( \\displaystyle\\quad k = \\frac{{n + 1}}{{2}} "
                f"= \\frac{{{n} + 1}}{{2}} = {k + 1} \\)"
            ])
            steps_mathjax.extend([
                f"Therefore, the median is the "
                f"\\( \\text{{{str_ordinal(k + 1)}}} \\) value in the "
                "sorted data. That is, ",
                f"\\( \\qquad P_{{50}} = {p50} \\)",
                "This value is also highlighted in the sorted data below."
            ])
            array_colored = one_d_array_stack(data, color_idx=[k])
            steps_mathjax.append(f"\\[ {array_colored} \\]")
            
        return p50, steps_mathjax
    
    if p == 25:
        data_lower, _ = data_split(data_sorted)
        p50, steps_temp = _median(data_lower)
        steps_mathjax.extend(steps_temp)
    elif p == 50:
        p50, steps_temp = _median(data_sorted)
        steps_mathjax.extend(steps_temp)
    elif p == 75:
        _, data_upper = data_split(data_sorted)
        p50, steps_temp = _median(data_upper)
        steps_mathjax.extend(steps_temp)
    else:
        raise ValueError(f"Expected 'p' to be one of 25, 50 or 75, got {p}")
        
    # Step 5: Interpretation
    
    steps_temp = _interpretation_percentile(p=p, percentile_value=p50, step_n=3)
    steps_mathjax.extend(steps_temp)

    return steps_mathjax
    
    
def percentile_latex(
    data: SequenceArrayLike,
    p: float,
    quartiles_use_linear: False,
    decimals: int = 4
) -> list[str]:
    """
    Calculate the p-th percentile of a dataset.
    
    Parameters
    ----------
    data : list of float
        Dataset of numbers.
    p : float
        Percentile (0-100)
    decimals: int, default=4
        Number of decimal points.
    
    Returns
    -------
    steps_mathjax : list[str]
        Step by step mathjax calculations.
    
    Notes
    -----
    Uses the same formula as NumPy's default `np.percentile(method="linear")`:
        h = (n - 1) * p/100
        P = x_floor + fraction * (x_ceil - x_floor)
    where:
        n = number of observations
        h = fractional rank position (0 ≤ h ≤ n-1)
        x_floor = value at position floor(h) in sorted data
        x_ceil = value at position ceil(h) in sorted data
        fraction = h - floor(h)
    """
    steps_mathjax = []
    
    if not (0 <= p <= 100):
        raise ValueError(f"Expected 'p' to be between 0 and 100, got {p}.")
    
    if p in [25, 50, 75] and not quartiles_use_linear:
        return quartiles_latex(data=data, p=p)
    
    # Step 1: Sort the data
    data_sorted = sort(data)
    
    # (steps_mathjax DONE ELSEWHERE)
    
    # Step 2: Calculate the fractional rank position
    steps_mathjax.append(
        html_style_bg(title="STEP 2: Calculate Fractional Rank Position")
    )
    
    n = data_sorted.size
    h = (n - 1) * p / 100
    h_rnd = round(h, decimals)
    p_rnd = round(p / 100, decimals)
    
    steps_mathjax.extend([
        "The fractional rank position \\( h \\) is calculated using the "
        "formula:",
        f"\\(  \\displaystyle h = (n - 1) \\times \\frac{{p}}{{100}} \\)",
        f"\\( \\displaystyle\\quad = ({n} - 1) "
        f"\\times \\frac{{{p_rnd}}}{{100}} \\)",
        f"\\( \\displaystyle\\quad = {n - 1} \\times {p_rnd} \\)",
        f"\\( \\quad = {h_rnd} {StemConstants.CHECKMARK} \\)"
    ])
    
    # Step 3: Identify positions for interpolation
    lower_idx = int(floor(h))  # floor index (0-based)
    upper_idx = min(lower_idx + 1, n - 1)  # ceil index
    fraction = h - lower_idx
    fraction_rnd = round(fraction, decimals)
    
    steps_mathjax.append(
        html_style_bg(title="STEP 3: Find the Rank Positions")
    )
    steps_mathjax.extend([
        "Evaluate the lower and upper positions using the value of \\( h \\) "
        "calculated above.",
        f"Floor position: ",
        f"\\( \\quad\\lfloor h \\rfloor = \\text{{int(floor({h_rnd}))}} "
        f"= {lower_idx} \\)",
        f"Ceil position: ",
        f"\\( \\quad\\lceil h \\rceil = \\text{{min({h_rnd} + 1, {n} - 1)}} "
        f"= {upper_idx} \\)",
        # f"Fractional part: ",
        # f"\\( \\quad h - \\lfloor h \\rfloor = {h_rnd} - {lower_idx} "
        # f"= {fraction_rnd} \\)",
        f"The corresponding values in the sorted array are:",
        f"\\( \\quad x_{{\\lfloor h \\rfloor}} = x_{{{lower_idx}}} "
        f"= {data_sorted[lower_idx]} \\)",
        f"\\( \\quad x_{{\\lceil h \\rceil}} = x_{{{upper_idx}}} "
        f"= {data_sorted[upper_idx]} \\)"
    ])
    
    vals_colored = one_d_array_stack(
        data=data_sorted, color_idx=[lower_idx, upper_idx]
    )
    
    steps_mathjax.extend([
        "The above values are highlighted in the sorted data as shown below.",
        f"\\[ {vals_colored} \\]"
    ])

    # Step 4: Linear interpolation
    steps_mathjax.append(html_style_bg(title="STEP 4: Linear Interpolation"))
    
    lower_value = data_sorted[lower_idx]
    lower_value_rnd = round(lower_value, decimals)
    upper_value = data_sorted[upper_idx]
    upper_value_rnd = round(upper_value, decimals)
    percentile_value = lower_value + fraction * (upper_value - lower_value)
    percentile_rnd = round(percentile_value, decimals)
    
    steps_mathjax.extend([
        "The percentile is calculated using linear interpolation as "
        "follows",
        "\\( P_p = x_{\\lfloor h \\rfloor} + (h - \\lfloor h \\rfloor) \\times "
        "\\left(x_{\\lceil h \\rceil} - x_{\\lfloor h \\rfloor}\\right) \\)",
        f"\\( P_{{{p_rnd * 100}}} = {lower_value_rnd} + ({h_rnd} - {lower_idx}) "
        f"\\times ({upper_value_rnd} - {lower_value_rnd}) \\)",
        f"\\( \\quad = {lower_value_rnd} + {fraction_rnd} "
        f"\\times {upper_value - lower_value_rnd} \\)",
        f"\\( \\quad = {lower_value_rnd} "
        f"+ {round(fraction * (upper_value - lower_value), decimals)} \\)",
        f"\\( \\quad = {percentile_rnd} {StemConstants.CHECKMARK * 2} \\)",
    ])
    
    # Step 5: Interpretation
    steps_temp = _interpretation_percentile(
        p=p,
        percentile_value=percentile_rnd,
        step_n=5
    )
    steps_mathjax.extend(steps_temp)
    
    return steps_mathjax


def _interpretation_percentile(
    p: int | float,
    percentile_value: float,
    step_n: int
) -> list[str]:
    
    steps_mathjax = []
    steps_mathjax.append(html_style_bg(title=f"STEP {step_n}: Interpretation"))
    p_rnd2 = round(p, 2)
    steps_mathjax.extend([
        f"The \\( \\text{{{str_ordinal(p)}}} \\) percentile "
        f"\\( \\text{{P}}_{{{p_rnd2}}} = {percentile_value} \\) means that:",
        f"- Approximately \\( {p_rnd2}\% \\) of the observations are less "
        f"than or equal to \\( {percentile_value} \\)",
        f"- Approximately \\( {100 - p_rnd2}\% \\) of the observations are "
        f"greater than or equal to \\( {percentile_value} \\)",
    ])
    return steps_mathjax


def stats_percentiles(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_bg: bool = True,
    decimals: int = 4
) -> AnswerStepsResult:
    return StatsPosition(statistic="xxx", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed, steps_bg=steps_bg, decimals=decimals)._stats_percentiles(
        steps_compute=steps_compute, steps_bg=steps_bg, decimals=decimals
    )


def stats_p25(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_bg:bool = True,
    use_linear: bool = False,
    decimals: int = 4
) -> AnswerStepsResult:
    return StatsPosition(statistic="xxx", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed, steps_bg=steps_bg, decimals=decimals)._stats_p25(
        steps_compute=steps_compute,
        use_linear=use_linear,
        decimals=decimals
    )


def stats_p50(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_bg: bool = True,
    use_linear: bool = False,
    decimals: int = 4
) -> AnswerStepsResult:
    return StatsPosition(statistic="xxx", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed, steps_bg=steps_bg, decimals=decimals)._stats_p50(
        steps_compute=steps_compute,
        use_linear=use_linear,
        decimals=decimals
    )


def stats_p75(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_bg: bool = True,
    use_linear: bool = False,
    decimals: int = 4
) -> AnswerStepsResult:
    return StatsPosition(statistic="xxx", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed, steps_bg=steps_bg, decimals=decimals)._stats_p75(
        steps_compute=steps_compute,
        use_linear=use_linear,
        decimals=decimals
    )


def stats_zscores(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_bg: bool = True,
    decimals: int = 4
) -> AnswerStepsResult:
    return StatsPosition(statistic="xxx", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed, steps_bg=steps_bg, decimals=decimals)._stats_zscores(
        steps_compute=steps_compute, steps_bg=steps_bg, decimals=decimals
    )