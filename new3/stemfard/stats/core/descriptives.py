from functools import cached_property, wraps
from typing import Callable, Literal

from numpy import (
    around, array, count_nonzero, float64, floor, median, percentile, sort, sqrt
)
from numpy.typing import NDArray
from pandas import Series
from scipy import stats
from sympy import flatten
from stemcore import arr_to_numeric, numeric_format

from stemfard.core._type_aliases import (
    ScalarSequenceArrayLike, SequenceArrayLike
)
from stemfard.core.models import (AnswerStepsResult, StatsDescriptives)
from stemfard.stats.descriptives._freq_tally import sta_freq_tally
from stemfard.core._strings import str_ordinal, str_remove_tzeros
from stemfard.core.arrays_highlight import one_d_array_stack
from stemfard.core._html import html_style_bg
from stemfard.core.constants import StemConstants
from stemfard.core._enumerate import ColorCSS
from stemfard.stats.descriptives._base import StatsMeanConfidenceInterval


class Descriptives:
    """
    Compute descriptive statistics for one-dimensional numeric data.

    This class provides a comprehensive set of descriptive statistical
    measures including location, dispersion, shape, and frequency-based
    summaries. Each statistic is returned together with optional
    step-by-step mathematical explanations formatted for MathJax 
    rendering.

    Parameters
    ----------
    data : SequenceArrayLike
        One-dimensional sequence of numeric observations.
    decimals : int, default=4
        Number of decimal places used when rounding results.

    Notes
    -----
    - Input data is coerced to numeric form using ``arr_to_numeric``.
    - Cached properties are used to avoid recomputation of core statistics.
    - Step-by-step explanations are accumulated in ``steps_mathjax``.
    """
    def __init__(self, data: SequenceArrayLike, decimals: int = 4):
        self.data = arr_to_numeric(data=data)
        self.decimals = decimals
    
    @property
    def n(self) -> int:
        return len(self.data)
    
    @property
    def data_sorted(self) -> NDArray:
        return sort(self.data)
        
    @cached_property
    def total(self) -> int | float:
        return self.data.sum()
    
    @cached_property
    def total_rnd(self) -> int | float:
        return numeric_format(around(self.total, self.decimals))
    
    @cached_property
    def mean(self) -> float:
        return self.data.mean()
    
    @cached_property
    def mean_rnd(self) -> int | float:
        return numeric_format(around(self.mean, self.decimals))
    
    @cached_property
    def var(self) -> float:
        return self.data.var(ddof=1)
    
    @cached_property
    def var_rnd(self) -> int | float:
        return numeric_format(around(self.var, self.decimals))
    
    @cached_property
    def std(self) -> float:
        return self.data.std(ddof=1)
    
    @cached_property
    def std_rnd(self) -> float:
        return numeric_format(around(self.std, self.decimals))
    
    
    def stats_total(self) -> AnswerStepsResult:
        """
        Calculate the total (sum) of all values in the dataset.

        Returns
        -------
        AnswerStepsResult
            answer : float
                The sum of all data values.
            steps : list[str]
                Step-by-step explanation of the calculation.

        Examples
        --------
        >>> import stemfard as stm
        >>> data = [10, 20, 30, 40]
        >>> result = stm.stats_total(data)
        >>> result.answer
        100
        >>> result.steps
        [
            "\\( \\displaystyle Total = \\sum_{i=1}^{n} x_i \\)",
            "\\( \\quad = 10 + 20 + 30 + 40 \\)",
            "\\( \\quad = 100 \\)"
        ]
        """
        steps_mathjax: list[str] = []
        
        data_joined = (
            " + ".join(map(str, numeric_format(self.data)))
            .replace("+ -", "- ")
        )
        steps = [
            f"\\( \\displaystyle Total = \\sum_{{i\\:=\\:1}}^{{n}} x_{{i}} \\)",
            f"\\( \\quad = {data_joined} \\)",
            f"\\( \\quad = {self.total_rnd} \\)"
        ]
        steps_mathjax.extend(steps)
        
        return AnswerStepsResult(
            answer=self.total_rnd,
            steps=steps_mathjax
        )
        

    def stats_tally(
        self,
        class_width: int | float,
        start_from: int | float | None = None,
        show_values: bool = False,
        include_cumfreq: bool = False,
        conf_level: float = 0.95
    ) -> StatsDescriptives:
        """
        Generate a frequency tally (grouped frequency table) for the
        dataset.

        This method groups the numeric data into classes of specified
        width and computes the frequency for each class. Optional
        cumulative frequency and class values can be included.
        A confidence level can also be specified for interval
        calculations.

        Parameters
        ----------
        class_width : int or float
            The width of each class interval for grouping the data.
        start_from : int or float, optional
            The starting point for the first class. Defaults to the 
            minimum of the data.
        show_values : bool, default=False
            If True, include the actual data points in each class in
            the output.
        include_cumfreq : bool, default=False
            If True, include cumulative frequencies in the output.
        conf_level : float, default=0.95
            Confidence level for interval calculations, if applicable.

        Returns
        -------
        StatsDescriptives
            An object containing the frequency table, class intervals, 
            frequencies, cumulative frequencies (if requested), and any
            additional statistics computed.

        Examples
        --------
        >>> import stemfard as stm
        >>> data = [
            38, 40, 54, 43, 43, 56, 46, 32, 37, 38, 52, 45, 45, 43, 38,
            56, 46, 26, 48, 38, 33, 40, 34, 36, 37, 29, 49, 43, 33, 52,
            45, 40, 49, 44, 41, 42, 46, 42, 40, 39, 36, 40, 32, 59, 52,
            33, 39, 38, 48, 41
        ]
        >>> tally = stm.stats_tally(data=data, class_width=5)
        >>> tally.table
                 Class                 Tally  Frequency
        1  25 ≤ x < 30                    //          2
        2  30 ≤ x < 35             ///// . /          6
        3  35 ≤ x < 40     ///// . ///// . /         11
        4  40 ≤ x < 45  ///// . ///// . ////         14
        5  45 ≤ x < 50       ///// . ///// .         10
        6  50 ≤ x < 55                  ////          4
        7  55 ≤ x < 60                   ///          3
        
        >>> tally.cumfreq
        array([ 2,  8, 19, 33, 43, 47, 50], dtype=int64)
        
        >>> result.stats.mean
        41.92
        
        >>> result.stats.mode
        {'mode': 38.0, 'count': 5}
        
        >>> result.stats.mean_ci
        {'lci': 39.86914794328127, 'uci': 43.97085205671873}
        
        >>> result.stats.mean_ci["lower"]
        39.86914794328127
        
        >>> result.stats.mean_ci["upper"]
        43.97085205671873
        """
        result = sta_freq_tally(
            data=self._data,
            class_width=class_width,
            start_from=start_from,
            show_values=show_values,
            include_cumfreq=include_cumfreq,
            conf_level=conf_level,
            decimals=self.decimals
        )
        return result
    

    def stats_dfn(self, k: Literal[1, 2] = 1) -> AnswerStepsResult:
        """
        Calculate the degrees of freedom for the dataset.

        The degrees of freedom (df) for a sample is defined as:

            df = n - k

        where `n` is the number of observations. This is commonly used
        in statistical formulas such as sample variance, t-tests, and
        confidence intervals.

        Examples
        --------
        >>> import stemfard as stm
        >>> data = [42, 35, 20, 54, 38, 65, 42, 78, 29]
        >>> result = stm.stats_dfn(data)
        >>> result.answer
        8
        >>> result.steps
        ['\\( df = n - 1 \\)', '\\( \\quad = 9 - 1 \\)', '\\( \\quad = 8 \\)']
        """
        steps_mathjax: list[str] = []
        
        if k not in (1, 2):
            raise ValueError(f"Expected 'k' to be either 1 or 2, got {k!r}")
        
        steps_mathjax.extend([
            "Degrees of freedom for a sample is calculated by subtracting "
            f"\\( {k} \\) from the sample size, \\( n \\) (number of observations). "
            "That is,",
            f"\\( \\text{{df}} = n - {k} \\)",
            f"\\( \\quad = {self.n} - {k} \\)",
            f"\\( \\quad = {self.n - k} \\)"
        ])
        return AnswerStepsResult(answer=self.n - k, steps=steps_mathjax)


    def stats_min(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        data_min = numeric_format(self.data.min())
        steps = one_d_array_stack(self.data, ncols=10, color_vals=data_min)
        min_vals = self.data[self.data == data_min]
        splural = "value is" if len(min_vals) == 1 else "values are"
        steps = [
            f"The minimum {splural} highlighted in the data below.",
            f"\\( {steps} \\)"
        ]
        steps_mathjax.extend(steps)
        
        return AnswerStepsResult(answer=data_min, steps=steps_mathjax)


    def stats_max(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        data_max = numeric_format(self.data.max())
        steps = one_d_array_stack(self.data, ncols=10, color_vals=data_max)
        min_vals = self.data[self.data == data_max]
        splural = "value is" if len(min_vals) == 1 else "values are"
        steps = [
            f"The maximum {splural} highlighted in the data below.",
            f"\\( {steps} \\)"
        ]
        steps_mathjax.extend(steps)
        
        return AnswerStepsResult(answer=data_max, steps=steps_mathjax)


    def stats_range(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        min_vals = self.data.min()
        max_vals = self.data.max()
        data_range = numeric_format(max_vals - min_vals)
        min_idx = self.data.argmin()
        max_idx = self.data.argmax()
        idx = (min_idx, max_idx)
        m = count_nonzero(self.data == min_vals)
        n = count_nonzero(self.data == max_vals)
        multiple = ""
        if (m + n) > 2:
            multiple = " (only the first occurances are highlighted)"
        steps = one_d_array_stack(self.data, ncols=10, color_idx=idx)
        steps = [
            "The minimum and maximum values are highlighted in the data "
            f"below{multiple}.",
            f"\\( {steps} \\)"
        ]
        steps_mathjax.extend(steps)
        
        return AnswerStepsResult(answer=data_range, steps=steps_mathjax)


    def stats_percentiles(
        self,
        p: ScalarSequenceArrayLike | None = None,
        quartiles_use_linear: bool = True
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        percentiles = p
        if percentiles is None:
            percentiles = [25, 50, 75]
        else:
            if isinstance(percentiles, (int, float)):
                percentiles = [percentiles]
        percentiles = numeric_format(percentiles)
            
        data_percentiles = numeric_format(percentile(self.data, percentiles))
        answer = dict(zip([f"p{p}" for p in percentiles], data_percentiles))
        
        # Step 1: Sort the data
        step1_sort = []
        step1_sort.append(html_style_bg(title="STEP 1: Sort the Data"))
        latex_str = one_d_array_stack(data=self.data_sorted)
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
                decimals=self.decimals
            )
            p_list.extend(perc_list)

        steps_mathjax.extend(array(flatten(p_list), dtype=object))
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_p25(self, use_linear: bool = False) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        p = 25
        answer = percentile(self.data, p)
        perc_list = percentile_latex(
            data=self._data,
            p=p,
            quartiles_use_linear=use_linear,
            decimals=self.decimals
        )
        steps_mathjax.extend(perc_list)
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_p50(self, use_linear: bool = False) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        p = 50
        answer = percentile(self.data, p)
        perc_list = percentile_latex(
            data=self._data,
            p=p,
            quartiles_use_linear=use_linear,
            decimals=self.decimals
        )
        steps_mathjax.extend(perc_list)
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_p75(self, use_linear: bool = False) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        p = 75
        answer = percentile(self.data, p)
        perc_list = percentile_latex(
            data=self._data,
            p=p,
            quartiles_use_linear=use_linear,
            decimals=self.decimals
        )
        steps_mathjax.extend(perc_list)
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_iqr(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = percentile(self.data, 75) - percentile(self.data, 25)
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_iqd(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = (percentile(self.data, 50) - percentile(self.data, 25)) / 2
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_mode(self) -> AnswerStepsResult:
        """Find the mode"""
        
        steps_mathjax: list[str] = []
        
        res_mode = stats.mode(self.data, keepdims=False)
        answer = {"mode": res_mode.mode, "count": res_mode.count}
        arr_str = one_d_array_stack(data=self._data, color_vals=res_mode)
        steps_mathjax.append(f"\\( {arr_str} \\)")
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_median(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = median(self.data)
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_mean(
        self,
        assumed_mean: int | float | None = None
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = self.mean
        if assumed_mean:
            steps_mathjax.extend(
                stats_mean_steps(data=self._data, decimals=self.decimals)
            )
        else:
            data_joined = str_remove_tzeros(" + ".join(map(str, self.data)))
            steps_temp = [
                f"\\( \\displaystyle\\bar{{x}} "
                f"= \\frac{{1}}{{n}} \\sum_{{i\\:=\\:1}}^{{n}} x_{{i}} \\)",
                f"\\( \\displaystyle\\quad = \\frac{{{data_joined}}}{{{self.n}}} \\)",
                f"\\( \\displaystyle\\quad = \\frac{{{self.total_rnd}}}{{{self.n}}} \\)",
                f"\\( \\quad = {self.mean_rnd} \\)"
            ]
            steps_mathjax.extend(steps_temp)
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_mean_ci(
        self, conf_level, show_mean_steps: bool = False
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        alpha = 1.0 - conf_level
        t_critical = stats.t.ppf(1 - alpha / 2, df=self.n - 1)
        stderror = self.std / sqrt(self.n)
        margin = t_critical * stderror
        margin_rnd = round(margin, self.decimals)
        
        lower = self.mean - margin
        lower_rnd = round(lower, self.decimals)
        upper = self.mean + margin
        upper_rnd = round(upper, self.decimals)
        
        steps_mathjax.extend([
            f"The {conf_level} confidence interval for the mean is "
            "calculated as follows.",
            f"\\( \\displaystyle 95% \\text{{CI}} "
            f"= \\bar{{x}} \\pm t_{{{conf_level}}}"
            f"\\:\\frac{{sigma}}{{\\sqrt{{n}}}} \\)",
        ])
        
        if show_mean_steps:
            steps_mathjax.extend(self.stats_mean(assumed_mean=None))
            
        steps_mathjax.extend([
            f"\\( \\displaystyle 95% \\text{{CI}} "
            f"= {self.mean_rnd} \\pm {t_critical}"
            f"\\:\\frac{{{self.std_rnd}}}{{\\sqrt{{{self.n}}}}} \\)",
            f"\\( \\quad = {self.mean_rnd} \\pm {margin_rnd} \\)",
            "Therefore,",
            f"\\( 95% \\text{{LCI}} = {self.mean_rnd} - {margin_rnd} \\)",
            f"\\( \\quad = {lower_rnd} \\)",
            f"\\( 95% \\text{{UCI}} = {self.mean_rnd} + {margin_rnd} \\)",
            f"\\( \\quad = {upper_rnd} \\)"
        ])
        
        answer = {"lower": lower, "upper": upper}
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_var(self, ddof: int = 1) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = self.data.var()
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_std(self, ddof: int = 1) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = self.data.std()
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_stderror(self, ddof: int = 1) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = self.data.std(ddof=1) / sqrt(self.n)
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_sem(self, ddof: int = 1) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = Series(self.data).sem(ddof=1)
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_cv(self, ddof: int = 1) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = self.std / self.mean
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_skew(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        answer = Series(self.data).skew()
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)


    def stats_kurt(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] = []
        
        data_kurt = Series(self.data).kurt()
        answer = {"kurt": data_kurt, "kurt + 3": data_kurt + 3}
        
        return AnswerStepsResult(answer=answer, steps=steps_mathjax)

# Wrapper factory (after the class)

def _make_wrapper(method_name: str) -> Callable:
    """
    Create a functional wrapper for a Descriptives instance method.

    This wrapper allows calling the method directly in a functional style:

        result = stats_mean(data)

    instead of manually creating a Descriptives instance:

        desc = Descriptives(data)
        result = desc.stats_mean()

    The wrapper also preserves the original method's docstring and 
    examples.
    """

    method = getattr(Descriptives, method_name)

    @wraps(method)
    def wrapper(data: SequenceArrayLike, decimals: int = 4, **kwargs):
        desc = Descriptives(data=data, decimals=decimals)
        result = getattr(desc, method_name)(**kwargs)

        # stats_tally returns StatsDescriptives directly
        if not isinstance(result, AnswerStepsResult):
            return result

        return AnswerStepsResult(
            answer=result.answer,
            steps=result.steps,
        )

    wrapper.__name__ = method_name
    wrapper.__qualname__ = method_name
    return wrapper

# Public API declaration

_STATS_METHODS = [
    "stats_total", "stats_tally", "stats_dfn", "stats_min", "stats_max",
    "stats_range", "stats_percentiles", "stats_p25", "stats_p50", "stats_p75",
    "stats_iqr", "stats_iqd", "stats_mode", "stats_median", "stats_mean",
    "stats_mean_ci", "stats_var", "stats_std", "stats_stderror", "stats_sem",
    "stats_cv", "stats_skew", "stats_kurt"
]

# Auto-generate wrappers

__all__ = []

for _method_name in _STATS_METHODS:
    globals()[_method_name] = _make_wrapper(_method_name)
    __all__.append(_method_name)


def stats_mean_steps(data: SequenceArrayLike, decimals: int = 4) -> list[str]:
    steps_mathjax = []
    steps_mathjax.append("To update...")
    
    return steps_mathjax


def stats_var_steps(
    data: SequenceArrayLike,
    formula: Literal[1, 2, 3] = 1,
    use_table: bool = False,
    decimals: int = 4
) -> list[str]:
    steps_mathjax = []
    steps_mathjax.append("To update...")
    
    return steps_mathjax