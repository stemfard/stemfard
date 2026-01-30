from stemcore import numeric_format

from stemfard.core.models import AnswerStepsResult, StatsDescriptives
from stemfard.stats.descriptives._base import BaseDescriptives
from stemfard.core._strings import str_remove_tzeros
from stemfard.stats.descriptives._freq_tally import sta_freq_tally
from stemfard.core._type_aliases import SequenceArrayLike


class StatsOthers(BaseDescriptives):
    
    def _stats_dfn(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.extend([
                "Degrees of freedom for a sample is calculated by subtracting "
                "\\( 1 \\) from the sample size, \\( n \\) (number of "
                "observations). That is,",
                f"\\( \\text{{df}} = n - 1 \\)",
                f"\\( \\quad = {self.n} - 1 \\)",
                f"\\( \\quad = {self.n - 1} \\)"
            ])
        
        return AnswerStepsResult(
            answer=self.n - 1,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    
    
    def _stats_dfn2(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.extend([
                "Degrees of freedom for a sample is calculated by subtracting "
                "\\( 2 \\) from the sample size, \\( n \\) (number of "
                "observations). That is,",
                f"\\( \\text{{df}} = n - 2 \\)",
                f"\\( \\quad = {self.n} - 2 \\)",
                f"\\( \\quad = {self.n - 2} \\)"
            ])
            
        return AnswerStepsResult(
            answer=self.n - 2,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    
    
    def _stats_tally(
        self,
        class_width: int | float,
        start_from: int | float | None = None,
        show_values: bool = False,
        include_cumfreq: bool = False,
        conf_level: float = 0.95,
        decimals: int = 4
    ) -> StatsDescriptives: # used `StatsDescriptives` since its output is different from the others
        
        return sta_freq_tally(
            data=self._data,
            class_width=class_width,
            start_from=start_from,
            show_values=show_values,
            include_cumfreq=include_cumfreq,
            conf_level=conf_level,
            decimals=decimals
        )


    def _stats_total(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            data_joined = " + ".join(map(str, self._data))
            steps_mathjax.append(
                "\\( \\displaystyle Total "
                f"= \\sum_{{i\\:=\\:1}}^{{n}} x_{{i}} \\)"
            )
            steps_mathjax.append(f"\\( \\quad = {data_joined} \\)")
            steps_mathjax.append(f"\\( \\quad = {self.total_rnd} \\)")
        
        return AnswerStepsResult(
            answer=self.total_rnd,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    

    def _stats_mean_ci(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.mean_ci,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    

    def _stats_sem(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.sem,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_cv(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.cv,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    
    
def stats_dfn(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    show_bg: bool = True
) -> StatsOthers:
    """
    Calculate the degrees of freedom for the dataset.

    The degrees of freedom (df) for a sample is defined as:

        df = n - 1

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
    result = StatsOthers(
        statistic="dfn",
        data=data,
        steps_compute=steps_compute,
        show_bg=show_bg
    )
    
    return result._stats_dfn()


def stats_dfn2(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    show_bg: bool = True
) -> StatsOthers:
    """
    Calculate the degrees of freedom for the dataset.

    The degrees of freedom (df) for a sample is defined as:

        df = n - 2

    where `n` is the number of observations. This is commonly used
    in statistical formulas such as sample variance, t-tests, and
    confidence intervals.

    Examples
    --------
    >>> import stemfard as stm
    >>> data = [42, 35, 20, 54, 38, 65, 42, 78, 29]
    >>> result = stm.stats_dfn2(data)
    >>> result.answer
    8
    >>> result.steps
    ['\\( df = n - 2 \\)', '\\( \\quad = 9 - 2 \\)', '\\( \\quad = 7 \\)']
    """
    result = StatsOthers(
        statistic="xxx",
        data=data,
        steps_compute=steps_compute,
        show_bg=show_bg
    )
    
    return result._stats_dfn2()


def stats_tally(
    data: SequenceArrayLike,
    *,
    class_width: int | float,
    start_from: int | float | None = None,
    show_values: bool = False,
    include_cumfreq: bool = False,
    conf_level: float = 0.95,
    decimals: int = 4
) -> StatsOthers:
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
    result = StatsOthers(
        statistic="tally",
        data=data
    )
    
    return result._stats_tally(
        class_width=class_width, start_from=start_from, show_values=show_values,
        include_cumfreq=include_cumfreq, conf_level=conf_level, decimals=decimals
    )


def stats_total(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsOthers:
    result = StatsOthers(
        statistic="xxx", data=data, steps_compute=steps_compute,
        show_bg=show_bg, decimals=decimals
    )
    
    return result._stats_total()
    
    
def stats_mean_ci(
    data: SequenceArrayLike,
    *,
    conf_level: float = 0.95,
    steps_compute: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsOthers:
    result = StatsOthers(
        statistic="mean_ci", data=data, conf_level=conf_level,
        steps_compute=steps_compute, show_bg=show_bg, decimals=decimals
    )
    return result._stats_mean_ci()
    
    
def stats_sem(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsOthers:
    return StatsOthers(
        statistic="xxx", data=data, steps_compute=steps_compute,
        steps_detailed=steps_detailed, show_bg=show_bg, decimals=decimals
    )._stats_sem()
    
    
def stats_cv(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsOthers:
    return StatsOthers(
        statistic="xxx", data=data, steps_compute=steps_compute,
        steps_detailed=steps_detailed, show_bg=show_bg, decimals=decimals
    )._stats_cv()