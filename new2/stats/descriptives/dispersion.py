from typing import Literal
from numpy import count_nonzero

from stemfard.core.models import AnswerStepsResult
from stemfard.stats.descriptives._base import BaseDescriptives
from stemfard.core._html import html_bg_level1
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core.arrays_highlight import one_d_array_stack
from stemfard.stats.descriptives.position import stats_p25, stats_p75


class StatsDispersion(BaseDescriptives):
    "Methods for calculating measures of dispersion"
    def _stats_min(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            data_latex = one_d_array_stack(
                self._data,
                ncols=10,
                color_vals=self.min
            )
            min_vals = self._data[self._data == self.min]
            splural = (
                "this minimum value is" if len(min_vals) == 1 else "these "
                "minimum values are"
            )
            steps = [
                f"The minimum value is the smallest value in a set of values. "
                f"For the given data, {splural} highlighted below.",
                f"\\[ {data_latex} \\]"
            ]
            steps_mathjax.extend(steps)
        
        return AnswerStepsResult(
            answer=self.min,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    
    
    def _stats_max(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            data_latex = one_d_array_stack(
                self._data,
                ncols=10,
                color_vals=self.max
            )
            max_vals = self._data[self._data == self.max]
            splural = (
                "this maximum value is" if len(max_vals) == 1 else "these "
                "maximum values are"
            )
            steps = [
                f"The maximum value is the largest value in a set of values. "
                f"For the given data, {splural} highlighted below.",
                f"\\[ {data_latex} \\]"
            ]
            steps_mathjax.extend(steps)
        
        return AnswerStepsResult(
            answer=self.max,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_range(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            min_idx = self._data.argmin()
            max_idx = self._data.argmax()
            idx = (min_idx, max_idx)
            m = count_nonzero(self._data == self.min)
            n = count_nonzero(self._data == self.max)
            
            multiple = ""
            if (m + n) > 2:
                multiple = " (only the first occurances are highlighted)"
            
            data_latex = one_d_array_stack(self._data, ncols=10, color_idx=idx)
            steps_temp = [
                "The minimum and maximum values are highlighted in the data "
                f"below{multiple}.",
                f"\\[ {data_latex} \\]"
            ]
            steps_mathjax.extend(steps_temp)
            steps_mathjax.extend([
                "The range is calculated as the difference between the "
                "maximum and the minimum values. That is, ",
                f"\\( \\text{{Range}} = {{\\max(x)}} - {{\\min(x)}} \\)",
                f"\\( \\qquad = {self.max} - {self.min} \\)",
                f"\\( \\qquad = {self.max - self.min} \\)"
            ])
            
        return AnswerStepsResult(
            answer=self.range,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )


    def _stats_var(
        self,
        assumed_mean: int | float | None = None,
        formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
        var_formula: Literal[1, 2, 3] = 1,
        ddof: int = 1,
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.var(ddof=ddof),
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )


    def _stats_std(
        self,
        assumed_mean: int | float | None = None,
        formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
        var_formula: Literal[1, 2, 3] = 1,
        ddof: int = 1,
    ) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.std(ddof=ddof),
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_iqr(self, use_linear: bool = False) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            title = (
                f"1.) Calculate the Lower Quartile \\( \\text{{P}}_{{25}} \\)"
            )
            steps_mathjax.append(html_bg_level1(title=title))
            steps_temp = stats_p25(
                data=self._data,
                steps_compute=True,
                steps_detailed=False,
                use_linear=use_linear,
                decimals=self.decimals
            )
            steps_mathjax.extend(steps_temp.steps[:-4]) # `-4` excludes interpretation
            title = (
                f"2.) Calculate the Upper Quartile \\( \\text{{P}}_{{75}} \\)"
            )
            steps_mathjax.append(html_bg_level1(title=title))
            
            steps_temp = stats_p75(
                data=self._data,
                steps_compute=True,
                steps_detailed=False,
                use_linear=use_linear,
                decimals=self.decimals
            )
            steps_mathjax.extend(steps_temp.steps[:-4]) # `-4` excludes interpretation
            
            title = (f"3.) Calculate the Interquartile Range (IQR)")
            steps_mathjax.append(html_bg_level1(title=title))
            steps_mathjax.extend([
                f"The \\( \\text{{IQR}} \\) is calculated as the difference "
                f"between the \\( \\text{{upper}} \\) and "
                f"\\( \\text{{lower}} \\) interquartile range. This is done below.",
                f"\\( \\text{{IQR}} = \\text{{P}}_{{75}} - \\text{{P}}_{{25}} \\)",
                f"\\( \\qquad = {self.p75} - {self.p25} \\)",
                f"\\( \\qquad = {self.iqr} \\)"
            ])
        
        return AnswerStepsResult(
            answer=self.iqr,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_iqd(self, use_linear: bool = False) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_temp = self.stats_iqr(
                steps_compute=True,
                steps_detailed=False,
                use_linear=use_linear,
                decimals=self.decimals
            )
            steps_mathjax.extend(steps_temp.steps)
            title = "4.) Calculate the Interquartile Range Deviation (IQD)"
            steps_mathjax.append(html_bg_level1(title=title))
            steps_mathjax.extend([
                f"The \\( \\text{{IQD}} \\) is calculated by dividing the "
                f"\\( \\text{{interquartile range}} \\) calculated in "
                f"\\( \\text{{3.)}} \\) above by \\( 2 \\). This is done "
                "below. ",
                f"\\( \\displaystyle\\text{{IQD}} "
                f"= \\frac{{\\text{{IQR}}}}{{2}} \\)",
                f"\\( \\displaystyle\\qquad "
                f"= \\frac{{{self.iqr}}}{{2}} \\)",
                f"\\( \\qquad = {self.iqd} \\)"
            ])

        return AnswerStepsResult(
            answer=self.iqd,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    
    
    def _stats_mad(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.mean_abs_dev,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_mead(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.median_abs_dev,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
        
        
def stats_min(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True
) -> StatsDispersion:
    return StatsDispersion(
        statistic="min", data=data, steps_compute=steps_compute
    )._stats_min()


def stats_max(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
) -> StatsDispersion:
    return StatsDispersion(
        statistic="max", data=data, steps_compute=steps_compute    
    )._stats_max()


def stats_range(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_bg: bool = True,
    decimals: int = 4
) -> StatsDispersion:
    return StatsDispersion(
        statistic="range", data=data, steps_compute=steps_compute,
        steps_bg=steps_bg, decimals=decimals
    )._stats_range()
    

def stats_var(
    data: SequenceArrayLike,
    *,
    assumed_mean: int | float | None = None,
    formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
    var_formula: Literal[1, 2, 3] = 1,
    ddof: int = 1,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    decimals: int = 4
) -> StatsDispersion:
    
    result = StatsDispersion(statistic="xxx", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed, decimals=decimals)
    return result._stats_var(
            assumed_mean=assumed_mean,
            formula=formula,
            var_formula=var_formula,
            ddof=ddof
        )
    
    
def stats_std(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    assumed_mean: int | float | None = None,
    formula: Literal["x-a", "x/w-a", "(x-a)/w"] | None = None,
    var_formula: Literal[1, 2, 3] = 1,
    ddof: int = 1,
    decimals: int = 4
) -> StatsDispersion:
    result = StatsDispersion(
        statistic="xxx", data=data, steps_compute=steps_compute,
        steps_detailed=steps_detailed, decimals=decimals
    )
    return result._stats_std(
        assumed_mean=assumed_mean,
        formula=formula,
        var_formula=var_formula,
        ddof=ddof
    )
    

def stats_iqr(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 4
) -> StatsDispersion:
    result = StatsDispersion(
        statistic="xxx", data=data,steps_compute=steps_compute,
        steps_detailed=steps_detailed, decimals=decimals
    )
    return result._stats_iqr()


def stats_iqd(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 4
) -> StatsDispersion:
    result = StatsDispersion(
        statistic="xxx", data=data, steps_compute=steps_compute,
        steps_detailed=steps_detailed, steps_bg=steps_bg, decimals=decimals
    )
    return result._stats_iqd()


def stats_mad(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 4
) -> StatsDispersion:
    result = StatsDispersion(
        statistic="xxx", data=data,steps_compute=steps_compute,
        steps_detailed=steps_detailed, steps_bg=steps_bg, decimals=decimals
    )
    return result._stats_mad()


def stats_mead(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 4
) -> StatsDispersion:
    result = StatsDispersion(
        statistic="xxx", data=data, steps_compute=steps_compute,
        steps_detailed=steps_detailed, decimals=decimals
    )
    return result._stats_mead()