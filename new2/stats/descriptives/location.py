from stemcore import str_data_join
from stemfard.core.models import AnswerStepsResult
from stemfard.stats.descriptives._base import BaseDescriptives
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core.arrays_highlight import one_d_array_stack
from stemfard.stats.descriptives._steps_mean import ArithmeticMeanSteps
from stemfard.stats.descriptives._tables import QtnDownloadableTables


class StatsLocation(BaseDescriptives):

    def _stats_mean(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        params = {
            "statistic": self.statistic,
            "data": self._data,
            "freq": self.freq,
            "assumed_mean": self.assumed_mean,
            "steps_compute": self.steps_compute,
            "steps_detailed": self.steps_detailed,
            "show_bg": self.show_bg,
            "param_name": self.param_name,
            "decimals": self.decimals
        }
        
        if self.show_bg:
            steps_temp = ArithmeticMeanSteps(**params).bg_arithmetic_mean()
            steps_mathjax.extend(steps_temp)
            
        if self.steps_compute:
            
            if not self.steps_detailed:
                steps_temp = ArithmeticMeanSteps(
                    **params
                ).arithmetic_mean_reusable(border_before=True)
            
            steps_mathjax.append("For the data given below,")
            
            if self.freq is None:
                data_latex = one_d_array_stack(data=self.data_rnd)
                steps_mathjax.append(data_latex)
            else:
                table_latex = QtnDownloadableTables(**params)._table_qtn.rowise
                steps_mathjax.append(f"\\[ {table_latex} \\]")
                
            if self.assumed_mean:
                if self.is_calculate_amean:
                    steps_mathjax.append(
                        "The arithmetic mean given an assumed mean of is "
                        "\\( A \\) evaluated as follows."
                    )
                else:
                    steps_mathjax.append(
                        "The arithmetic mean using an assumed mean of "
                        f"\\( A = {self.assumed_mean_rnd} \\) is evaluated "
                        "as follows."
                    )
            else:
                steps_mathjax.append(
                    "The arithmetic mean is calculated as follows."
                )
            
            if self.freq is None:
                if not self.assumed_mean:
                    steps_temp = ArithmeticMeanSteps(
                        **params
                    ).mean_and_no_assumed_mean_and_no_freq()
                else:
                    steps_temp = ArithmeticMeanSteps(
                        **params    
                    ).mean_and_assumed_mean_and_no_freq()
            else:
                if not self.assumed_mean:
                    steps_temp = ArithmeticMeanSteps(
                        **params    
                    ).mean_and_no_assumed_mean_and_freq()
                else:
                    steps_temp = ArithmeticMeanSteps(
                        **params    
                    ).mean_and_assumed_mean_and_freq()
            
            steps_mathjax.extend(steps_temp)

        return AnswerStepsResult(
            answer=self.mean,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    

    def _stats_median(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.median,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_mode(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            mode = self.mode
            mode, count = mode.mode, mode.count
            splural = (
                "this value is" if count == 1 else "these values are"
            )
            steps_mathjax.append(
                "The mode is the most frequently occuring value in a set of "
                f"values. For the given data, {splural} highlighted below."
            )
            data_latex = one_d_array_stack(data=self._data, color_vals=mode)
            steps_mathjax.append(f"\\[ {data_latex} \\]")
            splural = "once" if count == 1 else f"\\( {count} \\) times"
            steps_mathjax.append(
                f"The mode is therefore \\( {mode} \\) and appears {splural}."
            )

        return AnswerStepsResult(
            answer=self.mode,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_multimode(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            multimode = self.multimode
            modes, count = multimode.modes, multimode.count
            splural = (
                "this value is" if count == 1 else "these values are"
            )
            steps_mathjax.append(
                "The mode is the most frequently occuring value in a set of "
                f"values. For the given data, {splural} highlighted below."
            )
            data_latex = one_d_array_stack(data=self._data, color_vals=modes)
            steps_mathjax.append(f"\\[ {data_latex} \\]")
            splural = "once" if count == 1 else f"\\( {count} \\) times"
            steps_mathjax.append(
                f"The modes are therefore \\( \\left( {str_data_join(modes)} "
                f"\\right) \\) and each appears {splural}."
            )

        return AnswerStepsResult(
            answer=self.multimode,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
        
    
    def _stats_harmonic(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.harmonic,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )

    
    def _stats_geometric(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.geometric,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
        
    
def stats_mean(
    data: SequenceArrayLike,
    *,
    freq: SequenceArrayLike | None = None,
    assumed_mean: int | float | None = None,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_name: str = "x",
    decimals: int = 4
) -> StatsLocation:
    """
    >>> data = [
        38, 40, 54, 43, 43, 56, 46, 32, 37, 38, 52, 45, 45, 43, 38, 56, 46,
        26, 48, 38, 33, 40, 34, 36, 37, 29, 49, 43, 33, 52, 45, 40, 49, 44,
        41, 42, 46, 42, 59, 39, 36, 40, 32, 59, 52, 33, 39, 38, 48, 41
    ]
    >>> f = [2, 0, 1, 2, 3, 2, 5, 6, 7, 5, 3, 2, 1, 1]
    >>> data = data[:len(f)]
    >>> result = stm.stats_mean(
        data=data,
        assumed_mean="auto",
        freq=None,
        steps_compute=False,
        steps_detailed=False,
        show_bg=False,
        param_name="marks"
    )
    >>> result.answer
    >>> stm.prlatex(result.steps)
    """
    return StatsLocation(
        statistic="mean", data=data, freq=freq, assumed_mean=assumed_mean,
        steps_compute=steps_compute, show_bg=show_bg, param_name=param_name,
        decimals=decimals
    )._stats_mean()
    
    
def stats_median(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsLocation:
    return StatsLocation(
        statistic="median", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed,
        show_bg=show_bg, decimals=decimals
    )._stats_median()
    
    
def stats_mode(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsLocation:
    return StatsLocation(
        statistic="mode", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed,
        show_bg=show_bg, decimals=decimals
    )._stats_mode()


def stats_multimode(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsLocation:
    return StatsLocation(
        statistic="multimode", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed,
        show_bg=show_bg, decimals=decimals
    )._stats_multimode()
    
    
def stats_harmonic(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsLocation:
    return StatsLocation(
        statistic="harmonic", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed,
        show_bg=show_bg, decimals=decimals
    )._stats_harmonic()


def stats_geometric(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsLocation:
    return StatsLocation(
        statistic="geometric", data=data, steps_compute=steps_compute, steps_detailed=steps_detailed,
        show_bg=show_bg, decimals=decimals
    )._stats_geometric()