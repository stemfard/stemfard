from stemfard.core.models import AnswerStepsResult
from stemfard.stats.descriptives._base import BaseDescriptives

from stemfard.core._type_aliases import SequenceArrayLike


class StatsShape(BaseDescriptives):

    def _stats_skew(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.skew,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    

    def _stats_kurt(self) -> AnswerStepsResult:
        
        steps_mathjax: list[str] | None = []
        
        if self.steps_compute:
            steps_mathjax.append("To update")

        return AnswerStepsResult(
            answer=self.kurt,
            steps=steps_mathjax if steps_mathjax else None,
            params=self.params
        )
    
    
def stats_skew(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsShape:
    result = StatsShape(
        statistic="xxx", data=data, steps_compute=steps_compute,
        steps_detailed=steps_detailed, show_bg=show_bg, decimals=decimals
    )
    return result._stats_skew()
    
    
def stats_kurt(
    data: SequenceArrayLike,
    *,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    decimals: int = 4
) -> StatsShape:
    result = StatsShape(
        statistic="xxx", data=data, steps_compute=steps_compute,
        steps_detailed=steps_detailed, show_bg=show_bg, decimals=decimals
    )
    return result._stats_kurt()