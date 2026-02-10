from stemfard.core.models import AnswerStepsResult
from stemfard.core._type_aliases import SequenceArrayLike


def test_statistics_t(
    data: SequenceArrayLike,
    decimals: int = 4
) -> AnswerStepsResult:
    
    steps_mathjax = []
    
    answer = 999
    
    return AnswerStepsResult(answer=answer, steps=steps_mathjax)


def test_statistics_chisquare(
    data: SequenceArrayLike,
    decimals: int = 4
) -> AnswerStepsResult:
    
    steps_mathjax = []
    
    answer = 999
    
    return AnswerStepsResult(answer=answer, steps=steps_mathjax)


def test_statistics_f(
    data: SequenceArrayLike,
    decimals: int = 4
) -> AnswerStepsResult:
    
    steps_mathjax = []
    
    answer = 999
    
    return AnswerStepsResult(answer=answer, steps=steps_mathjax)