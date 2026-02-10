from dataclasses import dataclass
from typing import Any

from pandas import DataFrame, Series
from sympy import (
    Float, Integer, Rational, Symbol, flatten
)
from stemcore import is_symexpr

from stemfard.core._latex import tex_to_latex
from stemfard.core._strings import str_color_math
from stemfard.core.constants import StemConstants
from stemfard.core._html import html_bg_level1
from stemfard.core._isinstance import is_system_of_expressions
from stemfard.core.rounding import fround


@dataclass(slots=True, frozen=True)
class AnswerRawCSV:
    answer: Any
    raw: str
    csv: str
    
    @property
    def n(self) -> int | None:
        if hasattr(self.answer, "__len__"):
            return len(self.answer)
        return 0
    
    def __repr__(self):
        return f"AnswerRawCSV(n={self.n}, answer, raw, csv)"


@dataclass(slots=True, frozen=True)
class FullAPIResults:
    question: str
    params_raw: dict[str, Any]
    params_parsed: dict[str, Any]
    answer: AnswerRawCSV
    steps_mathjax: list[str]
    steps_latex: list[str]
    steps_speech: list[str]
    related: dict[str, str]
    suggested: dict[str, str]
    syntax: dict[str, str]
    
    @property
    def n(self) -> int | None:
        if hasattr(self.steps_mathjax, "__len__"):
            return len(self.steps_latex)
        return 0
    
    def __repr__(self):
        return (
            f"FullAPIResults(n={self.n}, question, params_raw, "
            "params_parsed, answer, steps_mathjax, steps_latex, "
            "steps_speech, related, suggested, syntax)"
        )
    

def present_checked_answer(
    result: Any,
    decimals: int = -1,
    add_equal_sign: bool = True,
    n: int | None = 2,
    color: str = "red"
) -> str:
    """
    Final results for step by step procedure.
    """
    result_float = fround(x=result, decimals=decimals)
    add_equal_sign = f" = " if add_equal_sign else ""
    
    if color:
        result_float_latex = tex_to_latex(M=result_float)
        result_float = str_color_math(
            value=f"{add_equal_sign}{result_float_latex}"
        )
    
    if n:
        result_float = result_float + StemConstants.CHECKMARK * n
    
    return result_float


def result_to_csv(
    obj: Any,
    is_system: bool = False,
    index: bool = False,
    header: bool = False    
) -> str:
    
    dtypes = (str, int, float, Integer, Float, Symbol, Rational)
    obj_str = str(obj)
    
    if isinstance(obj, dtypes) or is_symexpr(obj) or "=" in obj_str or ">" in obj_str or "<" in obj_str or "!=" in obj_str:
        obj = DataFrame(data=[str(obj)])
        result = obj.to_csv(header=header, index=index)
        return result
    
    if isinstance(obj, (DataFrame, Series)):
        result = obj.to_csv(header=True, index=index)
    elif is_system and is_system_of_expressions(obj):
        result = ", ".join(map(str, flatten(obj)))
    else:
        if hasattr(obj, "tolist"):
            try:
                obj = DataFrame(data=obj.tolist())
            except ValueError:
                obj = DataFrame(data=[obj.tolist()])
            result = obj.to_csv(header=header, index=index)
        else:
            try:
                obj = DataFrame(data=obj)
            except (TypeError, ValueError, AttributeError):
                try:
                    obj = DataFrame(data=[obj])
                except (TypeError, ValueError, AttributeError):
                    obj = DataFrame(data=["Export to CSV failed"])
            result = obj.to_csv(header=header, index=index)
             
    return result.replace(",", ", ")


def present_full_api_results(
    result: Any, # this are different class objects
    param_operations: tuple[str, ...],
    map_operations: dict[str, str],
    steps_compute: bool,
    params_raw: dict[str, Any],
    params_parsed: dict[str, Any],
) -> FullAPIResults:

    if not steps_compute:
        return result

    # Build step-by-step output
    answer_list: list[Any] = []
    raw_list: list[Any] = []
    csv_list: list[Any] = []
    steps_mathjax: list[str] = []
    
    for idx, op in enumerate(param_operations, 1):
        title = f"{idx}) {map_operations[op]}"
        heading = html_bg_level1(title=title)
        answer_steps = result.steps_latex(op)
        
        answer_list.append(heading)
        answer_list.append(f"\\( {tex_to_latex(answer_steps.answer)} \\)")
        
        raw_list.append(str(answer_steps.answer).replace("'", ""))
        
        if len(param_operations) > 1:
            csv_list.append(title)
        csv_list.append(result_to_csv(answer_steps.answer))
        
        steps_mathjax.append(heading)
        steps_mathjax.extend(answer_steps.steps)
    
    if len(raw_list) == 1:
        raw_list = raw_list[0]
        
    csv_list = "\n".join(csv_list)
    
    steps_latex = steps_mathjax    
    steps_speech = steps_mathjax
    
    return FullAPIResults(
        question=answer_steps.question,
        params_raw=params_raw,
        params_parsed=params_parsed,
        answer=AnswerRawCSV(
            answer=answer_list,
            raw=raw_list,
            csv=csv_list
        ),
        steps_mathjax=steps_latex,
        steps_latex=steps_latex,
        steps_speech=steps_speech,
        related={"name", "url"},
        suggested={"name", "url"},
        syntax=answer_steps.syntax
    )