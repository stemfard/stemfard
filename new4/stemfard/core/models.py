from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from numpy import array, float64, int64
from numpy.typing import NDArray
from pandas import DataFrame

from stemfard.core._type_aliases import ScalarSequenceArrayLike, SequenceArrayLike


def dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively convert a dataclass to a dict.
    NumPy arrays are converted to lists for JSON serialization.
    """
    if is_dataclass(obj):
        result = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            result[f.name] = dataclass_to_dict(val)
        return result
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    elif hasattr(obj, "dtype") and hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    else:
        return obj
    

def dict_to_dataclass(cls: type[Any], data: dict) -> Any:
    """
    Recursively convert a dictionary to a dataclass instance.
    Handles nested dataclasses and NumPy arrays.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    init_kwargs = {}
    for f in fields(cls):
        fname = f.name
        ftype = f.type

        if fname not in data:
            continue

        value = data[fname]

        # Convert lists back to NumPy arrays if field expects NDArray
        if isinstance(value, list) and "NDArray" in str(ftype):
            if "float" in str(ftype):
                init_kwargs[fname] = array(value, dtype=float64)
            elif "int" in str(ftype):
                init_kwargs[fname] = array(value, dtype=int64)
            else:
                init_kwargs[fname] = array(value)
        # Nested dataclass
        elif is_dataclass(ftype):
            init_kwargs[fname] = dict_to_dataclass(ftype, value)
        else:
            init_kwargs[fname] = value

    return cls(**init_kwargs)


@dataclass(slots=True, frozen=True)
class CoreParamsResult:
    raw: dict[str, Any]
    parsed: dict[str, Any]
    
    def __repr__(self) -> str:
        return "CoreParamsResult(raw, parsed)"


@dataclass(slots=True, frozen=True)
class CoreTablesRawLatexCSV:
    raw: str | None = None
    latex: str | None = None
    rowise: str | None = None
    csv: str | None = None
    
    def __repr__(self) -> str:
        return "CoreTablesRawLatexCSV(raw, latex, rowise, csv)"
    
    
@dataclass(slots=True, frozen=True)
class Syntaxes:
    numpy: str | None = None
    sympy: str | None = None
    scipy: str | None = None
    statsmodels: str | None = None
    scikit_learn: str | None = None
    matlab: str | None = None
    maple: str | None = None
    r: str | None = None
    stata: str | None = None
    ggplot2: str | None = None
    matplotlib: str | None = None

    def __repr__(self) -> str:
        return (
            "Syntaxes(numpy, sympy, scipy, statsmodels, scikit_learn, "
            f"matlab, maple, r, stata, ggplot2, matplotlib)"
        )

@dataclass(slots=True, frozen=True)
class AnswerStepsMathjax:
    question: str | None = None
    answer: Any | None = None
    steps: list[str] | None = None
    syntax: Syntaxes | None = None
    
    def __repr__(self) -> str:            
        return f"AnswerStepsMathjax(question, answer, steps, syntax)"
    
    
@dataclass(slots=True, frozen=True)
class AnswerStepsResult:
    answer: Any
    steps: list[str] | None
    params: dict[str, dict[str, Any]] | None = None
    
    def __repr__(self) -> str:            
        return f"AnswerStepsResult(answer, steps)"


@dataclass(frozen=True)
class StatsDescriptives:
    k: int | None
    n: int
    dfn: int
    total: int | float | None
    min: float
    max: float
    range: float
    percentiles: ScalarSequenceArrayLike
    p25: float
    p50: float
    p75: float
    iqr: float
    iqd: float
    mode: dict[str, int | float]
    median: float
    mean: float
    conf_level: float
    mean_ci: dict[str, float]
    var: float
    std: float
    stderror: float
    sem: float
    cv: float
    skew: float
    kurt: dict[str, float]
    

@dataclass(frozen=True)
class FrequencyTallyResult:
    table: DataFrame
    class_limits: NDArray
    freq: NDArray
    cumfreq: NDArray[int64]
    columns: NDArray
    stats: StatsDescriptives
    params: SequenceArrayLike
    params_parsed: NDArray[float64]

    def __repr__(self) -> str:
        keys = ", ".join(self.__dataclass_fields__.keys())
        return f"{self.__class__.__name__}({keys})"