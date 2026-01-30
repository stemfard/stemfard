from dataclasses import dataclass
from typing import Literal
from numpy import int64
from stemcore import arr_to_numeric
from verifyparams import (
    verify_boolean, verify_decimals, verify_int, verify_int_or_float, verify_membership, verify_numeric,
    verify_sta_conf_level, verify_str_identifier
)

from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core.models import CoreParamsResult

_ALLOWED_STATISTICS = [
    # tally
    "tally",
    
    # dispersion
    "stats_harmonic", "stats_geometric", "stats_mean", "stats_median",
    "stats_mode", "multimode",
    
    # location
    "stats_iqd", "stats_iqr", "stats_mad", "stats_max", "stats_mead",
    "stats_min", "stats_range", "stats_std", "stats_var",
    
    # others
    "stats_cv", "stats_dfn", "stats_dfn2", "stats_sem", "stats_tally",
    "stats_mean_ci", "stats_total",
    
    # position
    "stats_p25", "stats_p50", "stats_p75", "stats_percentiles",
    "stats_zscores",
    
    # shape
    "stats_kurt", "stats_skew"
]

ALLOWED_STATISTICS = [stat.replace("stats_", "") for stat in _ALLOWED_STATISTICS]

@dataclass(slots=True, frozen=True)
class ParseDescriptives:
    params: CoreParamsResult
    statistic: str
    data: SequenceArrayLike
    freq: SequenceArrayLike
    assumed_mean: int | float | Literal["auto"] | None = None
    conf_level: float = 0.95
    var_formula: Literal[1, 2, 3] = 1
    ddof: int = 1
    p: int | float | None = None
    quartiles_use_linear: bool = True
    steps_compute: bool = True
    steps_detailed: bool = True
    show_bg: bool = True
    param_name: str = "x",
    decimals: int = 4
    
    def __repr__(self) -> str:
        return (
            "ParseDescriptives(statistic, data, freq, assumed_mean, "
            "var_formula, ddof, p, quartiles_use_linear, steps_compute, "
            "steps_detailed, show_bg, decimals)"
        )
    

def parse_descriptives(
    statistic: str,
    data: SequenceArrayLike,
    freq: SequenceArrayLike,
    assumed_mean: int | float | Literal["auto"] | None = None,
    conf_level: float = 0.95,
    var_formula: Literal[1, 2, 3] = 1,
    ddof: int = 1,
    p: int | float | None = None,
    quartiles_use_linear: bool = True,
    steps_compute: bool = True,
    steps_detailed: bool = True,
    show_bg: bool = True,
    param_name: str = "x",
    decimals: int = 4
) -> ParseDescriptives:
    
    raw_params = {}
    parsed_params = {}
    
    statistic = verify_membership(
        statistic, valid_items=ALLOWED_STATISTICS, param_name="statistic"
    )
    
    raw_params.update({"data": data})
    data = arr_to_numeric(data=data, kind="array", param_name="data")
    parsed_params.update({"data": data})
    
    # ----- mean parameters -----
    
    if statistic == "mean":
        raw_params.update({
            "freq": freq,
            "assumed_mean": assumed_mean,
            "conf_level": conf_level
        })
        if freq is not None:
            freq = arr_to_numeric(
                data=freq,
                kind="array",
                dtype=int64,
                all_integers=True,
                param_name="freq"
            )
        
        if assumed_mean is not None:
            if assumed_mean == "auto":
                assumed_mean = float((data.max() + data.min()) / 2.0)
            else:
                assumed_mean = verify_int_or_float(
                    value=assumed_mean,
                    param_name="assumed_mean"
                )
        
        conf_level = verify_sta_conf_level(value=conf_level)
            
        parsed_params.update({
            "freq": freq,
            "assumed_mean": assumed_mean,
            "conf_level": conf_level
        }) 
        
    # ----- variance / std parameters -----
    
    if statistic in ["var", "std", "sem", "cv", "zscores"]:
        
        raw_params.update({"ddof": ddof, "var_formula": var_formula})
        
        ddof = verify_numeric(
            ddof,
            limits=(0, data.size - ddof),
            is_integer=True,
            param_name="ddof"
        )
        
        var_formula = verify_membership(
            value=var_formula,
            valid_items=[1, 2, 3],
            param_name="var_formula"
        )
        
        parsed_params.update({"ddof": ddof, "var_formula": var_formula})
        
    if statistic == "percentiles":
        raw_params.update({"p": p})
        p = verify_int_or_float(p, param_name="p")
        parsed_params.update({"p": p})
        
    if statistic in ["percentiles", "p25", "p50", "p75"]:
        raw_params.update({"use_linear": quartiles_use_linear})
        quartiles_use_linear = verify_boolean(
            quartiles_use_linear, default=False
        )
        parsed_params.update({"use_linear": quartiles_use_linear})
    
    raw_params.update({
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_name": param_name
    })
    
    if statistic not in ["min", "max"]:
        raw_params.update({"decimals": decimals})
    
    steps_compute = verify_boolean(steps_compute, default=True)
    steps_detailed = verify_boolean(steps_detailed, default=True)
    if not steps_compute:
        steps_detailed = False
    show_bg = verify_boolean(show_bg, default=True)
    param_name = verify_str_identifier(param_name)
    
    if param_name != "x" and "(x)" not in param_name:
        param_name = f"{param_name} (x)".capitalize()
        
    decimals = verify_decimals(decimals)
    
    parsed_params.update({
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "show_bg": show_bg,
        "param_name": param_name
    })
    if statistic not in ["min", "max"]:
        parsed_params.update({"decimals": decimals})
    
    return ParseDescriptives(
        statistic=statistic,
        data=data,
        freq=freq,
        assumed_mean=assumed_mean,
        conf_level=conf_level,
        var_formula=var_formula,
        ddof=ddof,
        p=p,
        quartiles_use_linear=quartiles_use_linear,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        show_bg=show_bg,
        param_name=param_name,
        decimals=decimals,
        params=CoreParamsResult(raw=raw_params, parsed=parsed_params)
    )