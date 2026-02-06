from collections.abc import Iterable
from typing import Any

from stemcore import str_data_join


class ComputeParameterError(ValueError):
    pass


def verify_param_operations(
    param: Any, valid: list[str], param_name: str = "compute"
) -> list:
    
    try:
        if param is None: return []
        if isinstance(param, str): param = (param, )
        if "all" in param: return valid
        if isinstance(param, Iterable): return param
    except (ValueError, TypeError):
        raise ComputeParameterError(
            f"Expected {param_name!r} to be a list with at least one of: "
            f"{str_data_join(valid)}"
        )