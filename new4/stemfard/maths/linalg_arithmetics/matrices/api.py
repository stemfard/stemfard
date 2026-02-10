from stemfard.core._type_aliases import (
    ScalarSequenceArrayLike,
    SequenceArrayLike
)
from stemfard.maths.linalg_arithmetics.matrices.steps import MatrixArithmetics
from stemfard.maths.linalg_arithmetics.matrices._parse_params import parse_user_params
from stemfard.maths.linalg_arithmetics.matrices._docstring import (
    COMMON_DOCSTRING_PARAMS,
    inject_common_params
)
from stemfard.maths.linalg_arithmetics.matrices.utils import (
    MAP_OPERATION,
    MatrixOperations,
)
from stemfard.core.io._io import FullAPIResults, present_full_api_results


@inject_common_params(COMMON_DOCSTRING_PARAMS)
def linalg_matrix_arithmetics(
    A: SequenceArrayLike,
    B: ScalarSequenceArrayLike,
    broadcast: bool = False,
    operations: str | tuple[MatrixOperations, ...] = "add",
    param_names: tuple[str, str] = ("A", "B"),
    result_name: str = "ans",
    steps_compute: bool = True,
    steps_detailed: bool = True,
    steps_bg: bool = True,
    decimals: int = 12
) -> FullAPIResults:
    """
    Perform matrix arithmetic computations.

    Parameters
    ----------
    A : SequenceArrayLike
        First matrix or array-like input.
    B : ScalarSequenceArrayLike
        Second matrix or array-like input.
    broadcast : bool
        If True, allows broadcasting / 1x1 expansion.
    {common_params}
    """
    # Validate and parse user input
    parsed = parse_user_params(
        A=A,
        B=B,
        broadcast=broadcast,
        operations=operations,
        param_names=param_names,
        result_name=result_name,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        steps_bg=steps_bg,
        decimals=decimals
    )

    # Instantiate computation object
    result = MatrixArithmetics(
        A=parsed.A,
        B=parsed.B,
        broadcast=parsed.broadcast,
        operations=parsed.config.operations,
        param_names=parsed.config.param_names,
        result_name=parsed.config.result_name,
        steps_compute=parsed.config.steps_compute,
        steps_detailed=parsed.config.steps_detailed,
        steps_bg=parsed.config.steps_bg,
        decimals=parsed.config.decimals
    )
    
    # this returns a `FullAPIResults` object
    return present_full_api_results(
        result=result,
        steps_compute=parsed.config.steps_compute,
        param_operations=parsed.config.operations,
        map_operations=MAP_OPERATION,
        params_raw=parsed.params.raw,
        params_parsed=parsed.params.parsed
    )