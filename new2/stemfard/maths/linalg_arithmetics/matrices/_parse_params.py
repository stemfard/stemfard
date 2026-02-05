from dataclasses import dataclass
from typing import Any

from sympy import Matrix
from verifyparams import verify_array_or_matrix

from stemfard.core.models import CoreParamsResult
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core._validate import CommonParamsConfig, ComputeApis, ParamNames
from stemfard.core._parse_compute import parse_compute


VALID_MATRIX_ARITHMETICS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "raise",
    "matmul",
)


# -------------------------------------------------
# Parsed parameter container
# -------------------------------------------------

@dataclass(frozen=True, slots=True)
class MatrixArithmeticsParsedParams:
    A: Matrix
    B: Matrix
    config: CommonParamsConfig
    params: CoreParamsResult


# -------------------------------------------------
# Parser
# -------------------------------------------------

def parse_linalg_matrix_arithmetics(
    A: SequenceArrayLike,
    B: SequenceArrayLike,
    compute_apis: tuple[str, ...],
    result_name: str,
    param_names: tuple[str, str],
    steps_compute: bool,
    steps_detailed: bool,
    steps_bg: bool,
    decimals: int,
) -> MatrixArithmeticsParsedParams:
    """
    Parse and validate parameters for matrix arithmetic computations.
    """
    raw_params: dict[str, Any] = {
        "A": A,
        "B": B,
        "compute_apis": compute_apis,
        "param_names": param_names,
        "result_name": result_name,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "decimals": decimals,
    }

    # Normalize compute_apis parameter
    
    compute_apis = parse_compute(
        param=compute_apis,
        valid=VALID_MATRIX_ARITHMETICS
    )
    
    # Common validation

    config = CommonParamsConfig.validate(
        compute_apis=ComputeApis(
            apis=compute_apis,
            valid_items=VALID_MATRIX_ARITHMETICS,
            param_name="compute_apis",
        ),
        param_names=ParamNames(
            names=param_names,
            n=2,
        ),
        result_name=result_name,
        steps_compute=steps_compute,
        steps_detailed=steps_detailed,
        steps_bg=steps_bg,
        decimals=decimals,
    )

    # Matrix validation

    A_mat = verify_array_or_matrix(
        A=A,
        to_matrix=True,
        param_name="A",
    )
    B_mat = verify_array_or_matrix(
        A=B,
        to_matrix=True,
        param_name="B",
    )

    parsed_params = {
        "A": A_mat,
        "B": B_mat,
        "config": config,
    }

    return MatrixArithmeticsParsedParams(
        A=A_mat,
        B=B_mat,
        config=config,
        params=CoreParamsResult(
            raw=raw_params,
            parsed=parsed_params,
        )
    )