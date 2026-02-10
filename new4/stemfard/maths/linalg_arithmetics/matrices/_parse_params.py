# NOTE: COMPLETED

from dataclasses import dataclass
from typing import Any

from sympy import Matrix
from verifyparams import verify_array_or_matrix

from stemfard.core.models import CoreParamsResult
from stemfard.core._type_aliases import SequenceArrayLike
from stemfard.core._validate import CommonParamsConfig, ComputeApis, ParamNames
from stemfard.core._verify_param_operations import verify_param_operations
from stemfard.maths.linalg_arithmetics.matrices.utils import (
    MATRIX_OPS_ELEMENTWISE,
    MATRIX_OPERATIONS,
    broadcast_and_expand_matrices,
    check_matrix_shapes
)
from stemfard.core.io.serialize import serialize_matrix

# --------------------------
# Parsed parameter container
# --------------------------

@dataclass(frozen=True, slots=True)
class MatrixArithmeticsParsedParams:
    A: Matrix
    B: Matrix
    broadcast: bool
    config: CommonParamsConfig
    params: CoreParamsResult


# ----------------------------
# Parser: Validate user inputs
# ----------------------------

def parse_user_params(
    A: SequenceArrayLike,
    B: SequenceArrayLike,
    operations: tuple[str, ...],
    broadcast: bool,
    result_name: str,
    param_names: tuple[str, str],
    steps_compute: bool,
    steps_detailed: bool,
    steps_bg: bool,
    decimals: int,
) -> MatrixArithmeticsParsedParams:
    """
    Parse and validate parameters for matrix arithmetic computations.

    Notes
    -----
    Elementwise operations support broadcasting and 1×1 expansion.
    Matrix multiplication (matmul) is always strict (no broadcasting).
    """

    # --------------------------------
    # Store raw input for traceability
    # --------------------------------
    raw_params: dict[str, Any] = {
        "A": A,
        "B": B,
        "broadcast": broadcast,
        "operations": operations,
        "param_names": param_names,
        "result_name": result_name,
        "steps_compute": steps_compute,
        "steps_detailed": steps_detailed,
        "steps_bg": steps_bg,
        "decimals": decimals,
    }

    # -------------------------------
    # Normalize / validate operations
    # -------------------------------
    operations_normalized = verify_param_operations(
        param=operations,
        valid=MATRIX_OPERATIONS,
    )
    
    # -------------------------------
    # Common configuration validation
    # -------------------------------
    config = CommonParamsConfig.validate(
        operations=ComputeApis(
            apis=operations_normalized,
            valid_items=MATRIX_OPERATIONS,
            param_name="operations",
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

    # -----------------
    # Matrix validation
    # -----------------
    A_mat = verify_array_or_matrix(
        A=A,
        to_matrix=True,
        param_name=param_names[0],
    )
    B_mat = verify_array_or_matrix(
        A=B,
        to_matrix=True,
        param_name=param_names[1],
    )

    # ----------------------------------------------------------
    # Separate elementwise and matmul operations from user input
    # ----------------------------------------------------------
    matrix_ops_elementwise = tuple(
        op for op in operations_normalized if op in MATRIX_OPS_ELEMENTWISE
    )
    matmul_ops = tuple(op for op in operations_normalized if op == "matmul")

    # -----------------------------
    # Handle elementwise operations
    # -----------------------------
    A_broadcasted, B_broadcasted = A_mat, B_mat

    if matrix_ops_elementwise:
        if broadcast and A_mat.shape != B_mat.shape:
            # Apply broadcasting once for all elementwise operations.
            # All elementwise ops share the same target shape, so
            # we don't need to broadcast separately for each operation.
            op = matrix_ops_elementwise[0]  # pick first op for broadcasting
            A_broadcasted, B_broadcasted = broadcast_and_expand_matrices(
                A=A_mat, B=B_mat, operation=op, param_names=param_names
            )
        else:
            # Strict validation without broadcasting
            check_matrix_shapes(
                A=A_mat,
                B=B_mat,
                operations=matrix_ops_elementwise,
                param_names=param_names
            )

    # ----------------------------------------
    # Handle matmul operations (always strict)
    # ----------------------------------------
    if matmul_ops:
        if broadcast:
            raise TypeError(
                "Broadcasting is not supported for matrix multiplication "
                "(matmul)"
            )
        check_matrix_shapes(A_mat, B_mat, matmul_ops, param_names)

    # ------------------------
    # Parsed params dictionary
    # ------------------------
    parsed_params: dict[str, Any] = {
        "A": serialize_matrix(A_broadcasted),
        "B": serialize_matrix(B_broadcasted),
        "broadcast": broadcast,
        "operations": config.operations,
        "param_names": config.param_names,
        "result_name": config.result_name,
        "steps_compute": config.steps_compute,
        "steps_detailed": config.steps_detailed,
        "steps_bg": config.steps_bg,
        "decimals": config.decimals
    }

    # -----------------------
    # Return frozen dataclass
    # -----------------------
    return MatrixArithmeticsParsedParams(
        A=A_broadcasted,
        B=B_broadcasted,
        broadcast=broadcast,
        config=config,
        params=CoreParamsResult(
            raw=raw_params,
            parsed=parsed_params,
        )
    )