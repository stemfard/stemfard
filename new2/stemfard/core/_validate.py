from __future__ import annotations

from dataclasses import dataclass

from verifyparams import (
    verify_boolean,
    verify_decimals,
    verify_membership_iterable,
    verify_param_names,
)
from stemfard.core._strings import str_var_name


# -------------------------------------------------
# Helper input-spec dataclasses (validation intent)
# -------------------------------------------------

@dataclass(frozen=True, slots=True)
class ComputeApis:
    """
    Specification for validating compute APIs.
    """
    apis: tuple[str, ...]
    valid_items: tuple[str, ...]
    param_name: str


@dataclass(frozen=True, slots=True)
class ParamNames:
    """
    Specification for validating parameter names.
    """
    names: tuple[str, ...]
    n: int


# -------------------------------------------------
# Final validated configuration object
# -------------------------------------------------

@dataclass(frozen=True, slots=True)
class CommonParamsConfig:
    """
    Immutable, validated computation configuration.

    This object is safe to pass across multiple layers of the library.
    """
    compute_apis: tuple[str, ...]
    param_names: tuple[str, ...] | None
    result_name: str
    steps_compute: bool
    steps_detailed: bool
    steps_bg: bool
    decimals: int

    @staticmethod
    def validate(
        *,
        compute_apis: ComputeApis,
        param_names: ParamNames | None,
        result_name: str,
        steps_compute: bool,
        steps_detailed: bool,
        steps_bg: bool,
        decimals: int,
    ) -> CommonParamsConfig:
        """
        Validate common computation parameters and return a configuration
        object.

        Parameters
        ----------
        compute_apis : ComputeApis
            Operations to validate, along with allowed values.
        param_names : ParamNames | None
            Optional parameter names and expected count.
        result_name : str
            Name of the result variable.
        steps_compute : bool
            Whether to record computation steps.
        steps_detailed : bool
            Whether to include detailed steps.
        steps_bg : bool
            Whether to include background information.
        decimals : int
            Number of decimal places for numerical output.

        Returns
        -------
        CommonParamsConfig
            Fully validated, immutable configuration.
        """    
        validated_apis = verify_membership_iterable(
            value=compute_apis.apis,
            valid_items=compute_apis.valid_items,
            param_name=compute_apis.param_name,
        )

        validated_param_names: tuple[str, ...] | None = None
        if param_names is not None:
            validated_param_names = verify_param_names(
                value=param_names.names,
                n=param_names.n
            )

        return CommonParamsConfig(
            compute_apis=validated_apis,
            param_names=validated_param_names,
            result_name=str_var_name(result_name),
            steps_compute=verify_boolean(steps_compute, default=True),
            steps_detailed=verify_boolean(steps_detailed, default=True),
            steps_bg=verify_boolean(steps_bg, default=True),
            decimals=verify_decimals(decimals),
        )