from typing import Callable, Dict
import numpy as np
from stemlab.core.display import Result
from stemlab.core.arraylike import conv_to_arraylike
from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.validators.errors import RequiredError
from stemlab.core.base.constraints import max_rows

# Dispatcher
class Interpolator:
    _METHODS: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Register an interpolation method."""
        def decorator(func: Callable):
            cls._METHODS[name] = func
            return func
        return decorator

    @classmethod
    def interp(
        cls,
        x,
        y,
        x0,
        *,
        yprime=None,
        p0=None,
        method: str = "newton-divided",
        expr_variable: str = "x",
        poly_order: int = 1,
        qs_constraint=0,
        end_points=None,
        exp_type: str = "b*exp(ax)",
        sat_type: str = "ax/(x+b)",
        plot_x0: bool = False,
        diff_order: int = 0,
        plot_deriv: bool = False,
        truncate_terms: float = 1e-16,
        auto_display: bool = True,
        decimal_points: int = 12,
    ) -> Result:

        # check method exists
        if method not in cls._METHODS:
            raise ValueError(f"Unknown interpolation method: {method}")

        # normalize and validate arrays
        x = conv_to_arraylike(x, to_ndarray=True, par_name='x')
        y = conv_to_arraylike(y, to_ndarray=True, par_name='y')

        if len(x) > max_rows():
            raise ValueError(f"len(x) = {len(x)} exceeds maximum allowed rows {max_rows()}")
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        # numeric validations
        x0 = ValidateArgs.check_numeric('x0', x0, limits=[min(x), max(x)])
        diff_order = ValidateArgs.check_numeric('diff_order', diff_order, limits=[0, 9], is_integer=True, is_positive=True)
        truncate_terms = ValidateArgs.check_numeric('truncate_terms', truncate_terms, limits=[0, .1])
        poly_order = ValidateArgs.check_numeric('poly_order', poly_order, limits=[1, 9]) if method == "polynomial" else poly_order

        # boolean validations
        plot_x0 = ValidateArgs.check_boolean(plot_x0, default=False)
        plot_deriv = ValidateArgs.check_boolean(plot_deriv, default=False)
        auto_display = ValidateArgs.check_boolean(auto_display, default=False)

        # string/member validations
        expr_variable = ValidateArgs.check_string('expr_variable', expr_variable, default='x')
        if method == "exponential":
            exp_type = ValidateArgs.check_member('exp_type', exp_type, ['b*exp(ax)', 'b*10^ax', 'ab^x'])
        if method == "saturation":
            sat_type = ValidateArgs.check_member('sat_type', sat_type, ['x/(ax+b)', 'ax/(x+b)'])

        # method-specific requirements
        if method == "hermite" and yprime is None:
            raise RequiredError(par_name='yprime', required_when='method=hermite')

        if yprime is not None:
            yprime = conv_to_arraylike(yprime, to_ndarray=True, par_name='yprime')
            if len(yprime) != len(x):
                raise ValueError("yprime must have the same length as x")

        if method == "clamped-cubic-splines" and end_points is None:
            raise RequiredError(par_name='end_points', required_when='method=clamped-cubic-splines')

        if end_points is not None:
            end_points = conv_to_arraylike(end_points, to_ndarray=True, par_name='end_points', n=2)

        # create a context dict
        context = {
            "x": x,
            "y": y,
            "x0": x0,
            "yprime": yprime,
            "p0": p0,
            "expr_variable": expr_variable,
            "poly_order": poly_order,
            "qs_constraint": qs_constraint,
            "end_points": end_points,
            "exp_type": exp_type,
            "sat_type": sat_type,
            "plot_x0": plot_x0,
            "diff_order": diff_order,
            "plot_deriv": plot_deriv,
            "truncate_terms": truncate_terms,
            "auto_display": auto_display,
            "decimal_points": decimal_points
        }

        # dispatch to the registered method
        return cls._METHODS[method](**context)