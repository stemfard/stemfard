from typing import Callable, Dict, Literal, Union, Optional
import numpy as np
from sympy import Expr
from your_validator_module import IVPValidator
from your_constants_module import START_VALUES_DICT, ADAPTIVE_VARIABLE_STEP

# Solver methods
IVPMethod = Literal[
    'taylor1','taylor2','taylor3','taylor4','taylor5','taylor6','taylor7','taylor8','taylor9',
    'feuler','meuler','beuler','rkmidpoint','rkmeuler','ralston2','heun3','nystrom3','rk3','rk4','rk38',
    'rkmersen','rk5','rkbeuler','rktrapezoidal','rk1stage','rk2stage','rk3stage','ab2','ab3','ab4','ab5',
    'am2','am3','am4','eheun','abm2','abm3','abm4','abm5','hamming','msimpson','mmsimpson','rkf45','rkf54',
    'rkv','adamsvariablestep','extrapolation','tnewton'
]

# Start methods for multistep/predictor solvers
StartMethod = Literal['feuler', 'meuler', 'heun3', 'rk4']

class IVPSolver:
    _METHODS: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        """Register an ODE solver method."""
        def decorator(func: Callable):
            cls._METHODS[name] = func
            return func
        return decorator
    
    @staticmethod
    def _resolve_start_values(method, start_values_or_method):
        """
        Determine the actual start values or method for multistep/predictor solvers.

        Returns either a list of start values or a validated start method string.
        """
        if method in START_VALUES_DICT.values():
            if isinstance(start_values_or_method, list):
                # User provided actual start values
                return start_values_or_method
            else:
                # Validate string method
                start_method = IVPValidator.method(start_values_or_method)
                if start_method not in START_VALUES_DICT.values():
                    raise ValueError(
                        f"'{start_method}' is not a valid start method for '{method}'. "
                        f"Choose one of: {list(START_VALUES_DICT.values())}"
                    )
                return start_method
        else:
            # Methods that do NOT require start values can ignore this
            return start_values_or_method

    @classmethod
    def ivps(
        cls,
        *,
        method: IVPMethod = 'rk4',
        odeqtn: Union[str, list, Callable] = '-1.2 * y + 7 * np.exp(-0.3 * t)',
        exactsol: Optional[Union[str, Expr, Callable]] = None,
        derivs: Optional[list[str]] = None,
        vars: list[str] = ['t', 'y'],
        start_values_or_method: Union[list[float], StartMethod] = 'rk4',
        time_span: list[float] = [0, 1],
        y0: float = None,
        stepsize: Optional[Union[float, list[float]]] = None,
        nsteps: int = 10,
        tolerance: float = 1e-6,
        maxit: int = 10,
        show_iters: Optional[int] = None,
        auto_display: bool = True,
        decimal_points: int = 8,
    ):
        """
        Solve an IVP for an ODE.

        Parameters
        ----------
        method : str, optional
            Solver method. Autocompletion supported.
        start_values_or_method : list[float] | str
            Start values for multistep/predictor methods or a method string.
            Autocompletion suggests: 'feuler', 'meuler', 'heun3', 'rk4'.
        ...
        """

        # Validate solver method
        method = IVPValidator.method(method)

        # Resolve start values/method
        start_values_or_method = cls._resolve_start_values(method, start_values_or_method)

        # Continue with normal validation
        fty, number_of_odes = IVPValidator.odeqtn(odeqtn, method, vars)
        ft = IVPValidator.exactsol(exactsol, vars)
        derivs_arr = IVPValidator.taylor_derivs(method, derivs)
        t0, tf = IVPValidator.time_span(time_span)

        t_vals, h, n_actual = IVPValidator.stepsize_or_nsteps(
            stepsize, nsteps, t0, tf, method, ADAPTIVE_VARIABLE_STEP, hmin_hmax=stepsize
        )

        y0_val, n_odes_check = IVPValidator.y0(y0, number_of_odes=number_of_odes)
        tol = IVPValidator.tolerance(tolerance)
        max_iterations = IVPValidator.maxit(maxit, n_actual)
        dp = IVPValidator.decimals(decimal_points)
        auto_disp = IVPValidator.boolean(auto_display)

        if show_iters is None:
            show_iters_val = n_actual
        else:
            show_iters_val = IVPValidator.maxit(show_iters, n_actual)

        # Dispatch to the registered solver
        if method not in cls._METHODS:
            raise ValueError(f"Unknown IVP method: {method}")

        return cls._METHODS[method](
            odeqtn=fty,
            exactsol=ft,
            derivs=derivs_arr,
            vars=vars,
            start_values_or_method=start_values_or_method,
            time_span=[t0, tf],
            y0=y0_val,
            stepsize=h,
            nsteps=n_actual,
            tolerance=tol,
            maxit=max_iterations,
            show_iters=show_iters_val,
            auto_display=auto_disp,
            decimal_points=dp,
        )
