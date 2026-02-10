from sympy import sympify, Expr
from numpy import asfarray, array
from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.arraylike import conv_to_arraylike
from stemlab.core.validators.errors import LowerGteUpperError, SympifyError
from stemlab.core.base.strings import str_singular_plural

# Constants (from your code)
VALID_IVP_METHODS = [...]
SYSTEM_OF_EQUATIONS = [...]
START_VALUES_DICT = {...}
TAYLOR_N = [f'taylor{i}' for i in range(1, 10)]


class IVPValidator:
    """
    Comprehensive validation for ODE solver inputs.
    """

    @staticmethod
    def method(method: str) -> str:
        return ValidateArgs.check_member(
            par_name="method",
            valid_items=VALID_IVP_METHODS,
            user_input=method
        )

    @staticmethod
    def odeqtn(odeqtn, method: str, vars: list[str]):
        try:
            fty = sympify(odeqtn)
        except Exception:
            raise SympifyError(par_name="odeqtn", user_input=odeqtn)

        if isinstance(fty, (list, tuple)):
            if method not in SYSTEM_OF_EQUATIONS:
                raise TypeError(
                    f"System of equations only allowed for methods: {', '.join(SYSTEM_OF_EQUATIONS)}"
                )
            f_list = conv_to_arraylike(array_values=odeqtn, to_ndarray=True,
                                       n=len(odeqtn), par_name="odeqtn")
            return f_list, len(f_list)
        else:
            return fty, 1

    @staticmethod
    def exactsol(exactsol, vars):
        if exactsol is None:
            return None
        try:
            ft = sympify(exactsol) if isinstance(exactsol, str) else exactsol
            if not isinstance(ft, Expr):
                ft = sympify(ft)
            return ft
        except Exception:
            raise SympifyError(par_name="exactsol", user_input=exactsol)

    @staticmethod
    def taylor_derivs(method: str, derivs, taylor_order: int = None):
        """
        Validate derivatives for Taylor methods.
        """
        if method not in TAYLOR_N:
            return None

        if taylor_order is None:
            taylor_order = int(method[-1])

        if derivs is None:
            s = str_singular_plural(n=taylor_order)
            raise ValueError(
                f"'{method}' expects {taylor_order} derivative{s} for the specified ODE "
                f"but got None"
            )

        # If derivs provided as string, attempt to convert to sympy
        if isinstance(derivs, str):
            try:
                derivs = sympify(derivs.split(','))
            except Exception:
                raise SympifyError(par_name="derivs", user_input=derivs)

        # Convert to array-like with correct shape
        derivs_arr = conv_to_arraylike(
            derivs,
            n=taylor_order - 1,
            par_name="derivs"
        )
        return derivs_arr

    @staticmethod
    def time_span(time_span):
        t_span = conv_to_arraylike(array_values=time_span, n=2, to_ndarray=True,
                                   par_name="time_span")
        t0, tf = t_span
        t0 = ValidateArgs.check_numeric("t0", to_float=True, user_input=t0)
        tf = ValidateArgs.check_numeric("tf", to_float=True, user_input=tf)
        if t0 >= tf:
            raise LowerGteUpperError(par_name="time_span", lower_par_name="t0",
                                     upper_par_name="tf", user_input=[t0, tf])
        return t0, tf

    @staticmethod
    def stepsize_or_nsteps(stepsize, nsteps, t0, tf, method, adaptive_methods, hmin_hmax=None):
        if method not in adaptive_methods:
            if stepsize is None:
                n = ValidateArgs.check_numeric("nsteps", is_integer=True, user_input=nsteps)
                n = abs(n)
                h = (tf - t0) / n
                from numpy import linspace
                t = linspace(t0, tf, n + 1)
            else:
                h = ValidateArgs.check_numeric("stepsize", to_float=True, user_input=stepsize)
                from stemlab.core.arraylike import arr_abrange
                t = arr_abrange(t0, tf, h)
                n = len(t)
        else:
            # adaptive methods with hmin/hmax
            hmin_hmax = conv_to_arraylike(hmin_hmax, n=2, par_name="hmin_hmax")
            hmin = ValidateArgs.check_numeric("- hmin_hmax[0]", to_float=True, user_input=hmin_hmax[0])
            hmax = ValidateArgs.check_numeric("- hmin_hmax[1]", to_float=True, user_input=hmin_hmax[1])
            if hmin >= hmax:
                raise LowerGteUpperError("hmin_hmax", "hmin_hmax[0]", "hmin_hmax[1]", [hmin, hmax])
            h = hmin
            from stemlab.core.arraylike import arr_abrange
            t = arr_abrange(t0, tf, hmin)
            n = len(t)
        return t, h, n

    @staticmethod
    def y0(y0, number_of_odes=None):
        try:
            y_val = sympify(y0)
        except Exception:
            raise TypeError(f"Expected numeric or array_like y0 but got: {y0}")

        if isinstance(y_val, (list, tuple)):
            y_val = conv_to_arraylike(array_values=y_val, n=len(y_val), to_ndarray=True, par_name="y0")
            n_odes = len(y_val)
        else:
            y_val = ValidateArgs.check_numeric("y0", to_float=True, user_input=y_val)
            n_odes = 1

        if number_of_odes and n_odes != number_of_odes:
            raise ValueError(f"y0 length {n_odes} does not match number of ODEs {number_of_odes}")
        return y_val, n_odes

    @staticmethod
    def tolerance(tol):
        return ValidateArgs.check_numeric("tolerance", boundary="exclusive", to_float=True, user_input=tol)

    @staticmethod
    def maxit(maxit, n):
        return ValidateArgs.check_numeric("maxit", limits=[1, n], is_integer=True, user_input=maxit)

    @staticmethod
    def decimals(dp):
        return ValidateArgs.check_decimals(x=dp)

    @staticmethod
    def boolean(val, default=True):
        return ValidateArgs.check_boolean(user_input=val, default=default)