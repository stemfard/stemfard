from sympy import symbols, lambdify

from stemfard.interpolation.core.utils import make_plot, make_table
from stemfard.interpolation.dispatcher import InterpDispatcher
from stemfard.interpolation.result import InterpResult

@InterpDispatcher.register("saturation")
def saturation(req):
    x_sym = symbols(req.expr_variable)
    a, b = 1, 1  # simple default parameters
    if req.sat_type == "ax/(x+b)":
        poly = a*x_sym/(x_sym+b)
    else:
        poly = x_sym/(a*x_sym+b)

    fx = poly.subs(x_sym, req.x0) if req.symbolic else float(lambdify(x_sym, poly, "numpy")(req.x0))

    table = make_table(req, poly)
    plot = make_plot(req, poly, x0=req.x0)

    return InterpResult(
        table=table,
        polynomial=poly,
        value_at_x0=fx,
        plot=plot
    )