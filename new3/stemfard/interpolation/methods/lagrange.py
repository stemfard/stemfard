from sympy import symbols, interpolate, simplify, lambdify

from stemfard.interpolation.core.utils import make_plot, make_table
from stemfard.interpolation.dispatcher import InterpDispatcher
from stemfard.interpolation.result import InterpResult

@InterpDispatcher.register("lagrange")
def lagrange(req):
    x_sym = symbols("x")
    points = list(zip(req.x, req.y))
    poly = simplify(interpolate(points, x_sym))

    fx = poly.subs(x_sym, req.x0) if req.symbolic else float(lambdify(x_sym, poly, "numpy")(req.x0))

    table = make_table(req, poly)
    plot = make_plot(req, poly, x0=req.x0)

    return InterpResult(
        table=table,
        polynomial=poly,
        value_at_x0=fx,
        plot=plot
    )
