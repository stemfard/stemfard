from sympy import symbols, Poly, simplify, lambdify

from stemfard.interpolation.core.utils import make_plot, make_table
from stemfard.interpolation.dispatcher import InterpDispatcher
from stemfard.interpolation.result import InterpResult

@InterpDispatcher.register("polynomial")
def polynomial(req):
    x_sym = symbols("x")
    poly = Poly(req.y, x_sym).as_expr()  # simple polynomial fit for demo
    poly = simplify(poly)

    fx = poly.subs(x_sym, req.x0) if req.symbolic else float(lambdify(x_sym, poly, "numpy")(req.x0))

    derivative = poly.diff(x_sym) if req.diff_order > 0 else None
    dfx = derivative.subs(x_sym, req.x0) if derivative else None

    table = make_table(req, poly, derivative)
    plot = make_plot(req, poly, derivative, x0=req.x0, plot_deriv=(req.diff_order>0))

    return InterpResult(
        table=table,
        polynomial=poly,
        value_at_x0=fx,
        derivative=derivative,
        derivative_at_x0=dfx,
        plot=plot
    )
