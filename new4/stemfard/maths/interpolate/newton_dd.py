from sympy import symbols, simplify
import numpy as np
from stemlab.core.display import Result

@Interpolator.register("newton-divided")
def newton_divided(*, x, y, x0, expr_variable, diff_order, decimal_points, **_):
    n = len(x)
    xsym = symbols(expr_variable)
    
    # Divided differences table
    dd = np.zeros((n, n))
    dd[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            dd[i,j] = (dd[i+1,j-1] - dd[i,j-1]) / (x[i+j] - x[i])
    
    # Newton polynomial
    poly = dd[0,0]
    term = 1
    for j in range(1, n):
        term *= (xsym - x[j-1])
        poly += dd[0,j] * term
    
    poly = simplify(poly)
    fx0 = float(poly.subs(xsym, x0))
    
    if diff_order > 0:
        dpoly = poly.diff(xsym, diff_order)
        dfx0 = float(dpoly.subs(xsym, x0))
    else:
        dpoly = None
        dfx0 = None
    
    # Optional: generate a table if you have a helper
    table, dframe = None, None
    
    return Result(table=table, dframe=dframe, f=poly, fx=fx0, df=dpoly, dfx=dfx0)
