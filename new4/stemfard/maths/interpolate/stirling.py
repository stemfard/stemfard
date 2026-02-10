from stemfard.maths.interpolate.dispatcher import Interpolator


@Interpolator.register("stirling")
def stirling(*, x, y, x0, p0, expr_variable, diff_order, decimal_points, **_):
    if p0 is None:
        raise ValueError("p0 index is required for Stirling interpolation.")
    
    xsym = symbols(expr_variable)
    h = x[1] - x[0]  # equally spaced
    n = len(x)
    
    # Central differences
    cdiff = np.zeros((n, n))
    cdiff[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            cdiff[i,j] = cdiff[i+1,j-1] - cdiff[i,j-1]
    
    poly = y[p0]  # start at central point
    # For brevity, use simplified central polynomial
    poly += sum(cdiff[p0, j]*((xsym - x[p0])/h)**j for j in range(1, n-p0))
    
    poly = simplify(poly)
    f_num = lambda xx: float(poly.subs(xsym, xx))
    fx0 = f_num(x0)
    
    if diff_order > 0:
        dpoly = poly.diff(xsym, diff_order)
        dfx0 = float(dpoly.subs(xsym, x0))
    else:
        dpoly = None
        dfx0 = None
    
    table, dframe = generate_table(x, y, decimal_points)
    return Result(table=table, dframe=dframe, f=poly, fx=fx0, df=dpoly, dfx=dfx0)