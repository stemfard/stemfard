@Interpolator.register("gauss-forward")
def gauss_forward(*, x, y, x0, p0, expr_variable, diff_order, decimal_points, **_):
    if p0 is None:
        raise ValueError("p0 index is required for Gauss Forward.")
    
    xsym = symbols(expr_variable)
    h = x[1] - x[0]  # assumes equally spaced
    n = len(x)
    # Forward differences
    fwd_diff = np.zeros((n, n))
    fwd_diff[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            fwd_diff[i,j] = fwd_diff[i+1,j-1] - fwd_diff[i,j-1]

    # Build polynomial (simplified version)
    poly = fwd_diff[p0,0]
    term = 1
    u = (xsym - x[p0]) / h
    for j in range(1, n - p0):
        term *= (u - j + 1)/j
        poly += fwd_diff[p0, j] * term
    
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
