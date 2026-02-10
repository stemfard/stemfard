# Explicit Euler
@IVPSolver.register('feuler')
def feuler(*, odeqtn, y0, time_span, nsteps, exactsol=None, decimal_points=8, **kwargs):
    t0, tf = time_span
    h = (tf-t0)/nsteps
    t_vals = np.linspace(t0, tf, nsteps+1)
    y_vals = np.zeros(nsteps+1)
    y_vals[0] = y0

    if isinstance(odeqtn, str):
        f = lambda t, y: eval(odeqtn, {"t":t, "y":y, "np":np})
    elif callable(odeqtn):
        f = odeqtn
    else:
        raise ValueError("odeqtn must be string or callable")

    for i in range(nsteps):
        y_vals[i+1] = y_vals[i] + h * f(t_vals[i], y_vals[i])

    y_exact = None
    if exactsol:
        y_exact = np.array([eval(exactsol, {"t":ti,"np":np}) if isinstance(exactsol,str)
                            else exactsol(ti) for ti in t_vals])
    table, dframe = generate_ivp_table(t_vals, y_vals, y_exact, decimal_points)
    return IVPResult(table, dframe, y_vals[-1])
