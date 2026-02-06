@IVPSolver.register("rk4")
def rk4(*, odeqtn, y0, time_span, nsteps, exactsol=None, decimal_points=8, **kwargs):
    t0, tf = time_span
    h = (tf - t0)/nsteps
    t_vals = np.linspace(t0, tf, nsteps+1)
    y_vals = np.zeros(nsteps+1)
    y_vals[0] = y0

    # Convert odeqtn to callable
    if isinstance(odeqtn, str):
        f = lambda t, y: eval(odeqtn, {"t": t, "y": y, "np": np})
    elif callable(odeqtn):
        f = odeqtn
    else:
        raise ValueError("odeqtn must be a string or callable")

    # RK4 loop
    for i in range(nsteps):
        t = t_vals[i]
        y = y_vals[i]
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y_vals[i+1] = y + (k1 + 2*k2 + 2*k3 + k4)/6

    # Exact solution if provided
    if exactsol is not None:
        if isinstance(exactsol, str):
            y_exact = np.array([eval(exactsol, {"t": t, "np": np}) for t in t_vals])
        elif callable(exactsol):
            y_exact = np.array([exactsol(t) for t in t_vals])
        else:
            y_exact = np.array([float(exactsol.subs({'t': ti})) for ti in t_vals])
    else:
        y_exact = None

    table, dframe = generate_ivp_table(t_vals, y_vals, y_exact, decimal_points)
    return IVPResult(table=table, dframe=dframe, answer=float(y_vals[-1]))
