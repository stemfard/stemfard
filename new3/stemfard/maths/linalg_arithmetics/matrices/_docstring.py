# NOTE: COMPLETED

COMMON_DOCSTRING_PARAMS = """
    param_names : str | tuple[str, ...]
        Name(s) for the key input parameters, used in computation 
        display.
    result_name : str, default="ans"
        Name of the result variable in outputs.
    steps_compute : bool, default=True
        Whether to generate step-by-step output. If False, only the
        computation result is returned.
    steps_detailed : bool, default=True
        Include detailed intermediate steps in the output
        (if steps_compute is True).
    steps_bg : bool, default=True
        Include background/contextual information in the output
        (if steps_compute is True).
    decimals : int, default=12
        Number of decimal places to round numerical outputs.
    """.strip()


def inject_common_params(doc: str):
    """
    Decorator to inject a reusable docstring snippet into a function's docstring.

    Parameters
    ----------
    doc : str
        Docstring fragment to inject where {common_params} appears.

    Returns
    -------
    decorator : Callable
        Function decorator.
    """
    def decorator(func):
        if func.__doc__ is None:
            func.__doc__ = ""
        func.__doc__ = func.__doc__.format(common_params=doc.strip())
        return func
    return decorator