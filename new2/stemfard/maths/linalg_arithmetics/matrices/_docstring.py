COMMON_DOCSTRING_PARAMS = """
    param_names : tuple[str, str], default=("A", "B")
        Names for the input parameters, used in computation display.
    result_name : str, default="M"
        Name of the result variable in outputs.
    steps_compute : bool, default=True
        Whether to generate step-by-step output. If False, only the computation result is returned.
    steps_detailed : bool, default=True
        Include detailed intermediate steps in the output (if steps_compute is True).
    steps_bg : bool, default=True
        Include background/contextual information in the output (if steps_compute is True).
    decimals : int, default=12
        Number of decimal places to round numerical outputs.
"""

def inject_common_params(doc: str):
    def decorator(func):
        func.__doc__ = func.__doc__.format(common_params=doc)
        return func
    return decorator
