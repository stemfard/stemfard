from typing import Any, Iterable
import re
from decimal import Decimal, ROUND_HALF_UP

from numpy import ndarray

from stemfard.core._enumerate import ColorCSS


def str_ordinal(n: int) -> str:
    """Convert integer to ordinal string (1st, 2nd, 3rd, etc.)."""
    n = int(n)
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    
    return f"{n}{suffix}"


def str_remove_tzeros(
    value: Iterable[Any],
    remove_decimal: bool = True,
    precision: int = None
) -> str:
    """
    Clean numeric strings in mathematical expressions by:
      - Removing trailing zeros
      - Optionally removing decimal points for integers
      - Supporting rounding to a given precision
      - Preserving scientific notation

    Parameters
    ----------
    expr : Iterable[Any]
        The iterable containing numerical values.
    remove_decimal : bool, optional
        If True, remove the decimal point if the number is an integer.
        Default is True.
    precision : int or None, optional
        If given, round numbers to this many decimal places before
        removing trailing zeros. Default is None.

    Returns
    -------
    str
        The cleaned string with numeric values simplified.

    Examples
    --------
    >>> str_remove_tzeros('10.00 + 20.000')
    '10 + 20'
    >>> str_remove_tzeros('3.000 * 5.0 / 2.0')
    '3 * 5 / 2'
    >>> str_remove_tzeros('2.0 + 3.000', remove_decimal=False)
    '2.0 + 3.0'
    >>> str_remove_tzeros('1.2000e+03 + 2.000E-02')
    '1.2e+03 + 2e-02'
    >>> str_remove_tzeros('3.14159 + 2.71828', precision=2)
    '3.14 + 2.72'
    """
    expr = str(value)

    # Regex to match integers, floats, and scientific notation
    float_pattern = re.compile(
        r'\b\d+(\.\d+)?([eE][+-]?\d+)?\b'
    )

    def _clean(match: re.Match) -> str:
        num_str = match.group(0)

        try:
            # Use Decimal for precision-safe rounding
            num = Decimal(num_str)
            if precision is not None:
                q = Decimal("1e-{0}".format(precision))
                num = num.quantize(q, rounding=ROUND_HALF_UP)
            num_str = format(num, "f")
        except Exception:
            # fallback: keep original string
            return match.group(0)

        # Split base and exponent if scientific notation
        sci_match = re.match(r"([-+]?\d*\.?\d+)([eE][-+]?\d+)?", num_str)
        if sci_match:
            base, exponent = sci_match.groups()
        else:
            base, exponent = num_str, None

        # Remove trailing zeros and optional decimal
        if "." in base:
            base = base.rstrip("0").rstrip(".")
            if not base:
                base = "0"
            elif not remove_decimal and "." not in base:
                base += ".0"

        # Reattach exponent (normalize 'E' -> 'e')
        if exponent:
            exponent = exponent.replace("E", "e")
            return f"{base}{exponent}"
        return base

    # Apply replacement
    result = float_pattern.sub(_clean, expr)

    result = (
        result.replace("+ -", " - ")
              .replace("- +", " - ")
              .replace("- -", " + ")
              .replace("+ +", " + ")
    )

    return result


def str_color_values(
    value: Any,
    color: str | None = None,
    add_box: bool = False,
    remove_italic: bool = False,
    add_tags: bool = False
) -> str:
    """
    Colorizes the given value with the specified color and optionally
    adds a box.
    
    Parameters
    ----------
    value : any
        The value to be colored.
    color : str, optional (default=None)
        The name or HEX code of the color to be applied.
    add_box : bool, default=False
        Whether to add a box around the value.
    
    Returns
    -------
    colored : str
        A colored string with optional box.
    
    Examples
    --------
    >>> str_color_math('Hello', color='red', add_box=False)
    '{\\color{red}{Hello}}'
    >>> str_color_math('World', color='blue', add_box=True)
    '\\fbox{\\color{blue}{World}}'
    >>> str_color_math('Test', add_box=True)
    '\\fbox{\\color{#1E90FF}{Test}}'  # Assuming ColorCSS.BLUE = '#1E90FF'
    """
    if color is None:
        color = ColorCSS.COLORDEFAULT.value
        
    if remove_italic:
        value = f"\\text{{{value}}}"
    
    colored_text = f"{{\\color{{{color}}}{{{value}}}}}"
    
    if add_box:
        colored_text = f"\\boxed{{{colored_text}}}"
        
    if add_tags:
        colored_text = f"\\( {colored_text} \\)"
    return colored_text


def str_color_text(
    value: Any, color: str | None = None, add_box: bool = False
) -> str:
    return str_color_values(
        value=value,
        color=color,
        add_box=add_box,
        remove_italic=True,
        add_tags=True
    )


def str_color_math(
    value: Any,
    color: str | None = None,
    add_box: bool = False
) -> str:
    return str_color_values(
        value=value,
        color=color,
        add_box=add_box,
        remove_italic=False,
        add_tags=False
    )
    
    
def str_eqtn_number(num: str) -> str:
    color = ColorCSS.COLORDEFAULT.value
    return (
        f"{{\\color{{{color}}}{{\\qquad\\qquad \\cdots "
        f"\\qquad\\qquad ({num})}}}}"
    )


def str_omitted(msg: str | None = None) -> str:
    
    steps_mathjax: list[str] = []
    
    steps_mathjax.append("\\( \\qquad\\qquad\\vdots \\)")
    if msg:
        steps_mathjax.append(f"\\( \\qquad\\textit{{{msg} omitted}} \\)")
    else:
        steps_mathjax.append(f"\\( \\qquad\\textit{{some output omitted}} \\)")
    steps_mathjax.append("\\( \\qquad\\qquad\\vdots \\)")
    
    return steps_mathjax


def str_caption(
    label: str = "Table", num: str = ..., caption: str = ...
) -> str:
    caption_str = (
        f"{str_color_text(f'{label} {num}:')} \\( \\textit{{{caption}}} \\)"
    )
    return f"<div>{caption_str}</div>"
    

def str_data_join(
    values: list | tuple | ndarray,
    delim: str = ", ",
    is_quoted: bool = False,
    use_map: bool = True,
    use_and: bool = False
) -> str:
    """
    Join a list of values into a string.
    
    Parameters
    ----------
    values : list, tuple, or ndarray
        Values to join.
    delim : str, default ", "
        Delimiter between values.
    is_quoted : bool, default False
        Whether to quote each value.
    use_map : bool, default True
        Whether to use map() for faster joining.
    use_and : bool, default False
        Whether to use "and" before the last item.
    
    Returns
    -------
    str
        Joined string.
    """
    if len(values) == 0:
        return ""
    
    if len(values) == 1:
        val = str(values[0])
        return f'"{val}"' if is_quoted else val
    
    if isinstance(values, ndarray):
        values = values.tolist() 
    
    if use_and and len(values) > 1:
        values_copy = list(values)
        if len(values_copy) > 2:
            values_copy.insert(-1, "and")
            delim = ", "
        else:
            return f"{values_copy[0]} and {values_copy[1]}"
        values = values_copy
    
    if use_map:
        if is_quoted:
            str_values = map(lambda x: f"'{x}'", values)
        else:
            str_values = map(str, values)
        strng = delim.join(str_values)
    else:
        if is_quoted:
            str_values = [f"'{v}'" for v in values]
        else:
            str_values = [str(v) for v in values]
        strng = delim.join(str_values)
        
    strng = strng.replace(", 'and',", " and").replace(", and,", " and")
    
    return strng

    
def str_data_join_contd(
    values: list | tuple | ndarray,
    max_show: int = 10,
    use_map: bool = True,
    is_quoted: bool = False
) -> str:
    """
    Format excluded values for warning messages.
    
    Parameters
    ----------
    values : ndarray
        Array of excluded values.
    max_show : int, default 10
        Maximum number of values to show before truncating.
    
    Returns
    -------
    str
        Formatted string of values.
    """
    if len(values) == 0:
        return ""
    
    kwargs = {
        "use_map": use_map,
        "is_quoted": is_quoted
    }
    
    if len(values) <= max_show or len(values) <= 10:
        return str_data_join(values, **kwargs)
    else:
        first_values = str_data_join(values[:5], **kwargs)
        last_values = str_data_join(values[-3:], **kwargs)
        
        return f"{first_values}, ..., {last_values}"