from numpy import ndarray


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