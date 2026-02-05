from typing import Any, Literal
from IPython.display import display, HTML

from stemfard.core._enumerate import ColorCSS


def prlatex(result: list[str]) -> None:
    result = ["None"] if result is None else result
    result = f'<p>{"</p><p>".join(result)}</p>'
    display(HTML(result))
    
    
def html_border(
    loc: Literal['top', 'bottom', 'left', 'right'] = "top", 
    width: int = 1, 
    type: Literal['solid', 'dashed', 'dotted'] = "dashed", 
    color: str = "#ddd", 
    margin: str = "13px 0px"
) -> str:
    """
    Generate an HTML border.

    Parameters
    ----------
    loc : str, optional (default="top")
        The location of the border (top, bottom, left, right).
    width : int, optional (default=1)
        The width of the border in pixels.
    type : str, optional (default="dashed")
        The type of the border (solid, dashed, dotted).
    color : str, optional (default="#ddd")
        The color of the border in hexadecimal or named color format.
    margin : str, optional (default="13px -5px")
        The margin of the border in CSS format.

    Returns
    -------
    str
        The HTML code for the border.
    """
    border = (
        f'<div style="border-{loc}:{width}px {type} {color}; '
        f'margin:{margin};"></div> <!-- TEXBORDER {type}|{color} -->'
    )
    return border


def html_border_solid(
    loc: Literal['top', 'bottom', 'left', 'right'] = "top", 
    width: int = 1,
    color: str = "#ddd", 
    margin: str = "13px 0px"
) -> str:
    
    return (
        f'<div style="border-{loc}:{width}px solid {color}; '
        f'margin:{margin};"></div> <!-- TEXBORDER solid|{color} -->'
    )
    

def html_border_dashed(
    loc: Literal['top', 'bottom', 'left', 'right'] = "top", 
    width: int = 1,
    color: str = "#ddd", 
    margin: str = "13px 0px"
) -> str:
    
    return (
        f'<div style="border-{loc}:{width}px dashed {color}; '
        f'margin:{margin};"></div> <!-- TEXBORDER dashed|{color} -->'
    )


def html_border_dotted(
    loc: Literal['top', 'bottom', 'left', 'right'] = "top", 
    width: int = 1,
    color: str = "#ddd", 
    margin: str = "13px 0px"
) -> str:
    
    return (
        f'<div style="border-{loc}:{width}px dotted {color}; '
        f'margin:{margin};"></div> <!-- TEXBORDER dotted|{color} -->'
    )
        

def html_style_bg(
    title: str,
    n: str = '',
    bg: str ='#EBF4F7',
    bg_tex: str = "teal",
    bcolor: str = '#ccc',
    lw: int = 1,
    fw: str = '600',
    mt: int = 15,
    mb: int = 15,
    lspace: int = 1
) -> str:
    try:
        bg = bg.value
    except:
        pass
    
    style = (
        f'border-top:{lw}px solid {bcolor}; '
        f'border-bottom:{lw}px solid {bcolor}; '
        f'background: {bg}; '
        f'margin-top:{mt}px; '
        f'margin-bottom: {mb}px; '
        f'padding: 2px 8px; '
        f'font-weight:{fw};'
    )
    # <span>{title}</span> is used for extraction of title by `regex`
    title_styled = (
        f'<span style="letter-spacing:{lspace}px;"><span>{title}</span></span>'
    )
    if n:
        n = f' {n}'
    result = (
        f'<div style="{style}">{title_styled}{n}</div>'
        f'<!-- TEXBGCOLOR {bg_tex} -->'
    )
    
    return result


def html_bg_level1(title: str, n: str = "") -> str:
    bg = ColorCSS.COLORD8F0F8.value
    return html_style_bg(title=title, n=n, bg=bg, lw=2)


def html_bg_level2(title: str, n: str = "") -> str:
    bg = ColorCSS.BGDEFAULT.value
    return html_style_bg(title=title, n=n, bg=bg, lw=1)