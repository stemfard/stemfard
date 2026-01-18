def html_style_bg(
    title: str,
    n: str = '',
    bg: str ='#EBF4F7',
    bg_tex: str = "teal",
    bcolor: str = '#ccc',
    fw: str = '600',
    mt: int = 15,
    mb: int = 15,
    lspace: int = 1
) -> str:
    style = (
        f'border-top:1px solid {bcolor}; '
        f'border-bottom:1px solid {bcolor}; '
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