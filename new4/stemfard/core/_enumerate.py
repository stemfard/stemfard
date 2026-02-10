from enum import Enum

class ColorCSS(str, Enum):
    """
    Enumeration of supported options with validation rules.
    """
    COLORDEFAULT = "#01B3D1"
    BGDEFAULT = "#EBF4F7"
    BLUE = "blue"
    GRAY = "gray"
    ORANGE = "orange"
    COLOR01B3D1 = "#01B3D1"
    COLOR87CEFA = "#87CEFA"
    COLOR76D3F5 = "#76D3F5"
    COLORD8F0F8 = "#D8F0F8"
    COLOREBF4F7 = "#F0F2F3"
    COLORF5F5F5 = "#F5F5F5"