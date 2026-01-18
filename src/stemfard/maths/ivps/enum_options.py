from enum import Enum


class EnumIVPs(str, Enum):
    """
    Enumeration of supported options with validation rules.
    """
    XXX = "taylor1"
    XXX = "taylor2"
    XXX = "taylor3"
    XXX = "taylor4"
    XXX = "taylor5"
    XXX = "taylor6"
    XXX = "taylor7"
    XXX = "taylor8"
    XXX = "taylor9"
    XXX = "feuler"
    XXX = "meuler"
    XXX = "beuler"
    XXX = "rkmidpoint"
    XXX = "rkmeuler"
    XXX = "ralston2"
    XXX = "heun3"
    XXX = "nystrom3"
    XXX = "rk3"
    XXX = "rk4"
    XXX = "rk38"
    XXX = "rkmersen"
    XXX = "rk5"
    XXX = "rkbeuler"
    XXX = "rktrapezoidal"
    XXX = "rk1stage"
    XXX = "rk2stage"
    XXX = "rk3stage"
    XXX = "ab2"
    XXX = "ab3"
    XXX = "ab4"
    XXX = "ab5"
    XXX = "am2"
    XXX = "am3"
    XXX = "am4",
    XXX = "eheun"
    XXX = "abm2"
    XXX = "abm3"
    XXX = "abm4"
    XXX = "abm5"
    XXX = "hamming"
    XXX = "msimpson"
    XXX = "mmsimpson"
    XXX = "rkf45"
    XXX = "rkf54"
    XXX = "rkv"
    XXX = "avs"
    XXX = "extrapolation"
    XXX = "tnewton"
    
    @classmethod
    def _missing_(cls, value):
        """
        Raise a descriptive error if an invalid option is passed.
        """
        raise ValueError(
            f"'{value}' is not a valid initial value problem method. "
            f"Choose from: {", ".join([e.value for e in cls])}"
        )
        

class EnumStartMethod(str, Enum):
    """
    Enumeration of supported options with validation rules.
    """
    XXX = "taylor1"
    XXX = "taylor2"
    XXX = "taylor3"
    XXX = "taylor4"
    
    @classmethod
    def _missing_(cls, value):
        """
        Raise a descriptive error if an invalid option is passed.
        """
        raise ValueError(
            f"'{value}' is not a valid IVPs start method method. "
            f"Choose from: {", ".join([e.value for e in cls])}"
        )