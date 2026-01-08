import copy
from typing import Any


class FrequencyTallyWarning(UserWarning):
    """
    Allow users to filter warnings as follows.
    warnings.filterwarnings("ignore", category=FrequencyTallyWarning)
    """
    pass


STATS = ("nrows", "ncols", "n", "min", "max", "range", "mean", "var", "std")

class ResultDict(dict):
    """
    Container object for function results.

    It is a dictionary subclass that supports both key-based and attribute-
    based access to stored values.

    Attributes may be added dynamically and are stored directly in the
    underlying dictionary.
    """
    def to_dict(self) -> dict[str, Any]:
        """
        Return a deep copy of the stored results, ideal for serialization.
        """
        return copy.deepcopy(dict(self))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            if name in STATS:
                raise AttributeError(
                    f"'ResultDict' object has no attribute {name!r}, "
                    f"use obj.stats.{name} instead"
                )
            raise AttributeError(
                f"'ResultDict' object has no attribute {name!r}"
            ) from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            # i.e. values starting with `_` are not shown when 
            # `__repr__` is called
            super().__setattr__(name, value)
        else:
            self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            if name in STATS:
                raise AttributeError(
                    f"'ResultDict' object has no attribute {name!r}, "
                    f"use del obj.stats.{name} instead"
                )
            raise AttributeError(
                f"'ResultDict' object has no attribute {name!r}"
            ) from None

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self))
        return f"{self.__class__.__name__}({keys})"