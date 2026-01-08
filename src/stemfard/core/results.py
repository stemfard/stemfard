import copy
from typing import Any


class ResultDict(dict):
    """
    Container object for function results.

    It is a dictionary subclass that supports both key-based and attribute-
    based access to stored values.

    Attributes may be added dynamically and are stored directly in the
    underlying dictionary.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy of the stored results."""
        return copy.deepcopy(dict(self))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self))
        return f"{self.__class__.__name__}({keys})"