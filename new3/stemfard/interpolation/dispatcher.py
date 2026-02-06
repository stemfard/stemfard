METHOD_MAP = {}

class InterpDispatcher:
    @staticmethod
    def register(name: str):
        def decorator(func):
            METHOD_MAP[name.lower()] = func
            return func
        return decorator

    @staticmethod
    def dispatch(req):
        func = METHOD_MAP.get(req.method.lower())
        if func is None:
            raise ValueError(f"No implementation for method '{req.method}'")
        return func(req)