from stemlab.core.validators.validate import ValidateArgs
from stemlab.core.validators.errors import RequiredError, MaxNError
from stemlab.core.base.constraints import max_rows
from stemlab.core.arraylike import conv_to_arraylike
from numpy import full, nan

INTERPOLATION_METHODS = [
    'straight-line', 'lagrange', 'hermite',
    'newton-backward', 'newton-forward',
    'gauss-backward', 'gauss-forward',
    'newton-divided', 'neville', 'stirling', 'bessel', 'laplace-everett',
    'linear-splines', 'quadratic-splines', 'natural-cubic-splines', 
    'clamped-cubic-splines', 'not-a-knot-splines', 
    'linear-regression', 'polynomial', 
    'exponential', 'power', 'saturation', 'reciprocal'
]

class InterpolationValidator:

    @staticmethod
    def array(arr, name, ncols=None):
        arr_conv = conv_to_arraylike(array_values=arr, to_ndarray=True, par_name=name)
        if len(arr_conv) > max_rows():
            raise MaxNError(par_name=f'len({name})', user_input=len(arr_conv), maxn=max_rows())
        if ncols is not None and arr_conv.shape[1] != ncols:
            raise ValueError(f'{name} should have {ncols} columns, got {arr_conv.shape[1]}')
        return arr_conv

    @staticmethod
    def numeric(value, par_name, limits=None, is_integer=False, is_positive=False, boundary='inclusive'):
        return ValidateArgs.check_numeric(
            user_input=value,
            par_name=par_name,
            limits=limits,
            is_integer=is_integer,
            is_positive=is_positive,
            boundary=boundary
        )

    @staticmethod
    def member(value, par_name, valid_items):
        return ValidateArgs.check_member(par_name=par_name, user_input=value, valid_items=valid_items)

    @staticmethod
    def boolean(value, default=False):
        return ValidateArgs.check_boolean(user_input=value, default=default)

    @classmethod
    def method_specific(cls, method, yprime=None, poly_order=None,
                        qs_constraint=None, end_points=None, exp_type=None, sat_type=None, x_len=None):
        
        result = {}
        
        if method == 'hermite':
            if yprime is None:
                raise RequiredError(par_name='yprime', required_when='method=hermite')
            result['yprime'] = cls.array(yprime, 'yprime')

        if method == 'polynomial':
            result['poly_order'] = cls.numeric(poly_order, 'poly_order', limits=[1, 9], boundary='inclusive')

        if method == 'quadratic-splines':
            result['qs_constraint'] = cls.numeric(qs_constraint, 'qs_constraint')
            result['qs_constraint'] = f'a0-{result["qs_constraint"]}'

        if method == 'clamped-cubic-splines':
            if end_points is None:
                raise RequiredError(par_name='end_points', required_when='method=clamped-cubic-splines')
            result['end_points'] = cls.array(end_points, 'end_points', ncols=2)

        if method == 'exponential':
            result['exp_type'] = cls.member(exp_type, 'exp_type', ['b*exp(ax)', 'b*10^ax', 'ab^x'])

        if method == 'saturation':
            result['sat_type'] = cls.member(sat_type, 'sat_type', ['x/(ax+b)', 'ax/(x+b)'])

        return result
