METHODS_DICT = {
    'taylor1': 'taylor1',
    'taylor2': 'taylor2',
    'taylor3': 'taylor3',
    'taylor4': 'taylor4',
    'taylor5': 'taylor5',
    'taylor6': 'taylor6',
    'taylor7': 'taylor7',
    'taylor8': 'taylor8',
    'taylor9': 'taylor9',
    'forward-euler': 'feuler',
    'modified-euler': 'meuler',
    'backward-euler': 'beuler',
    'midpoint-runge-kutta': 'rkmidpoint',
    'modified-euler-runge-kutta': 'rkmeuler',
    'second-order-ralston': 'ralston2',
    'third-order-heun': 'heun3',
    'third-order-nystrom': 'nystrom3',
    'third-order-runge-kutta': 'rk3',
    'fourth-order-runge-kutta': 'rk4',
    'fourth-order-runge-kutta - 38': 'rk38',
    'fourth-order-runge-kutta-mersen': 'rkmersen',
    'fiexactsolh-order-runge-kutta': 'rk5',
    'backward-euler': 'rkbeuler',
    'trapezoidal': 'rktrapezoidal',
    'one-stage-gauss-legendre': 'rk1stage',
    'two-stage-gauss-legendre': 'rk2stage',
    'three-stage-gauss-legendre': 'rk3stage',
    'adams-bashforth - 2-step': 'ab2',
    'adams-bashforth - 3-step': 'ab3',
    'adams-bashforth-4-step': 'ab4',
    'adams-bashforth-5-step': 'ab5',
    'adams-moulton - 2-step': 'am2',
    'adams-moulton - 3-step': 'am3',
    'adams-moulton - 4-step': 'am4',
    'euler-heun': 'eheun',
    'adams-bashforth-moulton - 2-step': 'abm2',
    'adams-bashforth-moulton - 3-step': 'abm3',
    'adams-bashforth-moulton - 4-step': 'abm4',
    'adams-bashforth-moulton - 5-step': 'abm5',
    'msimpson': 'ms',
    'modified-msimpson': 'mms',
    'hamming': 'hamming',
    'runge-kutta-fehlberg-45': 'rkf45',
    'runge-kutta-fehlberg-54': 'rkf54',
    'runge-kutta-verner': 'rkv',
    'adams-variable-step-size': 'avs',
    'extrapolation': 'extrapolation',
    'trapezoidal-with-newton-approximation': 'tnewton'
}

START_VALUES_DICT = {
    'explicit-euler': 'feuler',
    'modified-euler': 'meuler',
    'third-order-heun': 'heun3',
    'fourth-order-runge-kutta':'rk4'
}

TAYLOR_N = [f'taylor{order + 1}' for order in range(9)]
EULER_METHODS = ['feuler', 'meuler', 'beuler']
EXPLICIT_RK = [
    'rkmidpoint', 
    'rkmeuler', 
    'ralston2', 
    'heun3', 
    'nystrom3', 
    'rk3', 
    'rk4', 
    'rk38', 
    'rkmersen', 
    'rk5'
]
IMPLICIT_RK = ['rkbeuler', 'rktrapezoidal', 'rk1stage', 'rk2stage', 'rk3stage']
EXPLICIT_MULTISTEP = ['ab2', 'ab3', 'ab4', 'ab5']
IMPLICIT_MULTISTEP = ['am2', 'am3', 'am4']
PREDICTOR_CORRECTOR = [
    'eheun', 'abm2','abm3','abm4', 'abm5', 'hamming', 'msimpson', 'mmsimpson'
]
ADAPTIVE_VARIABLE_STEP = [
    'rkf45', 'rkf54', 'rkv', 'avs', 'extrapolation', 'tnewton'
]

VALID_IVP_METHODS = (
    TAYLOR_N
    + EULER_METHODS 
    + EXPLICIT_RK 
    + IMPLICIT_RK 
    + EXPLICIT_MULTISTEP 
    + IMPLICIT_MULTISTEP 
    + PREDICTOR_CORRECTOR 
    + ADAPTIVE_VARIABLE_STEP
)
START_VALUE_METHODS = (
    EXPLICIT_MULTISTEP + IMPLICIT_MULTISTEP + PREDICTOR_CORRECTOR
)

# system of equations is only allowed for the following method
SYSTEM_OF_EQUATIONS = (
    TAYLOR_N + EULER_METHODS + EXPLICIT_MULTISTEP + PREDICTOR_CORRECTOR
)