"""
========
stemfard
========

stemfard is a Python library for performing mathematical computations.
It aims to become a first choice library for trainers and students
in Science, Technology, Engineering and Mathematics.

`Pedagogical and Reproducible STEM Computation Enabled by Step-Wise Execution APIs
-A Framework for Transparent and Inspectable Reasoning`

How to use the documentation
----------------------------
Documentation is available in two forms: 
    - Docstrings provided with the code
    - stemfard homepage <https://stemfard.stemfard.org>`

The docstring examples assume that `stemfard` has been imported as ``stm``::

>>> import stemfard as stm
"""

import sys

if sys.version_info < (3, 6):
    raise ImportError(
        'stemfard requires installation of Python version 3.6 or above.'
    )
del sys

__version__ = '0.0.2a1'
__author__ = 'STEM Research'
__credits__ = 'STEM Research'
__email__ = "stemfard@stemfard.org"


from .core import *
from .stats.descriptives import *
from .stats.regression_linear import *
from .maths.linalg_arithmetics import *
from .maths.linalg_iterative import *

#========================================================================#
# STEM RESEARCH :: AI . APIs . Innovate :: https://stemfard.stemfard.org #
#========================================================================#