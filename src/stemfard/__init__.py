'''
=======
stemfard
=======

stemfard is a Python library for performing mathematical computations.
It aims to become a first choice library for trainers and students
in Science, Technology, Engineering and Mathematics.

How to use the documentation
----------------------------
Documentation is available in two forms: 
    - Docstrings provided with the code
    - stemfard homepage <https://stemfard.org>`

The docstring examples assume that `stemfard` has been imported as ``stm``::

>>> import stemfard as stm

Code snippets are indicated by three greater-than signs::

>>> x = 42
>>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

>>> help(stm.arr_abrange)
... # doctest: +SKIP

To search for documents containing a keyword, do::

>>> np.lookfor('keyword')
... # doctest: +SKIP

Viewing documentation using IPython
-----------------------------------

Start IPython and import `stemfard` 

>>> import stemfard as stm

- To see which functions are available in `stemfard`, enter ``stm.<TAB>`` 
(where ``<TAB>`` refers to the TAB key), or use
``stm.*rk*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to 
filter functions that contain the characters *rk*.

- To view the docstring for a specified function, use
``np.arr_abrange?<ENTER>`` (to view the docstring) and 
``np.cos??<ENTER>`` (to view the source code).

'''

import sys

if sys.version_info < (3, 6):
    raise ImportError('stemfard requires installation of Python version 3.6 or above.')
del sys

__version__ = '0.0.2a1'
__author__ = 'STEM Research'
__credits__ = 'STEM Research'
__email__ = "stemfard@stemfard.org"

from .stats import sta_eda_grouped, sta_eda_ungrouped, sta_correlation

#===========================================================================#
#                                                                           #
# STEM RESEARCH :: AI . APIs . Cloud :: https://stemfard.org        #
#																			#
#===========================================================================#