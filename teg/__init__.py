from .lang.integrable_program import *
from .lang.operator_overloads import *

import os
# Export data path.
__data_path__ = f'{os.path.dirname(os.path.dirname(__file__))}{os.path.sep}data'
assert os.path.exists(__data_path__), f'Couldn\'t find data directory at {__data_path__}'
# Export runtime includes path.
__include_path__ = f'{os.path.dirname(__file__)}{os.path.sep}include'
assert os.path.exists(__include_path__), f'Couldn\'t find include path directory at {__include_path__}'
