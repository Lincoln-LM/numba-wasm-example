"""Example test functions compiled originally in main.py"""

import numpy as np

from .util import wasm_function


@wasm_function
def square(input_value: np.float64) -> np.float64:
    """Basic double square function example"""


@wasm_function
def new_array_function() -> np.ndarray[1, np.uint32]:
    """Basic new array creation function example"""


@wasm_function
def modify_array_function(
    input_array: np.ndarray[1, np.uint32]
) -> np.ndarray[1, np.uint32]:
    """Basic array modification function example"""
