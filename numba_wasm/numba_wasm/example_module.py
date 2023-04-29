import numpy as np

from .util import wasm_function


@wasm_function
def square(input_value: np.float64) -> np.float64:
    """Basic double square function example"""


@wasm_function
def new_array_function() -> np.ndarray[1, np.uint32]:
    """Basic double square function example"""
