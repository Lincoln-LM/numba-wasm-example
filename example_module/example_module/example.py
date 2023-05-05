"""Example numba-compiled functions"""

import numpy as np
from numba_wasm.util import njit_wasm


@njit_wasm
def square(input_value: np.float64) -> np.float64:
    """Basic double square function example"""
    return input_value**2


@njit_wasm
def new_array_function() -> np.ndarray[1, np.uint32]:
    """Basic array creation function"""
    array = np.zeros(np.int32(123), np.uint32)
    # arbitrary values
    array[0] = 1
    array[1] = 2
    array[3] = 3
    array[4] = 4
    return array


@njit_wasm
def modify_array_function(
    input_array: np.ndarray[1, np.uint32]
) -> np.ndarray[1, np.uint32]:
    """Basic array modification function example"""
    # arbitrary scalar addition
    input_array += np.uint32(10)
    return input_array


@njit_wasm
def modify_array_in_place_function(input_array: np.ndarray[1, np.uint32]):
    """Basic in-place array modification function example"""
    # arbitrary scalar multiplication
    input_array *= np.uint32(2)


@njit_wasm
def new_and_modify_array_function() -> np.ndarray[1, np.uint32]:
    """Basic array creation and modification function example
    which calls previously defined functions"""
    array = new_array_function()
    return modify_array_function(array)
