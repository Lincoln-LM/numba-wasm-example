"""Example numba-compiled functions"""

import numpy as np
from numba_wasm.util import njit_wasm, global_variable

global_counter_getter, global_counter_setter, global_counter_spec = global_variable(
    "global_counter", 0, np.uint32
)


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


@njit_wasm(symbol="specially_named_new_array_function")
def specially_named_new_array_function() -> np.ndarray[1, np.uint32]:
    """Basic array creation function"""
    array = np.zeros(np.int32(5), np.uint32)
    # arbitrary values
    array[0] = 12345
    array[1] = 23451
    array[2] = 34512
    array[3] = 45123
    array[4] = 51234
    return array


@njit_wasm
def increment_global_counter_function():
    """Function that increments the global variable ``global_counter``"""
    global_counter_setter(global_counter_getter() + 1)


@njit_wasm
def get_global_counter() -> np.uint32:
    """Function that returns the global variable ``global_counter``"""
    return global_counter_getter()
