"""Utility functions"""

import sys
import typing
import ctypes
import numpy as np

if sys.platform == "emscripten":
    import js
else:
    js = None


class NumpyHolder:
    """Holder class for WASM-created numpy array"""

    def __init__(self, data_pointer: int, T: typing.Type, shape: tuple) -> None:
        self.__array_interface__ = {
            "data": (data_pointer, False),
            "typestr": T.str,
            "shape": shape,
        }


def np_array_from_spec_pointer(
    spec_pointer: int, array_type: typing.Type
) -> np.ndarray:
    """Convert a specification pointer to an ndarray without copying the underlying data.

    "aray_type" is expected to be an annotated np.ndarray type with the number of dimensions
    and item type specified.

    For example, a 2d array of float64 must be declared as np.ndarray[2, np.float64]."""
    ndim, T = array_type.__args__
    if not isinstance(T, np.dtype):
        T = np.dtype(T)
    pointer_uint32 = ctypes.POINTER(ctypes.c_uint32)

    data_pointer = ctypes.cast(spec_pointer + 16, pointer_uint32).contents.value
    shape = tuple(
        ctypes.cast(spec_pointer + offset * 4 + 20, pointer_uint32).contents.value
        for offset in range(ndim)
    )

    holder = NumpyHolder(data_pointer, T, shape)
    return np.array(holder)


def np_array_to_spec_pointer(input_array: np.ndarray) -> int:
    """Convert an ndarray to a "spec pointer."

    A spec pointer is a pointer to an array which details the information of the array.
    This is the format numba uses, and thus, the format numba-compiled functions expect.
    """
    # TODO: is this system safe for memory? possible concerns about deallocation?
    meminfo_ptr = js.global_functions.NRT_MemInfo_alloc_safe_aligned(
        input_array.nbytes, 32
    )
    ndim = input_array.ndim
    pointer, _ = input_array.__array_interface__["data"]
    spec = np.zeros(5 + (ndim * 2), np.uint32)
    spec[0] = meminfo_ptr
    # spec[1] = pointer to parent python object
    spec[2] = input_array.size
    spec[3] = input_array.itemsize
    spec[4] = pointer
    for i in range(ndim):
        spec[5 + i] = input_array.shape[i]
        spec[5 + ndim + i] = input_array.strides[i]
    spec_pointer, _ = spec.__array_interface__["data"]
    return spec_pointer


def convert_inputs(func, args: tuple) -> tuple:
    """Convert arguments to the format WASM expects.

    Currently, this only involves turning ndarrays into pointers."""
    return tuple(
        np_array_to_spec_pointer(arg_value)
        if arg_type.__name__ == "ndarray"
        else arg_value
        for arg_type, arg_value in zip(func.__annotations__.values(), args)
    )


def wasm_function(func):
    """Decorator for calling a WASM function from python.

    This decorator assumes all the arguments of the function and return type are annotated
    appropriately.

    More specifically, the arrays must be marked as ndarrays with the proper amount of
    dimensions and item type.

    For example, a 2d array of float64 must be declared as np.ndarray[2, np.float64]."""
    return_type = func.__annotations__["return"]
    # functions which return arrays must have the ndarray created from the returned pointer
    if return_type.__name__ == "ndarray":

        def wrap(*args):
            # functions with array arguments must be converted to pointers
            inputs = convert_inputs(func, args)
            result_pointer = getattr(js.global_functions, func.__name__)(*inputs)
            return np_array_from_spec_pointer(
                result_pointer,
                return_type,
            )

    else:

        def wrap(*args):
            # ...
            inputs = convert_inputs(func, args)
            return getattr(js.global_functions, func.__name__)(*inputs)

    return wrap
