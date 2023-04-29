"""Utility functions"""

import sys
import typing
import ctypes
import numpy as np

from . import util

if sys.platform == "emscripten":
    import js
else:
    js = None


def np_array_from_spec_pointer(
    spec_pointer: int, array_type: typing.Type
) -> np.ndarray:
    ndim, T = array_type.__args__
    if not isinstance(T, np.dtype):
        T = np.dtype(T)
    pointer_uint32 = ctypes.POINTER(ctypes.c_uint32)

    data_pointer = ctypes.cast(spec_pointer + 16, pointer_uint32).contents.value
    shape = tuple(
        ctypes.cast(spec_pointer + offset * 4 + 20, pointer_uint32).contents.value
        for offset in range(ndim)
    )

    buff = {"data": (data_pointer, False), "typestr": T.str, "shape": shape}

    class NumpyHolder:
        """Holder class for numpy array"""

        def __init__(self) -> None:
            self.__array_interface__ = buff

    holder = NumpyHolder()
    return np.array(holder)


def wasm_function(func):
    return_type = func.__annotations__["return"]
    if return_type.__name__ == "ndarray":

        def wrap(*args, **kwargs):
            return util.np_array_from_spec_pointer(
                getattr(js.global_functions, func.__name__)(*args, **kwargs),
                return_type,
            )

    else:

        def wrap(*args, **kwargs):
            return getattr(js.global_functions, func.__name__)(*args, **kwargs)

    return wrap
