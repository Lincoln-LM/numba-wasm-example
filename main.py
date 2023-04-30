import numba
import numpy as np
from numba.pycc import CC

from util import CustomCompiler

# Example test functions
# --------------------------------------------------------------------------------
cc = CC("my_module")


@cc.export("square", numba.float64(numba.float64))
def square(input_value: np.float64) -> np.float64:
    """Basic double square function example"""
    return input_value**2


@cc.export("new_array_function", numba.uint32[::1]())
def new_array_function() -> np.ndarray[1, np.uint32]:
    """Basic array creation function"""
    array = np.zeros(np.int32(123), np.uint32)
    # arbitrary values
    array[0] = 1
    array[1] = 2
    array[3] = 3
    array[4] = 4
    return array


@cc.export("modify_array_function", numba.uint32[::1](numba.uint32[::1]))
def modify_array_function(
    input_array: np.ndarray[1, np.uint32]
) -> np.ndarray[1, np.uint32]:
    """Basic array modification function example"""
    # arbitrary scalar addition
    input_array += np.uint32(10)
    return input_array


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # compile functions to LLVM IR
    compiler = CustomCompiler(cc._export_entries, cc._basename)
    compiler.external_init_function = cc._init_function
    library = compiler._cull_exports()
    with open("out.ll", "w+", encoding="utf-8") as output_file:
        output_file.write(str(library._final_module))
