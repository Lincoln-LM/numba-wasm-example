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
    return np.empty(np.int32(123), np.uint32)


# --------------------------------------------------------------------------------

if __name__ == "__main__":
    compiler = CustomCompiler(cc._export_entries, cc._basename)
    compiler.external_init_function = cc._init_function
    library = compiler._cull_exports()
    with open("out.ll", "w+", encoding="utf-8") as output_file:
        output_file.write(str(library._final_module))
