"""Script to build the LLVM IR for the numba functions of this module"""

import os

os.environ["BUILD_WASM_IR"] = "1"
# pylint: disable=wrong-import-position
from numba_wasm.util import build_wasm_ir_module  # noqa: E402
from example_module import (  # noqa: E402
    square,
    new_array_function,
    modify_array_function,
    modify_array_in_place_function,
    new_and_modify_array_function,
)

# pylint: enable=wrong-import-position

with open("example_module.ll", "w+", encoding="utf-8") as out_file:
    out_file.write(
        build_wasm_ir_module(
            # TODO: should these need to be explicitly included like this?
            # the cc.export decorator automatically adds them to its export list
            (
                square,
                new_array_function,
                modify_array_function,
                modify_array_in_place_function,
                new_and_modify_array_function,
            )
        )
    )
