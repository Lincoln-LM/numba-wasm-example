# numba-wasm-example

Example of compiling numba functions to wasm for use in JS or alongside pyodide

## This repository contains two python packages set up with poetry:

### [numba_wasm](./numba_wasm/) - A very basic library that provides compilation to and interfacing with WebAssembly.
The library provides a custom @njit_wasm decorator which handles both compiling to wasm and interfacing between pyodide and the wasm compiled functions.

When called from a regular interpreter, the decorator acts mostly the same as the traditional numba.njit, with the exception that it infers the function signature from type annotations rather than relying on it being explicitly declared. This allows easier testing of compiled functions as you do not need to compile to wasm test in a pyodide environment to test simple things like logic.

When called from a regular interpreter with the "BUILD_WASM_IR" environment variable set (ex. [build_numba_functions.py](./example_module/build_numba_functions.py)), the decorator compiles to LLVM IR targetted at a 32-bit memory system (with the help of [some patches to trick numba](./numba_wasm/numba_wasm/wasm_compilation_util.py)) in order to allow emscripten to compile it to wasm32.

When called from a pyodide interpreter, the decorator ignores the actual contents of the function and instead interfaces with the appropriate compiled WebAssembly function, converting inputs to the format the compiled code expects (ex. arrays are converted into pointers when appropriate).

The combination of these three different functionalities allows the same module that contains your code to be used for local testing, compilation, and use within pyodide itself.

### [example_module](./example_module/) - An incredibly simple example library containing basic functions to be compiled to WebAssembly via numba_wasm.
The library provides basic functions marked with the @njit_wasm decorator to test arithmetic and array creation/modification

It also contains the aforementoned [build_numba_functions.py](./example_module/build_numba_functions.py) script used to generate and export LLVM IR for the functions.

This IR can then be compiled with emscripten as a wasm side module to be loaded directly into the browser via pyodide's interface as is done in the [example page](https://lincoln-lm.github.io/numba-wasm-example/) ([src](https://github.com/Lincoln-LM/numba-wasm-example/tree/gh-pages)).

With the compiled wasm side module loaded, using the functions is as simple as loading and importing the wheels for both numba_wasm and example_module into the pyodide interpreter and directly calling the functions (ex. example_module.example.square(2.0)).

## Very minimal (console) example page: https://lincoln-lm.github.io/numba-wasm-example/ ([src](https://github.com/Lincoln-LM/numba-wasm-example/tree/gh-pages))
