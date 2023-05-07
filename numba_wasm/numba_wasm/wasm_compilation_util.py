"""Utility for compiling numba functions for WASM.
Importing this module will rewrite constants to attempt to mock wasm32."""

import sys
from functools import partial, cached_property
from inspect import getmodule

import numba
from numba import _dynfunc
from llvmlite import ir
from numba.core import codegen, config, runtime, types, utils, compiler_lock
from numba.core.cpu import cgutils, CPUContext
from numba.core.registry import CPUTarget, CPUDispatcher
from numba.core.target_extension import (
    Generic,
    target_registry,
    dispatcher_registry,
    jit_registry,
    jit,
)

# Do our best to convince numba that we are compiling on/for a 32-bit memory system.
# This is neccesary for the IR to be built correctly as size_t would otherwise be incorrect
# --------------------------------------------------------------------------------
sys.platform = "emscripten"

config.MACHINE_BITS = 32
config.IS_32BITS = True
utils.MACHINE_BITS = 32


runtime.nrt.nrtdynmod._word_type = ir.IntType(32)
runtime.nrt.nrtdynmod._meminfo_struct_type = ir.LiteralStructType(
    [
        runtime.nrt.nrtdynmod._word_type,
        runtime.nrt.nrtdynmod._pointer_type,
        runtime.nrt.nrtdynmod._pointer_type,
        runtime.nrt.nrtdynmod._pointer_type,
        runtime.nrt.nrtdynmod._word_type,
    ]
)

types.intp = types.int32
types.uintp = types.uint32
types.intc = types.int32
types.uintc = types.uint32
types.ssize_t = types.int32
types.size_t = types.uint32
cgutils.intp_t = ir.IntType(32)


# --------------------------------------------------------------------------------


# Patch code libraries to disable full optimizations.
# These optimizations assume 64-bit on 64-bit machines and causes issues for the same reason as the
# previous note.
# I am not familiar enough with LLVM module passes to figure out how to fix this,
# so I am settling for simply disabling these
# TODO: fix this
# --------------------------------------------------------------------------------


def _function_pass_manager(self, llvm_module, **kwargs):
    pm = codegen.ll.create_function_pass_manager(llvm_module)
    self._tm.add_analysis_passes(pm)
    # with self._pass_manager_builder(**kwargs) as pmb:
    #     pmb.populate(pm)
    if config.LLVM_REFPRUNE_PASS:
        pm.add_refprune_pass(codegen._parse_refprune_flags())
    return pm


class PatchedJITCodeLibrary(codegen.JITCodeLibrary):
    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        # ...
        self._codegen._mpm_full.run = lambda *args, **kwargs: None
        self._codegen._function_pass_manager = partial(
            _function_pass_manager, self._codegen
        )


codegen.JITCodeLibrary = PatchedJITCodeLibrary
codegen.JITCPUCodegen._library_class = PatchedJITCodeLibrary


# --------------------------------------------------------------------------------


# Custom context for WASM
# --------------------------------------------------------------------------------


class WASMContext(CPUContext):
    def __init__(self, typingctx, target="cpu"):
        super().__init__(typingctx, target)

    def create_cfunc_wrapper(self, library, fndesc, _env, _call_helper):
        """Custom cfunc wrapper for generating WASM/JS-accessible functions"""

        wrapper_module = self.create_module("cfunc_wrapper")
        fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
        wrapper_callee = ir.Function(wrapper_module, fnty, fndesc.llvm_func_name)

        # If an argument is an array, it must be a pointer
        ll_argtypes = []
        arg_pointer_flags = []
        for arg_type in fndesc.argtypes:
            if isinstance(arg_type, numba.core.types.npytypes.Array):
                ll_argtypes.append(ir.PointerType(self.get_value_type(arg_type)))
                arg_pointer_flags.append(True)
            else:
                ll_argtypes.append(self.get_value_type(arg_type))
                arg_pointer_flags.append(False)

        ll_return_type = (
            ir.VoidType()
            if fndesc.restype is numba.types.void
            else self.get_value_type(fndesc.restype)
        )

        returns_array = isinstance(fndesc.restype, numba.core.types.npytypes.Array)
        # If the function returns an array, it must be returned as a pointer
        if returns_array:
            ll_return_type = ir.PointerType(ll_return_type)

        wrapty = ir.FunctionType(ll_return_type, ll_argtypes)
        wrapfn = ir.Function(wrapper_module, wrapty, fndesc.llvm_cfunc_wrapper_name)
        builder = ir.IRBuilder(wrapfn.append_basic_block("entry"))

        args = []
        for arg_type, arg_var, pointer_flag in zip(
            ll_argtypes, wrapfn.args, arg_pointer_flags
        ):
            # derference pointer arguments
            if pointer_flag:
                arg_var = builder.load(arg_var)
            args.append(arg_var)

        _status, result = self.call_conv.call_function(
            builder,
            wrapper_callee,
            fndesc.restype,
            fndesc.argtypes,
            args,
            attrs=("noinline",),
        )

        # allocate memory and store the array specification
        if returns_array:
            # TODO: if the struct contains pointers this will be wrong on 64-bit machines
            # this likely stems from the same issue as the LLVM optimization,
            # but is not major as it just results in extra memory being allocated
            size_of_struct = builder.ptrtoint(
                builder.gep(
                    ll_return_type("null"),
                    (cgutils.int32_t(1),),
                ),
                cgutils.int32_t,
            )

            fnty = ir.FunctionType(ir.PointerType(ir.IntType(8)), [cgutils.int32_t])
            fn = cgutils.get_or_insert_function(builder.module, fnty, name="malloc")
            fn.return_value.add_attribute("noalias")

            pointer_int8 = builder.call(
                fn, [builder.add(size_of_struct, cgutils.int32_t(4))]
            )

            pointer = builder.bitcast(pointer_int8, ll_return_type)
            builder.store(result, pointer)
            builder.ret(pointer)
        elif fndesc.restype == numba.types.none:
            builder.ret_void()
        else:
            builder.ret(result)

        # TODO: name mangling
        wrapfn.name = f"{fndesc.modname}.{fndesc.qualname}"
        library.add_ir_module(wrapper_module)


class WASMTarget(CPUTarget):
    @cached_property
    def _toplevel_target_context(self):
        return WASMContext(self.typing_context, self._target_name)


wasm_target = WASMTarget("cpu")


class WASMDispatcher(CPUDispatcher):
    targetdescr = wasm_target


class WASM(Generic):
    """Mark the target as WASM CPU."""


# replace "cpu" target with the wasm cpu target
# it would likely be more ideal to have wasm be its own target
# but doing it this way maintains the CPU overloads
# for functions like np.empty
del target_registry["cpu"]
target_registry["cpu"] = WASM
dispatcher_registry[WASM] = WASMDispatcher
jit_registry[WASM] = jit

# --------------------------------------------------------------------------------


# General utility
# --------------------------------------------------------------------------------


@compiler_lock.global_compiler_lock
def build_wasm_ir_module(njit_functions: tuple) -> str:
    """Build a WASM-compatible ir module from list of njit functions"""
    library = codegen.JITCPUCodegen("function_library").create_library(
        "function_library"
    )
    for function in njit_functions:
        # assume only 1 signature
        function_module = tuple(function.overloads.values())[0].library._final_module
        library.add_llvm_module(function_module)
        # assign custom symbol
        if function.symbol is not None:
            library.get_function(
                f"{getmodule(function).__name__}.{function.__name__}"
            ).name = function.symbol
    library.finalize()
    return str(library._final_module)


# --------------------------------------------------------------------------------
