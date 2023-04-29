import sys
from functools import partial

import numba
from llvmlite import ir
from numba.core import codegen, config, runtime, types, utils
from numba.core.cpu import cgutils
from numba.pycc.compiler import (
    Flags,
    Linkage,
    ModuleCompiler,
    compile_extra,
    global_compiler_lock,
    nrtdynmod,
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


# Patch code libraries to disable full module-level optimizations.
# These optimizations assume 64-bit on 64-bit machines and causes issues for the same reason as the previous note.
# I am not familiar enough with LLVM module passes to figure out how to fix this,
# so I am settling for simply disabling these
# --------------------------------------------------------------------------------
class PatchedAOTCodeLibrary(codegen.AOTCodeLibrary):
    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        self._codegen._mpm_full.run = lambda *args, **kwargs: None


class PatchedJITCodeLibrary(codegen.JITCodeLibrary):
    def __init__(self, codegen, name):
        super().__init__(codegen, name)
        # ...
        self._codegen._mpm_full.run = lambda *args, **kwargs: None


codegen.AOTCodeLibrary = PatchedAOTCodeLibrary
codegen.AOTCPUCodegen._library_class = PatchedAOTCodeLibrary
codegen.JITCodeLibrary = PatchedJITCodeLibrary
codegen.JITCPUCodegen._library_class = PatchedJITCodeLibrary

codegen.AOTCodeLibrary = PatchedAOTCodeLibrary
codegen.AOTCPUCodegen._library_class = PatchedAOTCodeLibrary
codegen.JITCodeLibrary = PatchedJITCodeLibrary
codegen.JITCPUCodegen._library_class = PatchedJITCodeLibrary


# --------------------------------------------------------------------------------


# Custom cfunc wrapper
# --------------------------------------------------------------------------------
def create_cfunc_wrapper(self, library, fndesc, _env, _call_helper):
    """Custom cfunc wrapper for generating WASM/JS-accessible functions"""

    wrapper_module = self.create_module("cfunc_wrapper")
    fnty = self.call_conv.get_function_type(fndesc.restype, fndesc.argtypes)
    wrapper_callee = ir.Function(wrapper_module, fnty, fndesc.llvm_func_name)

    ll_argtypes = [self.get_value_type(ty) for ty in fndesc.argtypes]
    ll_return_type = self.get_value_type(fndesc.restype)

    returns_array = isinstance(fndesc.restype, numba.core.types.npytypes.Array)
    # If the function returns an array, it must be returned as a pointer
    if returns_array:
        ll_return_type = ir.PointerType(ll_return_type)

    wrapty = ir.FunctionType(ll_return_type, ll_argtypes)
    wrapfn = ir.Function(wrapper_module, wrapty, fndesc.llvm_cfunc_wrapper_name)
    builder = ir.IRBuilder(wrapfn.append_basic_block("entry"))

    _status, result = self.call_conv.call_function(
        builder,
        wrapper_callee,
        fndesc.restype,
        fndesc.argtypes,
        wrapfn.args,
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
    else:
        builder.ret(result)

    library.add_ir_module(wrapper_module)


# --------------------------------------------------------------------------------


# Custom compiler for WASM
# --------------------------------------------------------------------------------
class CustomCompiler(ModuleCompiler):
    """Custom compiler for WASM"""

    def __init__(self, export_entries, module_name, **aot_options):
        super().__init__(export_entries, module_name, use_nrt=True, **aot_options)
        self.context.create_cfunc_wrapper = partial(create_cfunc_wrapper, self.context)

    @global_compiler_lock
    def _cull_exports(self):
        self.exported_function_types = {}
        self.function_environments = {}
        self.environment_gvs = {}

        codegen = self.context.codegen()
        library = codegen.create_library(self.module_name)

        flags = Flags()
        flags.no_compile = True
        flags.no_cpython_wrapper = True

        if self.use_nrt:
            flags.nrt = True
            # Compile NRT helpers
            nrt_module, _ = nrtdynmod.create_nrt_module(self.context)
            library.add_ir_module(nrt_module)

        for entry in self.export_entries:
            cres = compile_extra(
                self.typing_context,
                self.context,
                entry.function,
                entry.signature.args,
                entry.signature.return_type,
                flags,
                locals={},
                library=library,
            )

            func_name = cres.fndesc.llvm_func_name
            cfunc = cres.library.get_function(f"cfunc.{func_name}")

            cfunc.name = entry.symbol
            # Export custom cfunc
            self.dll_exports.append(entry.symbol)

        # Hide all functions except those explicitly exported
        library.finalize()
        for fn in library.get_defined_functions():
            if fn.name not in self.dll_exports:
                if fn.linkage in {Linkage.private, Linkage.internal}:
                    # Private/Internal linkage must have "default" visibility
                    fn.visibility = "default"
                else:
                    fn.visibility = "hidden"
        return library


# --------------------------------------------------------------------------------
