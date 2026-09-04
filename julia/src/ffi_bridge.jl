# FFI Bridge to the Rust Core
#
# When the PhiFlow Rust library is compiled as a shared object (.so/.dll/.dylib),
# Julia can call it directly via Libdl. This gives you the speed of Rust
# with the interactivity of Julia.
#
# To build the Rust library as a .so:
#   cargo build --release --lib
#   # On Linux: target/release/libphiflow.so
#   # On macOS: target/release/libphiflow.dylib
#   # On Windows: target/release/phiflow.dll
#
# Then set the PHI_LIB environment variable to the path.

module FFIBridge

using Libdl

export compile_phi, run_phi, compile_to_openqasm, lib_available

# Find the shared library
function _find_lib()::String
    env_path = get(ENV, "PHI_LIB", "")
    if !isempty(env_path) && isfile(env_path)
        return env_path
    end

    candidates = [
        "target/release/libphiflow.so",
        "target/release/libphiflow.dylib",
        "target/release/phiflow.dll",
        "../target/release/libphiflow.so",
        "../target/release/libphiflow.dylib",
    ]

    for c in candidates
        if isfile(c)
            return abspath(c)
        end
    end

    return ""
end

"Check if the Rust library is available"
function lib_available()::Bool
    return !isempty(_find_lib())
end

"""
    compile_phi(source::String) -> String

Compile a PhiFlow program to PhiIR using the Rust core.
Returns the IR as a JSON string.

Requires: cargo build --release --lib
"""
function compile_phi(source::String)::String
    lib = _find_lib()
    isempty(lib) && error("PhiFlow library not found. Run: cargo build --release --lib")

    handle = Libdl.dlopen(lib)
    try
        sym = Libdl.dlsym(handle, "compile_and_run_phi_ir")
        result = ccall(sym, Cstring, (Cstring,), source)
        return unsafe_string(result)
    finally
        Libdl.dlclose(handle)
    end
end

"""
    compile_to_openqasm(source::String, optimize_depth::Bool=false) -> String

Compile a PhiFlow program to OpenQASM 3.0 using the Rust core.
"""
function compile_to_openqasm(source::String, optimize_depth::Bool=false)::String
    lib = _find_lib()
    isempty(lib) && error("PhiFlow library not found. Run: cargo build --release --lib")

    handle = Libdl.dlopen(lib)
    try
        sym = Libdl.dlsym(handle, "compile_to_openqasm")
        result = ccall(sym, Cstring, (Cstring, Cint), source, optimize_depth ? 1 : 0)
        return unsafe_string(result)
    finally
        Libdl.dlclose(handle)
    end
end

"""
    run_phi(source::String) -> String

Compile and run a PhiFlow program. Returns the output as JSON.
"""
function run_phi(source::String)::String
    return compile_phi(source)
end

end # module FFIBridge
