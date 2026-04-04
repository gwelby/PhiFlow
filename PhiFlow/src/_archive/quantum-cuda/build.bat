@echo off
set VS_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set MSVC_PATH=%VS_PATH%\VC\Tools\MSVC\14.43.34808
set WINDOWS_SDK=C:\Program Files (x86)\Windows Kits\10
set WINDOWS_SDK_VERSION=10.0.22621.0

:: Set up Visual Studio environment
call "%VS_PATH%\VC\Auxiliary\Build\vcvars64.bat"

:: Add Windows SDK lib paths
set LIB=%WINDOWS_SDK%\Lib\%WINDOWS_SDK_VERSION%\um\x64;%WINDOWS_SDK%\Lib\%WINDOWS_SDK_VERSION%\ucrt\x64;%LIB%

:: Add CUDA to path
set PATH=%CUDA_PATH%\bin;%PATH%

:: Add Rust to path
set PATH=%USERPROFILE%\.cargo\bin;%PATH%

:: Create build directory
if not exist "build" mkdir build

:: Compile CUDA code to PTX
nvcc --ptx ^
    -O3 ^
    --use_fast_math ^
    -arch=sm_75 ^
    -I"%CUDA_PATH%\include" ^
    -I"%MSVC_PATH%\include" ^
    -I"%WINDOWS_SDK%\Include\%WINDOWS_SDK_VERSION%\ucrt" ^
    -I"%WINDOWS_SDK%\Include\%WINDOWS_SDK_VERSION%\um" ^
    -I"%WINDOWS_SDK%\Include\%WINDOWS_SDK_VERSION%\shared" ^
    src\cuda\quantum_kernels.cu ^
    -o build\quantum_kernels.ptx

:: Copy PTX to output directory
if not exist "target\debug" mkdir target\debug
copy /Y build\quantum_kernels.ptx target\debug\

:: Build with cargo
cargo build
