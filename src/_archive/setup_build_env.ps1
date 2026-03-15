$vsPath = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvarsall = Join-Path $vsPath "VC\Auxiliary\Build\vcvarsall.bat"

# Run vcvarsall.bat and get its environment variables
$vcvars = cmd /c "`"$vcvarsall`" x64 & set"

# Parse environment variables
$vcvars | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        $name = $matches[1]
        $value = $matches[2]
        Set-Item "env:$name" $value
    }
}

# Set CUDA paths
$env:Path += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

# Set Rust paths
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# Set cargo target directory
$env:CARGO_TARGET_DIR = "$env:TEMP\quantum_target"
