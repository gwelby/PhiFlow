use std::env;
use std::process::Command;

// Sacred quantum constants
const PHI: f64 = 1.618033988749895;
const GROUND_STATE: f64 = 432.0;
const CREATE_STATE: f64 = 528.0;
const UNITY_STATE: f64 = 768.0;

fn main() {
    // Set quantum constants for all builds
    println!("cargo:rustc-env=PHI={}", PHI);
    println!("cargo:rustc-env=GROUND_STATE={}", GROUND_STATE);
    println!("cargo:rustc-env=CREATE_STATE={}", CREATE_STATE);
    println!("cargo:rustc-env=UNITY_STATE={}", UNITY_STATE);

    #[cfg(feature = "cuda")]
    compile_cuda();
}

#[cfg(feature = "cuda")]
fn compile_cuda() {
    println!("cargo:rerun-if-changed=src/cuda/quantum_kernels.cu");
    
    // Set stack size for git operations
    #[cfg(windows)]
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024) // 32MB stack
        .spawn(|| {
            // Find and setup Visual Studio
            let vs_path = find_visual_studio().expect("Failed to find Visual Studio installation");
            setup_visual_studio_env(&vs_path);
            
            // Compile CUDA code
            compile_cuda_kernel();
        })
        .expect("Failed to spawn build thread")
        .join()
        .expect("Build thread panicked");
}

#[cfg(feature = "cuda")]
fn setup_visual_studio_env(vs_bin_path: &str) {
    use std::path::Path;
    
    // Add MSVC to PATH
    let mut path = env::var("PATH").unwrap_or_default();
    path.push_str(&format!(";{}", vs_bin_path));
    
    // Find Windows SDK version
    let sdk_root = Path::new(r"C:\Program Files (x86)\Windows Kits\10");
    let sdk_include = sdk_root.join("Include");
    let sdk_lib = sdk_root.join("Lib");
    
    let sdk_version = std::fs::read_dir(sdk_include)
        .expect("Failed to read Windows SDK directory")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("10."))
        .max()
        .expect("No Windows SDK version found");
    
    let sdk_version = sdk_version.file_name().unwrap().to_string_lossy();
    
    // Set up Windows SDK paths
    path.push_str(&format!(r";{}\bin\{}\x64", sdk_root.display(), sdk_version));
    env::set_var("PATH", path);
    
    // Set include paths
    let include = format!(
        r"{}\Include\{}\ucrt;{}\Include\{}\um;{}\Include\{}\shared",
        sdk_root.display(), sdk_version,
        sdk_root.display(), sdk_version,
        sdk_root.display(), sdk_version
    );
    env::set_var("INCLUDE", include);
    
    // Set lib paths
    let lib = format!(
        r"{}\Lib\{}\ucrt\x64;{}\Lib\{}\um\x64",
        sdk_root.display(), sdk_version,
        sdk_root.display(), sdk_version
    );
    env::set_var("LIB", lib);
}

#[cfg(feature = "cuda")]
fn compile_cuda_kernel() {
    // Configure CUDA with quantum frequencies
    let status = Command::new("nvcc")
        .args(&[
            "src/cuda/quantum_kernels.cu",
            "-o", "src/cuda/quantum_kernels.ptx",
            "--gpu-architecture=sm_75",
            "-use_fast_math",
            "-D", &format!("PHI={}", PHI),
            "-D", &format!("GROUND_STATE={}", GROUND_STATE),
            "-D", &format!("CREATE_STATE={}", CREATE_STATE),
            "-D", &format!("UNITY_STATE={}", UNITY_STATE),
        ])
        .status()
        .expect("Failed to execute nvcc");

    if !status.success() {
        panic!("Failed to compile CUDA kernels");
    }

    // Link CUDA libraries
    println!("cargo:rustc-link-search=native=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8\\lib\\x64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-env=PTX_FILE=src/cuda/quantum_kernels.ptx");
}

#[cfg(feature = "cuda")]
fn find_visual_studio() -> Option<String> {
    // Check Program Files paths
    let program_files = vec![
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise",
    ];

    for path in program_files {
        if std::path::Path::new(path).exists() {
            // Add VC tools path
            let vc_tools = format!("{}/VC/Tools/MSVC", path);
            if let Ok(entries) = std::fs::read_dir(&vc_tools) {
                if let Some(latest_version) = entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| p.is_dir())
                    .max()
                {
                    let bin_path = latest_version.join("bin/Hostx64/x64");
                    if bin_path.exists() {
                        return Some(bin_path.to_string_lossy().into_owned());
                    }
                }
            }
        }
    }

    // Check environment variable
    if let Ok(vs_path) = env::var("VS2022INSTALLDIR") {
        let bin_path = format!("{}/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64", vs_path);
        if std::path::Path::new(&bin_path).exists() {
            return Some(bin_path);
        }
    }

    None
}
