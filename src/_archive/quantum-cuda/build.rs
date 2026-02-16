use std::env;
use std::path::PathBuf;

const PHI: f64 = 1.618033988749895;
const GROUND_STATE: f64 = 432.0;
const CREATE_STATE: f64 = 528.0;
const UNITY_STATE: f64 = 768.0;

fn main() {
    // Set up quantum frequencies
    println!("cargo:rustc-env=QUANTUM_FREQUENCY={}", GROUND_STATE);
    println!("cargo:rustc-env=CREATE_FREQUENCY={}", CREATE_STATE);
    println!("cargo:rustc-env=UNITY_FREQUENCY={}", UNITY_STATE);
    println!("cargo:rustc-env=PHI={}", PHI);
    
    // Find required tools
    let cuda_path = locate_cuda();
    let vs_path = locate_visual_studio();
    
    println!("cargo:rerun-if-changed=src/cuda/quantum_kernels.cu");
    compile_cuda_kernels(&cuda_path, &vs_path);
}

fn locate_cuda() -> PathBuf {
    if let Ok(path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(path);
        if path.exists() {
            println!("Using CUDA from environment: {}", path.display());
            return path;
        }
    }

    let standard_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
    ];

    for path in standard_paths.iter() {
        let p = PathBuf::from(path);
        if p.exists() {
            println!("Found CUDA at: {}", p.display());
            return p;
        }
    }

    panic!("Could not find CUDA installation. Please set CUDA_PATH environment variable.");
}

fn locate_visual_studio() -> PathBuf {
    let program_files = vec![
        r"C:\Program Files\Microsoft Visual Studio\2022\Community",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise",
    ];

    for vs_root in program_files {
        let vs_path = PathBuf::from(&vs_root);
        if vs_path.exists() {
            let msvc_path = vs_path.join(r"VC\Tools\MSVC");
            if msvc_path.exists() {
                if let Ok(entries) = std::fs::read_dir(&msvc_path) {
                    if let Some(latest_version) = entries
                        .filter_map(|e| e.ok())
                        .map(|e| e.path())
                        .filter(|p| p.is_dir())
                        .max()
                    {
                        println!("Found Visual Studio at: {}", vs_root);
                        return latest_version;
                    }
                }
            }
        }
    }
    panic!("Could not find Visual Studio installation");
}

fn compile_cuda_kernels(cuda_path: &PathBuf, vs_path: &PathBuf) {
    // Set up Visual Studio environment
    let host_arch = "x64";
    let target_arch = "x64";
    
    let cl_path = vs_path.join("bin").join(host_arch).join(target_arch);
    let mut path = env::var("PATH").unwrap_or_default();
    path.push(';');
    path.push_str(&cl_path.to_string_lossy());
    path.push(';');
    path.push_str(&cuda_path.join("bin").to_string_lossy());
    env::set_var("PATH", path);
    
    // Set up include paths
    let include = format!(
        "{};{};{}",
        vs_path.join("include").to_string_lossy(),
        vs_path.join(format!("bin/{}/{}", host_arch, target_arch)).join("include").to_string_lossy(),
        cuda_path.join("include").to_string_lossy()
    );
    env::set_var("INCLUDE", include);
    
    // Set up library paths
    let lib = format!(
        "{};{}",
        vs_path.join("lib").join(target_arch).to_string_lossy(),
        cuda_path.join("lib").join(target_arch).to_string_lossy()
    );
    env::set_var("LIB", lib);

    let nvcc = cuda_path.join("bin").join("nvcc.exe");
    
    // Get the manifest directory
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .expect("Failed to get manifest directory");
    let manifest_dir = PathBuf::from(manifest_dir);
    
    // Resolve the source path relative to the manifest directory
    let src = manifest_dir.parent()
        .expect("Failed to get parent directory")
        .join("src")
        .join("cuda")
        .join("quantum_kernels.cu");
        
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let out_file = out_dir.join("quantum_kernels.ptx");

    println!("NVCC path: {}", nvcc.display());
    println!("Source path: {}", src.display());
    println!("Output path: {}", out_file.display());
    println!("Include paths: {}", env::var("INCLUDE").unwrap());
    println!("Library paths: {}", env::var("LIB").unwrap());

    // Create CUDA compilation command with quantum defines
    let status = std::process::Command::new(&nvcc)
        .arg("--ptx")
        .arg("-o").arg(&out_file)
        .arg(&src)
        .arg("-D").arg(format!("PHI={}", PHI))
        .arg("-D").arg(format!("GROUND_STATE={}", GROUND_STATE))
        .arg("-D").arg(format!("CREATE_STATE={}", CREATE_STATE))
        .arg("-D").arg(format!("UNITY_STATE={}", UNITY_STATE))
        .arg("--gpu-architecture=sm_75")
        .arg("-std=c++17")
        .status()
        .unwrap_or_else(|e| panic!("Failed to execute nvcc: {}", e));

    if !status.success() {
        panic!("Failed to compile CUDA kernels");
    }

    // Tell cargo to link CUDA
    println!("cargo:rustc-link-search=native={}", cuda_path.join("lib/x64").display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
