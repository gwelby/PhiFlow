use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use rustacuda::function::Function;
use num_complex::Complex64;
use ndarray::{Array3, Array4};
use std::error::Error;
use std::ffi::CString;
use crate::quantum::quantum_nccl::QuantumNcclAccelerator;

const BLOCK_SIZE: u32 = 256;

pub struct QuantumCudaAccelerator {
    context: Context,
    module: Module,
    stream: Stream,
    nccl: Option<QuantumNcclAccelerator>,
}

impl QuantumCudaAccelerator {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        
        // Get first device
        let device = Device::get_device(0)?;
        
        // Create CUDA context
        let context = Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, 
            device
        )?;
        
        // Create CUDA stream
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        // Load PTX module
        let module_data = CString::new(include_str!("../cuda/quantum_kernels.ptx"))?;
        let module = Module::load_from_string(&module_data)?;
        
        // Initialize NCCL if multiple GPUs are available
        let device_count = Device::get_device_count()?;
        let nccl = if device_count > 1 {
            Some(QuantumNcclAccelerator::new(0, device_count as i32)?)
        } else {
            None
        };

        Ok(Self {
            context,
            module, 
            stream,
            nccl,
        })
    }

    pub fn transform_quantum_state(&self, time_field: &Array4<Complex64>) -> Result<Array3<Complex64>, Box<dyn Error>> {
        if let Some(nccl) = &self.nccl {
            // Use NCCL accelerator for multi-GPU computation
            nccl.transform_quantum_state(time_field)
        } else {
            let (t, h, w, c) = time_field.dim();
            
            // Allocate device memory
            let mut d_time = unsafe { 
                DeviceBox::new(&time_field.as_slice().unwrap()).unwrap()
            };
            
            let mut d_freq = unsafe {
                DeviceBox::new_zeroed::<Complex64>((h * w * c) as usize).unwrap()
            };
            
            // Calculate grid dimensions
            let block_size = BLOCK_SIZE;
            let grid_size = ((h * w * c) as u32 + block_size - 1) / block_size;
            
            // Launch quantum FFT kernel
            let fft = self.module.get_function("quantum_fft")?;
            
            unsafe {
                launch!(
                    fft<<<grid_size, block_size, 0, self.stream>>>(
                        d_time.as_device_ptr(),
                        d_freq.as_device_ptr(),
                        t as u32,
                        h as u32, 
                        w as u32,
                        c as u32
                    )
                )?;
            }
            
            // Synchronize stream
            self.stream.synchronize()?;
            
            // Copy result back to host
            let mut host_result = Array3::<Complex64>::zeros((h, w, c));
            unsafe {
                d_freq.copy_to(host_result.as_slice_mut().unwrap())?;
            }
            
            Ok(host_result)
        }
    }
    
    pub fn apply_sacred_frequencies(&self, field: &Array3<Complex64>) -> Result<Array3<Complex64>, Box<dyn Error>> {
        if let Some(nccl) = &self.nccl {
            // Use NCCL accelerator for multi-GPU computation
            nccl.apply_sacred_frequencies(field)
        } else {
            let (h, w, c) = field.dim();
            
            // Allocate device memory
            let mut d_field = unsafe {
                DeviceBox::new(field.as_slice().unwrap()).unwrap()
            };
            
            // Calculate grid dimensions  
            let block_size = BLOCK_SIZE;
            let grid_size = ((h * w * c) as u32 + block_size - 1) / block_size;
            
            // Launch sacred frequencies kernel
            let sacred = self.module.get_function("apply_sacred_frequencies")?;
            
            unsafe {
                launch!(
                    sacred<<<grid_size, block_size, 0, self.stream>>>(
                        d_field.as_device_ptr(),
                        h as u32,
                        w as u32, 
                        c as u32
                    )
                )?;
            }
            
            // Synchronize stream
            self.stream.synchronize()?;
            
            // Copy result back to host
            let mut host_result = Array3::<Complex64>::zeros((h, w, c));
            unsafe {
                d_field.copy_to(host_result.as_slice_mut().unwrap())?;
            }
            
            Ok(host_result)
        }
    }
    
    pub fn compute_quantum_coherence(&self, field: &Array3<Complex64>) -> Result<Array3<f64>, Box<dyn Error>> {
        if let Some(nccl) = &self.nccl {
            // Use NCCL accelerator for multi-GPU computation
            nccl.compute_quantum_coherence(field)
        } else {
            let (h, w, c) = field.dim();
            
            // Allocate device memory
            let mut d_field = unsafe {
                DeviceBox::new(field.as_slice().unwrap()).unwrap()
            };
            
            let mut d_coherence = unsafe {
                DeviceBox::new_zeroed::<f64>((h * w * c) as usize).unwrap()
            };
            
            // Calculate grid dimensions
            let block_size = BLOCK_SIZE;
            let grid_size = ((h * w * c) as u32 + block_size - 1) / block_size;
            
            // Launch coherence kernel
            let coherence = self.module.get_function("quantum_coherence")?;
            
            unsafe {
                launch!(
                    coherence<<<grid_size, block_size, 0, self.stream>>>(
                        d_field.as_device_ptr(),
                        d_coherence.as_device_ptr(),
                        h as u32,
                        w as u32,
                        c as u32
                    )
                )?;
            }
            
            // Synchronize stream
            self.stream.synchronize()?;
            
            // Copy result back to host
            let mut host_result = Array3::<f64>::zeros((h, w, c));
            unsafe {
                d_coherence.copy_to(host_result.as_slice_mut().unwrap())?;
            }
            
            Ok(host_result)
        }
    }
}

impl Drop for QuantumCudaAccelerator {
    fn drop(&mut self) {
        // Context is automatically dropped when it goes out of scope
    }
}
