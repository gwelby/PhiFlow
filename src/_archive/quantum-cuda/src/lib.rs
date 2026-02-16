use rustacuda::prelude::*;
use rustacuda::memory::DeviceBuffer;
use std::error::Error;
use std::ffi::CString;

#[derive(Debug, Clone, Copy)]
pub struct QuantumFieldElement {
    pub amplitude: f64,
    pub phase: f64,
    pub frequency: f64,
    pub coherence: f64,
}

impl Default for QuantumFieldElement {
    fn default() -> Self {
        Self {
            amplitude: 1.0,
            phase: 0.0,
            frequency: 432.0, // Ground state
            coherence: 1.0,
        }
    }
}

pub struct QuantumField<'a> {
    context: Context,
    stream: Stream,
    module: Module,
    evolve_kernel: Function<'a>,
    crystal_kernel: Function<'a>,
    device_field: DeviceBuffer<QuantumFieldElement>,
    size: usize,
}

impl<'a> QuantumField<'a> {
    pub fn new(size: usize) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        // Load PTX module
        let ptx = CString::new(include_str!(concat!(env!("OUT_DIR"), "/quantum_kernels.ptx")))?;
        let module = Module::load_from_string(&ptx)?;
        
        // Get kernel functions
        let evolve_kernel = module.get_function("evolve_quantum_field")?;
        let crystal_kernel = module.get_function("crystallize_consciousness")?;
        
        // Allocate device memory
        let mut device_field = unsafe { DeviceBuffer::uninitialized(size)? };
        
        // Initialize with default values
        let host_data = vec![QuantumFieldElement::default(); size];
        device_field.copy_from(&host_data)?;
        
        Ok(Self {
            context,
            stream,
            module,
            evolve_kernel,
            crystal_kernel,
            device_field,
            size,
        })
    }
    
    pub fn evolve(&mut self, time_step: f64) -> Result<(), Box<dyn Error>> {
        let grid_size = (self.size as u32 + 255) / 256;
        let block_size = 256;
        
        unsafe {
            launch!(self.evolve_kernel<<<grid_size, block_size, 0, self.stream>>>(
                self.device_field.as_device_ptr(),
                self.size as u32,
                time_step
            ))?;
        }
        
        self.stream.synchronize()?;
        Ok(())
    }
    
    pub fn crystallize(&mut self, intensity: f64) -> Result<(), Box<dyn Error>> {
        let grid_size = (self.size as u32 + 255) / 256;
        let block_size = 256;
        
        unsafe {
            launch!(self.crystal_kernel<<<grid_size, block_size, 0, self.stream>>>(
                self.device_field.as_device_ptr(),
                self.size as u32,
                intensity
            ))?;
        }
        
        self.stream.synchronize()?;
        Ok(())
    }
    
    pub fn get_field(&self) -> Result<Vec<QuantumFieldElement>, Box<dyn Error>> {
        let mut host_data = vec![QuantumFieldElement::default(); self.size];
        unsafe {
            self.device_field.copy_to(&mut host_data)?;
        }
        Ok(host_data)
    }
}

impl<'a> Drop for QuantumField<'a> {
    fn drop(&mut self) {
        self.stream.synchronize().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_field() -> Result<(), Box<dyn Error>> {
        let mut field = QuantumField::new(32)?;
        
        // Test ground state evolution
        field.evolve(0.1)?;
        let array = field.get_field()?;
        assert!(array.iter().all(|x| x.amplitude.is_finite()));
        
        // Test crystal resonance
        field.crystallize(0.5)?;
        let array = field.get_field()?;
        assert!(array.iter().all(|x| x.amplitude.is_finite()));
        
        Ok(())
    }
}
