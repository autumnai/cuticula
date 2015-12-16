use co::backend::{Backend, BackendConfig};
use co::framework::IFramework;
use co::frameworks::Native;
use co::memory::MemoryType;
use co::tensor::SharedTensor;

/// The Transformer Trait
///
/// Gets implemented for all Transformable Data Types.
/// Allows all Transformable Data Types to get transformed into a `SharedTensor`
pub trait Transformer {

    /// Transforms non-numeric data into a numeric `SharedTensor`
    ///
    /// The shape attribute is used to controll the dimensions/shape of the SharedTensor.
    /// It returns an Error, when the expected capacity (defined by the shape) differs from the
    /// observed one.
    fn transform(&self, shape: Vec<usize>) -> Result<SharedTensor<f32>, TransformerError> {
        let framework = Native::new();
        let hardwares = framework.hardwares();
        let backend_config = BackendConfig::new(framework, hardwares);
        let backend = Backend::new(backend_config).unwrap();
        let mut tensor = SharedTensor::<f32>::new(backend.device(), &shape).unwrap();
        match self.write_into_tensor(tensor.get_mut(backend.device()).unwrap()) {
            Ok(_) => Ok(tensor),
            Err(e) => Err(e)
        }
    }

    /// Transforms the non-numeric data into a numeric `Vec`
    fn transform_to_vec(&self) -> Vec<f32>;

    /// Writes into `SharedTensor`s' data
    fn write_into_tensor(&self, mem: &mut MemoryType) -> Result<(), TransformerError> {
        let data = self.transform_to_vec();
        match mem {
            &mut MemoryType::Native(ref mut mem) => {
                if mem.byte_size() / 4 == data.capacity() {
                    let mut mem_buffer = mem.as_mut_slice::<f32>();
                    for (index, datum) in data.iter().enumerate() {
                        mem_buffer[index] = *datum;
                    }
                    Ok(())
                } else {
                    Err(TransformerError::InvalidShape)
                }
            },
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => panic!()
        }
    }
}

#[derive(Debug, Copy, Clone)]
/// The Transformer Errors
pub enum TransformerError {
    /// When the speficied shape capacitiy differs from the actual capacity of the numeric Vec
    InvalidShape,
    /// When The Image Pixel Buffer can't be converted to a RGB Image
    InvalidRgbPixels,
    /// When The Image Pixel Buffer can't be converted to a RGBA Image
    InvalidRgbaPixels,
    /// When The Image Pixel Buffer can't be converted to a greyscale Image
    InvalidLumaPixels,
    /// When The Image Pixel Buffer can't be converted to a greyscale Alpha Image
    InvalidLumaAlphaPixels,
}
