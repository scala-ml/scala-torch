// Fixture generator only: not a dependency of scala-torch runtime, build or published jars.
use candle_transformers::models::stable_diffusion::unet_2d_blocks;
mod reference_vae;
use candle::{DType,Device,Tensor};
use candle_nn::VarBuilder;
use reference_vae::{AutoEncoderKL,AutoEncoderKLConfig};
use std::io::Write;
fn save(output: &std::path::Path,name: &str,value: &Tensor) -> anyhow::Result<()> {
    let values = value.flatten_all()?.to_vec1::<f32>()?;
    let mut file = std::fs::File::create(output.join(format!("{name}-f32.bin")))?;
    for v in &values { file.write_all(&v.to_le_bytes())?; }
    println!("{name}: shape={:?} first={:?}",value.dims(),&values[..4]);
    Ok(())
}
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let output = std::path::Path::new(&args[2]);
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("diffusion_pytorch_model.safetensors")],DType::F32,&device)? };
    let config = AutoEncoderKLConfig { block_out_channels: vec![32,64], ..Default::default() };
    let model = AutoEncoderKL::new(vb,3,3,config)?;
    for (batch,height,width) in [(1,6,10),(1,7,11),(2,8,8)] {
        let values: Vec<f32> = (0..batch*3*height*width).map(|i| ((i as f64)*0.13).sin() as f32).collect();
        // Candle 0.9.1 CPU VAE batched mid-block differs from independent rows.
        // Construct fresh tensors, avoiding both batched kernels and sliced layouts.
        let rows: Vec<Tensor> = values.chunks(3*height*width).map(|row| {
            model.reference_moments(&Tensor::from_vec(row.to_vec(),(1,3,height,width),&device)?)
        }).collect::<candle::Result<_>>()?;
        save(output,&format!("encode-b{batch}-{height}x{width}"),&Tensor::cat(&rows,0)?)?;
    }
    for (batch,height,width) in [(1,3,5),(1,4,4),(2,3,5)] {
        let values: Vec<f32> = (0..batch*4*height*width).map(|i| ((i as f64)*0.17).cos() as f32).collect();
        let rows: Vec<Tensor> = values.chunks(4*height*width).map(|row| {
            model.decode(&Tensor::from_vec(row.to_vec(),(1,4,height,width),&device)?)
        }).collect::<candle::Result<_>>()?;
        save(output,&format!("decode-b{batch}-{height}x{width}"),&Tensor::cat(&rows,0)?)?;
    }
    Ok(())
}
