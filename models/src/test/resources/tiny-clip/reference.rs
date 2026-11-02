// Unmodified Candle0.9.1 CLIP math, with config.rs appended to expose the tiny configuration.
mod reference_clip;
use candle::{DType,Device,Tensor};
use candle_nn::{Module,VarBuilder};
use reference_clip::{Activation,ClipTextTransformer,tiny_config};
use std::io::Write;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let output = std::path::Path::new(&args[2]);
    let device = Device::Cpu;
    for (name,batch,skip,activation) in [
        ("gelu-b1",1,false,Activation::GeluErf),
        ("gelu-b2",2,false,Activation::GeluErf),
        ("gelu-skip1",1,true,Activation::GeluErf),
        ("quick-b1",1,false,Activation::QuickGelu),
        ("tanh-b1",1,false,Activation::Gelu),
    ] {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")],DType::F32,&device)? };
        let norm = candle_nn::layer_norm(32,1e-5,vb.pp("text_model.final_layer_norm"))?;
        let model = ClipTextTransformer::new(vb,&tiny_config(activation))?;
        let ids: Vec<u32> = (0..batch*77).map(|i| ((i*17+3)%997) as u32).collect();
        // Candle0.9.1's diffusion CLIP mask does not broadcast for batch>1.
        // Evaluate independent rows and concatenate, leaving upstream arithmetic unchanged.
        let mut rows = Vec::new();
        for row in ids.chunks(77) {
            let input = Tensor::from_slice(row,(1,77),&device)?;
            let result = if skip {
                let (_,intermediate) = model.forward_until_encoder_layer(&input,usize::MAX,3)?;
                norm.forward(&intermediate)?
            } else { model.forward(&input)? };
            rows.push(result);
        }
        let result = Tensor::cat(&rows,0)?;
        let values = result.flatten_all()?.to_vec1::<f32>()?;
        let mut file = std::fs::File::create(output.join(format!("{name}-f32.bin")))?;
        for v in &values { file.write_all(&v.to_le_bytes())?; }
        println!("{name}: count={} first={:?}",values.len(),&values[..4]);
    }
    Ok(())
}
