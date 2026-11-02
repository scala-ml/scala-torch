use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use std::io::Write;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let directory = std::path::Path::new(&args[1]);
    let cfg: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(directory.join("config.json"))?)?;
    let config = cfg.into_config(false);
    let device = Device::Cpu;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[directory.join("model.safetensors")], DType::F32, &device)? };
    let model = Llama::load(vb, &config)?;
    let ids = [1u32, 42, 17, 321];
    let mut file = std::fs::File::create(&args[2])?;
    for count in 1..=ids.len() {
        let mut cache = Cache::new(false, DType::F32, &config, &device)?;
        let input = Tensor::new(&ids[..count], &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0, &mut cache)?.flatten_all()?.to_vec1::<f32>()?;
        println!("prefix={} logits={} argmax={}", count, logits.len(), logits.iter().enumerate().max_by(|a,b|a.1.total_cmp(b.1)).unwrap().0);
        for value in logits { file.write_all(&value.to_le_bytes())?; }
    }
    Ok(())
}
