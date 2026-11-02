use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Llama, LlamaConfig};
use std::io::Write;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let directory = std::path::Path::new(&args[1]);
    let output = std::path::Path::new(&args[2]);
    let cfg: LlamaConfig = serde_json::from_str(&std::fs::read_to_string(directory.join("config.json"))?)?;
    let config = cfg.into_config(false);
    let device = Device::Cpu;
    let tokenizer = tokenizers::Tokenizer::from_file(directory.join("tokenizer.json")).unwrap();
    let prompt = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";
    let prompt_ids = tokenizer.encode(prompt, false).unwrap().get_ids().to_vec();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[directory.join("model.safetensors")], DType::F32, &device)? };
    let model = Llama::load(vb, &config)?;
    let mut file = std::fs::File::create(output.join("logits-f32.bin"))?;
    let mut cache = Cache::new(true, DType::F32, &config, &device)?;
    let mut chunk = prompt_ids.clone();
    let mut position = 0;
    let mut ids = Vec::new();
    for step in 0..16 {
        let input = Tensor::new(chunk.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, position, &mut cache)?.flatten_all()?.to_vec1::<f32>()?;
        if step < 4 { for value in &logits { file.write_all(&value.to_le_bytes())?; } }
        let next = logits.iter().enumerate().max_by(|a,b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0))).unwrap().0 as u32;
        println!("step={step} next={next}");
        ids.push(next);
        position += chunk.len();
        chunk = vec![next];
        if next == 2 { break; }
    }
    let csv = |v: &[u32]| v.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",");
    std::fs::write(output.join("ids.txt"), format!("{}\n{}\n", csv(&prompt_ids),csv(&ids)))?;
    let text = tokenizer.decode(&ids, true).unwrap();
    std::fs::write(output.join("text.txt"), &text)?;
    println!("continuation={text:?}");
    Ok(())
}
