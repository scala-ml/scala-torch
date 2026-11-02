// Independent CPU oracle: Candle 0.9.1 plus candle-erf.patch (GEGLU erf semantics).
// See docs/research/pretrained-unet.md. No Python runtime.
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::stable_diffusion::unet_2d::{BlockConfig, UNet2DConditionModel, UNet2DConditionModelConfig};
use std::io::Write;
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let output = std::path::Path::new(&args[2]);
    let device = Device::Cpu;
    let config = UNet2DConditionModelConfig {
        blocks: vec![
            BlockConfig { out_channels: 32, use_cross_attn: None, attention_head_dim: 8 },
            BlockConfig { out_channels: 64, use_cross_attn: Some(1), attention_head_dim: 8 },
        ],
        cross_attention_dim: 32,
        ..Default::default()
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("diffusion_pytorch_model.safetensors")], DType::F32, &device)? };
    let model = UNet2DConditionModel::new(vb, 4, 4, false, config)?;
    for (height, width, time) in [(3, 5, 7.5), (4, 4, 0.0), (4, 6, 19.0)] {
        let xs: Vec<f32> = (0..4*height*width).map(|i| ((i as f64)*0.17).sin() as f32).collect();
        let cs: Vec<f32> = (0..3*32).map(|i| ((i as f64)*0.13).cos() as f32).collect();
        let x = Tensor::from_vec(xs, (1,4,height,width), &device)?;
        let c = Tensor::from_vec(cs, (1,3,32), &device)?;
        let y = model.forward(&x, time, &c)?.flatten_all()?.to_vec1::<f32>()?;
        let mut file = std::fs::File::create(output.join(format!("output-{height}x{width}-f32.bin")))?;
        for v in &y { file.write_all(&v.to_le_bytes())?; }
        println!("shape=1x4x{height}x{width} timestep={time} output_count={} first={:?}",y.len(),&y[..4]);
    }
    Ok(())
}
