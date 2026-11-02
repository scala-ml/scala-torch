// Append to an unchanged copy of Candle0.9.1 stable_diffusion/vae.rs.
// Return the existing encoder/quant output before distribution construction; no arithmetic changes.
impl AutoEncoderKL {
    pub fn reference_moments(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.encoder.forward(xs)?;
        match &self.quant_conv {
            None => Ok(xs),
            Some(quant_conv) => quant_conv.forward(&xs),
        }
    }
}
