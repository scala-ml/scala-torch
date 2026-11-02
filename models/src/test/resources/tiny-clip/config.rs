// Append to an unchanged copy of Candle0.9.1 stable_diffusion/clip.rs.
// This only exposes configuration of the existing model; no arithmetic changes.
pub fn tiny_config(activation: Activation) -> Config {
    Config {
        vocab_size: 1000,
        embed_dim: 32,
        activation,
        intermediate_size: 37,
        max_position_embeddings: 77,
        pad_with: None,
        num_hidden_layers: 5,
        num_attention_heads: 4,
        projection_dim: 32,
    }
}
