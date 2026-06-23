//! Local embedding provider backed by candle (pure Rust).
//!
//! Compiled only with `--features semantic`. The default build pulls in none
//! of this and keeps the binary self-contained. Model weights are local
//! safetensors (fetched once into `~/.engram/models/` or user-provided), so
//! inference is fully offline after the first run.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use std::path::Path;
use tokenizers::Tokenizer;

/// Produces fixed-dim, L2-normalized embeddings for text.
pub trait EmbeddingProvider: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dim(&self) -> usize;
    fn model_id(&self) -> &str;
}

/// BERT sentence embedder running on candle/CPU.
pub struct CandleBertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    dim: usize,
    model_id: String,
}

impl CandleBertEmbedder {
    /// Load from a local directory containing `config.json`, `tokenizer.json`,
    /// and `model.safetensors`.
    pub fn from_local(dir: &Path, model_id: &str) -> Result<Self> {
        let device = Device::Cpu;
        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(dir.join("config.json"))
                .with_context(|| format!("read config.json in {}", dir.display()))?,
        )?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)
            .context("load tokenizer.json")?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DTYPE, &device)
                .context("mmap model.safetensors")?
        };
        let dim = config.hidden_size;
        let model = BertModel::load(vb, &config).context("load BertModel")?;
        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            model_id: model_id.to_string(),
        })
    }
}

impl EmbeddingProvider for CandleBertEmbedder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            let enc = self
                .tokenizer
                .encode(*t, true)
                .map_err(anyhow::Error::msg)?;
            // batch=1, no padding → mean over all tokens, no attention mask needed.
            let ids = Tensor::new(enc.get_ids(), &self.device)?.unsqueeze(0)?; // [1, seq]
            let type_ids = ids.zeros_like()?;
            let n_tokens = ids.dim(1)?;
            let hidden = self.model.forward(&ids, &type_ids, None)?; // [1, seq, hidden]
            let mean = (hidden.sum(1)? / n_tokens as f64)?; // [1, hidden]
            let v: Vec<f32> = mean.squeeze(0)?.to_vec1()?;
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            out.push(if norm > 0.0 {
                v.iter().map(|x| x / norm).collect()
            } else {
                v
            });
        }
        Ok(out)
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }
}

/// Ensure model files exist in `dir`; fetch from the HuggingFace Hub on first
/// run when missing. Air-gapped users provide the files (or `model_path`) and
/// no network is touched.
pub fn ensure_model(model_id: &str, dir: &Path) -> Result<()> {
    let needed = ["config.json", "tokenizer.json", "model.safetensors"];
    if needed.iter().all(|f| dir.join(f).exists()) {
        return Ok(());
    }
    std::fs::create_dir_all(dir).with_context(|| format!("create model dir {}", dir.display()))?;
    let api = hf_hub::api::sync::Api::new().context("init hf-hub api")?;
    let repo = api.model(model_id.to_string());
    for f in needed {
        if dir.join(f).exists() {
            continue;
        }
        let cached = repo
            .get(f)
            .with_context(|| format!("download {f} for {model_id}"))?;
        std::fs::copy(&cached, dir.join(f))
            .with_context(|| format!("copy {f} into {}", dir.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires local model in ~/.engram/models; run manually with --features semantic"]
    fn embeds_and_normalizes() {
        let dir = dirs::home_dir()
            .unwrap()
            .join(".engram/models/all-MiniLM-L6-v2");
        ensure_model("sentence-transformers/all-MiniLM-L6-v2", &dir).unwrap();
        let e =
            CandleBertEmbedder::from_local(&dir, "sentence-transformers/all-MiniLM-L6-v2").unwrap();
        let v = e.embed(&["hello world"]).unwrap();
        assert_eq!(v[0].len(), e.dim());
        let norm: f32 = v[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-3,
            "must be L2-normalized, got {norm}"
        );
    }
}
