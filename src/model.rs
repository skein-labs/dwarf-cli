use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

// ── RMSNorm ────────────────────────────────────────────────────────────

struct RMSNorm {
    scale: Tensor,
    eps: f64,
}

impl RMSNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(dim, "scale")?;
        Ok(Self { scale, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let rms = x_f32
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .affine(1.0, self.eps)?
            .sqrt()?
            .recip()?;
        let normed = x_f32.broadcast_mul(&rms)?;
        Ok(normed
            .to_dtype(x.dtype())?
            .broadcast_mul(&self.scale)?)
    }
}

// ── Rotary Embedding ───────────────────────────────────────────────────

struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::new(t, device)?;

        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;

        let cos_cache = emb.cos()?;
        let sin_cache = emb.sin()?;

        Ok(Self { cos_cache, sin_cache })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos_cache.i(..seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin_cache.i(..seq_len)?.unsqueeze(0)?.unsqueeze(0)?;

        let q_rot = self.rotate_and_apply(q, &cos, &sin)?;
        let k_rot = self.rotate_and_apply(k, &cos, &sin)?;
        Ok((q_rot, k_rot))
    }

    fn rotate_and_apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let neg_x2 = x2.neg()?;
        let rotated = Tensor::cat(&[&neg_x2, &x1], D::Minus1)?;
        let a = x.broadcast_mul(cos)?;
        let b = rotated.broadcast_mul(sin)?;
        Ok(a.add(&b)?)
    }
}

// ── GQA Attention ──────────────────────────────────────────────────────

struct GQAAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rope: RotaryEmbedding,
    n_heads: usize,
    n_kv_heads: usize,
    n_groups: usize,
    head_dim: usize,
}

fn linear_with_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    let bias = vb.get(out_dim, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

impl GQAAttention {
    fn load(cfg: &DwarfConfig, vb: VarBuilder) -> Result<Self> {
        let q_proj = linear_with_bias(cfg.d_model, cfg.n_heads * cfg.head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_with_bias(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_with_bias(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(cfg.n_heads * cfg.head_dim, cfg.d_model, vb.pp("o_proj"))?;
        let rope = RotaryEmbedding::new(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta, vb.device())?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            n_groups: cfg.n_heads / cfg.n_kv_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, t, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, t, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, t, self.n_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let (q, k) = self.rope.apply(&q, &k)?;

        let (k, v) = if self.n_groups > 1 {
            let k = k.repeat(&[1, self.n_groups, 1, 1])?;
            let v = v.repeat(&[1, self.n_groups, 1, 1])?;
            (k, v)
        } else {
            (k, v)
        };

        let scale = (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // ── Causal mask (broadcast to all heads) ──────
        let (_, h, t_attn, _) = attn.dims4()?;

        let mut mask_data = Vec::with_capacity(t_attn * t_attn);
        for i in 0..t_attn {
            for j in 0..t_attn {
                mask_data.push(if j > i { f32::NEG_INFINITY } else { 0f32 });
            }
        }
        let mask = Tensor::from_vec(mask_data, (t_attn, t_attn), attn.device())?
            .reshape((1, 1, t_attn, t_attn))?;

        // Broadcast mask to match attention heads
        let mask = if h > 1 {
            let mut mask_heads = Vec::with_capacity(h);
            for _ in 0..h {
                mask_heads.push(mask.clone());
            }
            Tensor::cat(&mask_heads.iter().collect::<Vec<&Tensor>>(), 1)?
        } else {
            mask
        };

        let attn = (attn + mask)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let attn = attn.to_dtype(v.dtype())?;

        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, t, self.n_heads * self.head_dim))?;
        Ok(self.o_proj.forward(&out)?)
    }
}

// ── SwiGLU FFN ─────────────────────────────────────────────────────────

struct SwiGLUFFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl SwiGLUFFN {
    fn load(cfg: &DwarfConfig, vb: VarBuilder) -> Result<Self> {
        let gate_proj = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("down_proj"))?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        Ok(self.down_proj.forward(&(gate * up)?)?)
    }
}

// ── Transformer Block ──────────────────────────────────────────────────

struct DwarfBlock {
    norm_attn: RMSNorm,
    attn: GQAAttention,
    norm_ffn: RMSNorm,
    ffn: SwiGLUFFN,
}

impl DwarfBlock {
    fn load(cfg: &DwarfConfig, vb: VarBuilder) -> Result<Self> {
        let norm_attn = RMSNorm::load(cfg.d_model, cfg.norm_eps, vb.pp("norm_attn"))?;
        let attn = GQAAttention::load(cfg, vb.pp("attn"))?;
        let norm_ffn = RMSNorm::load(cfg.d_model, cfg.norm_eps, vb.pp("norm_ffn"))?;
        let ffn = SwiGLUFFN::load(cfg, vb.pp("ffn"))?;
        Ok(Self { norm_attn, attn, norm_ffn, ffn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.attn.forward(&self.norm_attn.forward(x)?)?;
        let x = (residual + x)?;
        let residual = &x;
        let x = self.ffn.forward(&self.norm_ffn.forward(&x)?)?;
        Ok((residual + x)?)
    }
}

// ── Config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DwarfConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    #[serde(default = "default_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_eps")]
    pub norm_eps: f64,
}

fn default_theta() -> f64 { 10000.0 }
fn default_eps() -> f64 { 1e-5 }

// ── Full Model ─────────────────────────────────────────────────────────

pub struct DwarfModel {
    embed_tokens: Embedding,
    layers: Vec<DwarfBlock>,
    norm: RMSNorm,
    lm_head_weight: Tensor,
}

impl DwarfModel {
    pub fn load(cfg: &DwarfConfig, vb: VarBuilder) -> Result<Self> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.d_model, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(DwarfBlock::load(cfg, vb.pp(format!("layers.{i}")))?);
        }

        let norm = RMSNorm::load(cfg.d_model, cfg.norm_eps, vb.pp("norm"))?;

        // Weight-tied: lm_head.weight == embed_tokens.weight
        let lm_head_weight = vb
            .get((cfg.vocab_size, cfg.d_model), "lm_head.weight")
            .unwrap_or_else(|_| embed_tokens.embeddings().clone());

        Ok(Self { embed_tokens, layers, norm, lm_head_weight })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
    let (b, t) = input_ids.dims2()?;
    let mut x = self.embed_tokens.forward(input_ids)?;
    for layer in &self.layers {
        x = layer.forward(&x)?;
    }
    let x = self.norm.forward(&x)?;
    
    // Fix: get vocab_size from the weight dimensions
    let vocab_size = self.lm_head_weight.dim(0)?;
    let d_model = self.lm_head_weight.dim(1)?;
    
    // Reshape [B, T, D] → [B*T, D] for 2D matmul
    let x_flat = x.reshape((b * t, d_model))?;
    // [B*T, D] @ [D, V] → [B*T, V]
    let logits_flat = x_flat.matmul(&self.lm_head_weight.t()?)?;
    // [B*T, V] → [B, T, V]
    let logits = logits_flat.reshape((b, t, vocab_size))?;
    
    Ok(logits)
    }
}