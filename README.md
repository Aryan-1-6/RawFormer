<div align="center">

```
       ██████╗  █████╗ ██╗    ██╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗██████╗
       ██╔══██╗██╔══██╗██║    ██║██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔══██╗
       ██████╔╝███████║██║ █╗ ██║█████╗  ██║   ██║██████╔╝██╔████╔██║█████╗  ██████╔╝
       ██╔══██╗██╔══██║██║███╗██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ██╔══██╗
       ██║  ██║██║  ██║╚███╔███╔╝██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗██║  ██║
       ╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝ ╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
```

### *No PyTorch. No autograd. No shortcuts. Just raw GPU math.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![CuPy](https://img.shields.io/badge/CuPy-GPU--native-green?style=flat-square)](https://cupy.dev)
[![Framework](https://img.shields.io/badge/Framework-None-red?style=flat-square)]()

</div>

---

> **"Wanted to understand transformers at the deepest level possible — so threw away every framework and built on concepts directly."**

RawFormer is a fully working **decoder-only transformer** trained on GPU, with zero dependency on PyTorch, TensorFlow, or any autograd engine. Every weight update, every backward pass, every kernel call — written from scratch in CuPy.

This is a fully functional implementation trained on real data (Penn Treebank),
achieving meaningful perplexity while keeping every computation transparent.

---

## 🔬 Purpose

Purpose of this repo is for anyone with a keen interest in LLM fundamentals and wants to understand and most importantly experiment with free hands on high-scale Transformer models. This project gives you full control — down to the lowest-level math — to explore, modify, and extend the model.

---

## 🧠 The Architecture

```
Input Token IDs  (B, T)
       │
       ▼
 ┌─────────────────────────────────────────────┐
 │          Token Embeddings  (B, T, D)         │
 │       + Sinusoidal Positional Encoding       │
 └─────────────────────┬───────────────────────┘
                       │
          ┌────────────▼────────────┐
          │                         │  × N layers
          │     DecoderBlock        │
          │                         │
          │  ┌─────────────────┐    │
          │  │  LayerNorm      │    │
          │  │  ↓              │    │
          │  │  QKV Attention  │    │  ← Fused matmul
          │  │  (causal mask)  │    │
          │  │  ↓              │    │
          │  │  + Residual     │    │  ← Pre-LN style
          │  └─────────────────┘    │
          │  ┌─────────────────┐    │
          │  │  LayerNorm      │    │
          │  │  ↓              │    │
          │  │  FFN (4x dim)   │    │
          │  │  ↓              │    │
          │  │  + Residual     │    │
          │  └─────────────────┘    │
          └────────────┬────────────┘
                       │
               Final LayerNorm
                       │
                    LM Head
               (weight-tied ↕)
                       │
                   Softmax
                       │
             Token Probabilities  (B, T, V)
```

---

## ✍️ Debug gradient flow

Getting to debug gradients open up many architectural issues and might foster model development at scale. This is what the attention backward pass looks like as an example (approx.):

```python
def backward(self, dvalues):
    # dV = softmax_weights.T @ d_attn_out
    dV = np.matmul(self.attn_weights.transpose(0,2,1), dvalues)

    # d_softmax = d_attn_out @ V.T
    d_attn_weights = np.matmul(dvalues, self.V.transpose(0,2,1))
    self.softmax.backward(d_attn_weights)
    d_scores = self.softmax.dinputs
    d_scores *= (self.mask[:, :T, :T] > -1e8)   # zero masked positions

    # dQ = d_scores @ K * scale
    # dK = d_scores.T @ Q * scale
    dQ = np.matmul(d_scores, self.K) * self.scale
    dK = np.matmul(d_scores.transpose(0,2,1), self.Q) * self.scale

    # Reverse the QKV split from forward — concatenate and backprop as one
    d_qkv = np.concatenate([dQ, dK, dV], axis=-1)
    self.qkv_layer.backward(d_qkv)

    return self.qkv_layer.dinputs
```
No abstractions. Just math.

---

## 📁 Project Structure

```
RawFormer/
│
├── rawformer/                  # 🧱 Core model package
│   ├── layers.py               #    Linear layer + LayerNorm (+ backward)
│   ├── activations.py          #    ReLU, Leaky ReLU, Softmax (+ backward)
│   ├── loss.py                 #    Cross-entropy loss (+ backward)
│   ├── optimizer.py            #    Adam with warmup, SGD
│   ├── attention.py            #    Fused QKV causal self-attention
│   ├── feedforward.py          #    FFN block (4× expansion)
│   ├── blocks.py               #    DecoderBlock (Pre-LN residuals)
│   ├── decoder.py              #    Full Decoder model
│   └── __init__.py             #    Clean public API
│
├── data/
│   └── dataloader.py           # 📦 PTB loader, windowing, flatten
│
├── config.py                   # ⚙️  All hyperparameters in one place
├── train.py                    # 🏋️  Training loop + early stopping
├── evaluate.py                 # 📊  Test perplexity
├── generate.py                 # 💬  Autoregressive text generation
├── checkpoint.py               # 💾  Save / load weights (.pkl)
├── requirements.txt
└── .gitignore
```

---

## 🚀 Quickstart

### 1. Install
```bash
pip install cupy-cuda12x nltk gensim scikit-learn
python -c "import nltk; nltk.download('punkt_tab')"
```
> Change `cupy-cuda12x` to match your CUDA version: `cupy-cuda11x`, `cupy-cuda117`, etc.

### 2. Dataset
Download the Penn Treebank and place files here:
```
ptbdataset/
    ptb.train.txt
    ptb.test.txt
    ptb.valid.txt
```

### 3. Configure
Open `config.py` and set your model size and training config.

### 4. Train
```bash
python train.py
```

### 5. Evaluate
```bash
python evaluate.py
```

### 6. Generate text
```bash
python generate.py --prompt "the stock market" --max_len 30
```

---

## ⚙️ Recommended Config

```python
# config.py — tuned for PTB 10k sentences
EMBD_DIM      = 256
NUM_LAYERS    = 4
CONTEXT       = 128
BATCH_SIZE    = 64
LEARNING_RATE = 0.0003
WARMUP_STEPS  = 200
EPOCHS        = 50
```

<!-- ### Expected Training Curve

```
Epoch   1  │████░░░░░░░░░░░░░░░░│  val ppl ~8000   (random ≈ vocab size)
Epoch  10  │████████░░░░░░░░░░░░│  val ppl ~600
Epoch  30  │████████████████░░░░│  val ppl ~300
Epoch  50  │████████████████████│  val ppl ~150-200
``` -->

---

## 🔧 Learning Highlights

**Fused QKV Projection**
Instead of 3 separate matmuls for Q, K, V — one single projection then split.
Fewer GPU kernel launches = faster training.

**Pre-LN Residual Connections (GPT-2 style)**
```
x = x + Attention(LayerNorm(x))   ← norm before, not after
x = x + FFN(LayerNorm(x))
```
Focused on stabilizing training

**Weight Tying**
The LM head shares weights with the token embedding matrix.
Fewer parameters, better generalization, standard in modern LMs.

**GPU-native Embedding Gradients**
```python
cupyx.scatter_add(self.dembeddings, flat_ids, flat_grads)
```
Avoids slow unbuffered `np.add.at` — stays entirely on GPU.

**Adam with Linear Warmup**
LR ramps from 0 → peak over the first N steps, then stays constant.
Prevents early large gradient updates from corrupting random initialization.

---

## 📦 Dependencies

```
cupy       — GPU array library (drop-in NumPy for CUDA)
nltk       — tokenization
gensim     — optional Word2Vec embedding init
sklearn    — metrics utilities
```

No PyTorch. No TensorFlow. No JAX.

---

## 🗺️ Roadmap

- [ ] Multi-head attention (currently single-head with `n_heads` placeholder)
- [ ] Top-k / nucleus sampling in generation
- [ ] Gradient norm clipping
- [ ] BPE tokenizer support
- [ ] Mixed precision (float16 forward, float32 backward)
- [ ] Learning rate decay schedule
- [ ] Multi-GPU training (data parallelism)

---

## 🤝 Collab

Any type of contribution or suggestion is welcomed.

---

<!-- ## 📄 License

None — use it, fork it, learn from it. -->


<div align="center">

**Built without frameworks. Understood from first principles.**

*If you found this useful, consider starring the repo ⭐*

</div>
