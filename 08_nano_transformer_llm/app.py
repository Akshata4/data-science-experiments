"""Nano Transformer LLM
==========================
A real, from-scratch, decoder-only Transformer language model — the same
architecture family (causal self-attention, autoregressive next-token
prediction) as production LLMs, just ~6 orders of magnitude smaller so it
trains from random initialization on a laptop CPU in under a minute. Built
with PyTorch, no shortcuts: real attention heads, real backprop, real
sampling.

Trained on **Tiny Shakespeare** — the real ~1.1MB corpus of Shakespeare's
plays that Andrej Karpathy's char-rnn/nanoGPT tutorials made famous.

Run:
    streamlit run app.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import theme  # noqa: E402

DATA_PATH = Path(__file__).resolve().parent / "data" / "tinyshakespeare.txt"
torch.manual_seed(1337)


# ---------------------------------------------------------------------------
# Model — a compact nanoGPT-style decoder-only Transformer
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        return wei @ v


class MultiHead(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(), nn.Linear(4 * n_embd, n_embd))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.sa = MultiHead(n_embd, n_head, n_embd // n_head, block_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


@torch.no_grad()
def attention_map(model: NanoGPT, idx: torch.Tensor):
    """Recompute block[-1]'s first attention head's weights for one sequence,
    for visualization — a small, self-contained forward pass replay."""
    x = model.token_emb(idx) + model.pos_emb(torch.arange(idx.size(1)))
    for block in model.blocks[:-1]:
        x = block(x)
    last_block = model.blocks[-1]
    xn = last_block.ln1(x)
    head = last_block.sa.heads[0]
    B, T, C = xn.shape
    k, q = head.key(xn), head.query(xn)
    wei = q @ k.transpose(-2, -1) * C ** -0.5
    wei = wei.masked_fill(head.tril[:T, :T] == 0, float("-inf"))
    return F.softmax(wei, dim=-1)[0].numpy()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load_corpus():
    text = DATA_PATH.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return text, chars, stoi, itos, data


def get_batch(data, block_size, batch_size, split_idx):
    ix = torch.randint(0, split_idx - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@st.cache_resource(show_spinner=True)
def train_model(n_embd: int, n_head: int, n_layer: int, block_size: int, steps: int, batch_size: int = 32, lr: float = 3e-3):
    text, chars, stoi, itos, data = load_corpus()
    vocab_size = len(chars)
    n_train = int(len(data) * 0.9)
    train_data, val_data = data[:n_train], data[n_train:]

    model = NanoGPT(vocab_size, n_embd, n_head, n_layer, block_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = []
    t0 = time.perf_counter()
    for step in range(steps):
        xb, yb = get_batch(train_data, block_size, batch_size, len(train_data))
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 40) == 0 or step == steps - 1:
            with torch.no_grad():
                xv, yv = get_batch(val_data, block_size, batch_size, len(val_data))
                _, val_loss = model(xv, yv)
            history.append({"step": step, "train_loss": loss.item(), "val_loss": val_loss.item()})
    elapsed = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    return model, pd.DataFrame(history), elapsed, n_params, (train_data, val_data)


def encode(s, stoi):
    return [stoi.get(c, 0) for c in s]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def main():
    theme.apply("🔮", "Nano Transformer LLM")
    theme.hero(
        "🔮", "Nano Transformer LLM",
        "A real, from-scratch decoder-only Transformer (causal self-attention, "
        "autoregressive generation) — the same architecture family as production LLMs, "
        "scaled down ~6 orders of magnitude to train on a laptop CPU in under a minute, "
        "on the real Tiny Shakespeare corpus.",
        ["PyTorch", "Self-attention from scratch", "Char-level GPT", "Trains live in-browser"],
    )

    text, chars, stoi, itos, data = load_corpus()

    with st.sidebar:
        st.subheader("📁 Corpus")
        st.metric("Characters", f"{len(text):,}")
        st.metric("Vocabulary size", len(chars))
        st.caption(
            "Tiny Shakespeare — the real ~1.1MB corpus of Shakespeare's plays used in "
            "Andrej Karpathy's char-rnn / nanoGPT tutorials. Character-level tokenization: "
            f"{len(chars)} unique characters, no subword vocabulary needed."
        )

    tabs = st.tabs([f"{icon} {label}" for label, icon in theme.CRISP_DM_PHASES])

    # 1. Business Understanding --------------------------------------------------
    with tabs[0]:
        st.subheader("What this demonstrates")
        st.markdown(
            """
Every modern LLM — GPT, Claude, Llama — is fundamentally the same
architecture idea: a stack of **causal self-attention** blocks trained to
predict the next token, over and over, at massive scale. This project
strips away the scale (billions of parameters, trillions of tokens, GPU
clusters) and keeps the **architecture** — a real multi-head self-attention
Transformer, real backpropagation, real temperature/top-k sampling —
small enough (~160K parameters vs. GPT-3's 175 *billion*) to train from
random weights on a laptop CPU in well under a minute.

**What you'll see is genuinely learned, not scripted:** early in training
the model outputs random characters; by a few hundred steps it has learned
English word shapes, character n-gram statistics, and Shakespeare's
distinctive `CHARACTER:\\n` dialogue formatting — entirely from predicting
one character at a time.

**Success criteria:** falling train/validation loss (cross-entropy over
next-character prediction) and, qualitatively, generated text that *looks*
like Shakespeare even though it's mostly nonsense semantically — exactly
what you'd expect from a model this small trained this briefly.
            """
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="ds-card"><div class="ds-metric-label">Architecture</div>'
                     '<div class="ds-metric-value">Decoder-only Transformer</div></div>', unsafe_allow_html=True)
        c2.markdown('<div class="ds-card"><div class="ds-metric-label">Task</div>'
                     '<div class="ds-metric-value">Next-char prediction</div></div>', unsafe_allow_html=True)
        c3.markdown('<div class="ds-card"><div class="ds-metric-label">Primary metric</div>'
                     '<div class="ds-metric-value">Cross-entropy / perplexity</div></div>', unsafe_allow_html=True)

    # 2. Data Understanding ---------------------------------------------------------
    with tabs[1]:
        st.subheader("The corpus")
        st.code(text[:400], language="text")
        freq = pd.Series(list(text)).value_counts().head(30)
        fig = px.bar(freq, title="Top 30 most frequent characters (includes space, newline)")
        theme.style_fig(fig, 340); fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')

    # 3. Data Preparation ------------------------------------------------------------
    with tabs[2]:
        st.subheader("Character-level tokenization")
        st.markdown(
            f"""
Unlike production LLMs (which use subword tokenizers like BPE), this nano
model tokenizes at the **character** level — the entire vocabulary is just
the {len(chars)} unique characters in the corpus, each mapped to an integer:
            """
        )
        vocab_preview = pd.DataFrame({"char": [repr(c) for c in chars[:20]], "id": list(range(20))})
        st.dataframe(vocab_preview, width='stretch', hide_index=True)
        st.markdown(
            """
**Chronological 90/10 split** — the first 90% of the text trains the model,
the last 10% validates it. (Shuffling would leak future Shakespeare into
training, the same time-ordering discipline as
[project 05](../05_time_series_forecasting).)

**Blocks & batches** — the model never sees the whole corpus at once; each
training step samples a batch of short, fixed-length windows (`block_size`
characters) at random starting positions, each paired with the same window
shifted one character forward as the prediction target — the model learns
to predict character *t+1* from characters *1..t*.
            """
        )
        st.code(
            "x = data[i : i+block_size]      # input window\n"
            "y = data[i+1 : i+block_size+1]  # same window, shifted 1 char = the target",
            language="python",
        )

    # 4. Modeling -----------------------------------------------------------------------
    with tabs[3]:
        st.subheader("Train the model live")
        c1, c2, c3 = st.columns(3)
        n_layer = c1.select_slider("Layers (n_layer)", [1, 2, 3, 4], value=3)
        n_embd = c2.select_slider("Embedding dim (n_embd)", [32, 64, 128], value=64)
        steps = c3.select_slider("Training steps", [100, 200, 400, 600, 1000, 1500], value=600)
        n_head = 4
        block_size = 64

        model, history, elapsed, n_params, (train_data, val_data) = train_model(n_embd, n_head, n_layer, block_size, steps)

        m1, m2, m3 = st.columns(3)
        m1.metric("Parameters", f"{n_params:,}")
        m2.metric("Training time", f"{elapsed:.1f}s")
        m3.metric("Final val loss", f"{history['val_loss'].iloc[-1]:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["step"], y=history["train_loss"], mode="lines", name="train loss", line=dict(color="#6ee7f2")))
        fig.add_trace(go.Scatter(x=history["step"], y=history["val_loss"], mode="lines", name="val loss", line=dict(color="#fb7185")))
        fig.update_layout(title="Cross-entropy loss during training", xaxis_title="step", yaxis_title="loss")
        theme.style_fig(fig, 400)
        st.plotly_chart(fig, width='stretch')
        st.caption(f"Architecture: {n_layer} blocks × {n_head} heads × {n_embd}-dim embeddings, "
                    f"{block_size}-character context window. Random-init loss ≈ ln({len(chars)}) ≈ {np.log(len(chars)):.2f}.")

    # 5. Evaluation -----------------------------------------------------------------------
    with tabs[4]:
        st.subheader("Perplexity & attention")
        model, history, elapsed, n_params, (train_data, val_data) = train_model(64, 4, 3, 64, 600)
        val_ppl = float(np.exp(history["val_loss"].iloc[-1]))
        st.metric("Validation perplexity", f"{val_ppl:.1f}",
                  help="exp(cross-entropy loss) — roughly 'how many characters the model is choosing between, on average, at each step'. Random guessing over 65 characters would be ~65.")

        st.markdown("#### What is one attention head looking at?")
        sample_prompt = st.text_input("Sample text to visualize attention over (≤64 chars)", "ROMEO: wherefore art thou")[:64]
        idx = torch.tensor([encode(sample_prompt, stoi)])
        try:
            wei = attention_map(model, idx)
            fig = px.imshow(wei, x=list(sample_prompt), y=list(sample_prompt), color_continuous_scale="Tealgrn",
                             title="Last block, head 0 — attention weight from each row-character back to each column-character")
            theme.style_fig(fig, 460)
            st.plotly_chart(fig, width='stretch')
            st.caption("Bright cells show which earlier characters this head attends to most when predicting the "
                       "next character — the lower-triangular structure is the causal mask (a character can only "
                       "attend to itself and earlier characters, never the future).")
        except Exception:
            st.info("Type at least 2 characters to visualize attention.")

    # 6. Deployment -----------------------------------------------------------------------
    with tabs[5]:
        st.subheader("Generate text")
        model, history, elapsed, n_params, (train_data, val_data) = train_model(64, 4, 3, 64, 600)
        prompt = st.text_area("Prompt", "ROMEO:", height=80)
        c1, c2, c3 = st.columns(3)
        max_tokens = c1.slider("Length (new characters)", 50, 500, 200, 10)
        temperature = c2.slider("Temperature", 0.1, 2.0, 0.8, 0.1, help="Lower = more repetitive/confident; higher = more random.")
        top_k = c3.slider("Top-k", 1, 65, 20, help="Only sample from the k most likely next characters.")

        if st.button("🔮 Generate", type="primary"):
            idx = torch.tensor([encode(prompt or "\n", stoi)])
            out = model.generate(idx, max_tokens, temperature=temperature, top_k=top_k)
            generated = decode(out[0].tolist(), itos)
            st.markdown("#### Output")
            st.code(generated, language="text")
            st.caption("This is a ~160K-parameter model trained for well under a minute — expect Shakespeare-*flavored* "
                       "gibberish (real word shapes, real dialogue formatting), not coherent plot. That gap between "
                       "'the architecture is right' and 'the model is actually good' is exactly what scale (parameters "
                       "× data × compute) buys you in real LLMs.")

    theme.footer("08 · Nano Transformer LLM")


if __name__ == "__main__":
    main()
