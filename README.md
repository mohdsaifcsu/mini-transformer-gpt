# MINI-TRANFORMER-GPT
## A minimal, educational purpose GPT-style Transformer built from scratch in Python and PyTorch
This project implements a lightweight GPT-style language model from scratch, including:

- Character level Transformer
- GPT-2 BPE tokenizer
- A mini GPT-2–style architecture
- Training loop
- Text generation with multinomial sampling
- Google Colab compatible training
- Local + Colab interoperability for checkpoints (.pth)

This codebase is designed for deep learning education, LLM internals learning, and research experiments on small-scale models.

---

## Project Structure

```bash
mini-transformer-gpt/
│
├── char_gpt/                 # Character level tokenizer + model
│   ├── checkpoints/          # Model weights (.pth)
│   ├── data/                 # Dataset files
│   └── src/                  # Model + tokenizer + train + inference scripts
│
├── gpt2_bpe_tiny/            # Tiny GPT-2 BPE model (4 Layer)
│   ├── checkpoints/          # Model weights (.pth)
│   ├── data/                 # Dataset files (BPE-encoded)
│   ├── vocab/                # BPE merges + vocab.json
│   └── src/                  # Model + tokenizer + train + inference scripts
│
├── gpt2_bpe_mini/            # Mini GPT-2 BPE model (8 Layer)
│   ├── checkpoints/          # Model weights (.pth)
│   ├── data/                 # Dataset files (BPE-encoded)
│   ├── vocab/                # BPE merges + vocab.json
│   └── src/                  # Model + tokenizer + train + inference scripts
│
├── requirements.txt
└── README.md
```

---

## Features
### From-scratch Transformer implementation

Attention, softmax scaling, positional embeddings, feed-forward layers, etc.

### GPT-2 BPE tokenizer (custom implementation)

- Vocabulary JSON
- Merges file
- Encoding/decoding

### Mini GPT-2 model

A highly reduced GPT-2 architecture:
- Multi-head causal self-attention
- Feed-forward MLP
- LayerNorm
- Learned positional embeddings

### Training Pipeline

- Works locally (CPU)
- Works remotely (Colab GPU - T4)
- Saves checkpoints
- Can resume training

### Inference Engine

Generates text with:
- Temperature
- Top-k
- Multinomial sampling
- EOS-token stopping

---

## Installation
```bash
git clone https://github.com/mohdsaifcsu/mini-transformer-gpt.git
cd mini-transformer-gpt
pip install -r requirements.txt
```

---

## Run Inference (Local)

Example:
```bash
python -m gpt2_bpe_mini.src.inference_mini
```
You will be prompted:
```bash
Enter your prompt: Once upon a time
```
Output:
```bash
===== RESULT =====
Onceuponatime ....... <generated text>
==================
```

---

## Train the Model (Local)
```bash
python -m gpt2_bpe_mini.src.train_gpt2_mini
```
This trains for several epochs depending on your dataset size.

---

## Train in Google Colab (Highly Recommended for fast training)

1. Upload your project zip
2. Unzip inside Colab
3. Add project path to sys.path
4. Run training:

```bash
!python -m gpt2_bpe_mini.src.train_gpt2_mini
```
All checkpoints will be saved to:
```bash
gpt2_bpe_mini/checkpoints/
```

---

## Using Checkpoints Between Local & Colab
### Download from Colab:
```bash
from google.colab import files
files.download("gpt2_bpe_mini/checkpoints/model_epoch2.pth")
```
### Then place in local folder:
```bash
gpt2_bpe_mini/checkpoints/
```
#### Local inference automatically loads it.

---

## Goals of This Project

- Understand Transformer internals
- Learn how GPT models tokenize text
- Practice training small language models
- Build a smooth CPU <> GPU workflow

---

## Future Improvements

- Add top-p (nucleus sampling)
- Add weight initialization exactly like GPT-2
- Add AdamW with learning-rate warmup
- Add model quantization
- Add dataset loader for custom text corpora
- Add more notebooks (training visualization, tokenizer analysis)


---

##  Author

**Mohd Saif**  
Master’s Student - Colorado State University  
GitHub: https://github.com/mohdsaifcsu

---

##  License

This project is for **academic and educational use** only.

---



























