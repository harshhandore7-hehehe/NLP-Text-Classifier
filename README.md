# 🧙 Magic Vendor NPC — NLP Intent Classifier

> A custom Natural Language Processing neural network built from scratch in Python and PyTorch, designed to act as the "brain" of a grizzled Magic Vendor NPC in an Unreal Engine game. The player types open-ended text — the model reads it, understands the intent, and triggers the correct C++ game logic.

---

## 🎮 Project Context

In most RPGs, NPC dialogue is driven by rigid button menus. This project replaces that with a **free-text input system** — the player types anything they want, and the neural network classifies their intent in real time so the vendor can respond intelligently.

Instead of pressing `[BUY]`, the player might type:

- *"Show me your wares, old man"*
- *"Are you trying to rob me blind?"*
- *"Where did this ancient tome come from?"*

The model reads those sentences and maps them to one of five actionable intents that the Unreal Engine C++ game logic can act on directly.

---

## 🧠 How It Works — Word Embeddings vs Bag-of-Words

Most beginner NLP tutorials use **Bag-of-Words** or **TF-IDF** — techniques that represent text as giant lists of word counts. This project deliberately avoids that approach.

### The Problem with Bag-of-Words

Imagine giving every word in your vocabulary a checkbox. The sentence *"I want to buy a scroll"* becomes a list of 10,000 checkboxes — most unchecked, a few checked. This has three fatal flaws:

- `"buy"` and `"purchase"` are completely unrelated — just two different checkboxes. The model cannot know they mean the same thing.
- Word **order** is destroyed. *"The scroll ate the wizard"* is identical to *"The wizard ate the scroll"*.
- The representation is enormous (10,000+ numbers) but almost entirely empty.

### The Solution — Word Embeddings

This model uses `nn.Embedding` — a PyTorch layer that acts like a learned dictionary. Instead of a checkbox, every word gets a small dense vector of floating-point numbers:

```
"buy"      → [ 0.82, -0.11,  0.44, ... ]   (64 numbers)
"purchase" → [ 0.79, -0.08,  0.41, ... ]   (64 numbers — nearly identical!)
"dragon"   → [-0.90,  0.55, -0.22, ... ]   (64 numbers — very different)
```

Words with similar **meanings** end up with similar **coordinates**. These coordinates are not hand-crafted — they are **learned automatically** during training and improve with every epoch.

---

## 🔁 The Full Pipeline

```
Player types:  "Are you trying to rob me blind?"
      │
      ▼  tokenize()
["are", "you", "trying", "to", "rob", "me", "blind"]
      │
      ▼  vocab.encode()
[4, 12, 88, 3, 201, 17, 55]
      │
      ▼  pad_sequence(max_len=20)
[4, 12, 88, 3, 201, 17, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      │
      ▼  nn.Embedding(vocab_size=291, dim=64)
20 vectors × 64 floats   →   shape: (1, 20, 64)
      │
      ▼  Bidirectional LSTM (hidden=128, layers=2)
Reads left→right AND right→left → single 256-float context vector
      │
      ▼  Linear(256→128) + ReLU
      │
      ▼  Linear(128→5)
[0.03, 0.02, 0.91, 0.02, 0.02]   ←  5 raw scores
      │
      ▼  Softmax + Argmax
intent_haggle  ✓
```

---

## 🏗️ Model Architecture

| Layer | Type | Details |
|---|---|---|
| 1 | `nn.Embedding` | `vocab_size × 64` lookup table. Learns semantic word coordinates. `<PAD>` token is frozen at zero. |
| 2 | `nn.LSTM` | Bidirectional, 2 layers, hidden size 128. Reads the word sequence and compresses it into a single context vector. |
| 3 | `nn.Dropout` | Rate 0.4. Randomly disables neurons during training to prevent memorisation. |
| 4 | `nn.Linear` + ReLU | 256 → 128. Intermediate interpretation layer. |
| 5 | `nn.Linear` | 128 → 5. Outputs one raw score per intent. |

**Total trainable parameters: ~645,000**

### Why Bidirectional LSTM?

A standard LSTM reads left to right. A **bidirectional** LSTM reads the sentence *forwards and backwards simultaneously*, then combines both passes. This gives the model full context from both directions — critical for catching meaning that depends on word position, like *"you must be joking"* at the end of a sentence signalling disbelief rather than curiosity.

---

## 🎯 Intents

| ID | Intent | Example Player Input |
|---|---|---|
| `0` | `intent_buy` | *"Show me your scrolls"*, *"I'll take the healing potion"* |
| `1` | `intent_sell` | *"Take these potions off my hands"*, *"I want to offload some gear"* |
| `2` | `intent_haggle` | *"Are you trying to rob me blind?"*, *"That price is outrageous"* |
| `3` | `intent_lore` | *"Where did this tome come from?"*, *"Is this scripture authentic?"* |
| `4` | `intent_other` | *"What's the weather like?"*, *"asdfghjkl"* |

---

## 📊 Training Results

The model trains for 200 epochs with the Adam optimiser and a `ReduceLROnPlateau` scheduler that halves the learning rate when loss plateaus.

```
Epoch [  1/200]  Loss: 1.6133  Accuracy: 20.0%
Epoch [ 10/200]  Loss: 0.8014  Accuracy: 65.0%
Epoch [ 20/200]  Loss: 0.3198  Accuracy: 94.0%
Epoch [ 50/200]  Loss: 0.0630  Accuracy: 99.0%
Epoch [100/200]  Loss: 0.0008  Accuracy: 100.0%
Epoch [200/200]  Loss: 0.0024  Accuracy: 100.0%
```

### Sample Inference Output

```
INPUT      : "Are you trying to rob me blind?"
PREDICTION : INTENT_HAGGLE
CONFIDENCE : 100.0%

  intent_haggle   1.0000  ##############################
  intent_buy      0.0000
  intent_sell     0.0000
  intent_lore     0.0000
  intent_other    0.0000
```

---

## 📦 ONNX Export for Unreal Engine

After training, the model is exported to the **ONNX** (Open Neural Network Exchange) format — a universal, runtime-agnostic model format. Unreal Engine's **Neural Network Inference (NNI)** plugin can load `.onnx` files and run them natively in C++ with **zero Python dependency at runtime**.

Two files are produced:

| File | Purpose |
|---|---|
| `magic_vendor_brain.onnx` | The trained model with all weights baked in |
| `magic_vendor_brain_config.json` | Vocabulary map and config so C++ can tokenise player text identically to Python |

### Unreal Engine C++ Integration (Sketch)

```cpp
// 1. Load the model once — e.g., in BeginPlay()
UNeuralNetwork* VendorBrain = NewObject<UNeuralNetwork>();
VendorBrain->Load(TEXT("/Game/AI/magic_vendor_brain.onnx"));

// 2. On player text submission — tokenise, encode, pad using VocabTable
TArray<float> Input = TokenizeAndEncode(PlayerInput, VocabTable, /*MaxLen=*/20);
VendorBrain->SetInputFromArrayCopy(Input);
VendorBrain->Run();

// 3. Read the 5 output scores and dispatch the winning intent
TArray<float> Scores = VendorBrain->GetOutputTensor(0, ...);
int32 IntentID = ArgMax(Scores);
DispatchVendorIntent(static_cast<EVendorIntent>(IntentID));
```

The `_config.json` file contains the full vocabulary (word → integer ID) and sequence length so your C++ tokeniser produces byte-for-byte identical input tensors to what the Python training pipeline used.

---

## 🚀 Getting Started

### Requirements

```
Python 3.10+
torch
numpy
onnx
```

### Install & Run

```bash
git clone https://github.com/YOUR_USERNAME/magic-vendor-nlp.git
cd magic-vendor-nlp
pip install torch numpy onnx
python magic_vendor_nlp.py
```

The script is fully self-contained. It will auto-install `onnx` if it isn't present, train the model, run inference tests, and export both output files to the working directory.

---

## 🔧 Hyperparameters

| Parameter | Value | What It Controls |
|---|---|---|
| `MAX_SEQ_LEN` | 20 | Max words per player sentence. Longer = truncated, shorter = padded. |
| `EMBEDDING_DIM` | 64 | How many floats represent each word's meaning. |
| `HIDDEN_DIM` | 128 | Internal LSTM memory size. Larger = more capacity, slower training. |
| `BATCH_SIZE` | 16 | Training examples processed per gradient step. |
| `EPOCHS` | 200 | Full passes through the dataset. |
| `LEARNING_RATE` | 0.001 | Step size for weight updates. Adam adjusts this automatically. |
| `DROPOUT_RATE` | 0.4 | Fraction of neurons disabled each step to prevent overfitting. |

---

## 🐛 Known Issues Fixed

| Error | Root Cause | Fix Applied |
|---|---|---|
| `TypeError: verbose` | `verbose` removed from `ReduceLROnPlateau` in PyTorch ≥ 2.2 | Removed the argument |
| `ModuleNotFoundError: onnxscript` | PyTorch 2.x defaults to new dynamo exporter which requires `onnxscript` | Set `dynamo=False` to use stable TorchScript path |
| `ModuleNotFoundError: onnx` | `onnx` is a separate package not bundled with PyTorch | Auto-install bootstrap via `subprocess` at module load |
| `do_constant_folding` deprecation | Removed from `torch.onnx.export` in PyTorch ≥ 2.6 | Removed the argument |
| `intent_haggle` misclassification | Words like `"gold"`, `"joking"`, `"hundred"` had no haggle signal in training data | Added 7 targeted haggle training examples |
| JSON integer key coercion | `json.dump()` silently converts `int` keys to strings | Pre-converted `LABEL_TO_INTENT` keys to `str` explicitly |

---

## 📁 File Structure

```
magic-vendor-nlp/
│
├── magic_vendor_nlp.py              # Full training + inference + export script
├── magic_vendor_brain.onnx          # Exported model (generated on run)
├── magic_vendor_brain_config.json   # Vocab map + config for C++ (generated on run)
└── README.md
```

---

## 💡 Extending the Model

**More training data** — The single biggest improvement you can make. 200–300 examples per intent will dramatically improve generalisation to novel player phrasing.

**Pre-trained embeddings** — Replace the random `nn.Embedding` with GloVe or FastText vectors as the starting point. The model will already understand synonyms before training begins.

**More intents** — Add `intent_inspect` (examining an item), `intent_quest` (asking about rumours), or `intent_threaten` by following the same pattern: add labelled examples to `RAW_DATA` and increment `NUM_INTENTS`.

**Confidence threshold** — In C++, if `max(Scores) < 0.6`, fall back to a generic *"Speak plainly, traveller"* dialogue line rather than guessing a low-confidence intent.

---

## 📄 License

MIT — free to use in commercial and personal game projects.
