# 📚 Long Document Summarization with Fine-Tuned DistilBART

This project fine-tunes the [`sshleifer/distilbart-xsum-12-6`](https://huggingface.co/sshleifer/distilbart-xsum-12-6) model on the [CNN/DailyMail dataset](https://huggingface.co/datasets/cnn_dailymail) for **abstractive text summarization of long documents**. It includes evaluation using ROUGE and compares pretrained and fine-tuned performance.

---

## 🧠 Model Summary

- **Base Model**: `sshleifer/distilbart-xsum-12-6`  
  - A distilled version of BART (`6 encoder`, `6 decoder` layers).
  - Pretrained on the [XSum dataset](https://huggingface.co/datasets/xsum) (short summaries, ~23 words).
- **Target Dataset**: CNN/DailyMail (long summaries, 3-4 sentences).
- **Goal**: Adapt XSum-trained DistilBART for long-document summarization tasks.

---

## 🗂 Project Structure

```
Summarization-Model-for-Long-Documents/
│
├── data/
│   └── cnn_dailymail (loaded from HuggingFace Datasets)
├── models/
│   └── distilbart_fine_tuned/
│       ├── model_final/
│       └── model_epoch/
|
├── Abstractive_Summarization.ipynb
|
├── README.md
├── LICENSE.txt
└── requirements.txt

```

---

## ⚙️ Installation & Setup

- Python 3.9 or later

Steps:  

1) Download and install Miniconda from [here](https://www.anaconda.com/docs/main#quick-command-line-install)

2) Create a new enviroment using the following command:
```bash
$ conda create -n hugging-face-env python=3.9
```

3) Activate the enviroment:
```bash
$ conda activate hugging-face-env
```

4) Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run Pretrained Model
Use pretrained DistilBART on a CNN article:
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")
model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-xsum-12-6")
```

### 2. Fine-Tune on CNN/DailyMail
The training script fine-tunes using HuggingFace `Accelerate`:
- Max source length: 512
- Target length: 128
- Batch size: 16
- Optimizer: `AdamW`
- LR: `2.2e-5`
- Scheduler: `Linear`
- Epochs: 1

Intermediate checkpoints are saved every ~1/4 of an epoch.

### 3. Evaluate Fine-Tuned Model
- Reload model from disk
- Compute ROUGE score on test samples
- Compare performance to pretrained

---

## 📊 Evaluation

| Metric     | Pretrained (`xsum`) | Fine-Tuned (`CNN`) |
|------------|---------------------|--------------------|
| ROUGE-1    | ~25%                | ~38.5%             |
| ROUGE-2    | ~8%                 | ~17.1%             |
| ROUGE-L    | ~19%                | ~27.2%             |

> Fine-tuned model generates **longer and more contextually appropriate summaries** for CNN articles.

---

## 📁 Checkpoints

Saved under `models/distilbart_fine_tuned/`:
- `model_epoch_quarter/` — quarter epoch checkpoint
- `model_epoch/` — full epoch
- `model_final/` — final weights and tokenizer
- `learning_rate.txt` — stored learning rate used

---

## 📌 Notes

- Evaluation metric: `evaluate.load("rouge")` (with stemming).
- All summaries generated using `num_beams=4`, `max_length=128`, `early_stopping=True`.
- Uses `Accelerator` for hardware abstraction and mixed precision training.

## 📜 License

This project is licensed under the **Apache License 2.0**.
