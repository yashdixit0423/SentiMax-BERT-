# 🎯 Amazon Product Review Sentiment Analysis

> Fine-tuned BERT transformer model for classifying Amazon product reviews into **Positive**, **Neutral**, or **Negative** sentiments.

---

## 📌 Overview

This project fine-tunes a pre-trained **BERT (bert-base-uncased)** model on Amazon product review data to perform 3-class sentiment classification. It uses a PyTorch training pipeline with Hugging Face Transformers and includes a Gradio-based web interface for real-time inference.

| Label | Rating | Numeric |
|-------|--------|---------|
| Negative | ≤ 2 stars | 0 |
| Neutral | 3 stars | 1 |
| Positive | ≥ 4 stars | 2 |

---

## 📊 Model Performance

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.73 | 0.45 | 0.55 |
| Neutral | 0.48 | 0.37 | 0.42 |
| Positive | 0.97 | 0.99 | 0.98 |

> The model achieves strong performance on the Positive class (F1: 0.98), with room for improvement on minority classes (Negative and Neutral) through data balancing techniques.

---

## 🗂️ Project Structure

```
sentiment-analysis/
│
├── sentiment.ipynb          # Main notebook: preprocessing, training & evaluation
├── saved_model/             # Fine-tuned BERT model weights & tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer_config.json
├── 1429_1.csv               # Amazon product reviews dataset
└── README.md
```

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Deep Learning**: PyTorch
- **NLP**: Hugging Face Transformers (`bert-base-uncased`)
- **Data**: pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: scikit-learn
- **UI**: Gradio

---

## 🚀 Usage

### Run the Notebook

Open `sentiment.ipynb` in Jupyter and run all cells. The notebook walks through:
1. Data loading & cleaning
2. Label engineering
3. BERT tokenization
4. PyTorch Dataset/DataLoader setup
5. Model training
6. Saving & loading the model
7. Inference & evaluation
8. Gradio web UI

---

## 🔌 Deployable Inference Function

Once the model is trained and saved, use the following function to predict sentiment on any review text. Load the model once and call `predict_sentiment()` as many times as needed.

**Input:** A raw review string  
**Output:** `"Positive"`, `"Neutral"`, or `"Negative"`

```python
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load saved model
model = BertForSequenceClassification.from_pretrained("./saved_model")
tokenizer = BertTokenizer.from_pretrained("./saved_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_sentiment(review_text):
    encoding = tokenizer(
        review_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return label_map[predicted_class]

# Example
print(predict_sentiment("This product is amazing!"))   # → Positive
print(predict_sentiment("It was okay, not great."))    # → Neutral
print(predict_sentiment("Worst purchase ever."))        # → Negative
```

---

## 🔍 Data Preprocessing

1. **Column selection**: Kept `reviews.text`, `reviews.rating`, and `reviews.title`
2. **Null removal**: Dropped rows with any missing values
3. **Text combination**: Concatenated title + review body into a single `combined_text` field for richer context
4. **Label mapping**: Converted star ratings → 3-class numeric labels
5. **Train/test split**: 80/20 split with `random_state=0`

---

## 🧠 Model Architecture & Training

- **Base model**: `bert-base-uncased` (110M parameters)
- **Classification head**: Linear layer → 3 output logits
- **Tokenization**: Max length 128 tokens, with padding & truncation
- **Optimizer**: AdamW (`lr=2e-5`, `eps=1e-8`, `weight_decay=0.01`)
- **Scheduler**: Linear warmup over 10% of total training steps
- **Epochs**: 3
- **Batch size**: 16
- **Loss**: CrossEntropy
