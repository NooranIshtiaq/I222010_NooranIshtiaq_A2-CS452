# 🧾 Legal Clause Semantic Similarity – Assignment 2  

**Name:** Nooran Ishtiaq  
**Roll No:** 22i-2010  
**Section:** DS-B  

---

## 📘 Project Overview  
This project develops and evaluates **three baseline NLP architectures** to identify **semantic similarity between legal clauses**.  
The models are built **from scratch (no pre-trained transformers)** and compared based on their ability to capture legal clause semantics.  

---

## 📂 Dataset Description  
The dataset consists of multiple CSV files containing labeled legal clauses.  

**Columns:**
- `clause_text` – raw legal clause text  
- `clause_type` – categorical label for clause type  

**Pair Construction:**
- Positive pairs → same clause type  
- Negative pairs → different clause types  

**Dataset Split:**
| Split | Purpose | Percentage |
|--------|-----------|------------|
| Train | Model training | 70% |
| Validation | Model tuning | 15% |
| Test | Final evaluation | 15% |

---

## ⚙️ Preprocessing  
- Lowercased text and tokenized via regex  
- Vocabulary size: 30,000 (min frequency = 2)  
- Sequence length: 128 (padded/truncated)  

---

## 🧠 Baseline Models  

### 1️⃣ Siamese Self-Attention Encoder  
A Siamese architecture with shared **Self-Attention Encoders** that learn contextual representations for each clause.  

**Key Components:**  
- Embedding dimension = 200  
- Multi-head attention (4 heads)  
- Layer normalization + feedforward  
- Feature fusion: concatenation, abs diff, element-wise product, cosine similarity  
- MLP classifier: [256 → 128 → 1]  
- Dropout = 0.2  

**Training Configuration:**  
- Optimizer: Adam (lr=0.001)  
- Epochs: 50  
- Batch size: 64  
- Loss: BCEWithLogits  
- Device: GPU (if available)  

**Results:**  
| Metric | Score |
|--------|--------|
| Accuracy | 0.9438 |
| Precision | 0.9479 |
| Recall | 0.9391 |
| F1-score | 0.9435 |
| ROC-AUC | 0.9850 |
| PR-AUC | 0.9838 |

🟢 **Strengths:** Captures long-range dependencies and shows balanced precision/recall.  
🔴 **Weaknesses:** Slightly limited for very long or complex legal clauses.  

---

### 2️⃣ Siamese TextCNN  
A lightweight convolutional baseline using multi-kernel CNNs for feature extraction.  

**Architecture:**  
- Embedding dim: 200  
- Conv1D with kernel sizes [3, 4, 5] (128 channels each)  
- Global max pooling + concatenation  
- MLP: [256 → 128 → 1]  
- Dropout = 0.2  

**Training Setup:**  
- Optimizer: Adam (lr=1e-3)  
- Epochs: 4–8  
- Loss: BCEWithLogitsLoss  

**Results:**  
| Metric | Score |
|--------|--------|
| Accuracy | 0.9225 |
| F1-score | 0.9224 |
| ROC-AUC | 0.9720 |
| PR-AUC | 0.9691 |

🟢 **Strengths:** Simple, fast, efficient.  
🔴 **Weaknesses:** Relies mainly on surface-level lexical patterns.  

---

### 3️⃣ Siamese BiLSTM  
A recurrent neural baseline leveraging sequential context for clause similarity.  

**Architecture:**  
- Embedding dim: 200  
- BiLSTM hidden size: 128 (per direction)  
- MLP: [256 → 128 → 1]  
- Dropout: 0.2  

**Training Setup:**  
- Optimizer: Adam (lr=1e-3)  
- Batch size: 64  
- Epochs: 4–8  

**Results:**  
| Metric | Score |
|--------|--------|
| Accuracy | 0.9371 |
| F1-score | 0.9363 |
| ROC-AUC | 0.9801 |
| PR-AUC | 0.9770 |

🟢 **Strengths:** Models word order and syntax effectively.  
🔴 **Weaknesses:** Slow due to sequential processing; limited long-range memory.  

---

## 📊 Comparative Analysis  

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Training Time | Params |

| Siamese Self-Attention | **0.9438** | **0.9479** | **0.9391** | **0.9435** | **0.9850** | **0.9838** | Moderate | ~2.1M |

| Siamese BiLSTM | 0.9371 | 0.9412 | 0.9315 | 0.9363 | 0.9801 | 0.9770 | Slowest | ~2.8M |

| Siamese TextCNN | 0.9225 | 0.9278 | 0.9172 | 0.9224 | 0.9720 | 0.9691 | **Fastest** | ~1.4M |

---

## 🧩 Key Observations  
- **Best Overall:** Siamese Self-Attention Encoder (highest accuracy and F1).  
- **Best Efficiency:** TextCNN (lightweight and fast).  
- **Balanced:** BiLSTM (captures syntax well but slower).  

🧠 **Semantic Generalization:**  
The attention-based model generalizes best across paraphrased or reordered legal clauses.  

🚀 **Future Improvements:**  
- Combine BiLSTM + Attention (hybrid).  
- Use subword tokenization.  
- Explore domain-specific embeddings.  

---

## 🖥️ Run Instructions  

```bash
# Run Self-Attention Encoder
python -u baseline1_self_attention.py --data_dir csv --epochs 10

# Run Siamese BiLSTM
python -u baseline2_siamese_bilstm.py --data_dir csv --epochs 10

# Run Siamese TextCNN
python -u baseline3_siamese_textcnn.py --data_dir csv --epochs 10
