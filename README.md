# Cross-Lingual Natural Language Inference (English → Hebrew)

**Author:** Gil Leibovici\
**Course:** Natural Language Processing, Reichman University (2025)\
📄 Full technical details are available in the attached **project report
(PDF)**.

------------------------------------------------------------------------

## 📌 Project Overview

This project explores **cross-lingual Natural Language Inference (NLI)**
in a challenging and under-researched language pair:\
**English Premise → Hebrew Hypothesis**.

The goal is to determine whether a Hebrew sentence is an **entailment**,
**contradiction**, or **neutral** with respect to an English sentence.

Real-world applications:
- Cross-language question answering
- International news analysis
- Global customer support

The project evaluates:
- Machine translation pipelines
- Prompt-based LLM inference
- Multilingual sentence embeddings
- Ensemble learning strategies

------------------------------------------------------------------------

## 📊 Dataset

**Dataset:** HebNLI\
Derived from **MultiNLI** and machine-translated into Hebrew.

### Translation Variants

-   **Original:**
    -   Premise: English
    -   Hypothesis: Hebrew
-   **Translate-Hypothesis:**
    -   Premise: English
    -   Hypothesis: English
-   **Translate-Premise:**
    -   Premise: Hebrew
    -   Hypothesis: Hebrew

### Translation Quality (LaBSE Cosine Similarity)

-   Hebrew → English: **0.8878**
-   English → Hebrew: **0.8762**

------------------------------------------------------------------------

## 🧠 Models Used

-   **mBERT** (`bert-base-multilingual-cased`)
-   **mDeBERTa-v3**
-   **XLM-RoBERTa Large**
-   **FLAN-T5 Large** (Prompting)
-   **Multilingual SBERT**
-   **DictaBERT** (Hebrew)

------------------------------------------------------------------------

## ⚙️ Methods

-   Direct Cross-Lingual Inference (EN → HE)
-   Translation-Based Inference (HE → EN, EN → HE)
-   Fine-Tuning with AdamW + fp16
-   Prompt-Based Zero-Shot / Few-Shot Learning
-   Sentence Embedding Classification
-   Ensemble Voting with Confidence Tie-Breaking

------------------------------------------------------------------------

## 📈 Results

### Fine-Tuned Models (Test Set)

-   **mBERT**
    -   Accuracy: 80.7%
    -   Macro-F1: 80.6%
-   **mDeBERTa-v3**
    -   Accuracy: 89.8%
    -   Macro-F1: 89.7%
-   **XLM-RoBERTa**
    -   Accuracy: 91.5%
    -   Macro-F1: 91.4%

### Ensemble Performance

-   **Heterogeneous Ensemble**
    -   Accuracy: 90.9%
    -   Macro-F1: 90.8%
-   **Homogeneous Ensemble**
    -   Accuracy: **91.97%**
    -   Macro-F1: **91.89%**

### Prompt-Based (FLAN-T5)

-   Zero-shot: \~80% Macro-F1
-   Few-shot: \~79--80% Macro-F1

------------------------------------------------------------------------

## ✅ Key Takeaways

-   Fine-tuning multilingual models yields top performance
-   Translation-based pipelines remain strong baselines
-   Ensembles outperform individual classifiers
-   Prompt-based inference is flexible but less accurate
-   Sentence embeddings underperform discriminative transformers

------------------------------------------------------------------------

## 🚀 Future Work

-   Improve Hebrew translation using instruction-tuned LLMs
-   Expand ensembles with additional diverse models
-   Explore contrastive multilingual embedding training
-   Extend the system to additional low-resource languages

------------------------------------------------------------------------

## 🔁 Reproducibility

-   Implemented in **Google Colab**
-   Built with **Hugging Face Transformers**
-   Cached translations for deterministic results
-   Modular training and evaluation pipelines

------------------------------------------------------------------------

## 📄 Citation

    Leibovici, G. (2025). Cross-lingual Entailment Detection for English–Hebrew.
    Final Project Report, Reichman University.
