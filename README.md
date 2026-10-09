# Hallucination Detection for Language Models Using Lightweight Multi-Signal Semantic Consistency

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-HaluEval-orange)](https://huggingface.co/datasets/pminervini/HaluEval)

---

## 1. Project Title
**Hallucination Detection for Language Models Using Lightweight Multi-Signal Semantic Consistency**  
*A Senior Capstone / College NLP Research Project & Live Interactive Demonstration*

---

## 2. Abstract
Large Language Models (LLMs) frequently generate plausible yet factually unsupported or contradictory statements, commonly known as hallucinations. Existing detection paradigms often rely on fine-tuning massive generative models (7B+ parameters) or repeatedly querying proprietary commercial LLMs as black-box judges. These methods introduce high computational overhead, multi-second latencies, opacity, and the risk of hallucination recursion. 

In this research project, we investigate whether a composite framework of lightweight, interpretable NLP signals—comprising lexical surface grounding, dense sentence embeddings, cross-encoder natural language inference (NLI), named entity consistency, and sentence-level anomaly decomposition—can effectively detect hallucinations on consumer-grade hardware (standard CPU, 8 GB RAM). Evaluated on the standardized **HaluEval** Question Answering benchmark, our proposed multi-signal detector achieves **96.67% accuracy** and a **0.9669 F1-score** (ROC-AUC: **0.9948**), outperforming single-modality baselines such as pure TF-IDF (57.67% accuracy) and pure semantic similarity (70.00% accuracy) while maintaining sub-millisecond classification latency per sample.

---

## 3. Problem Statement
Generative language models optimized for next-token prediction prioritize fluency and statistical plausibility over strict factual fidelity. Consequently, in Retrieval-Augmented Generation (RAG) and open-ended question answering, models can fabricate relationships, dates, entities, or entire propositions that directly contradict or fail to find support within provided reference texts. 

Detecting such errors without relying on another heavy LLM is a fundamental challenge for resource-constrained applications, where models must run locally or on modest edge servers without dedicated GPUs.

---

## 4. Research Question
> *"Can a lightweight combination of lexical similarity, semantic similarity, entity consistency, and contradiction signals effectively detect hallucinated language-model-generated answers without requiring expensive LLM fine-tuning?"*

---

## 5. Motivation
1. **Hardware Accessibility:** Real-world educational and enterprise deployments frequently operate within strict hardware limits (e.g., quad-core Intel i5, 8 GB RAM, no discrete GPU).
2. **Interpretability:** Professors, domain experts, and auditors require verifiable evidence explaining *why* an answer is deemed hallucinated, rather than an ungrounded probability score from an opaque neural network.
3. **Reproducibility:** A scientific research project must rely on reproducible, open-source computational pipelines rather than proprietary, non-deterministic APIs.

---

## 6. Objectives
1. Implement independent single-modality baselines: Lexical (TF-IDF), Dense Semantic (`all-MiniLM-L6-v2`), and Directional NLI (`nli-deberta-v3-xsmall`).
2. Engineer an interpretable multi-signal feature extraction pipeline capturing surface overlap, embedding alignment, entailment/contradiction probabilities, named entity coverage, and sentence-level consistency.
3. Train an interpretable classifier (Logistic Regression) to learn feature importances and compute exact log-odds contributions for transparent evidence generation.
4. Execute rigorous evaluations on HaluEval, including confusion matrix analysis, ROC curve benchmarking, leave-one-group-out ablation studies, and diagnostic error analysis.
5. Provide a full-stack, real-time Streamlit demonstration interface displaying document- and sentence-level verification.

---

## 7. Proposed Methodology
Our methodology posits that factual faithfulness is multi-faceted:
- **Lexical Overlap ($S_{\text{lex}}$):** Verifies verbatim grounding against the reference context.
- **Dense Semantic Similarity ($S_{\text{sem}}$):** Verifies topical alignment using sentence embeddings.
- **Directional NLI ($S_{\text{nli}}$):** Directly measures whether the reference text logically entails or contradicts the candidate answer.
- **Named Entity Consistency ($S_{\text{ner}}$):** Detects fabricated or ungrounded named entities (people, places, organizations).
- **Sentence-Level Consistency ($S_{\text{sent}}$):** Isolates individual erroneous propositions within multi-sentence answers.

```
Reference Context + Generated Answer
                 │
  ┌──────────────┼──────────────┬──────────────┐
  ▼              ▼              ▼              ▼
TF-IDF         MiniLM         DeBERTa        spaCy
Lexical        Semantic         NLI           NER
  │              │              │              │
  └──────────────┴──────┬───────┴──────────────┘
                        │
          Standardized Logistic Classifier
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
   Binary Prediction        Mathematical Evidence
(SUPPORTED / HALLUCINATED)     & Sentence Flags
```

---

## 8. System Architecture & Project Structure

```
hallucination-detection/
├── README.md                          # Academic project documentation
├── requirements.txt                   # Dependency definitions
├── config.yaml                        # Centralized experiment configuration
├── .gitignore                         # Version control exclusions
│
├── data/
│   ├── raw/                           # Cached raw HaluEval benchmark
│   └── processed/                     # Stratified train/val/test splits & cached features
│
├── models/
│   ├── tfidf_vectorizer.joblib        # Fitted TF-IDF vocabulary & IDF matrix
│   └── hallucination_detector.joblib  # Trained Multi-Signal Detector pipeline
│
├── src/
│   ├── __init__.py                    # Package initialization
│   ├── data_loader.py                 # Dataset loading, schema validation, & splitting
│   ├── preprocessing.py               # Text cleaning & sentence segmentation
│   ├── feature_extraction.py          # High-throughput batched multi-signal featurizer
│   ├── entity_analysis.py             # spaCy Named Entity consistency analyzer
│   ├── nli_analyzer.py                # Cross-Encoder DeBERTa NLI inference module
│   ├── semantic_analyzer.py           # MiniLM-L6-v2 dense semantic similarity analyzer
│   ├── detector.py                    # Lightweight detector with evidence generation
│   ├── evaluation.py                  # Academic metrics calculation & table formatting
│   ├── error_analysis.py              # Diagnostic error profiling (TP/TN/FP/FN)
│   └── visualization.py               # Publication-ready Matplotlib/Seaborn figures
│
├── experiments/
│   ├── smoke_test.py                  # End-to-end pipeline verification test
│   ├── run_baselines.py               # Benchmark runner for Baselines 1, 2, 3, 4
│   ├── run_proposed_model.py          # Proposed detector training and evaluation
│   ├── run_ablation.py                # Leave-one-group-out ablation experiment
│   └── run_all.py                     # Full automated research pipeline orchestrator
│
├── notebooks/
│   ├── 01_dataset_analysis.ipynb      # Dataset exploratory analysis
│   ├── 02_baselines.ipynb             # Baselines execution and comparison
│   ├── 03_proposed_method.ipynb       # Proposed detector training and ablation
│   └── 04_error_analysis.ipynb        # Detailed failure mode investigation
│
├── results/
│   ├── figures/                       # Publication charts (ROC, CM, Ablation, etc.)
│   ├── metrics/                       # JSON & CSV empirical performance reports
│   └── predictions/                   # Out-of-sample test predictions & error analysis
│
├── docs/
│   ├── research_notes.md              # Research problem, hypothesis, and methodology
│   └── results_summary.md             # Automatically generated empirical results summary
│
└── app/
    └── streamlit_app.py               # Interactive research demonstration UI
```

---

## 9. Primary Dataset: HaluEval
- **Dataset Name:** HaluEval (Question Answering subset)
- **Source:** `pminervini/HaluEval` via HuggingFace Datasets (Li et al., 2023)
- **Total Benchmark Samples:** 10,000 paired records
- **Class Balance:** Perfectly balanced (5,010 Hallucinated [50.10%], 4,990 Supported [49.90%])
- **Standardized Schema:**
  - `context`: Ground-truth reference knowledge passage
  - `query`: User prompt / question
  - `answer`: Candidate LM-generated response
  - `label`: Target binary class (`1`: HALLUCINATED, `0`: SUPPORTED)
- **Experimental Splits (Stratified):**
  - **Train:** 1,500 samples (751 Hallucinated, 749 Supported)
  - **Validation:** 300 samples (151 Hallucinated, 149 Supported)
  - **Test:** 300 samples (150 Hallucinated, 150 Supported)

---

## 10. Baseline Implementations
All baselines were trained on the exact same training split and evaluated on the held-out test split:
1. **Baseline 1 (TF-IDF Similarity):** Fits a `TfidfVectorizer` (max 5,000 n-grams, range 1-2) on reference passages and answers, computes cosine similarity, and fits an $L_2$-regularized Logistic Regression classifier.
2. **Baseline 2 (Semantic Similarity):** Encodes contexts and answers using `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings), computes cosine similarity, and fits a Logistic Regression classifier.
3. **Baseline 3 (NLI Signals):** Feeds $(Context, Answer)$ pairs into `cross-encoder/nli-deberta-v3-xsmall` to extract probabilities for `contradiction`, `entailment`, and `neutral`, followed by Logistic Regression.
4. **Baseline 4 (Lexical + Semantic):** Concatenates TF-IDF and MiniLM similarity features.

---

## 11. Proposed Method: Lightweight Multi-Signal Detector
Our proposed detector combines 14 quantitative signals:
- **Lexical:** `tfidf_similarity`, `jaccard_token_overlap`
- **Semantic:** `semantic_similarity`, `min_sentence_semantic_sim`, `mean_sentence_semantic_sim`
- **NLI:** `nli_contradiction_prob`, `nli_entailment_prob`, `nli_neutral_prob`, `max_sentence_contradiction_prob`
- **Entity Consistency:** `entity_unsupported_count`, `entity_unsupported_ratio`, `entity_jaccard_overlap`
- **Sentence-Level / Linguistic:** `unsupported_sentence_ratio`, `answer_length_tokens`

### Two Implementations:
- **Method A (Learned Combination - Primary):** Standardized Logistic Regression that fits feature weights $\beta_j$ with balanced class weighting.
- **Method B (Heuristic Weighted Score):** Configurable weighted formula in `config.yaml` for rule-based comparison.

### Mathematical Evidence Generation (No LLM):
The detector calculates the linear log-odds contribution of each feature:
$$\text{contribution}_j = \beta_j \cdot \left(\frac{x_j - \mu_j}{\sigma_j}\right)$$
This transparently isolates the exact factors driving a prediction (e.g., high contradiction probability $+0.89$, unsupported entity presence $+1.45$).

---

## 12. Evaluation Metrics
Evaluated using standard academic metrics:
- **Accuracy:** Proportion of overall correct classifications
- **Precision:** $\frac{TP}{TP + FP}$ (purity of flagged hallucinations)
- **Recall:** $\frac{TP}{TP + FN}$ (coverage of actual hallucinations)
- **F1-Score & Macro F1:** Harmonic mean balancing both classes
- **ROC-AUC:** Area under the Receiver Operating Characteristic curve
- **False Positive Rate (FPR):** $\frac{FP}{FP + TN}$ (false alarm rate on supported text)
- **False Negative Rate (FNR):** $\frac{FN}{FN + TP}$ (missed hallucinations rate)
- **Inference Latency:** Per-sample inference time in milliseconds

---

## 13. Experimental Setup & Hardware Constraints
- **Operating System:** Windows 11
- **CPU:** Standard Quad-Core Intel i5
- **System Memory:** 8 GB RAM
- **GPU:** CPU fallback execution (no dedicated high-end VRAM required)
- **Frameworks:** PyTorch 2.12 (CPU), HuggingFace Transformers 5.18, spaCy 3.8, scikit-learn 1.7

---

## 14. Empirical Results

### Quantitative Benchmark Comparison Table

| Model / Methodology | Accuracy | Precision | Recall | F1-Score | Macro F1 | ROC-AUC | FPR | FNR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline 1: TF-IDF Similarity** | 57.67% | 0.5891 | 0.5067 | 0.5448 | 0.5746 | 0.6295 | 35.33% | 49.33% |
| **Baseline 2: Semantic Similarity** | 70.00% | 0.6744 | 0.7733 | 0.7205 | 0.6984 | 0.7828 | 37.33% | 22.67% |
| **Baseline 3: NLI (Entailment/Contradiction)** | 74.67% | 0.7176 | 0.8133 | 0.7625 | 0.7455 | 0.8113 | 32.00% | 18.67% |
| **Baseline 4: TF-IDF + Semantic** | 67.67% | 0.6532 | 0.7533 | 0.6997 | 0.6748 | 0.7539 | 40.00% | 24.67% |
| **Proposed Multi-Signal Detector (Learned)** | **96.67%** | **0.9605** | **0.9733** | **0.9669** | **0.9667** | **0.9948** | **4.00%** | **2.67%** |
| Proposed Method: Heuristic Weighted Score | 48.00% | 0.4826 | 0.5533 | 0.5155 | 0.4772 | 0.5464 | 59.33% | 44.67% |

### Key Experimental Findings:
1. **The Multi-Signal Synergy:** The proposed combination outperforms the strongest baseline (NLI) by **+22.0 percentage points** in accuracy and **+20.4 points** in F1-score.
2. **Failure of Pure Lexical Overlap:** TF-IDF achieves only 57.67% accuracy because hallucinations in HaluEval reuse words from the context in false configurations.
3. **Deceptive Semantic Similarity:** Pure dense semantic embeddings achieve 70.00% accuracy, but suffer from high false positive rates (37.33%) because topically relevant hallucinations often have high cosine similarity.
4. **Learned Weights vs Heuristics:** The learned Logistic Regression model vastly outperforms static heuristic weighting (96.67% vs 48.00%), demonstrating the necessity of empirical weight optimization.

---

## 15. Feature Ablation Study

To isolate the contribution of each feature group, we conducted a systematic leave-one-group-out ablation on the test set:

| Configuration | Test Accuracy | F1-Score | Macro F1 | Delta F1 ($\Delta$) |
|:---|:---:|:---:|:---:|:---:|
| **Full Proposed Model** | **96.67%** | **0.9669** | **0.9667** | **0.0000** |
| w/o Lexical Features | 97.00% | 0.9703 | 0.9700 | +0.0034 |
| w/o Semantic Features | 97.67% | 0.9767 | 0.9767 | +0.0098 |
| w/o NLI Features | 97.00% | 0.9699 | 0.9700 | +0.0030 |
| **w/o Entity Features** | **95.33%** | **0.9539** | **0.9533** | **-0.0130** |
| **w/o Sentence-Level Features** | **88.33%** | **0.8845** | **0.8833** | **-0.0824** |

### Ablation Takeaway:
- Removing **Sentence-Level Features** causes the largest drop in F1-score (**-8.24%**), confirming that detecting localized contradictions within sentences is critical.
- Removing **Entity Features** produces a notable **-1.30%** drop, indicating that named entity checking acts as a crucial guardrail against entity fabrication.

---

## 16. Diagnostic Error Analysis
Out of 300 test samples, the proposed model achieved:
- **True Positives (TP):** 146 (48.67%)
- **True Negatives (TN):** 144 (48.00%)
- **False Positives (FP):** 6 (2.00%)
- **False Negatives (FN):** 4 (1.33%)

Detailed case studies are automatically logged to `results/predictions/error_analysis.csv`.
- **False Positive Pattern:** Occurs when a supported answer uses external colloquial phrasing or indirect references, causing spurious contradiction signals or low surface overlap.
- **False Negative Pattern:** Occurs when a single subtle entity is swapped into a context with high surrounding topical overlap, causing the contradiction score to stay below the decision threshold.

---

## 17. Academic Limitations
1. **Context-Dependent Faithfulness:** The framework detects whether an answer is faithful to the *provided reference text*. It does not evaluate open-domain truth against the broader web without a context passage.
2. **NER Entity Boundary Limitations:** spaCy's small model (`en_core_web_sm`) may occasionally miss domain-specific terminology or obscure proper nouns.
3. **Context Truncation:** For extremely long passages (> 256 tokens), text is truncated to satisfy strict RAM limitations on consumer hardware.

---

## 18. Future Work
1. **Coreference Resolution Integration:** Incorporating lightweight fast-coreference to resolve pronouns before entity matching.
2. **Multi-Hop Graph Verification:** Constructing small subject-verb-object knowledge subgraphs from reference context for relational verification.
3. **Quantized Edge Deployment:** Exporting the pipeline to ONNX / INT8 format for sub-5ms mobile and browser execution.

---

## 19. Academic References
1. **Li et al. (2023):** *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models.* Proceedings of EMNLP 2023.
2. **Reimers & Gurevych (2019):** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* Proceedings of EMNLP-IJCNLP 2019.
3. **He et al. (2021):** *DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing.* arXiv:2111.09543.
4. **Honnibal et al. (2020):** *spaCy: Industrial-strength Natural Language Processing in Python.*
5. **Maynez et al. (2020):** *On Faithfulness and Factuality in Abstractive Summarization.* Proceedings of ACL 2020.

---

## 20. Installation & Quickstart Commands

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Dataset Preparation & Validation
```bash
python -m src.data_loader
```

### 3. Run All Research Experiments & Generate Visualizations
```bash
python experiments/run_all.py
```

### 4. Run Individual Baselines or Ablation
```bash
python experiments/run_baselines.py
python experiments/run_ablation.py
```

### 5. Launch Interactive Research Demonstration (Streamlit)
```bash
streamlit run app/streamlit_app.py
```
