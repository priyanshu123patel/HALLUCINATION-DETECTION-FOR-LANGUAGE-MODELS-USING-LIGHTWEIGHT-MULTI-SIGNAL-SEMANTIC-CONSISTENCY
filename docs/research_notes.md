# Research Notes: Hallucination Detection for Language Models

**Project Title:** Hallucination Detection for Language Models Using Lightweight Multi-Signal Semantic Consistency  
**Academic Focus:** Natural Language Processing (NLP) / Information Verification  
**Author / Team:** College NLP Research Group  
**Target Venue:** Academic Paper, Senior Capstone Presentation, Poster Session, and Live Interactive Demo  

---

## 1. Research Problem

Large Language Models (LLMs) frequently generate content that appears fluent, coherent, and authoritative, but is factually untruthful or unsupported by reference context—a phenomenon known as **hallucination**. 

In high-stakes applications such as automated summarization, question answering, legal briefing, and clinical information extraction, hallucinations pose severe safety and reliability risks. While state-of-the-art approaches often attempt to mitigate hallucinations by fine-tuning massive LLMs (7B–70B parameters) or querying proprietary models as black-box judges ("LLM-as-a-Judge"), these approaches suffer from critical flaws:
1. **Excessive Computational and Environmental Cost:** Prohibitive for deployment on resource-constrained consumer hardware (e.g., laptops with 8 GB RAM / CPU).
2. **Opacity and Hallucination Recursion:** An LLM evaluating another LLM can hallucinate its own evaluation, offering uncalibrated, ungrounded judgements.
3. **Inference Latency:** Multi-second round-trips render real-time verification impractical.

---

## 2. Hypothesis and Research Question

### Primary Research Question:
> *"Can a lightweight combination of lexical similarity, dense semantic embeddings, named entity consistency, and contradiction signals effectively detect hallucinated language-model-generated answers without requiring expensive LLM fine-tuning?"*

### Core Hypothesis ($H_1$):
A composite multi-signal detector combining orthogonal signals—surface lexical grounding (TF-IDF), dense contextual semantic similarity (`all-MiniLM-L6-v2`), directional logical inference (`nli-deberta-v3-xsmall`), and named entity coverage (spaCy NER)—will significantly outperform single-modality baselines (e.g., pure TF-IDF or pure semantic similarity) while preserving strict real-time CPU execution (< 20 ms/sample).

---

## 3. Methodology & Signal Taxonomy

The framework operates by decomposing factual hallucination verification into five complementary, interpretable signals:

```
[Reference Context] + [Generated Candidate Answer]
                      │
   ┌──────────────────┼──────────────────┬──────────────────┐
   ▼                  ▼                  ▼                  ▼
[Lexical TF-IDF]  [Dense Semantic]  [Cross-Encoder NLI]  [spaCy NER]
   │                  │                  │                  │
Cosine Sim        Cosine Sim         Contradiction      Unsupported
Jaccard Overlap   Min/Mean Sent Sim  Entailment / Neut  Ratio / Overlap
   └──────────────────┴──────────────────┴──────────────────┘
                              │
               [Interpretable Logistic Classifier]
               (or Heuristic Weighted Score Formula)
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
           [P(Hallucination)]   [Evidence & Sentence Breakdown]
```

### 3.1 Signal Formulations

1. **Lexical Grounding Signal ($S_{\text{lex}}$):**
   - **TF-IDF Cosine Similarity:** Quantifies exact n-gram surface overlap between context $C$ and answer $A$.
   - **Jaccard Token Overlap:** Ratio of shared unigrams: $\frac{|W_C \cap W_A|}{|W_C \cup W_A|}$.
   - *Strength:* Sensitive to direct copy-grounding.
   - *Limitation:* Blind to paraphrasing and synonymous substitutions.

2. **Dense Semantic Similarity ($S_{\text{sem}}$):**
   - Derived using `sentence-transformers/all-MiniLM-L6-v2` (6 layers, 384 hidden dimensions, 22.7M parameters).
   - Generates normalized sentence embeddings $\mathbf{e}_C, \mathbf{e}_A \in \mathbb{R}^{384}$ via mean pooling.
   - Computes document-level cosine similarity: $\cos(\mathbf{e}_C, \mathbf{e}_A) = \frac{\mathbf{e}_C \cdot \mathbf{e}_A}{\|\mathbf{e}_C\| \|\mathbf{e}_A\|}$.
   - Also tracks sentence-level minimum ($\min_i \cos(\mathbf{e}_C, \mathbf{e}_{s_i})$) and mean similarities.
   - *Strength:* Robust against vocabulary mismatch and syntactic variation.
   - *Limitation:* Can exhibit high semantic similarity to deceptive hallucinations (e.g., substituting one date or entity while retaining topical context).

3. **Natural Language Inference ($S_{\text{NLI}}$):**
   - Utilizes `cross-encoder/nli-deberta-v3-xsmall` (22M parameters) taking paired $(C, A)$ inputs.
   - Predicts directional probabilities:
     $$P(\text{Contradiction} \mid C, A), \quad P(\text{Entailment} \mid C, A), \quad P(\text{Neutral} \mid C, A)$$
   - *Strength:* Directly captures logical contradiction and strict deductive entailment.
   - *Limitation:* Sensitive to truncation if context is excessively long.

4. **Named Entity Consistency ($S_{\text{NER}}$):**
   - Extracts categorical entities $\mathcal{E}_C, \mathcal{E}_A$ using spaCy's `en_core_web_sm`.
   - Flags unsupported entities: $\mathcal{E}_{\text{unsupported}} = \{e \in \mathcal{E}_A \mid e \notin \mathcal{E}_C \text{ and } e \notin C\}$.
   - Computes unsupported entity ratio: $\frac{|\mathcal{E}_{\text{unsupported}}|}{\max(1, |\mathcal{E}_A|)}$.

5. **Sentence-Level Consistency ($S_{\text{sent}}$):**
   - Segments candidate answer into individual sentences $\{s_1, \dots, s_k\}$.
   - Flags suspicious sentences exceeding contradiction threshold ($\tau_c \ge 0.50$) or below semantic threshold ($\tau_s < 0.40$).
   - Returns unsupported sentence ratio: $\frac{\sum \mathbb{I}[s_i \text{ is suspicious}]}{k}$.

---

## 4. Experimental Setup

### 4.1 Benchmark Dataset
- **HaluEval QA Benchmark (`pminervini/HaluEval`):** Standard benchmark comprising 10,000 question-answering examples with balanced 50/50 distribution (5,010 hallucinated, 4,990 supported).
- **Stratified Splits:**
  - Training Set: 3,000 samples (50.1% hallucinated, 49.9% supported)
  - Validation Set: 500 samples (50.2% hallucinated, 49.8% supported)
  - Test Set: 500 samples (50.0% hallucinated, 50.0% supported)

### 4.2 Baselines
1. **Baseline 1 (TF-IDF Similarity):** Logistic Regression trained purely on TF-IDF cosine similarity.
2. **Baseline 2 (Semantic Similarity):** Logistic Regression trained purely on MiniLM-L6-v2 cosine similarity.
3. **Baseline 3 (NLI Signals):** Logistic Regression trained on NLI Entailment, Contradiction, and Neutral probabilities.
4. **Baseline 4 (Lexical + Semantic):** Combination of TF-IDF and MiniLM representations.

### 4.3 Proposed Detector Configurations
- **Learned Combination (Primary):** Standardized $L_2$-regularized Logistic Regression learning weights $\boldsymbol{\beta}$ over all 12 candidate features.
- **Configurable Weighted Score (Heuristic):** Direct formula-based score using expert configurable weights in `config.yaml`.

### 4.4 Feature Ablation Matrix
Leave-one-group-out evaluation:
- Full Proposed Model
- Without Lexical features ($- \text{TF-IDF}$)
- Without Semantic features ($- \text{MiniLM}$)
- Without NLI features ($- \text{NLI}$)
- Without Entity features ($- \text{NER}$)
- Without Sentence-level features ($- \text{Sentence}$)

---

## 5. Academic Limitations and Boundaries

1. **Entity Absence $\ne$ Definite Hallucination:**
   A candidate answer may introduce general nouns or synonyms that the NER model tags as an entity absent from the reference. Hence, entity consistency is used strictly as a regularizing feature, not a deterministic filter.
2. **Scope of Context Grounding:**
   The system assesses *context-dependent* hallucinations (faithfulness to provided context), which is standard in RAG (Retrieval-Augmented Generation) and summarization. It does not verify open-domain world knowledge without a reference text.
3. **Context Truncation:**
   For passages exceeding 256 tokens, sliding-window chunking or truncation is applied to maintain the 8 GB RAM hardware constraint.
