"""
Utility script to generate comprehensive Jupyter Notebooks for the project.
"""

import json
from pathlib import Path

def create_notebook(cells, output_path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Created notebook: {output_path}")

def make_cell(cell_type, source):
    if isinstance(source, list):
        lines = [s if s.endswith("\n") else s + "\n" for s in source]
    else:
        lines = [s + "\n" for s in source.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        "outputs": [] if cell_type == "code" else None,
        "execution_count": None if cell_type == "code" else None
    }

# 1. 01_dataset_analysis.ipynb
cells_01 = [
    make_cell("markdown", [
        "# Hallucination Detection for Language Models",
        "## Notebook 01: Dataset Analysis & Exploration",
        "",
        "This notebook explores the primary benchmark dataset: **HaluEval (Question Answering subset)**.",
        "We inspect the raw schema, verify class distribution, examine text lengths, and establish stratified train/val/test splits."
    ]),
    make_cell("code", [
        "import sys",
        "from pathlib import Path",
        "sys.path.append('..')",
        "",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from src.data_loader import HaluEvalDataLoader, load_config",
        "from src.preprocessing import clean_text",
        "%matplotlib inline"
    ]),
    make_cell("markdown", [
        "### 1. Load Dataset Splits"
    ]),
    make_cell("code", [
        "config = load_config('../config.yaml')",
        "loader = HaluEvalDataLoader(config=config)",
        "train_df, val_df, test_df = loader.load_or_prepare_splits()",
        "print(f'Train samples: {len(train_df)}')",
        "print(f'Val samples:   {len(val_df)}')",
        "print(f'Test samples:  {len(test_df)}')",
        "train_df.head(3)"
    ]),
    make_cell("markdown", [
        "### 2. Verify Class Balance"
    ]),
    make_cell("code", [
        "print('Train class distribution:')",
        "print(train_df['label'].value_counts(normalize=True))",
        "print('\nTest class distribution:')",
        "print(test_df['label'].value_counts(normalize=True))"
    ]),
    make_cell("markdown", [
        "### 3. Text Length Distribution Analysis"
    ]),
    make_cell("code", [
        "train_df['ctx_words'] = train_df['context'].apply(lambda x: len(x.split()))",
        "train_df['ans_words'] = train_df['answer'].apply(lambda x: len(x.split()))",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4))",
        "sns.histplot(train_df['ctx_words'], bins=30, ax=axes[0], color='#2b5c8f')",
        "axes[0].set_title('Reference Context Word Count')",
        "sns.histplot(train_df['ans_words'], bins=20, ax=axes[1], color='#d95f02')",
        "axes[1].set_title('Generated Answer Word Count')",
        "plt.tight_layout()",
        "plt.show()"
    ])
]

# 2. 02_baselines.ipynb
cells_02 = [
    make_cell("markdown", [
        "# Hallucination Detection for Language Models",
        "## Notebook 02: Baseline Methods Benchmark",
        "",
        "This notebook evaluates independent baseline methods:",
        "1. **Baseline 1:** TF-IDF Lexical Cosine Similarity",
        "2. **Baseline 2:** Dense Semantic Similarity (`all-MiniLM-L6-v2`)",
        "3. **Baseline 3:** Natural Language Inference (`nli-deberta-v3-xsmall`)",
        "4. **Baseline 4:** Combined Lexical + Semantic"
    ]),
    make_cell("code", [
        "import sys",
        "sys.path.append('..')",
        "import pandas as pd",
        "from experiments.run_baselines import run_baselines",
        "from src.evaluation import print_metrics_table"
    ]),
    make_cell("markdown", [
        "### Run Baselines and Evaluate on Test Set"
    ]),
    make_cell("code", [
        "results = run_baselines('../config.yaml')"
    ]),
    make_cell("markdown", [
        "### Inspect Baselines Metrics"
    ]),
    make_cell("code", [
        "import json",
        "with open('../results/metrics/baselines_metrics.json') as f:",
        "    bm = json.load(f)",
        "pd.DataFrame(bm).T[['accuracy', 'precision', 'recall', 'f1_score', 'macro_f1', 'roc_auc']]"
    ])
]

# 3. 03_proposed_method.ipynb
cells_03 = [
    make_cell("markdown", [
        "# Hallucination Detection for Language Models",
        "## Notebook 03: Proposed Multi-Signal Detector & Ablation Study",
        "",
        "This notebook trains and analyzes the proposed **Lightweight Multi-Signal Detector**.",
        "We also execute the feature group ablation study to quantify the contribution of each signal component."
    ]),
    make_cell("code", [
        "import sys",
        "sys.path.append('..')",
        "import pandas as pd",
        "import matplotlib.pyplot as plt",
        "from experiments.run_proposed_model import run_proposed_model",
        "from experiments.run_ablation import run_ablation",
        "from src.detector import LightweightHallucinationDetector",
        "%matplotlib inline"
    ]),
    make_cell("markdown", [
        "### 1. Train and Benchmark Proposed Detector"
    ]),
    make_cell("code", [
        "proposed_res = run_proposed_model('../config.yaml')"
    ]),
    make_cell("markdown", [
        "### 2. Learned Feature Importances & Directionality"
    ]),
    make_cell("code", [
        "detector = LightweightHallucinationDetector.load('../models/hallucination_detector.joblib')",
        "fi_df = detector.get_feature_importances()",
        "fi_df"
    ]),
    make_cell("markdown", [
        "### 3. Feature Group Ablation Study"
    ]),
    make_cell("code", [
        "ablation_df = run_ablation('../config.yaml')",
        "ablation_df[['configuration', 'num_features', 'accuracy', 'f1_score', 'delta_f1']]"
    ])
]

# 4. 04_error_analysis.ipynb
cells_04 = [
    make_cell("markdown", [
        "# Hallucination Detection for Language Models",
        "## Notebook 04: Diagnostic Error Analysis & Failure Patterns",
        "",
        "This notebook examines prediction errors (False Positives and False Negatives)",
        "to understand the boundary conditions and failure modes of lightweight hallucination detection."
    ]),
    make_cell("code", [
        "import sys",
        "sys.path.append('..')",
        "import pandas as pd",
        "import seaborn as sns",
        "import matplotlib.pyplot as plt",
        "%matplotlib inline"
    ]),
    make_cell("markdown", [
        "### 1. Load Diagnostic Error Dataset"
    ]),
    make_cell("code", [
        "err_df = pd.read_csv('../results/predictions/error_analysis.csv')",
        "print('Prediction distribution by error category:')",
        "print(err_df['error_category'].value_counts())"
    ]),
    make_cell("markdown", [
        "### 2. Inspect Sample False Positives (False Alarms)"
    ]),
    make_cell("code", [
        "fps = err_df[err_df['error_category'].str.contains('False Positive')]",
        "for idx, row in fps.head(2).iterrows():",
        "    print(f'Context: {row[\"context\"][:120]}...')",
        "    print(f'Answer:  {row[\"answer\"]}')",
        "    print(f'P(Hallucination): {row[\"hallucination_probability\"]}')",
        "    print(f'Explanation: {row[\"diagnostic_explanation\"]}')",
        "    print('-' * 60)"
    ]),
    make_cell("markdown", [
        "### 3. Inspect Sample False Negatives (Missed Hallucinations)"
    ]),
    make_cell("code", [
        "fns = err_df[err_df['error_category'].str.contains('False Negative')]",
        "for idx, row in fns.head(2).iterrows():",
        "    print(f'Context: {row[\"context\"][:120]}...')",
        "    print(f'Answer:  {row[\"answer\"]}')",
        "    print(f'P(Hallucination): {row[\"hallucination_probability\"]}')",
        "    print(f'Explanation: {row[\"diagnostic_explanation\"]}')",
        "    print('-' * 60)"
    ])
]

create_notebook(cells_01, "notebooks/01_dataset_analysis.ipynb")
create_notebook(cells_02, "notebooks/02_baselines.ipynb")
create_notebook(cells_03, "notebooks/03_proposed_method.ipynb")
create_notebook(cells_04, "notebooks/04_error_analysis.ipynb")
print("All 4 notebooks successfully generated!")
