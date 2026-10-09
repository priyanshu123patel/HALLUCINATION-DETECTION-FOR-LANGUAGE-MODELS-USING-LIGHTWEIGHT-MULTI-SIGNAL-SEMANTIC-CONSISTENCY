"""
Dataset Loader and Preprocessing Pipeline for Hallucination Detection.

Primary Dataset: HaluEval (Question Answering subset)
Source: pminervini/HaluEval (HuggingFace Datasets)
Paper: HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models (Li et al., 2023)

Fields:
- knowledge: Reference context / ground-truth knowledge passage
- question: User question / query
- answer: Generated candidate answer
- hallucination: Label ('yes' -> 1 [HALLUCINATED], 'no' -> 0 [SUPPORTED])
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

from src.preprocessing import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Loads configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HaluEvalDataLoader:
    """
    Handles downloading, schema adaptation, sampling, stratification,
    and train/val/test splitting for HaluEval.
    """

    def __init__(self, config: Optional[dict] = None, config_path: str = "config.yaml"):
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config

        self.ds_config = self.config.get("dataset", {})
        self.random_seed = self.config.get("project", {}).get("random_seed", 42)
        self.raw_dir = Path(self.config.get("paths", {}).get("raw_data_dir", "data/raw"))
        self.processed_dir = Path(self.config.get("paths", {}).get("processed_data_dir", "data/processed"))

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_raw_dataset(self) -> pd.DataFrame:
        """
        Loads HaluEval from HuggingFace Datasets or cached local file.
        """
        ds_name = self.ds_config.get("name", "pminervini/HaluEval")
        subset = self.ds_config.get("subset", "qa_samples")
        split = self.ds_config.get("split", "data")

        logger.info(f"Loading dataset: {ds_name} (subset: '{subset}', split: '{split}')")

        try:
            hf_ds = load_dataset(ds_name, subset, split=split)
            df = hf_ds.to_pandas()
            logger.info(f"Successfully loaded {len(df)} records from HuggingFace Hub.")
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace directly: {e}")
            raw_cache = self.raw_dir / f"{subset}_raw.csv"
            if raw_cache.exists():
                logger.info(f"Loading cached raw dataset from {raw_cache}")
                df = pd.read_csv(raw_cache)
            else:
                raise RuntimeError(f"Could not load dataset from HuggingFace or local cache: {e}")

        # Save raw cache if enabled
        if self.ds_config.get("save_raw", True):
            raw_cache = self.raw_dir / f"{subset}_raw.csv"
            df.to_csv(raw_cache, index=False)
            logger.info(f"Cached raw dataset saved to {raw_cache}")

        return df

    def standardize_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates columns, standardizes field names, maps labels,
        and applies text cleaning.
        """
        ctx_col = self.ds_config.get("text_field_context", "knowledge")
        query_col = self.ds_config.get("text_field_query", "question")
        ans_col = self.ds_config.get("text_field_answer", "answer")
        label_col = self.ds_config.get("label_field", "hallucination")
        mapping = self.ds_config.get("label_mapping", {"yes": 1, "no": 0})

        # Validate that required fields exist in dataset
        required_cols = [ctx_col, ans_col, label_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' missing from dataset. Available columns: {list(df.columns)}"
                )

        standardized = pd.DataFrame()
        standardized["context"] = df[ctx_col].astype(str).apply(clean_text)
        standardized["query"] = df[query_col].astype(str).apply(clean_text) if query_col in df.columns else ""
        standardized["answer"] = df[ans_col].astype(str).apply(clean_text)

        # Map labels: 1 = HALLUCINATED, 0 = SUPPORTED
        if df[label_col].dtype == object or isinstance(df[label_col].iloc[0], str):
            standardized["label"] = df[label_col].astype(str).str.lower().map(mapping)
        else:
            standardized["label"] = df[label_col].astype(int)

        # Drop any records with missing critical fields or unrecognized labels
        before_drop = len(standardized)
        standardized = standardized.dropna(subset=["context", "answer", "label"])
        standardized["label"] = standardized["label"].astype(int)

        # Filter empty strings
        standardized = standardized[
            (standardized["context"].str.strip().str.len() > 5) &
            (standardized["answer"].str.strip().str.len() > 0)
        ].reset_index(drop=True)

        logger.info(
            f"Dataset standardized: {len(standardized)} valid records (retained {len(standardized)}/{before_drop})"
        )
        return standardized

    def create_splits(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Creates stratified Train, Validation, and Test splits according to sample_sizes in config.
        """
        sample_sizes = self.ds_config.get("sample_sizes", {"train": 3000, "val": 500, "test": 500})
        n_train = sample_sizes.get("train", 3000)
        n_val = sample_sizes.get("val", 500)
        n_test = sample_sizes.get("test", 500)
        total_requested = n_train + n_val + n_test

        if len(df) < total_requested:
            logger.warning(
                f"Requested total samples ({total_requested}) exceeds dataset size ({len(df)}). "
                f"Downscaling proportionally."
            )
            ratio_train = n_train / total_requested
            ratio_val = n_val / total_requested
            ratio_test = n_test / total_requested
            n_train = int(len(df) * ratio_train)
            n_val = int(len(df) * ratio_val)
            n_test = len(df) - n_train - n_val

        # Subsample total requested with stratification
        if total_requested < len(df):
            df_sampled, _ = train_test_split(
                df,
                train_size=total_requested,
                random_state=self.random_seed,
                stratify=df["label"] if self.ds_config.get("stratify", True) else None
            )
        else:
            df_sampled = df.copy()

        df_sampled = df_sampled.sample(frac=1.0, random_state=self.random_seed).reset_index(drop=True)

        # First split off train
        train_df, rem_df = train_test_split(
            df_sampled,
            train_size=n_train,
            random_state=self.random_seed,
            stratify=df_sampled["label"] if self.ds_config.get("stratify", True) else None
        )

        # Split remaining into val and test
        val_ratio = n_val / (n_val + n_test)
        val_df, test_df = train_test_split(
            rem_df,
            train_size=val_ratio,
            random_state=self.random_seed,
            stratify=rem_df["label"] if self.ds_config.get("stratify", True) else None
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        logger.info(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        if self.ds_config.get("save_processed", True):
            train_df.to_csv(self.processed_dir / "train.csv", index=False)
            val_df.to_csv(self.processed_dir / "val.csv", index=False)
            test_df.to_csv(self.processed_dir / "test.csv", index=False)
            logger.info(f"Saved splits to {self.processed_dir}")

        return train_df, val_df, test_df

    def load_or_prepare_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads pre-existing split files if they exist; otherwise runs the full preparation.
        """
        train_path = self.processed_dir / "train.csv"
        val_path = self.processed_dir / "val.csv"
        test_path = self.processed_dir / "test.csv"

        if train_path.exists() and val_path.exists() and test_path.exists():
            logger.info(f"Loading existing processed splits from {self.processed_dir}")
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            return train_df, val_df, test_df

        raw_df = self.fetch_raw_dataset()
        clean_df = self.standardize_and_validate(raw_df)
        return self.create_splits(clean_df)


def inspect_dataset():
    """
    Dataset inspection script as required by Section 21 of the specification:
    Prints:
    - Number of samples
    - Available columns / fields
    - Label names and mapping
    - Class distribution
    - 5 sample records
    """
    print("\n" + "=" * 70)
    print("DATASET VALIDATION & SCHEMA INSPECTION: HaluEval")
    print("=" * 70)

    loader = HaluEvalDataLoader()
    raw_df = loader.fetch_raw_dataset()

    print(f"\n[1] RAW DATASET SUMMARY")
    print(f"    - Total records: {len(raw_df)}")
    print(f"    - Raw columns: {list(raw_df.columns)}")

    clean_df = loader.standardize_and_validate(raw_df)

    print(f"\n[2] STANDARDIZED FIELDS")
    print(f"    - 'context': Reference knowledge passage")
    print(f"    - 'query': Input question / prompt")
    print(f"    - 'answer': Candidate LM generated answer")
    print(f"    - 'label': Binary target (0: SUPPORTED, 1: HALLUCINATED)")

    print(f"\n[3] CLASS DISTRIBUTION (Total: {len(clean_df)})")
    class_counts = clean_df["label"].value_counts().to_dict()
    for lbl, cnt in class_counts.items():
        name = "HALLUCINATED (1)" if lbl == 1 else "SUPPORTED (0)"
        pct = (cnt / len(clean_df)) * 100
        print(f"    - {name}: {cnt} samples ({pct:.2f}%)")

    print(f"\n[4] SAMPLE RECORDS (First 5):")
    for idx, row in clean_df.head(5).iterrows():
        status = "HALLUCINATED" if row["label"] == 1 else "SUPPORTED"
        print(f"\n    --- Record {idx + 1} [{status}] ---")
        print(f"    Context:  {row['context'][:120]}...")
        if row["query"]:
            print(f"    Query:    {row['query']}")
        print(f"    Answer:   {row['answer']}")
        print(f"    Label:    {row['label']} ({status})")

    print(f"\n[5] CREATING STRATIFIED TRAIN / VAL / TEST SPLITS...")
    train_df, val_df, test_df = loader.create_splits(clean_df)

    print(f"    - Train split: {len(train_df)} samples")
    print(f"      Distribution: {train_df['label'].value_counts().to_dict()}")
    print(f"    - Val split:   {len(val_df)} samples")
    print(f"      Distribution: {val_df['label'].value_counts().to_dict()}")
    print(f"    - Test split:  {len(test_df)} samples")
    print(f"      Distribution: {test_df['label'].value_counts().to_dict()}")
    print("\n" + "=" * 70)
    print("DATASET VALIDATION COMPLETE. READY FOR EXPERIMENTS.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    inspect_dataset()
