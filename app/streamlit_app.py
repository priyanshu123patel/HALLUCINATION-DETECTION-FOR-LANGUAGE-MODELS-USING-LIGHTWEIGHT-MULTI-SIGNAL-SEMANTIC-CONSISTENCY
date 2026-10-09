"""
Interactive Academic Research Demonstration Interface.

Hallucination Detection for Language Models
Provides real-time multi-signal analysis:
- Lexical TF-IDF similarity
- Dense MiniLM-L6-v2 semantic similarity
- Cross-Encoder NLI entailment / contradiction probabilities
- spaCy Named Entity Consistency
- Sentence-level anomaly decomposition
- Mathematically grounded, transparent explanations (No black-box LLM judge)
- Pure SVG vector icons for a crisp, publication-grade academic UI
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path so imports work cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Enable running both via 'streamlit run app/streamlit_app.py' and 'python app/streamlit_app.py'
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        if get_script_run_ctx() is None:
            from streamlit.web import cli as stcli
            os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
            os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
            sys.argv = [
                "streamlit", "run", str(Path(__file__).resolve()),
                "--server.headless", "true",
                "--server.port", "8501",
                "--browser.gatherUsageStats", "false"
            ]
            sys.exit(stcli.main())
    except Exception:
        pass

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_config
from src.feature_extraction import FeatureExtractor
from src.detector import LightweightHallucinationDetector


# ==============================================================================
# Pure SVG Icon Library (Vector Graphics - Zero Image Artifacts)
# ==============================================================================

def svg_icon(name: str, size: int = 18, color: str = "currentColor", margin_right: int = 8) -> str:
    """Returns an inline SVG element for crisp, resolution-independent rendering."""
    icons = {
        "search": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>'
        ),
        "shield_check": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>'
            f'<polyline points="9 12 11 14 15 10"></polyline></svg>'
        ),
        "shield_alert": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>'
            f'<line x1="12" y1="8" x2="12" y2="12"></line>'
            f'<line x1="12" y1="16" x2="12.01" y2="16"></line></svg>'
        ),
        "chart": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<line x1="18" y1="20" x2="18" y2="10"></line>'
            f'<line x1="12" y1="20" x2="12" y2="4"></line>'
            f'<line x1="6" y1="20" x2="6" y2="14"></line></svg>'
        ),
        "file_text": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>'
            f'<polyline points="14 2 14 8 20 8"></polyline>'
            f'<line x1="16" y1="13" x2="8" y2="13"></line>'
            f'<line x1="16" y1="17" x2="8" y2="17"></line>'
            f'<polyline points="10 9 9 9 8 9"></polyline></svg>'
        ),
        "tag": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path>'
            f'<line x1="7" y1="7" x2="7.01" y2="7"></line></svg>'
        ),
        "cpu": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect>'
            f'<rect x="9" y="9" width="6" height="6"></rect>'
            f'<line x1="9" y1="1" x2="9" y2="4"></line>'
            f'<line x1="15" y1="1" x2="15" y2="4"></line>'
            f'<line x1="9" y1="20" x2="9" y2="23"></line>'
            f'<line x1="15" y1="20" x2="15" y2="23"></line>'
            f'<line x1="20" y1="9" x2="23" y2="9"></line>'
            f'<line x1="20" y1="14" x2="23" y2="14"></line>'
            f'<line x1="1" y1="9" x2="4" y2="9"></line>'
            f'<line x1="1" y1="14" x2="4" y2="14"></line></svg>'
        ),
        "info": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<circle cx="12" cy="12" r="10"></circle>'
            f'<line x1="12" y1="16" x2="12" y2="12"></line>'
            f'<line x1="12" y1="8" x2="12.01" y2="8"></line></svg>'
        ),
        "check": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<polyline points="20 6 9 17 4 12"></polyline></svg>'
        ),
        "alert": (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 24 24" '
            f'fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" '
            f'style="vertical-align: middle; margin-right: {margin_right}px;">'
            f'<circle cx="12" cy="12" r="10"></circle>'
            f'<line x1="15" y1="9" x2="9" y2="15"></line>'
            f'<line x1="9" y1="9" x2="15" y2="15"></line></svg>'
        )
    }
    return icons.get(name, "")


st.set_page_config(
    page_title="Hallucination Detection for Language Models",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Academic styling with SVG integration
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color, #f8fafc);
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-title .gradient-text {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        font-weight: 800;
    }
    .sub-title {
        font-size: 1.05rem;
        color: var(--text-color, #94a3b8);
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .supported-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-flex;
        align-items: center;
        letter-spacing: 0.02em;
    }
    .hallucinated-badge {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.15rem;
        display: inline-flex;
        align-items: center;
        letter-spacing: 0.02em;
    }
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-color, #f1f5f9);
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .stButton > button {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.65rem 1.4rem;
        border-radius: 8px;
        background: linear-gradient(135deg, #1d4ed8, #2563eb);
        color: #ffffff !important;
        border: none;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.2);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e40af, #1d4ed8);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    """Caches pipeline initialization to ensure sub-second response on interactive runs."""
    config = load_config()
    extractor = FeatureExtractor(config=config)

    model_path = Path("models/hallucination_detector.joblib")
    if model_path.exists():
        detector = LightweightHallucinationDetector.load(str(model_path))
    else:
        detector = LightweightHallucinationDetector(config=config)
    return config, extractor, detector


config, extractor, detector = load_system()

# Sidebar: Research Information & Pre-loaded Academic Test Cases
st.sidebar.markdown(f'<div class="section-header">{svg_icon("cpu", 22, "#38bdf8")} Research Controls</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
**Project:** Hallucination Detection for Language Models  
**Methodology:** Lightweight Multi-Signal Consistency Framework  
**Underlying Models:**
- Lexical: TF-IDF n-gram Vectorizer
- Semantic: `all-MiniLM-L6-v2` (384-dim)
- NLI: `nli-deberta-v3-xsmall`
- NER: `spaCy en_core_web_sm`
""")

# Preloaded Test Cases
preset_cases = {
    "Custom Input": {
        "context": "",
        "answer": ""
    },
    "Case 1: Supported Answer (Factual Grounding)": {
        "context": (
            "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia "
            "in the 19th century. First for Women is a woman's magazine published by Bauer Media Group in the USA."
        ),
        "answer": "Arthur's Magazine was an American literary periodical published in Philadelphia."
    },
    "Case 2: Direct Contradiction Hallucination": {
        "context": (
            "Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia "
            "in the 19th century. First for Women is a woman's magazine published by Bauer Media Group in the USA."
        ),
        "answer": "First for Women was started first in 1840."
    },
    "Case 3: Unsupported Entity Fabrication": {
        "context": (
            "Cadmium Chloride is a hygroscopic solid that is highly soluble in water and slightly soluble in alcohol. "
            "Ethanol is also called alcohol."
        ),
        "answer": "Cadmium Chloride is dissolved primarily in sulfuric acid and liquid nitrogen."
    },
    "Case 4: Multi-Sentence Partially Hallucinated Answer": {
        "context": (
            "Allison Beth 'Allie' Goertz (born March 2, 1991) is an American musician. "
            "Goertz is known for her satirical songs based on pop culture, including songs about The Simpsons "
            "character Milhouse, whom Matt Groening named after President Richard Nixon."
        ),
        "answer": (
            "Allie Goertz is an American musician famous for satirical songs. "
            "Matt Groening named Milhouse after President Richard Nixon. "
            "She later won three Grammy Awards for her classical piano symphonies in Paris."
        )
    }
}

selected_preset = st.sidebar.selectbox("Load Benchmark Example:", list(preset_cases.keys()))
default_context = preset_cases[selected_preset]["context"]
default_answer = preset_cases[selected_preset]["answer"]

# Main Interface Header
st.markdown(
    f'''<div class="main-title">
        {svg_icon("shield_check", 36, "#38bdf8")}
        <span class="gradient-text">Hallucination Detection for Language Models</span>
    </div>''',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Lightweight, Interpretable Multi-Signal Verification Pipeline without Expensive LLM Fine-Tuning</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Reference Knowledge / Context")
    context_input = st.text_area(
        "Enter ground-truth passage or retrieved context:",
        value=default_context,
        height=180,
        placeholder="Paste reference context here..."
    )

with col2:
    st.subheader("Generated Candidate Answer")
    answer_input = st.text_area(
        "Enter language model generated response:",
        value=default_answer,
        height=180,
        placeholder="Paste candidate answer here..."
    )

# Primary action button with pure text (zero emoji image artifacts)
analyze_button = st.button("Run Hallucination Verification", type="primary", use_container_width=True)

if analyze_button or (context_input and answer_input and selected_preset != "Custom Input"):
    if not context_input.strip() or not answer_input.strip():
        st.warning("Please provide both Reference Context and Candidate Answer to perform analysis.")
    else:
        with st.spinner("Extracting multi-signal features (Lexical, Semantic, NLI, NER)..."):
            extracted = extractor.extract_single(context_input, answer_input)
            features = extracted["features"]
            sentence_analysis = extracted["sentence_analysis"]
            entity_details = extracted["entity_details"]

            # Run Detector
            if detector.is_fitted:
                explanation = detector.explain(features, sentence_analysis, entity_details)
                pred_label = explanation["prediction"]
                hallu_prob = explanation["hallucination_probability"]
            else:
                # Heuristic fallback if detector has not been trained yet
                w_score, w_pred = detector.compute_weighted_score(features)
                pred_label = "HALLUCINATED" if w_pred == 1 else "SUPPORTED"
                hallu_prob = w_score
                explanation = {
                    "prediction": pred_label,
                    "hallucination_probability": hallu_prob,
                    "supported_probability": 1.0 - hallu_prob,
                    "evidence": ["Detector running in heuristic weighted mode (model checkpoint not yet trained)."],
                    "sentence_analysis": sentence_analysis,
                    "feature_contributions": []
                }

        st.markdown("---")

        # 1. Prediction Banner
        res_col1, res_col2, res_col3 = st.columns([1.2, 1.5, 1.3])

        with res_col1:
            st.markdown("### System Verdict")
            if pred_label == "SUPPORTED":
                st.markdown(
                    f'<div class="supported-badge">{svg_icon("shield_check", 22, "#065f46")} SUPPORTED</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="hallucinated-badge">{svg_icon("shield_alert", 22, "#991b1b")} HALLUCINATED</div>',
                    unsafe_allow_html=True
                )

        with res_col2:
            st.markdown("### Hallucination Risk")
            st.metric(
                label="Probability of Hallucination P(H)",
                value=f"{hallu_prob * 100:.1f}%",
                delta=f"{'High Risk' if hallu_prob >= 0.50 else 'Grounded'}",
                delta_color="inverse"
            )
            st.progress(float(hallu_prob))

        with res_col3:
            st.markdown("### Grounding Level")
            st.metric(
                label="Factual Grounding Score",
                value=f"{(1.0 - hallu_prob) * 100:.1f}%"
            )

        st.markdown("---")

        # 2. Multi-Signal Quantitative Features
        st.markdown(
            f'<div class="section-header">{svg_icon("chart", 22, "#38bdf8")} Signal Feature Decomposition</div>',
            unsafe_allow_html=True
        )
        m1, m2, m3, m4, m5 = st.columns(5)

        with m1:
            st.metric("TF-IDF Similarity", f"{features.get('tfidf_similarity', 0.0):.3f}")
        with m2:
            st.metric("Semantic Similarity", f"{features.get('semantic_similarity', 0.0):.3f}")
        with m3:
            st.metric("NLI Entailment", f"{features.get('nli_entailment_prob', 0.0):.3f}")
        with m4:
            st.metric("NLI Contradiction", f"{features.get('nli_contradiction_prob', 0.0):.3f}")
        with m5:
            unsupp_cnt = int(features.get('entity_unsupported_count', 0))
            st.metric("Unsupported Entities", f"{unsupp_cnt}")

        st.markdown("---")

        # 3. Transparent Evidence Explanation
        st.markdown(
            f'<div class="section-header">{svg_icon("info", 22, "#38bdf8")} Why Was This Prediction Made?</div>',
            unsafe_allow_html=True
        )
        st.markdown("**Evidence Derived Directly from Feature Computations (Zero LLM Hallucination):**")

        for point in explanation.get("evidence", []):
            st.markdown(f"- {point}")

        st.markdown("---")

        # 4. Sentence-Level Breakdown
        st.markdown(
            f'<div class="section-header">{svg_icon("file_text", 22, "#38bdf8")} Sentence-Level Diagnostic Breakdown</div>',
            unsafe_allow_html=True
        )
        if sentence_analysis:
            sent_rows = []
            for idx, s in enumerate(sentence_analysis):
                status_label = "❌ SUSPICIOUS" if s["status"] == "SUSPICIOUS" else "✅ SUPPORTED"
                reasons = ", ".join(s.get("reasons", [])) if s.get("reasons") else "Well-aligned with context"
                sent_rows.append({
                    "Sentence #": idx + 1,
                    "Candidate Sentence": s["sentence"],
                    "Status": status_label,
                    "Contradiction Prob": f"{s['contradiction_prob']:.2f}",
                    "Semantic Sim": f"{s['semantic_similarity']:.2f}",
                    "Diagnostic Reason": reasons
                })
            sent_df = pd.DataFrame(sent_rows)
            st.table(sent_df)
        else:
            st.info("Answer consists of a single sentence or fragment.")

        st.markdown("---")

        # 5. Entity Consistency Details
        with st.expander("Named Entity Analysis Details"):
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown(f"**{svg_icon('tag', 16, '#38bdf8')} Entities in Reference Context:**", unsafe_allow_html=True)
                ctx_ents = entity_details.get("matched_entity_list", [])
                st.write(ctx_ents if ctx_ents else "No major named entities detected in context.")
            with ec2:
                st.markdown(f"**{svg_icon('alert', 16, '#dc2626')} Unsupported Entities in Generated Answer:**", unsafe_allow_html=True)
                unsupp_ents = entity_details.get("unsupported_entity_list", [])
                if unsupp_ents:
                    st.error(f"Unsupported entities: {unsupp_ents}")
                else:
                    st.success("All entities in answer are supported by reference context.")
