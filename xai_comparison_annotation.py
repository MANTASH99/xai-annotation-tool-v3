"""
Explainable AI Comparison Annotation Tool
==========================================
Paper 2: Comparing XAI methods for emotion classification.

Two-phase annotation per sample:
  Phase A — Word highlighting: annotator selects up to 5 words that express the emotion (no XAI shown)
  Phase B — Heatmap ranking: annotator ranks 4 anonymized XAI heatmaps from best to worst

Deployed on Streamlit Cloud.
"""

import streamlit as st
import json
import csv
import random
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from scipy.stats import kendalltau, spearmanr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLES_JSON = DATA_DIR / "selected_samples.json"
SAMPLES_JSON_R2 = DATA_DIR / "selected_samples_round2.json"
METHODS = ["shap", "lime", "ig", "attention"]
METHOD_DISPLAY = {"shap": "SHAP", "lime": "LIME", "ig": "Integrated Gradients", "attention": "Attention"}
MAX_HIGHLIGHT = 5
TOP_K = 5  # Number of top features shown in Round 2 Phase B

ANNOTATOR_NAMES = ["Benni", "Emilia", "Vanessa", "Anna"]

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_samples(round_num=1):
    path = SAMPLES_JSON if round_num == 1 else SAMPLES_JSON_R2
    with open(path) as f:
        return json.load(f)


def get_shuffled_method_order(sample_id):
    """Get a deterministic but shuffled method order for a sample."""
    rng = random.Random(sample_id * 7 + 13)
    order = list(METHODS)
    rng.shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Google Sheets helpers
# ---------------------------------------------------------------------------
GCP_SA_FILE = Path(__file__).parent.parent / "gcp_service_account.json"


def get_gsheet_client():
    if not GSPREAD_AVAILABLE:
        return None
    creds_dict = None
    # Try Streamlit secrets first (cloud deployment)
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
    except (FileNotFoundError, KeyError):
        pass
    # Fallback: local service account JSON
    if creds_dict is None and GCP_SA_FILE.exists():
        import json as _json
        with open(GCP_SA_FILE) as f:
            creds_dict = _json.load(f)
    if creds_dict is None:
        return None
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
    return gspread.authorize(creds)


def get_or_create_worksheet(client, sheet_name, worksheet_title, headers):
    sh = client.open(sheet_name)
    try:
        ws = sh.worksheet(worksheet_title)
        # Update headers if new columns were added
        existing_headers = ws.row_values(1)
        if len(headers) > len(existing_headers):
            ws.update(range_name='1:1', values=[headers])
            # Expand columns if needed
            if ws.col_count < len(headers):
                ws.resize(cols=len(headers))
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_title, rows=500, cols=len(headers))
        ws.append_row(headers)
    return ws


# ---------------------------------------------------------------------------
# Persistence: save annotations (local CSV + optional Google Sheets)
# ---------------------------------------------------------------------------
def _phase_key(phase, round_num):
    """Return prefixed phase name: 'phase_a' for Round 1, 'r2_phase_a' for Round 2."""
    return phase if round_num == 1 else f"r2_{phase}"


def save_annotation(annotator, sample_id, phase, data, round_num=1):
    """Save one annotation row. Works locally always; cloud if configured."""
    phase_key = _phase_key(phase, round_num)
    timestamp = datetime.now().isoformat()
    row = {
        "annotator": annotator,
        "sample_id": sample_id,
        "phase": phase_key,
        "timestamp": timestamp,
        **data,
    }

    # Local CSV
    csv_path = OUTPUT_DIR / f"annotations_{annotator.lower()}_{phase_key}.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Google Sheets (optional)
    client = get_gsheet_client()
    if client:
        try:
            ws = get_or_create_worksheet(
                client,
                "XAI_Comparison_Annotations",
                f"{phase_key}_{annotator}",
                list(row.keys()),
            )
            ws.append_row(list(row.values()))
        except Exception as e:
            st.warning(f"Cloud save failed: {e}")

    # Update session state cache so load_completed_samples sees this immediately
    gsheet_key = f"_gsheet_completed_{annotator}_{phase_key}"
    if gsheet_key in st.session_state:
        st.session_state[gsheet_key].add(sample_id)

    return True


def _load_completed_from_gsheet(annotator, phase_key):
    """Load completed sample IDs from Google Sheets (survives app restarts)."""
    client = get_gsheet_client()
    if not client:
        st.sidebar.caption(f"DEBUG: No gsheet client for {phase_key}_{annotator}")
        return set()
    try:
        sh = client.open("XAI_Comparison_Annotations")
        ws = sh.worksheet(f"{phase_key}_{annotator}")
        # sample_id is column 2 (annotator, sample_id, phase, timestamp, ...)
        all_values = ws.col_values(2)
        completed = set()
        for val in all_values[1:]:  # skip header
            try:
                completed.add(int(val))
            except (ValueError, TypeError):
                pass
        st.sidebar.caption(f"DEBUG: Loaded {len(completed)} from GSheet {phase_key}_{annotator}")
        return completed
    except Exception as e:
        st.sidebar.caption(f"DEBUG: GSheet error {phase_key}_{annotator}: {e}")
        return set()


def load_completed_samples(annotator, phase, round_num=1):
    """Load set of sample IDs already annotated for this phase.
    Checks local CSV (current session) + Google Sheets (persistent across restarts)."""
    phase_key = _phase_key(phase, round_num)
    # Google Sheets — load once per session, cached in session state
    gsheet_key = f"_gsheet_completed_{annotator}_{phase_key}"
    if gsheet_key not in st.session_state:
        st.session_state[gsheet_key] = _load_completed_from_gsheet(annotator, phase_key)
    completed = set(st.session_state[gsheet_key])

    # Local CSV — has current session's annotations
    csv_path = OUTPUT_DIR / f"annotations_{annotator.lower()}_{phase_key}.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(int(row["sample_id"]))
    return completed


# ---------------------------------------------------------------------------
# Heatmap rendering (HTML)
# ---------------------------------------------------------------------------
def score_to_color(score, max_abs, method_name):
    """Convert an importance score to a background color."""
    if max_abs == 0:
        return "rgba(255,255,255,0)"

    normalized = score / max_abs  # -1 to 1

    if method_name == "attention":
        # Attention is always positive: white to red
        intensity = min(abs(normalized), 1.0)
        r = 255
        g = int(255 * (1 - intensity * 0.7))
        b = int(255 * (1 - intensity * 0.85))
        return f"rgba({r},{g},{b},0.85)"
    else:
        # Positive = red, negative = blue
        intensity = min(abs(normalized), 1.0)
        if normalized >= 0:
            r = 255
            g = int(255 * (1 - intensity * 0.7))
            b = int(255 * (1 - intensity * 0.85))
        else:
            r = int(255 * (1 - intensity * 0.7))
            g = int(255 * (1 - intensity * 0.6))
            b = 255
        return f"rgba({r},{g},{b},0.85)"


def render_heatmap_html(words, scores, method_name, label=""):
    """Render a word-level heatmap as styled HTML."""
    abs_scores = [abs(s) for s in scores]
    max_abs = max(abs_scores) if abs_scores else 1.0

    html_parts = []
    if label:
        html_parts.append(f'<div style="font-weight:bold; margin-bottom:6px; font-size:14px; color:#444;">{label}</div>')

    html_parts.append('<div style="line-height:2.2; font-size:16px; font-family: \'Source Sans Pro\', sans-serif;">')
    for word, score in zip(words, scores):
        color = score_to_color(score, max_abs, method_name)
        html_parts.append(
            f'<span style="background-color:{color}; padding:3px 5px; margin:2px; '
            f'border-radius:4px; display:inline-block; border: 1px solid #ddd;">'
            f'{word}</span>'
        )
    html_parts.append('</div>')

    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Phase A: Word Highlighting
# ---------------------------------------------------------------------------
def render_phase_a(sample, annotator, round_num=1):
    sample_id = sample["id"]
    words = sample["words"]
    predicted = sample["predicted_class"]

    st.markdown("---")
    st.markdown("### Phase A: Word Highlighting")
    st.markdown(
        "**Task:** Select up to **5 words** that you think are most important for "
        "the model's prediction. No explanations are shown — use your own judgment."
    )

    # Predicted emotion
    st.markdown(f"**Predicted Emotion:** `{predicted}`")

    # Show sentence prominently
    st.markdown(
        f'<div style="background:#e8f0fe; padding:16px 20px; border-radius:10px; '
        f'border-left:5px solid #4a90d9; margin:12px 0; font-size:20px; '
        f'line-height:1.6; color:#1a1a1a;">{sample["sentence"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Click on words to select them** (up to 5):")

    # Word selection
    key_prefix = f"phase_a_{sample_id}"
    ranks_key = f"{key_prefix}_ranks"

    # Use a dict {word_idx: rank_number} so rank is an explicit integer,
    # not dependent on list/dict ordering which Streamlit can lose across reruns.
    if ranks_key not in st.session_state:
        st.session_state[ranks_key] = {}

    ranks = st.session_state[ranks_key]

    # Show max-words warning from previous click (callbacks can't call st.toast)
    if st.session_state.pop("_max_words_warning", False):
        st.toast(f"Maximum {MAX_HIGHLIGHT} words. Deselect one first.", icon="⚠️")

    # Render clickable words — use on_click callbacks (no st.rerun needed)
    def _toggle_word(w_idx, rk, max_h):
        if w_idx in rk:
            removed = rk.pop(w_idx)
            for idx in rk:
                if rk[idx] > removed:
                    rk[idx] -= 1
        elif len(rk) < max_h:
            rk[w_idx] = len(rk) + 1
        else:
            st.session_state["_max_words_warning"] = True

    words_per_row = 10
    for row_start in range(0, len(words), words_per_row):
        row_words = words[row_start:min(row_start + words_per_row, len(words))]
        cols = st.columns(len(row_words))
        for j, (col, word) in enumerate(zip(cols, row_words)):
            word_idx = row_start + j
            is_selected = word_idx in ranks
            with col:
                btn_label = f"{ranks[word_idx]}. {word}" if is_selected else word
                st.button(
                    btn_label,
                    key=f"{key_prefix}_word_{word_idx}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                    on_click=_toggle_word,
                    args=(word_idx, ranks, MAX_HIGHLIGHT),
                )

    # Get ordered selection (sorted by rank value, NOT by word index)
    selected_by_rank = sorted(ranks.items(), key=lambda x: x[1])
    n_selected = len(selected_by_rank)

    # DEBUG: show raw session state so we can verify ranking
    if n_selected > 0:
        st.caption(f"DEBUG ranks dict: {dict(ranks)} | sorted: {selected_by_rank}")

    # Show selection status with ranked order
    if n_selected > 0:
        rank_html = ' → '.join(
            [f'<span style="background:#4a90d9; color:white; padding:4px 10px; '
             f'border-radius:16px; font-size:16px; font-weight:bold;">'
             f'{rank}. {words[word_idx]}</span>'
             for word_idx, rank in selected_by_rank]
        )
        st.markdown(
            f'<div style="background:#f0f7ff; padding:14px 18px; border-radius:10px; '
            f'border:2px solid #4a90d9; margin:10px 0; line-height:2.4;">'
            f'<b>Importance ranking (most → least):</b><br>{rank_html}</div>',
            unsafe_allow_html=True,
        )
        st.success(f"Selected {n_selected}/{MAX_HIGHLIGHT}")
    else:
        st.info(f"Selected {n_selected}/{MAX_HIGHLIGHT} words — click in order of importance (most important first)")

    # Submit Phase A
    can_submit = n_selected >= 1
    if st.button("Submit Phase A", key=f"{key_prefix}_submit", disabled=not can_submit,
                 type="primary"):
        # Build save data from explicit rank numbers
        ranked_words = " | ".join(
            f"{rank}:{words[word_idx]}" for word_idx, rank in selected_by_rank
        )
        rank_data = {}
        for word_idx, rank in selected_by_rank:
            rank_data[f"rank_{rank}_word"] = words[word_idx]
            rank_data[f"rank_{rank}_index"] = word_idx
        for rank_num in range(n_selected + 1, MAX_HIGHLIGHT + 1):
            rank_data[f"rank_{rank_num}_word"] = ""
            rank_data[f"rank_{rank_num}_index"] = ""
        save_annotation(annotator, sample_id, "phase_a", {
            "sentence": sample["sentence"],
            "predicted_class": predicted,
            "ranked_words": ranked_words,
            **rank_data,
        }, round_num=round_num)
        st.session_state[f"sample_{sample_id}_phase_a_done"] = True
        st.success("Phase A saved! Scroll down for Phase B.")
        st.rerun()

    return st.session_state.get(f"sample_{sample_id}_phase_a_done", False)


# ---------------------------------------------------------------------------
# Phase B: Heatmap Ranking
# ---------------------------------------------------------------------------
def render_phase_b(sample, annotator):
    sample_id = sample["id"]
    words = sample["words"]

    st.markdown("---")
    st.markdown("### Phase B: Explanation Ranking")

    # Show sentence prominently
    st.markdown(f"**Predicted Emotion:** `{sample['predicted_class']}`")
    st.markdown(
        f'<div style="background:#e8f0fe; padding:16px 20px; border-radius:10px; '
        f'border-left:5px solid #4a90d9; margin:12px 0; font-size:20px; '
        f'line-height:1.6; color:#1a1a1a;">{sample["sentence"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Task:** Below are 4 different explanations of the same prediction, shown as colored heatmaps. "
        "**Red** = the word pushes toward the prediction, **Blue** = pushes against it, "
        "**White** = neutral. Rank them from **best (1)** to **worst (4)**."
    )

    # Get shuffled method order for this sample
    method_order = get_shuffled_method_order(sample_id)
    anon_labels = ["Method A", "Method B", "Method C", "Method D"]

    # Display 4 heatmaps in a 2x2 grid
    key_prefix = f"phase_b_{sample_id}"

    for row in range(2):
        col1, col2 = st.columns(2)
        for col_idx, col in enumerate([col1, col2]):
            method_idx = row * 2 + col_idx
            method_name = method_order[method_idx]
            anon_label = anon_labels[method_idx]
            scores = sample["methods"][method_name]

            with col:
                st.markdown(
                    f'<div style="background:#f8f9fa; padding:12px; border-radius:8px; '
                    f'border:2px solid #dee2e6; margin-bottom:8px;">'
                    f'{render_heatmap_html(words, scores, method_name, label=anon_label)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Ranking inputs
    st.markdown("**Rank each method** (1 = best, 4 = worst). Each rank must be unique:")

    rank_cols = st.columns(4)
    ranks = {}
    for i, (col, anon_label) in enumerate(zip(rank_cols, anon_labels)):
        with col:
            ranks[anon_label] = st.selectbox(
                anon_label,
                options=[1, 2, 3, 4],
                index=i,
                key=f"{key_prefix}_rank_{anon_label}",
            )

    # Validate unique ranks
    rank_values = list(ranks.values())
    ranks_valid = len(set(rank_values)) == 4

    if not ranks_valid:
        st.warning("Each rank (1-4) must be used exactly once.")

    # Submit Phase B
    if st.button("Submit Phase B", key=f"{key_prefix}_submit", disabled=not ranks_valid,
                 type="primary"):
        # Map anonymous labels back to real method names
        method_ranks = {}
        for i, anon_label in enumerate(anon_labels):
            real_method = method_order[i]
            method_ranks[real_method] = ranks[anon_label]

        save_annotation(annotator, sample_id, "phase_b", {
            "sentence": sample["sentence"],
            "label": sample["label"],
            "method_order": json.dumps(method_order),
            "rank_shap": method_ranks.get("shap"),
            "rank_lime": method_ranks.get("lime"),
            "rank_ig": method_ranks.get("ig"),
            "rank_attention": method_ranks.get("attention"),
        })
        st.session_state[f"sample_{sample_id}_phase_b_done"] = True
        # Clear current sample so resume logic finds the next incomplete one
        st.session_state.pop("_current_sample_id", None)
        st.success("Phase B saved! Sample complete.")
        st.rerun()

    return st.session_state.get(f"sample_{sample_id}_phase_b_done", False)


# ---------------------------------------------------------------------------
# Phase B Round 2: Top-5 bar visualization
# ---------------------------------------------------------------------------
def render_top5_bar_html(words, scores, label=""):
    """Render a top-5 horizontal bar chart as styled HTML.
    Shows the 5 words with highest absolute scores and their values."""
    # Pair words with scores, sort by absolute value descending
    paired = [(w, s) for w, s in zip(words, scores)]
    paired.sort(key=lambda x: abs(x[1]), reverse=True)
    top5 = paired[:TOP_K]

    # Max absolute score for bar scaling
    max_abs = max(abs(s) for _, s in top5) if top5 else 1.0

    html = []
    if label:
        html.append(
            f'<div style="font-weight:bold; margin-bottom:8px; font-size:15px; '
            f'color:#333;">{label}</div>'
        )

    for word, score in top5:
        bar_pct = abs(score) / max_abs * 100 if max_abs > 0 else 0
        if score >= 0:
            bar_color = "#e74c3c"  # red for positive
        else:
            bar_color = "#3498db"  # blue for negative
        html.append(
            f'<div style="display:flex; align-items:center; margin:4px 0; font-size:14px;">'
            f'<span style="width:90px; text-align:right; padding-right:8px; '
            f'font-weight:600; color:#222;">{word}</span>'
            f'<div style="flex:1; background:#eee; border-radius:4px; height:22px; position:relative;">'
            f'<div style="width:{bar_pct:.0f}%; background:{bar_color}; height:100%; '
            f'border-radius:4px; min-width:2px;"></div>'
            f'</div>'
            f'</div>'
        )

    return "".join(html)


def render_phase_b_round2(sample, annotator):
    """Phase B for Round 2: top-5 bar charts instead of full heatmaps."""
    sample_id = sample["id"]

    st.markdown("---")
    st.markdown("### Phase B: Explanation Ranking")

    st.markdown(f"**Predicted Emotion:** `{sample['predicted_class']}`")
    st.markdown(
        f'<div style="background:#e8f0fe; padding:16px 20px; border-radius:10px; '
        f'border-left:5px solid #4a90d9; margin:12px 0; font-size:20px; '
        f'line-height:1.6; color:#1a1a1a;">{sample["sentence"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Task:** Below are 4 different explanations showing each method's **top 5 most important words** "
        "and their contribution scores. **Red bars** = pushes toward the prediction, "
        "**Blue bars** = pushes against. Rank them from **best (1)** to **worst (4)**."
    )

    method_order = get_shuffled_method_order(sample_id)
    anon_labels = ["Method A", "Method B", "Method C", "Method D"]

    key_prefix = f"phase_b_{sample_id}"

    # Display 4 top-5 bar charts in a 2x2 grid
    for row in range(2):
        col1, col2 = st.columns(2)
        for col_idx, col in enumerate([col1, col2]):
            method_idx = row * 2 + col_idx
            method_name = method_order[method_idx]
            anon_label = anon_labels[method_idx]
            scores = sample["methods"][method_name]
            words = sample["words"]

            with col:
                st.markdown(
                    f'<div style="background:#f8f9fa; padding:14px; border-radius:8px; '
                    f'border:2px solid #dee2e6; margin-bottom:8px;">'
                    f'{render_top5_bar_html(words, scores, label=anon_label)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Ranking inputs (same as Round 1)
    st.markdown("**Rank each method** (1 = best, 4 = worst). Each rank must be unique:")

    rank_cols = st.columns(4)
    ranks = {}
    for i, (col, anon_label) in enumerate(zip(rank_cols, anon_labels)):
        with col:
            ranks[anon_label] = st.selectbox(
                anon_label,
                options=[1, 2, 3, 4],
                index=i,
                key=f"{key_prefix}_rank_{anon_label}",
            )

    rank_values = list(ranks.values())
    ranks_valid = len(set(rank_values)) == 4

    if not ranks_valid:
        st.warning("Each rank (1-4) must be used exactly once.")

    if st.button("Submit Phase B", key=f"{key_prefix}_submit", disabled=not ranks_valid,
                 type="primary"):
        method_ranks = {}
        for i, anon_label in enumerate(anon_labels):
            real_method = method_order[i]
            method_ranks[real_method] = ranks[anon_label]

        save_annotation(annotator, sample_id, "phase_b", {
            "sentence": sample["sentence"],
            "label": sample["label"],
            "method_order": json.dumps(method_order),
            "rank_shap": method_ranks.get("shap"),
            "rank_lime": method_ranks.get("lime"),
            "rank_ig": method_ranks.get("ig"),
            "rank_attention": method_ranks.get("attention"),
        }, round_num=2)
        st.session_state[f"sample_{sample_id}_phase_b_done"] = True
        st.session_state.pop("_current_sample_id", None)
        st.success("Phase B saved! Sample complete.")
        st.rerun()

    return st.session_state.get(f"sample_{sample_id}_phase_b_done", False)


# ---------------------------------------------------------------------------
# Phase C: Similarity Grouping
# ---------------------------------------------------------------------------
def render_phase_c(sample, annotator):
    sample_id = sample["id"]
    words = sample["words"]

    st.markdown("---")
    st.markdown("### Phase C: Similarity Grouping")

    # Show sentence prominently
    st.markdown(f"**Predicted Emotion:** `{sample['predicted_class']}`")
    st.markdown(
        f'<div style="background:#f0f2f6; padding:16px 20px; border-radius:10px; '
        f'border-left:5px solid #4a90d9; margin:12px 0; font-size:20px; '
        f'line-height:1.6;">{sample["sentence"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "**Task:** Look at the 4 explanations again. Group the ones that look **similar** to each other. "
        "Assign each method to a group (Group 1, Group 2, etc.). Methods in the same group "
        "highlight similar words in similar ways."
    )

    method_order = get_shuffled_method_order(sample_id)
    anon_labels = ["Method A", "Method B", "Method C", "Method D"]

    # Show heatmaps again (smaller, side by side)
    cols = st.columns(4)
    for i, col in enumerate(cols):
        method_name = method_order[i]
        scores = sample["methods"][method_name]
        with col:
            st.markdown(
                f'<div style="background:#f8f9fa; padding:8px; border-radius:6px; '
                f'border:2px solid #dee2e6; font-size:13px;">'
                f'{render_heatmap_html(words, scores, method_name, label=anon_labels[i])}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Group assignment
    st.markdown("**Assign each method to a group** (same group = similar explanations):")

    key_prefix = f"phase_c_{sample_id}"
    group_options = ["Group 1", "Group 2", "Group 3", "Group 4"]

    group_cols = st.columns(4)
    groups = {}
    for i, (col, anon_label) in enumerate(zip(group_cols, anon_labels)):
        with col:
            groups[anon_label] = st.selectbox(
                anon_label,
                options=group_options,
                index=0,
                key=f"{key_prefix}_group_{anon_label}",
            )

    # Show grouping preview
    group_members = defaultdict(list)
    for label, group in groups.items():
        group_members[group].append(label)

    n_groups = len(group_members)
    preview_parts = []
    for group_name in sorted(group_members.keys()):
        members = group_members[group_name]
        if len(members) > 0:
            preview_parts.append(f"**{group_name}:** {', '.join(members)}")

    st.markdown("**Preview:** " + " | ".join(preview_parts))

    if n_groups == 1:
        st.info("All methods in one group = all explanations look similar.")
    elif n_groups == 4:
        st.info("Each method in its own group = all explanations look different.")

    # Optional comment
    comment = st.text_area(
        "Optional comment (anything noteworthy about this sample):",
        key=f"{key_prefix}_comment",
        height=68,
    )

    # Submit Phase C
    if st.button("Submit Phase C & Go to Next Sample", key=f"{key_prefix}_submit",
                 type="primary"):
        # Map back to real methods
        method_groups = {}
        for i, anon_label in enumerate(anon_labels):
            real_method = method_order[i]
            method_groups[real_method] = groups[anon_label]

        save_annotation(annotator, sample_id, "phase_c", {
            "sentence": sample["sentence"],
            "label": sample["label"],
            "method_order": json.dumps(method_order),
            "group_shap": method_groups.get("shap"),
            "group_lime": method_groups.get("lime"),
            "group_ig": method_groups.get("ig"),
            "group_attention": method_groups.get("attention"),
            "n_groups": n_groups,
            "comment": comment,
        })
        st.session_state[f"sample_{sample_id}_phase_c_done"] = True
        st.success("All phases complete for this sample!")
        st.rerun()

    return st.session_state.get(f"sample_{sample_id}_phase_c_done", False)


# ---------------------------------------------------------------------------
# IAA Dashboard: data loading from Google Sheets
# ---------------------------------------------------------------------------
@st.cache_data(ttl=120, show_spinner=False)
def _load_all_annotations_from_gsheet():
    """Load all annotators' Phase A and Phase B data from Google Sheets.
    Cached for 2 minutes to avoid excessive API calls.
    Phase A values are stored as sorted lists (not sets) for cache compatibility.
    Returns dict: {phase: {annotator: {sample_id: data}}}"""
    client = get_gsheet_client()
    if not client:
        return None

    try:
        sh = client.open("XAI_Comparison_Annotations")
    except Exception:
        return None

    result = {"phase_a": {}, "phase_b": {}, "r2_phase_a": {}, "r2_phase_b": {}}

    for ws in sh.worksheets():
        title = ws.title
        if title == "Sheet1":
            continue
        # Parse worksheet title: "phase_a_Benni" or "r2_phase_a_Benni"
        if title.startswith("r2_"):
            # e.g. "r2_phase_a_Benni"
            rest = title[3:]  # "phase_a_Benni"
            sub_parts = rest.split("_", 2)
            if len(sub_parts) < 3:
                continue
            phase = f"r2_{sub_parts[0]}_{sub_parts[1]}"
            annotator = sub_parts[2]
        else:
            parts = title.split("_", 2)  # e.g. "phase_a_Benni"
            if len(parts) < 3:
                continue
            phase = f"{parts[0]}_{parts[1]}"
            annotator = parts[2]
        if phase not in result:
            continue

        try:
            records = ws.get_all_records()
        except Exception:
            continue

        ann_data = {}
        for row in records:
            try:
                sid = int(row["sample_id"])
            except (ValueError, KeyError):
                continue

            if phase.endswith("phase_a"):
                indices = []
                for r in range(1, 6):
                    idx = row.get(f"rank_{r}_index", "")
                    if idx != "" and idx is not None:
                        try:
                            indices.append(int(idx))
                        except (ValueError, TypeError):
                            pass
                ann_data[sid] = sorted(set(indices))
            elif phase.endswith("phase_b"):
                try:
                    ann_data[sid] = [
                        int(row["rank_shap"]),
                        int(row["rank_lime"]),
                        int(row["rank_ig"]),
                        int(row["rank_attention"]),
                    ]
                except (ValueError, KeyError):
                    pass

        if ann_data:
            result[phase][annotator] = ann_data

    return result


def _compute_iaa_metrics(all_data, samples_map):
    """Compute IAA metrics from loaded annotation data.
    Returns a dict with all metrics."""
    # Convert phase_a lists back to sets for set operations
    phase_a = {
        ann: {sid: set(indices) for sid, indices in data.items()}
        for ann, data in all_data["phase_a"].items()
    }
    phase_b = all_data["phase_b"]
    annotators_a = sorted(phase_a.keys())
    annotators_b = sorted(phase_b.keys())
    methods = ["shap", "lime", "ig", "attention"]

    metrics = {
        "annotator_progress": {},
        "phase_a_pairwise": [],
        "phase_a_krippendorff": {},
        "phase_b_pairwise": [],
        "phase_b_kendall_w": {},
        "phase_b_avg_ranks": {},
    }

    # --- Per-annotator progress ---
    for ann in set(annotators_a) | set(annotators_b):
        n_a = len(phase_a.get(ann, {}))
        n_b = len(phase_b.get(ann, {}))
        fully = len(set(phase_a.get(ann, {}).keys()) & set(phase_b.get(ann, {}).keys()))
        metrics["annotator_progress"][ann] = {
            "phase_a": n_a, "phase_b": n_b, "fully_done": fully,
        }

    # --- Phase A: pairwise Jaccard ---
    for a1, a2 in combinations(annotators_a, 2):
        common = set(phase_a[a1].keys()) & set(phase_a[a2].keys())
        if not common:
            continue
        jaccards = []
        overlaps = []
        exact = 0
        for sid in common:
            s1, s2 = phase_a[a1][sid], phase_a[a2][sid]
            union = len(s1 | s2)
            inter = len(s1 & s2)
            jaccards.append(inter / union if union > 0 else 0)
            overlaps.append(inter)
            if s1 == s2:
                exact += 1
        metrics["phase_a_pairwise"].append({
            "pair": f"{a1} vs {a2}",
            "n_common": len(common),
            "mean_jaccard": float(np.mean(jaccards)),
            "std_jaccard": float(np.std(jaccards)),
            "mean_overlap": float(np.mean(overlaps)),
            "exact_match": exact,
            "exact_match_pct": exact / len(common) * 100,
        })

    # --- Phase A: Krippendorff's alpha (binary per-word-position) ---
    if len(annotators_a) >= 2:
        common_all_a = set(phase_a[annotators_a[0]].keys())
        for ann in annotators_a[1:]:
            common_all_a &= set(phase_a[ann].keys())

        if common_all_a:
            items = []
            for sid in sorted(common_all_a):
                if sid not in samples_map:
                    continue
                n_words = len(samples_map[sid]["words"])
                for wpos in range(n_words):
                    row = [1 if wpos in phase_a[ann][sid] else 0 for ann in annotators_a]
                    items.append(row)
            items_arr = np.array(items)

            # Krippendorff's alpha (nominal)
            n_items, n_coders = items_arr.shape
            Do = 0
            n_pairs = 0
            for i in range(n_items):
                vals = items_arr[i]
                for c1 in range(n_coders):
                    for c2 in range(c1 + 1, n_coders):
                        if vals[c1] != vals[c2]:
                            Do += 1
                        n_pairs += 1
            Do /= n_pairs if n_pairs > 0 else 1
            total = n_items * n_coders
            cats = np.unique(items_arr)
            De = 1 - sum((np.sum(items_arr == c) / total) ** 2 for c in cats)
            alpha = 1 - Do / De if De > 0 else 1.0

            metrics["phase_a_krippendorff"] = {
                "alpha": float(alpha),
                "n_annotators": len(annotators_a),
                "n_samples": len(common_all_a),
            }

    # --- Phase B: pairwise Kendall's tau / Spearman ---
    for a1, a2 in combinations(annotators_b, 2):
        common = set(phase_b[a1].keys()) & set(phase_b[a2].keys())
        if not common:
            continue
        taus = []
        spears = []
        same_top = 0
        for sid in common:
            r1, r2 = phase_b[a1][sid], phase_b[a2][sid]
            tau, _ = kendalltau(r1, r2)
            rho, _ = spearmanr(r1, r2)
            taus.append(tau)
            spears.append(rho)
            if r1.index(1) == r2.index(1):
                same_top += 1
        metrics["phase_b_pairwise"].append({
            "pair": f"{a1} vs {a2}",
            "n_common": len(common),
            "mean_tau": float(np.mean(taus)),
            "std_tau": float(np.std(taus)),
            "mean_rho": float(np.mean(spears)),
            "std_rho": float(np.std(spears)),
            "same_top1": same_top,
            "same_top1_pct": same_top / len(common) * 100,
        })

    # --- Phase B: Kendall's W (all annotators) ---
    if len(annotators_b) >= 2:
        common_all_b = set(phase_b[annotators_b[0]].keys())
        for ann in annotators_b[1:]:
            common_all_b &= set(phase_b[ann].keys())

        if common_all_b:
            k = len(annotators_b)
            n = 4  # methods
            ws_list = []
            for sid in sorted(common_all_b):
                rank_sums = np.zeros(n)
                for ann in annotators_b:
                    for j, r in enumerate(phase_b[ann][sid]):
                        rank_sums[j] += r
                mean_rs = np.mean(rank_sums)
                S = np.sum((rank_sums - mean_rs) ** 2)
                W = 12 * S / (k ** 2 * (n ** 3 - n))
                ws_list.append(W)
            metrics["phase_b_kendall_w"] = {
                "mean_W": float(np.mean(ws_list)),
                "std_W": float(np.std(ws_list)),
                "n_perfect": int(sum(1 for w in ws_list if w == 1.0)),
                "n_samples": len(common_all_b),
                "n_annotators": k,
            }

    # --- Phase B: average rank per method per annotator ---
    for ann in annotators_b:
        ranks_by_method = {m: [] for m in methods}
        for sid, ranks in phase_b[ann].items():
            for j, m in enumerate(methods):
                ranks_by_method[m].append(ranks[j])
        metrics["phase_b_avg_ranks"][ann] = {
            m: float(np.mean(v)) for m, v in ranks_by_method.items()
        }

    # Overall average ranks
    all_ranks = {m: [] for m in methods}
    for ann in annotators_b:
        for sid, ranks in phase_b[ann].items():
            for j, m in enumerate(methods):
                all_ranks[m].append(ranks[j])
    metrics["phase_b_avg_ranks"]["Overall"] = {
        m: float(np.mean(v)) for m, v in all_ranks.items()
    }

    return metrics


# ---------------------------------------------------------------------------
# IAA Dashboard: rendering
# ---------------------------------------------------------------------------
def render_iaa_dashboard():
    """Render the IAA dashboard page."""
    st.title("Inter-Annotator Agreement Dashboard")

    col_round, col_refresh = st.columns([3, 1])
    with col_round:
        iaa_round = st.radio(
            "Round", [1, 2], horizontal=True,
            format_func=lambda x: f"Round {x}" + (" (heatmaps)" if x == 1 else " (top-5 bars)"),
        )
    with col_refresh:
        if st.button("Refresh data"):
            _load_all_annotations_from_gsheet.clear()
            st.rerun()

    samples = load_samples(iaa_round)
    samples_map = {s["id"]: s for s in samples}
    n_total = len(samples)

    with st.spinner("Loading annotations from Google Sheets..."):
        all_data = _load_all_annotations_from_gsheet()

    if all_data is None:
        st.error("Could not connect to Google Sheets. Check credentials.")
        return

    # Select the right phase keys for the chosen round
    if iaa_round == 1:
        round_data = {"phase_a": all_data["phase_a"], "phase_b": all_data["phase_b"]}
    else:
        round_data = {"phase_a": all_data["r2_phase_a"], "phase_b": all_data["r2_phase_b"]}

    if not round_data["phase_a"] and not round_data["phase_b"]:
        st.warning(f"No annotation data found for Round {iaa_round}.")
        return

    metrics = _compute_iaa_metrics(round_data, samples_map)

    # ===================== ANNOTATOR PROGRESS =====================
    st.markdown("## Annotator Progress")

    progress_data = metrics["annotator_progress"]
    if progress_data:
        prog_cols = st.columns(len(progress_data))
        for col, (ann, prog) in zip(prog_cols, sorted(progress_data.items())):
            with col:
                pct = prog["fully_done"] / n_total * 100
                st.metric(ann, f"{prog['fully_done']} / {n_total}", f"{pct:.0f}%")
                st.progress(prog["fully_done"] / n_total)
                st.caption(f"A: {prog['phase_a']}  |  B: {prog['phase_b']}")

    # ===================== PHASE A: WORD HIGHLIGHTING =====================
    st.markdown("---")
    st.markdown("## Phase A: Word Highlighting Agreement")

    if metrics["phase_a_pairwise"]:
        st.markdown("### Pairwise Jaccard Similarity")
        st.markdown(
            "Jaccard = |words in common| / |all words selected by either|. "
            "Range: 0 (no overlap) to 1 (identical selection)."
        )

        # Table
        rows = []
        for p in metrics["phase_a_pairwise"]:
            rows.append({
                "Pair": p["pair"],
                "Samples": p["n_common"],
                "Mean Jaccard": f"{p['mean_jaccard']:.3f}",
                "Std": f"{p['std_jaccard']:.3f}",
                "Mean Overlap": f"{p['mean_overlap']:.1f} / 5",
                "Exact Match": f"{p['exact_match']}/{p['n_common']} ({p['exact_match_pct']:.1f}%)",
            })
        st.table(rows)

        # Bar chart of Jaccard similarities
        chart_df = pd.DataFrame({
            "Pair": [p["pair"] for p in metrics["phase_a_pairwise"]],
            "Jaccard": [p["mean_jaccard"] for p in metrics["phase_a_pairwise"]],
        })
        st.bar_chart(chart_df.set_index("Pair"))

    if metrics["phase_a_krippendorff"]:
        ka = metrics["phase_a_krippendorff"]
        st.markdown("### Krippendorff's Alpha (binary, per word position)")
        st.markdown(
            f"**alpha = {ka['alpha']:.3f}** "
            f"({ka['n_annotators']} annotators, {ka['n_samples']} common samples)"
        )
        if ka["alpha"] < 0.667:
            st.warning("Alpha < 0.667: tentative agreement. Word selection is inherently subjective.")
        elif ka["alpha"] < 0.8:
            st.info("Alpha 0.667-0.8: acceptable agreement for exploratory research.")
        else:
            st.success("Alpha >= 0.8: good agreement.")

    # ===================== PHASE B: RANKING =====================
    st.markdown("---")
    st.markdown("## Phase B: Explanation Ranking Agreement")

    if metrics["phase_b_pairwise"]:
        st.markdown("### Pairwise Kendall's Tau & Spearman's Rho")
        st.markdown(
            "Kendall's tau measures rank correlation (-1 to 1). "
            "Same #1 = how often annotators pick the same best method."
        )

        rows = []
        for p in metrics["phase_b_pairwise"]:
            rows.append({
                "Pair": p["pair"],
                "Samples": p["n_common"],
                "Mean Tau": f"{p['mean_tau']:.3f}",
                "Mean Rho": f"{p['mean_rho']:.3f}",
                "Same #1": f"{p['same_top1']}/{p['n_common']} ({p['same_top1_pct']:.1f}%)",
            })
        st.table(rows)

    if metrics["phase_b_kendall_w"]:
        kw = metrics["phase_b_kendall_w"]
        st.markdown("### Kendall's W (overall concordance)")
        st.markdown(
            f"**W = {kw['mean_W']:.3f}** (std={kw['std_W']:.3f}) "
            f"across {kw['n_annotators']} annotators, {kw['n_samples']} common samples"
        )
        st.markdown(
            f"Perfect agreement on {kw['n_perfect']}/{kw['n_samples']} samples."
        )
        if kw["mean_W"] < 0.3:
            st.warning("W < 0.3: weak agreement.")
        elif kw["mean_W"] < 0.5:
            st.info("W 0.3-0.5: moderate agreement.")
        elif kw["mean_W"] < 0.7:
            st.info("W 0.5-0.7: moderate-good agreement.")
        else:
            st.success("W >= 0.7: strong agreement.")

    # --- Average rank per method ---
    if metrics["phase_b_avg_ranks"]:
        st.markdown("### Average Rank per Method")
        st.markdown("Lower rank = better (1 = best, 4 = worst).")

        method_labels = {"shap": "SHAP", "lime": "LIME", "ig": "Int. Gradients", "attention": "Attention"}
        rank_rows = []
        for ann in sorted(metrics["phase_b_avg_ranks"].keys()):
            if ann == "Overall":
                continue
            row = {"Annotator": ann}
            for m, label in method_labels.items():
                row[label] = f"{metrics['phase_b_avg_ranks'][ann][m]:.2f}"
            rank_rows.append(row)
        # Overall row
        row = {"Annotator": "Overall"}
        for m, label in method_labels.items():
            row[label] = f"{metrics['phase_b_avg_ranks']['Overall'][m]:.2f}"
        rank_rows.append(row)
        st.table(rank_rows)

        # Chart: average rank per method (overall)
        overall = metrics["phase_b_avg_ranks"]["Overall"]
        chart_df = pd.DataFrame({
            "Method": [method_labels[m] for m in overall],
            "Avg Rank": [overall[m] for m in overall],
        })
        st.bar_chart(chart_df.set_index("Method"))


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="XAI Comparison Annotation",
        page_icon="🔬",
        layout="wide",
    )

    # Top-level page selector in sidebar
    st.sidebar.markdown("## Mode")
    page = st.sidebar.radio("", ["Annotation", "IAA Dashboard"], label_visibility="collapsed")

    if page == "IAA Dashboard":
        render_iaa_dashboard()
        return

    # Round selection
    st.sidebar.markdown("## Round")
    round_num = st.sidebar.radio(
        "Annotation round", [1, 2],
        format_func=lambda x: f"Round {x}" + (" (heatmaps)" if x == 1 else " (top-5 bars)"),
        label_visibility="collapsed",
    )

    if round_num == 1:
        st.title("Explainable AI Comparison Annotation")
        st.markdown(
            "Compare how different explanation methods highlight important words "
            "in emotion predictions. **2 phases per sample.**"
        )
    else:
        st.title("XAI Annotation — Round 2")
        st.markdown(
            "Compare explanation methods using their **top-5 most important words**. "
            "**2 phases per sample.**"
        )

    # Load data for selected round
    samples = load_samples(round_num)
    sample_map = {s["id"]: s for s in samples}
    sample_ids = [s["id"] for s in samples]

    # Sidebar: annotator selection
    st.sidebar.markdown("## Settings")
    annotator = st.sidebar.selectbox("Annotator", ANNOTATOR_NAMES)

    # Progress tracking
    completed_a = load_completed_samples(annotator, "phase_a", round_num)
    completed_b = load_completed_samples(annotator, "phase_b", round_num)
    fully_done = completed_a & completed_b

    n_total = len(sample_ids)
    n_done = len(fully_done)

    st.sidebar.markdown("## Progress")
    st.sidebar.progress(n_done / n_total if n_total > 0 else 0)
    st.sidebar.markdown(f"**{n_done} / {n_total}** samples completed")
    st.sidebar.markdown(f"- Phase A: {len(completed_a)} done")
    st.sidebar.markdown(f"- Phase B: {len(completed_b)} done")

    # Emotion filter
    emotions = sorted(set(s["label"] for s in samples))
    emotion_filter = st.sidebar.selectbox("Filter by emotion", ["All"] + emotions)

    if emotion_filter != "All":
        filtered_ids = [s["id"] for s in samples if s["label"] == emotion_filter]
    else:
        filtered_ids = sample_ids

    # Show only incomplete samples option
    show_incomplete = st.sidebar.checkbox("Show only incomplete", value=True)
    if show_incomplete:
        filtered_ids = [sid for sid in filtered_ids if sid not in fully_done]

    if not filtered_ids:
        st.balloons()
        st.success("All samples are complete! Thank you!")
        return

    # ------------------------------------------------------------------
    # Navigation — tracked by SAMPLE ID, no selectbox (avoids state drift)
    # ------------------------------------------------------------------
    st.sidebar.markdown("## Navigation")
    filtered_id_set = set(filtered_ids)

    # Detect annotator or round change → reset navigation
    if st.session_state.get("_prev_annotator") != annotator or \
       st.session_state.get("_prev_round") != round_num:
        st.session_state["_prev_annotator"] = annotator
        st.session_state["_prev_round"] = round_num
        st.session_state.pop("_current_sample_id", None)

    # Determine which sample to show
    if "_current_sample_id" not in st.session_state or \
       st.session_state["_current_sample_id"] not in filtered_id_set:
        # First load OR current sample was just completed (dropped from list)
        # → find the first incomplete sample AFTER the last completed one
        if fully_done:
            last_done_pos = max(
                (sample_ids.index(sid) for sid in fully_done if sid in set(sample_ids)),
                default=-1,
            )
            resume_id = None
            for sid in filtered_ids:
                if sample_ids.index(sid) > last_done_pos:
                    resume_id = sid
                    break
            st.session_state["_current_sample_id"] = resume_id or filtered_ids[0]
        else:
            st.session_state["_current_sample_id"] = filtered_ids[0]

    current_id = st.session_state["_current_sample_id"]
    current_idx = filtered_ids.index(current_id)
    current_label = samples[sample_ids.index(current_id)]["label"]

    # Show current position
    st.sidebar.markdown(
        f"**Sample {current_id}** ({current_label})"
        f"  —  {current_idx + 1} / {len(filtered_ids)}"
    )

    # Navigation buttons
    nav_col1, nav_col2 = st.sidebar.columns(2)
    with nav_col1:
        if st.button("← Previous", disabled=current_idx == 0):
            st.session_state["_current_sample_id"] = filtered_ids[current_idx - 1]
            st.rerun()
    with nav_col2:
        if st.button("Next →", disabled=current_idx >= len(filtered_ids) - 1):
            st.session_state["_current_sample_id"] = filtered_ids[current_idx + 1]
            st.rerun()

    sample = sample_map[current_id]

    # Show sample header
    st.markdown(f"## Sample {current_id}")

    # Phase A
    phase_a_done = current_id in completed_a or render_phase_a(sample, annotator, round_num)
    if current_id in completed_a:
        st.markdown("---")
        st.markdown("### Phase A: Word Highlighting")
        st.success("Phase A already completed for this sample.")

    # Phase B (only show after Phase A) — Round 2 uses top-5 bar viz
    if phase_a_done:
        if round_num == 2:
            phase_b_done = current_id in completed_b or render_phase_b_round2(sample, annotator)
        else:
            phase_b_done = current_id in completed_b or render_phase_b(sample, annotator)
        if current_id in completed_b:
            st.markdown("---")
            st.markdown("### Phase B: Explanation Ranking")
            st.success("Phase B already completed for this sample.")
    else:
        st.info("Complete Phase A to unlock Phase B.")
        phase_b_done = False

    # After Phase B, sample is complete
    if phase_b_done:
        st.markdown("---")
        st.success("All phases complete for this sample!")


if __name__ == "__main__":
    main()
