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
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLES_JSON = DATA_DIR / "selected_samples.json"
METHODS = ["shap", "lime", "ig", "attention"]
METHOD_DISPLAY = {"shap": "SHAP", "lime": "LIME", "ig": "Integrated Gradients", "attention": "Attention"}
MAX_HIGHLIGHT = 5

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
def load_samples():
    with open(SAMPLES_JSON) as f:
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
def get_gsheet_client():
    if not GSPREAD_AVAILABLE:
        return None
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
    except (FileNotFoundError, KeyError):
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
def save_annotation(annotator, sample_id, phase, data):
    """Save one annotation row. Works locally always; cloud if configured."""
    timestamp = datetime.now().isoformat()
    row = {
        "annotator": annotator,
        "sample_id": sample_id,
        "phase": phase,
        "timestamp": timestamp,
        **data,
    }

    # Local CSV
    csv_path = OUTPUT_DIR / f"annotations_{annotator.lower()}_{phase}.csv"
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
                f"{phase}_{annotator}",
                list(row.keys()),
            )
            ws.append_row(list(row.values()))
        except Exception as e:
            st.warning(f"Cloud save failed: {e}")

    # Update session state cache so load_completed_samples sees this immediately
    gsheet_key = f"_gsheet_completed_{annotator}_{phase}"
    if gsheet_key in st.session_state:
        st.session_state[gsheet_key].add(sample_id)

    return True


def _load_completed_from_gsheet(annotator, phase):
    """Load completed sample IDs from Google Sheets (survives app restarts)."""
    client = get_gsheet_client()
    if not client:
        st.sidebar.caption(f"DEBUG: No gsheet client for {phase}_{annotator}")
        return set()
    try:
        sh = client.open("XAI_Comparison_Annotations")
        ws = sh.worksheet(f"{phase}_{annotator}")
        # sample_id is column 2 (annotator, sample_id, phase, timestamp, ...)
        all_values = ws.col_values(2)
        completed = set()
        for val in all_values[1:]:  # skip header
            try:
                completed.add(int(val))
            except (ValueError, TypeError):
                pass
        st.sidebar.caption(f"DEBUG: Loaded {len(completed)} from GSheet {phase}_{annotator}")
        return completed
    except Exception as e:
        st.sidebar.caption(f"DEBUG: GSheet error {phase}_{annotator}: {e}")
        return set()


def load_completed_samples(annotator, phase):
    """Load set of sample IDs already annotated for this phase.
    Checks local CSV (current session) + Google Sheets (persistent across restarts)."""
    # Google Sheets — load once per session, cached in session state
    gsheet_key = f"_gsheet_completed_{annotator}_{phase}"
    if gsheet_key not in st.session_state:
        st.session_state[gsheet_key] = _load_completed_from_gsheet(annotator, phase)
    completed = set(st.session_state[gsheet_key])

    # Local CSV — has current session's annotations
    csv_path = OUTPUT_DIR / f"annotations_{annotator.lower()}_{phase}.csv"
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
def render_phase_a(sample, annotator):
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

    # Use a dict {word_idx: rank_number} so rank is an explicit integer,
    # not dependent on list/dict ordering which Streamlit can lose across reruns.
    if f"{key_prefix}_ranks" not in st.session_state:
        st.session_state[f"{key_prefix}_ranks"] = {}

    ranks = st.session_state[f"{key_prefix}_ranks"]

    # Render clickable words
    words_per_row = 10
    for row_start in range(0, len(words), words_per_row):
        row_words = words[row_start:min(row_start + words_per_row, len(words))]
        cols = st.columns(len(row_words))
        for j, (col, word) in enumerate(zip(cols, row_words)):
            word_idx = row_start + j
            is_selected = word_idx in ranks
            with col:
                # Show rank number on selected buttons
                btn_label = f"{ranks[word_idx]}. {word}" if is_selected else word
                if st.button(
                    btn_label,
                    key=f"{key_prefix}_word_{word_idx}",
                    type="primary" if is_selected else "secondary",
                    use_container_width=True,
                ):
                    if word_idx in ranks:
                        # Deselect: remove and compact remaining ranks
                        removed_rank = ranks.pop(word_idx)
                        for idx in ranks:
                            if ranks[idx] > removed_rank:
                                ranks[idx] -= 1
                    elif len(ranks) < MAX_HIGHLIGHT:
                        # Select: assign next rank number
                        ranks[word_idx] = len(ranks) + 1
                    else:
                        st.toast(f"Maximum {MAX_HIGHLIGHT} words. Deselect one first.", icon="⚠️")
                    st.rerun()

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
        })
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
# Main App
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="XAI Comparison Annotation",
        page_icon="🔬",
        layout="wide",
    )

    st.title("Explainable AI Comparison Annotation")
    st.markdown(
        "Compare how different explanation methods highlight important words "
        "in emotion predictions. **2 phases per sample.**"
    )

    # Load data
    samples = load_samples()
    sample_map = {s["id"]: s for s in samples}
    sample_ids = [s["id"] for s in samples]

    # Sidebar: annotator selection
    st.sidebar.markdown("## Settings")
    annotator = st.sidebar.selectbox("Annotator", ANNOTATOR_NAMES)

    # Progress tracking
    completed_a = load_completed_samples(annotator, "phase_a")
    completed_b = load_completed_samples(annotator, "phase_b")
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

    # Sample navigation
    st.sidebar.markdown("## Navigation")

    nav_key = "_sample_nav_selectbox"

    # Apply programmatic navigation from Prev/Next buttons
    if "_nav_target" in st.session_state:
        target = st.session_state.pop("_nav_target")
        target = max(0, min(target, len(filtered_ids) - 1))
        st.session_state[nav_key] = target

    # Clamp existing selectbox state to valid range (list may have shrunk after completion)
    if nav_key in st.session_state:
        val = st.session_state[nav_key]
        if val >= len(filtered_ids):
            st.session_state[nav_key] = max(0, len(filtered_ids) - 1)

    current_idx = st.sidebar.selectbox(
        "Sample",
        range(len(filtered_ids)),
        format_func=lambda i: f"Sample {filtered_ids[i]} ({samples[sample_ids.index(filtered_ids[i])]['label']})",
        key=nav_key,
    )
    current_id = filtered_ids[current_idx]
    sample = sample_map[current_id]

    # Navigation buttons
    nav_col1, nav_col2 = st.sidebar.columns(2)
    with nav_col1:
        if current_idx > 0:
            if st.button("Previous"):
                st.session_state["_nav_target"] = current_idx - 1
                st.rerun()
    with nav_col2:
        if current_idx < len(filtered_ids) - 1:
            if st.button("Next"):
                st.session_state["_nav_target"] = current_idx + 1
                st.rerun()

    # Show sample header
    st.markdown(f"## Sample {current_id}")

    # Phase A
    phase_a_done = current_id in completed_a or render_phase_a(sample, annotator)
    if current_id in completed_a:
        st.markdown("---")
        st.markdown("### Phase A: Word Highlighting")
        st.success("Phase A already completed for this sample.")

    # Phase B (only show after Phase A)
    if phase_a_done:
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
