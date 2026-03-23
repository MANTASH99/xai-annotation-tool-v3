# XAI Comparison Annotation Tool

## Overview

A Streamlit app for human evaluation of 4 XAI explanation methods (SHAP, LIME, Integrated Gradients, Attention) on emotion classification predictions. Part of Paper 2.

**App file:** `annotation/xai_comparison/xai_comparison_annotation.py`

---

## Purpose

The tool collects three types of human judgment per sample:

1. **Phase A — Word Highlighting:** Unbiased ground truth. Annotator selects the 5 words most responsible for the predicted emotion, without seeing any XAI explanations.
2. **Phase B — Explanation Ranking:** Annotator sees 4 anonymized heatmaps and ranks them 1 (best) to 4 (worst).
3. **Phase C — Similarity Grouping:** Annotator groups heatmaps that look similar to each other.

These answer three research questions:
- **RQ3a:** Which method's top features align best with human intuition? (Phase A)
- **RQ3b:** Which method do humans prefer? (Phase B)
- **RQ3c:** Do humans perceive the same inter-method similarity as computational metrics? (Phase C)

---

## Data

### Input

| File | Description |
|------|-------------|
| `data/selected_samples.json` | 200 samples (20 per emotion × 10 emotions), each with all 4 methods' word-level scores |

**Sample selection:** Random, stratified by emotion, fixed seed (42). Script: `scripts/select_annotation_samples.py`.

**Emotions included (10):** anger, boredom, disgust, fear, joy, pride, relief, sadness, surprise, trust. Excluded: guilt (1 sample), no-emotion (7 samples).

### Output

| File | Phase | Contents |
|------|-------|----------|
| `output/annotations_{name}_phase_a.csv` | A | Annotator name, sample ID, selected word indices and words |
| `output/annotations_{name}_phase_b.csv` | B | Rankings per real method (rank_shap, rank_lime, rank_ig, rank_attention) |
| `output/annotations_{name}_phase_c.csv` | C | Group assignments per real method, number of groups, optional comment |

All CSVs also include: sentence, predicted emotion, timestamp, and (for B/C) the shuffled method order.

**Persistence:** Google Sheets is the primary data store (survives Streamlit Cloud restarts). Local CSVs are also written as backup. On startup, the app loads completed sample IDs from Google Sheets (cached per session in `st.session_state`), determines each annotator's progress, and resumes at the first incomplete sample after the last completed one.

---

## Annotation Phases

### Phase A — Word Highlighting

- **Shown:** Sentence (large, prominent), predicted emotion
- **NOT shown:** Confidence, correct/wrong, any XAI heatmaps
- **Task:** Click exactly 5 words that the annotator thinks are most responsible for the emotion
- **Design rationale:** No XAI visualizations are shown to avoid biasing annotators toward any method. This produces method-agnostic ground truth.
- **Analysis:** Compute precision@5 and recall@5 of each method's top-5 features against the human-selected words

### Phase B — Explanation Ranking

- **Shown:** Same sentence + 4 heatmaps in a 2×2 grid
- **Heatmap colors:** Red = pushes toward prediction, Blue = pushes against, White = neutral
- **Labels:** "Method A", "Method B", "Method C", "Method D" (anonymized, shuffled per sample)
- **Task:** Rank 1 (best explanation) to 4 (worst), each rank used exactly once
- **Design rationale:** Anonymization prevents name-recognition bias (e.g., favoring SHAP because it's well-known). Shuffling per sample prevents pattern-tracking across samples.
- **Analysis:** Average rank per method, Kendall's W for annotator agreement

### Phase C — Similarity Grouping

- **Shown:** Same 4 heatmaps displayed in a row
- **Task:** Assign each method to a group (Group 1–4). Same group = similar-looking explanations.
- **Design rationale:** Ranking only tells us "which is best", not "which look alike." Grouping captures perceived similarity, which we compare to computational FA@5 and Spearman metrics.
- **Analysis:** Co-occurrence matrix (how often each method pair is grouped together), compare with computational agreement metrics

---

## Method Anonymization

Methods are anonymized as "Method A/B/C/D" with the order **shuffled per sample** using a deterministic seed (`sample_id * 7 + 13`). This means:

- Sample 20: Method A=LIME, B=IG, C=SHAP, D=Attention
- Sample 21: Method A=Attention, B=LIME, C=IG, D=SHAP
- Sample 99: Method A=SHAP, B=LIME, C=Attention, D=IG

The annotator never sees real method names. When saving annotations, the app **maps back** to real method names automatically:
- Phase B saves: `rank_shap=2, rank_lime=1, rank_ig=3, rank_attention=4`
- Phase C saves: `group_shap=Group 1, group_lime=Group 1, group_ig=Group 2, group_attention=Group 2`

The `method_order` column in the CSV records the shuffle for each sample, allowing full reconstruction.

---

## Annotators

Same 4 annotators as Paper 1:

| Annotator | Background |
|-----------|------------|
| Benni | Business Informatics |
| Emilia | Digital Humanities |
| Vanessa | Digital Humanities |
| Anna | Digital Humanities |

**All 4 annotators see all 200 samples** (unlike Paper 1 where samples were split). This enables inter-annotator agreement computation on the full dataset.

---

## Method Sparsity (What Annotators Will See)

Different methods produce very different heatmap patterns:

| Method | Avg visible words | Pattern |
|--------|-------------------|---------|
| SHAP | 7.8 | Sparse — 1-3 strongly colored words, rest white |
| LIME | 11.5 | Moderate — several colored words, some blue (negative) |
| IG | 14.0 | Broad — many colored words, larger magnitude range |
| Attention | 16.9 | Diffuse — many words lightly colored, never blue (always positive) |

"Visible" = absolute score > 10% of the max for that method on that sample.

SHAP heatmaps will often show only 1-2 strongly highlighted words. Attention heatmaps will show many words with light coloring. This is a real methodological difference, not a bug.

---

## How to Run

### Locally
```bash
source ~/.venv/bin/activate
streamlit run annotation/xai_comparison/xai_comparison_annotation.py
```

### On Streamlit Cloud
1. Push the `annotation/xai_comparison/` directory to GitHub
   - **GitHub repo:** `MANTASH99/xai-annotation-tool-v3`
2. Streamlit Cloud runs `streamlit_app.py` by default, which redirects to `xai_comparison_annotation.py`
3. Configure Google Sheets secrets in Streamlit Cloud dashboard (Settings → Secrets):
   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key = "..."
   client_email = "..."
   ...
   ```
4. The app saves to Google Sheet **`XAI_Comparison_Annotations`** and creates worksheets automatically: `phase_a_{annotator}`, `phase_b_{annotator}`, `phase_c_{annotator}`

---

## File Structure

```
annotation/xai_comparison/
├── xai_comparison_annotation.py    # Main Streamlit app (all logic lives here)
├── streamlit_app.py                # Entry point redirect (imports main() from above)
├── requirements.txt                # Dependencies (streamlit, gspread, google-auth)
├── ANNOTATION_TOOL.md              # This file
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── data/
│   └── selected_samples.json       # 200 samples with all 4 methods' scores
└── output/
    └── annotations_*.csv           # Local CSV backup (created at runtime)
```

---

## Navigation & Session Persistence

### How it works
- Navigation is tracked by **sample ID** (`_current_sample_id` in `st.session_state`), not by list index
- On startup, the app loads completed samples from Google Sheets, finds the **last completed sample** for the annotator, and resumes at the next incomplete one
- Word buttons use `on_click` callbacks (not `st.rerun()`) to avoid double-reruns that corrupt widget state
- Previous/Next buttons update `_current_sample_id` directly
- Switching annotators resets navigation to the correct resume point for the new annotator

### Design decisions (2026-03-23)
- **No `st.selectbox` for navigation.** Streamlit ignores `st.session_state[key]` overrides after a widget's first render. This caused the selectbox to drift to a different sample on any button click.
- **No `number_input` with `on_change` for jumping.** When annotator switches change `max_value`, Streamlit silently clamps the stored value and fires `on_change`, overriding `_current_sample_id`.
- **`streamlit_app.py` is a redirect only.** Streamlit Cloud defaults to `streamlit_app.py` as the entry point. All logic is in `xai_comparison_annotation.py` to keep a single source of truth.

---

## Differences from Paper 1 Annotation Tool

| Aspect | Paper 1 | Paper 2 (this tool) |
|--------|---------|---------------------|
| Purpose | Judge SHAP explanations (correct/wrong reason) | Compare 4 XAI methods |
| Methods shown | Normal SHAP + Relational SHAP | SHAP, LIME, IG, Attention (anonymized) |
| Annotation type | Per-feature correct/wrong + overall judgment | Word selection + ranking + grouping |
| Samples | 2,861 (split across annotators) | 200 (all annotators see all) |
| XAI bias | Annotators see SHAP visualizations | Phase A has no visualizations; Phases B/C are anonymized |
| Visualization | Plotly bar charts | Inline HTML word heatmaps |

---

## Annotation Progress (as of 2026-03-23)

| Annotator | Phase A | Phase B | Fully Done | Resume At |
|-----------|---------|---------|------------|-----------|
| Benni     | 25      | 24      | 24         | 540       |
| Emilia    | 51      | 51      | 51         | 782       |
| Vanessa   | 51      | 51      | 51         | 782       |
| Anna      | 0       | 0       | 0          | 20        |

*Created: 2026-03-17, Updated: 2026-03-23*
