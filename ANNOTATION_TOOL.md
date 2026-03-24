# XAI Comparison Annotation Tool

## Overview

A Streamlit app for human evaluation of 4 XAI explanation methods (SHAP, LIME, Integrated Gradients, Attention) on emotion classification predictions. Part of Paper 2.

**App file:** `annotation/xai_comparison/xai_comparison_annotation.py`

---

## Purpose

The tool collects two types of human judgment per sample:

1. **Phase A — Word Highlighting:** Unbiased ground truth. Annotator selects up to 5 words most responsible for the predicted emotion, without seeing any XAI explanations.
2. **Phase B — Explanation Ranking:** Annotator sees 4 anonymized XAI explanations and ranks them 1 (best) to 4 (worst).

These answer two research questions:
- **RQ3a:** Which method's top features align best with human intuition? (Phase A)
- **RQ3b:** Which method do humans prefer? (Phase B)

---

## Annotation Rounds

The tool supports two annotation rounds with separate data and visualizations:

### Round 1 (heatmaps)
- **Data:** `data/selected_samples.json` — 200 samples, seed=42
- **Phase B visualization:** Full-word heatmaps (all words colored by attribution score)
- **Google Sheets worksheets:** `phase_a_{annotator}`, `phase_b_{annotator}`
- **Status:** Benni 48, Emilia 200, Vanessa 200 (as of 2026-03-24)

### Round 2 (top-5 bars)
- **Data:** `data/selected_samples_round2.json` — 200 NEW samples, seed=99, zero overlap with Round 1
- **Phase B visualization:** Top-5 horizontal bar charts (only the 5 most important words per method, proportional bars, no numeric scores)
- **Google Sheets worksheets:** `r2_phase_a_{annotator}`, `r2_phase_b_{annotator}`
- **Status:** Not started

### Why Round 2 changed the visualization

Raw attribution scores are **not comparable across methods**:
- IG scores are ~8x larger than SHAP scores (different mathematical properties: IG sums to `f(x) - f(baseline)`, SHAP sums to `f(x) - E[f(x)]`)
- LIME coefficients depend on local surrogate regularization
- Attention weights are probabilities (sum to 1 per head)

Round 1's full heatmaps also caused confusion: Attention colors almost every word (avg 16.9 visible), while SHAP highlights only 1-3 words. Round 2 shows only the top 5 words per method with proportional bars (no numbers), making methods directly comparable without magnitude bias.

---

## Data

### Input

| File | Round | Description |
|------|-------|-------------|
| `data/selected_samples.json` | 1 | 200 samples (20 per emotion x 10 emotions), seed=42 |
| `data/selected_samples_round2.json` | 2 | 200 samples (20 per emotion x 10 emotions), seed=99, no overlap with Round 1 |

**Sample selection:** Random, stratified by emotion, fixed seed. Script: `scripts/select_annotation_samples.py`.

**Emotions included (10):** anger, boredom, disgust, fear, joy, pride, relief, sadness, surprise, trust. Excluded: guilt (1 sample), no-emotion (7 samples).

### Output

| File | Phase | Contents |
|------|-------|----------|
| `output/annotations_{name}_phase_a.csv` | A (R1) | Annotator name, sample ID, ranked word indices and words |
| `output/annotations_{name}_phase_b.csv` | B (R1) | Rankings per real method (rank_shap, rank_lime, rank_ig, rank_attention) |
| `output/annotations_{name}_r2_phase_a.csv` | A (R2) | Same format as R1 Phase A |
| `output/annotations_{name}_r2_phase_b.csv` | B (R2) | Same format as R1 Phase B |

All CSVs also include: sentence, predicted emotion, timestamp, and (for B) the shuffled method order.

**Persistence:** Google Sheets is the primary data store (survives Streamlit Cloud restarts). Local CSVs are also written as backup. The app also supports a local service account fallback (`annotation/gcp_service_account.json`) when `st.secrets` is unavailable.

---

## Annotation Phases

### Phase A — Word Highlighting (same for both rounds)

- **Shown:** Sentence (large, prominent), predicted emotion
- **NOT shown:** Confidence, correct/wrong, any XAI explanations
- **Task:** Click up to 5 words in order of importance (most important first)
- **Design rationale:** No XAI visualizations are shown to avoid biasing annotators toward any method. This produces method-agnostic ground truth.
- **Analysis:** Compute precision@5 and recall@5 of each method's top-5 features against the human-selected words

### Phase B — Explanation Ranking

#### Round 1: Full heatmaps
- **Shown:** Same sentence + 4 heatmaps in a 2x2 grid (all words colored)
- **Heatmap colors:** Red = pushes toward prediction, Blue = pushes against, White = neutral

#### Round 2: Top-5 bars
- **Shown:** Same sentence + 4 top-5 bar charts in a 2x2 grid
- **Bar display:** Each method shows its 5 highest-contributing words as horizontal bars
- **Bar colors:** Red = pushes toward prediction, Blue = pushes against
- **No numeric scores** — bars are proportional within each method to avoid cross-method magnitude bias

#### Both rounds
- **Labels:** "Method A", "Method B", "Method C", "Method D" (anonymized, shuffled per sample)
- **Task:** Rank 1 (best explanation) to 4 (worst), each rank used exactly once
- **Design rationale:** Anonymization prevents name-recognition bias. Shuffling per sample prevents pattern-tracking.
- **Analysis:** Average rank per method, Kendall's W for annotator agreement

---

## IAA Dashboard

The tool includes a built-in Inter-Annotator Agreement dashboard (accessible via sidebar toggle).

### Metrics computed
- **Per-annotator progress:** Phase A count, Phase B count, fully completed count
- **Phase A — Word Highlighting Agreement:**
  - Pairwise Jaccard similarity (|common words| / |all words selected by either|)
  - Krippendorff's alpha (binary, per word position)
- **Phase B — Ranking Agreement:**
  - Pairwise Kendall's tau and Spearman's rho
  - Kendall's W (overall concordance across all annotators)
  - Same #1 pick rate
  - Average rank per method per annotator

### Round 1 IAA results (as of 2026-03-24, 3 annotators)

| Metric | Value |
|--------|-------|
| Krippendorff's alpha (Phase A, all 3) | 0.527 |
| Best pair Jaccard (Emilia-Vanessa) | 0.611 |
| Kendall's W (Phase B, all 3) | 0.617 |
| Best pair same #1 (Emilia-Vanessa) | 57.0% |
| SHAP avg rank (overall) | 1.73 (best) |

### Data source
- Loads live from Google Sheets (cached 2 min)
- Supports round selection (Round 1 / Round 2)
- Refresh button to reload latest data

---

## Method Anonymization

Methods are anonymized as "Method A/B/C/D" with the order **shuffled per sample** using a deterministic seed (`sample_id * 7 + 13`). This means:

- Sample 20: Method A=LIME, B=IG, C=SHAP, D=Attention
- Sample 21: Method A=Attention, B=LIME, C=IG, D=SHAP
- Sample 99: Method A=SHAP, B=LIME, C=Attention, D=IG

The annotator never sees real method names. When saving annotations, the app **maps back** to real method names automatically:
- Phase B saves: `rank_shap=2, rank_lime=1, rank_ig=3, rank_attention=4`

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

**All 4 annotators see all 200 samples per round** (unlike Paper 1 where samples were split). This enables inter-annotator agreement computation on the full dataset.

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
3. Configure Google Sheets secrets in Streamlit Cloud dashboard (Settings > Secrets):
   ```toml
   [gcp_service_account]
   type = "service_account"
   project_id = "..."
   private_key = "..."
   client_email = "..."
   ...
   ```
4. The app saves to Google Sheet **`XAI_Comparison_Annotations`** and creates worksheets automatically per round and phase

---

## File Structure

```
annotation/xai_comparison/
├── xai_comparison_annotation.py    # Main Streamlit app (all logic)
├── streamlit_app.py                # Entry point redirect
├── requirements.txt                # Dependencies (streamlit, gspread, google-auth, scipy, numpy, pandas)
├── ANNOTATION_TOOL.md              # This file
├── ANNOTATION_GUIDELINES.md        # Annotator instructions
├── .streamlit/
│   └── config.toml                 # Theme configuration
├── data/
│   ├── selected_samples.json       # Round 1: 200 samples (seed=42)
│   └── selected_samples_round2.json # Round 2: 200 samples (seed=99)
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
- Switching annotators or rounds resets navigation to the correct resume point

### Design decisions (2026-03-23)
- **No `st.selectbox` for navigation.** Streamlit ignores `st.session_state[key]` overrides after a widget's first render.
- **No `number_input` with `on_change` for jumping.** When annotator switches change `max_value`, Streamlit silently clamps the stored value.
- **`streamlit_app.py` is a redirect only.** Streamlit Cloud defaults to `streamlit_app.py` as the entry point.

---

## Differences from Paper 1 Annotation Tool

| Aspect | Paper 1 | Paper 2 (this tool) |
|--------|---------|---------------------|
| Purpose | Judge SHAP explanations (correct/wrong reason) | Compare 4 XAI methods |
| Methods shown | Normal SHAP + Relational SHAP | SHAP, LIME, IG, Attention (anonymized) |
| Annotation type | Per-feature correct/wrong + overall judgment | Word selection + ranking |
| Samples | 2,861 (split across annotators) | 200 per round (all annotators see all) |
| XAI bias | Annotators see SHAP visualizations | Phase A has no visualizations; Phase B is anonymized |
| Visualization | Plotly bar charts | R1: heatmaps, R2: top-5 bars (no scores) |

---

## Annotation Progress (as of 2026-03-24)

### Round 1

| Annotator | Phase A | Phase B | Fully Done |
|-----------|---------|---------|------------|
| Benni     | 48      | 47      | 47         |
| Emilia    | 196     | 200     | 196        |
| Vanessa   | 200     | 200     | 200        |
| Anna      | 0       | 0       | 0          |

### Round 2

Not started.

*Created: 2026-03-17, Updated: 2026-03-24*
