# Annotation Guidelines for Roman

## How to get started

1. Open the tool: [XAI Comparison Annotation Tool](https://xai-annotation-tool-v3.streamlit.app/)
2. Select **"Roman"** from the dropdown in the sidebar.
3. You will see your 20 assigned samples (2 per emotion).
4. You can switch between **Round 1** and **Round 2** using the round selector at the top.

## What this is about

We have an emotion classification model (BERT, 15 emotions) that predicts emotions from text. Four different explanation methods (SHAP, LIME, Integrated Gradients, Attention) try to show which words the model considers important. Your job is to (1) tell us which words **you** think are important, and (2) evaluate the explanations.

## Each sample has 2 phases

| Phase | What you do | What you see |
|-------|-------------|--------------|
| **Phase A** | Select the most important words | Sentence + predicted emotion only (no explanations) |
| **Phase B** | Rank 4 explanations | Sentence + 4 visualizations (anonymized as Method A/B/C/D) |

You must complete Phase A before Phase B unlocks.

## Phase A -- Word Highlighting

- You see the sentence and the predicted emotion. No explanations are shown.
- **Click up to 5 words** that you think are most important for expressing the predicted emotion.
- Click them in order of importance: 1st click = most important, 2nd = second most important, etc.
- Ask yourself: **Which words in this sentence carry the emotion?**

Example:
> "We had to organize a very large funeral." -- Predicted: **sadness**
>
> Good selection: **1. funeral** (signals death/loss)
> Poor selection: "organize" or "large" (logistics, not sadness)

Click **"Submit Phase A"** when done.

## Phase B -- Explanation Ranking

- You see the same sentence plus 4 explanation visualizations labeled Method A, B, C, D (anonymized, shuffled per sample).
- **Round 1** shows full-word heatmaps (all words colored by importance).
- **Round 2** shows top-5 bar charts (only the 5 most important words per method, no numeric scores).

In both rounds:
- **Red** = pushes toward the predicted emotion (supports the prediction)
- **Blue** = pushes against the predicted emotion
- **White / no color** = neutral

**Rank the 4 explanations from best (1) to worst (4).** Each rank used exactly once.

A good explanation highlights words that genuinely relate to the emotion. A bad one highlights irrelevant words or is confusing.

Click **"Submit Phase B"** when done.

## Saving and navigation

- Click **Submit** to save each phase. Your work is saved locally.
- Use Previous/Next buttons or the sidebar to navigate between samples.
- You can go back to any sample.
- The "Show only incomplete" checkbox (enabled by default) helps you find remaining samples.

## Tips

- Focus on the red/highlighted features -- do they make sense for the emotion?
- Do not try to guess which method is which. Just evaluate what you see.
- Some methods highlight many words lightly, others highlight just 1-2 words strongly. Both are normal.
- Trust your intuition. If an explanation feels wrong, rank it lower.

If anything is unclear, just ask. Thank you!
