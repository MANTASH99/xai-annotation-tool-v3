# XAI Comparison Annotation Guidelines

## 1. Welcome

Hi Benni, Emilia, Vanessa, and Anna! Thank you for helping with this annotation task. We have an emotion classification model (BERT) that predicts emotions from text. In this study, we are comparing **four different explanation methods** that try to show which words the model considers important. Your job is to tell us which words **you** think are important, and then evaluate the explanations.

---

## 2. How to Access the Tool

1. Open the annotation tool in your browser: **[XAI Comparison Annotation Tool](https://xai-annotation-tool-v2.streamlit.app/)**
2. In the **sidebar** on the left, select your name from the dropdown.
3. All 200 samples are visible to every annotator.

---

## 3. Overview

Each sample has **2 phases**:

| Phase | What you do | What you see |
|-------|-------------|--------------|
| **Phase A** | Select the most important words | Sentence + predicted emotion only (no explanations) |
| **Phase B** | Rank 4 explanation heatmaps | Sentence + 4 colored heatmaps (anonymized) |

You must complete Phase A before Phase B unlocks for each sample.

---

## 4. Phase A -- Word Highlighting

### What you see

- The **sentence** the model classified
- The **predicted emotion** (e.g., sadness, joy, anger)
- A row of clickable word buttons

### What you do NOT see

- No explanations, no heatmaps, no confidence scores. This is intentional -- we want your unbiased human judgment.

### Your task

**Click up to 5 words** that you think are the most important for expressing the predicted emotion. Click them **in order of importance**:

- **1st click** = most important word (Rank 1)
- **2nd click** = second most important (Rank 2)
- **3rd click** = third most important (Rank 3)
- and so on, up to 5

The rank number will appear on each button after you click it (e.g., "1. funeral", "2. sad").

### How to change your selection

- **Deselect a word:** Click a selected (blue) word again to remove it. The remaining words will be re-ranked automatically.
- **Change the order:** Deselect the words you want to reorder and click them again in the correct order.

### What to look for

Ask yourself: **Which words in this sentence carry the emotion?**

Example:

> "We had to organize a very large funeral."
> Predicted emotion: **sadness**

Good selection: **1. funeral** (directly signals death/loss/grief)

Poor selection: "organize" or "large" (these relate to logistics, not sadness)

### When you are done

Click **"Submit Phase A"** to save and unlock Phase B.

---

## 5. Phase B -- Explanation Ranking

### What you see

- The same sentence and predicted emotion
- **4 heatmaps** in a 2x2 grid, each from a different explanation method
- Methods are labeled **Method A, Method B, Method C, Method D** (anonymized -- you do not know which method is which, and the order changes for every sample)

### Reading the heatmaps

Each heatmap highlights words with colors:

| Color | Meaning |
|-------|---------|
| **Red / dark red** | Pushes the model **toward** the predicted emotion (supports the prediction) |
| **Blue / dark blue** | Pushes the model **against** the predicted emotion (contradicts the prediction) |
| **White / no color** | Neutral -- the model does not consider this word important |

Stronger color = stronger effect. Some methods produce sparse heatmaps (only 1--2 colored words), while others color many words lightly. This is a real difference between methods, not a bug.

### Your task

**Rank the 4 heatmaps from best (1) to worst (4).** Each rank must be used exactly once.

A **good explanation** is one where:
- The most strongly highlighted (red) words are genuinely related to the emotion
- The explanation makes intuitive sense -- you can understand **why** the model made its prediction
- Irrelevant words are not highlighted

A **bad explanation** is one where:
- The highlighted words seem unrelated to the emotion
- Too many irrelevant words are highlighted, making it hard to understand
- The explanation is confusing or misleading

Use the dropdown menus below the heatmaps to assign ranks 1--4.

### When you are done

Click **"Submit Phase B"** to save. The sample is now complete.

---

## 6. Navigation

| Feature | What it does |
|---------|-------------|
| **Sample dropdown** (sidebar) | Select a specific sample to annotate |
| **Previous / Next** buttons (sidebar) | Move between samples |
| **Filter by emotion** (sidebar) | Show only samples of a specific emotion |
| **Show only incomplete** (sidebar) | Hide already-completed samples (enabled by default) |
| **Progress bar** (sidebar) | Shows how many samples you have completed |

---

## 7. Saving Your Work

- Your annotations are saved **every time you click Submit**.
- Everything **persists across sessions** -- you can close the browser and come back later.
- You can annotate in any order -- you do not have to go sequentially.
- Once a phase is submitted for a sample, it is marked as done. If you need to redo a sample, let us know.

---

## 8. Example Walkthrough

### Sample sentence

> "I was so happy when my best friend surprised me at my birthday party."
> Predicted emotion: **joy**

### Phase A

You see the sentence and the word buttons. You click:

1. **happy** (Rank 1 -- directly expresses joy)
2. **surprised** (Rank 2 -- positive surprise contributes to joy)
3. **birthday** (Rank 3 -- birthday context supports joy)
4. **party** (Rank 4 -- festive context)
5. **friend** (Rank 5 -- social bond adds to joy)

The buttons show: "1. happy", "2. surprised", "3. birthday", "4. party", "5. friend"

You click **Submit Phase A**.

### Phase B

You now see 4 heatmaps. You evaluate each one:

- **Method A:** Strongly highlights "happy" and "surprised" in red. Makes sense. You give it **Rank 1**.
- **Method C:** Highlights "happy", "birthday", and "party". Reasonable. You give it **Rank 2**.
- **Method B:** Highlights many words lightly, including "was", "so", "at", "my". Diffuse and hard to interpret. You give it **Rank 3**.
- **Method D:** Highlights "I" and "me" in red. These are not emotion words. You give it **Rank 4**.

You click **Submit Phase B**. Sample complete!

---

## 9. Tips

- **Take breaks.** Annotation fatigue leads to inconsistent judgments. A few focused sessions are better than one long marathon.
- **Trust your intuition.** If an explanation feels wrong or confusing, rank it lower. There is no trick here.
- **Do not try to guess which method is which.** The methods are anonymized and shuffled for each sample. Just evaluate what you see.
- **Focus on the red features** in Phase B. Red words support the prediction -- ask yourself if those words are genuinely related to the emotion.
- **You do not have to select exactly 5 words** in Phase A. Select at least 1, but only select words that are truly relevant. If only 2 words carry the emotion, select only 2.
- **Some heatmaps will look very different from each other.** Some methods highlight only 1--2 words strongly. Others color many words lightly. Both patterns are normal.

---

## 10. Quick Reference

```
Phase A:
  1. Read the sentence and predicted emotion
  2. Click up to 5 words in order of importance (most important first)
  3. Click "Submit Phase A"

Phase B:
  1. Look at the 4 heatmaps
  2. Rank them 1 (best) to 4 (worst)
  3. Click "Submit Phase B"

Repeat for all 200 samples!
```

---

## 11. FAQ

**Q: What if I cannot decide which word is more important?**
A: Go with your first instinct. There is no single correct answer -- we are collecting human judgment.

**Q: What if the sentence is ambiguous or has no clear emotion words?**
A: Select the words that come closest to expressing the emotion, even if the connection is indirect. If truly nothing seems related, select the most plausible candidates.

**Q: What if two heatmaps look equally good (or equally bad)?**
A: You still need to assign different ranks. Pick whichever one you find slightly more convincing and rank it higher.

**Q: Can I change my annotation after submitting?**
A: Currently, submitted annotations cannot be edited in the tool. If you make a mistake, let us know and we can fix it.

**Q: How long will this take?**
A: Each sample takes about 30--60 seconds. The full 200 samples should take roughly 2--3 hours total across multiple sessions.

---

If you have any questions or run into issues, reach out anytime. Thank you for your help!
