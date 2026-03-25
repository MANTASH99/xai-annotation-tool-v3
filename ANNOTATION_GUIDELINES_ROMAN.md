# Annotation Guidelines for Roman


## How to get started

1. Open the tool: [XAI Comparison Annotation Tool](https://xai-annotation-tool-v3.streamlit.app/)
2. Select "Roman" from the dropdown in the sidebar.
3. You will see your 20 assigned samples (2 per emotion).
4. You can switch between Round 1 and Round 2 at the top. Both rounds have the same task, but Round 2 uses a different visualization style.

## What this is about

We have a BERT model that classifies sentences into 15 emotions. We ran four different explanation methods on each prediction. Each method tries to show which words the model relied on. We want to know (1) which words a human would pick as carrying the emotion, and (2) which explanation method looks most convincing.

## Each sample has 2 phases

You must finish Phase A before Phase B unlocks for a given sample.

**Phase A** -- you see only the sentence and the predicted emotion. No explanations. You pick the words you think carry the emotion.

**Phase B** -- you now see 4 explanation visualizations (labeled Method A/B/C/D, anonymized and shuffled). You rank them from best to worst.


## Phase A -- Word Highlighting

You see the sentence and the predicted emotion label. Nothing else.

Click up to 5 words that you think are most responsible for the emotion. Click them in order: first click = most important word, second click = second most important, and so on.

The question to ask yourself: **if I removed this word, would the sentence still clearly express this emotion?**

Example:

> "My best friend called me out of nowhere just to check on me."
> Predicted emotion: **love**

Here you would probably pick something like: 1. friend, 2. check (as in checking on someone), maybe 3. called. Words like "out", "of", "nowhere" are not really about the emotion.

You do not have to pick exactly 5. If only 2 words matter, pick 2.

Click "Submit Phase A" when you are done.


## Phase B -- Explanation Ranking

Now you see the same sentence again, plus 4 explanations side by side. The methods are anonymized (Method A, B, C, D) and the order is shuffled for each sample, so do not try to track which method is which.

The color coding:
- Red = this word pushes the model toward the predicted emotion
- Blue = this word pushes against the predicted emotion
- White or no color = the model does not care about this word

Round 1 shows full heatmaps where every word gets a color. Round 2 shows only the top 5 words per method as horizontal bars (no numbers).

Rank all four from 1 (best) to 4 (worst). Each rank exactly once. A good explanation highlights words that actually relate to the emotion. A bad explanation highlights irrelevant words or colors everything equally so you cannot tell what matters.

Click "Submit Phase B" when you are done.


## Saving and navigation

- Click Submit to save. Your progress is saved locally.
- You can use Previous/Next or the sidebar to jump around.
- "Show only incomplete" is on by default so you only see what is left.
- You can go back and redo any sample.


## A few things to keep in mind

- Some methods highlight many words lightly, others highlight just 1 or 2 words strongly. Both are normal, just different behavior.
- Do not overthink it. Go with your gut feeling. There is no single correct answer.
- The whole thing should take around 20-30 minutes for 20 samples.

If anything is unclear, just reach out.
