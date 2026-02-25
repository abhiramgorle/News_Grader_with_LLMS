# AI Prediction Pipeline — Metrics Explainer

This document explains every metric in the Baseline Evaluation Report, what it means in the context of your pipeline, and what ranges are considered poor, acceptable, or good.

---

## 🗂️ Dataset Summary Metrics

These are just counts describing the shape of your dataset — not performance metrics per se, but important for interpreting everything downstream.

| Metric | What It Means |
|--------|---------------|
| `total_rows` | Total rows in the output file (one row per prediction candidate). |
| `unique_articles` | How many distinct news articles were processed. |
| `total_predictions` | Total prediction candidates the system extracted across all articles. |
| `human_validated` | How many predictions a human manually confirmed as real predictions (your ground truth). |
| `graded_predictions` | How many predictions have been graded (TRUE / FALSE / PARTIALLY_TRUE) so far. |
| `past_deadline` | Predictions whose estimated verification deadline has already passed — these are eligible for grading. |
| `true_predictions` | Count of predictions graded TRUE. |
| `false_predictions` | Count of predictions graded FALSE. |
| `partially_true` | Count of predictions graded PARTIALLY_TRUE. |

**Your current numbers:** 86 human-validated out of 530 total extractions, meaning the system is extracting roughly 6× more candidates than humans consider valid. This is the core challenge your pipeline is trying to solve.

---

## 1. Prediction Extraction Metrics

These measure how well the system identifies real predictions in article text. Ground truth = human annotations.

---

### Precision
**What it is:** Of all predictions the system extracted, what fraction were actually valid (human-confirmed)?

**Formula:** `True Positives / (True Positives + False Positives)`

**Your value:** 0.288 — meaning only ~29% of what the system extracts is genuinely a prediction.

| Range | Interpretation |
|-------|---------------|
| < 0.3 | 🔴 Poor — system is extracting a lot of noise |
| 0.3 – 0.6 | 🟡 Acceptable — common in open-domain extraction |
| 0.6 – 0.8 | 🟢 Good |
| > 0.8 | 🟢 Excellent |

---

### Recall
**What it is:** Of all predictions humans marked as valid, what fraction did the system find?

**Formula:** `True Positives / (True Positives + False Negatives)`

**Your value:** 1.000 — the system found every single human-validated prediction (it just found a lot of extras too).

| Range | Interpretation |
|-------|---------------|
| < 0.5 | 🔴 Poor — missing many real predictions |
| 0.5 – 0.7 | 🟡 Acceptable |
| 0.7 – 0.9 | 🟢 Good |
| > 0.9 | 🟢 Excellent |

---

### F1 Score
**What it is:** The harmonic mean of Precision and Recall. Balances both — a high F1 requires both to be good.

**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`

**Your value:** 0.447 — pulled down heavily by low precision.

| Range | Interpretation |
|-------|---------------|
| < 0.4 | 🔴 Poor |
| 0.4 – 0.6 | 🟡 Acceptable for first-pass extraction |
| 0.6 – 0.75 | 🟢 Good |
| > 0.75 | 🟢 Excellent |

---

### Macro Per-Article F1 (Soft Match)
**What it is:** The average F1 computed separately per article, then averaged. "Soft match" means a prediction counts as a match if token-level overlap with the gold prediction is ≥ 50% (rather than requiring exact string match).

**Your value:** 0.462 ± 0.211 — the ± tells you performance is inconsistent across articles (high variance).

| Range | Interpretation |
|-------|---------------|
| < 0.4 | 🔴 Poor |
| 0.4 – 0.6 | 🟡 Acceptable |
| 0.6 – 0.75 | 🟢 Good |
| > 0.75 | 🟢 Excellent |

The standard deviation (0.211) is important: a high σ means performance is erratic — great on some articles, terrible on others.

---

### Score Calibration (Verifiability & Certainty)
**What it is:** Do the scores the system assigns actually distinguish real predictions from noise?

- **Verifiability Score:** How checkable/factual is the prediction?
- **Certainty Score:** How confident/definitive is the prediction language?

**Your values:** Human-validated predictions score 3.50 verifiability vs. 2.86 for non-validated — a modest but meaningful gap.

| Difference (validated vs. non-validated) | Interpretation |
|------------------------------------------|---------------|
| < 0.3 | 🔴 Scores are not discriminating |
| 0.3 – 0.7 | 🟡 Mild signal |
| > 0.7 | 🟢 Scores are useful for filtering |

---

## 2. Validation Quality Metrics

These measure the system's second-pass filter — after extraction, does it correctly reject non-predictions?

---

### Filter Precision
**What it is:** After the validation/filtering step, what fraction of accepted predictions are actually valid?

**Your value:** 0.162 — only 16% of what passes the filter is human-confirmed. The filter is not removing enough noise.

| Range | Interpretation |
|-------|---------------|
| < 0.2 | 🔴 Filter is weak — most accepted items are noise |
| 0.2 – 0.5 | 🟡 Moderate filtering |
| > 0.5 | 🟢 Good filtering |

---

### Deadline Coverage
**What it is:** What fraction of predictions received a deadline estimate?

**Your value:** 1.000 — every prediction got a deadline. Good.

| Range | Interpretation |
|-------|---------------|
| < 0.7 | 🔴 Many predictions are not getting deadlines |
| 0.7 – 0.9 | 🟡 Acceptable |
| > 0.9 | 🟢 Good |

---

### Avg Deadline Confidence
**What it is:** Average confidence score (1–5) the system assigns to its deadline estimates. 1 = very uncertain, 5 = explicit date given in the article.

**Your value:** 0.00 — this is a data issue (confidence values likely not being saved to the column used for analysis).

| Range | Interpretation |
|-------|---------------|
| 0.0 | 🔴 Data pipeline error — not being recorded |
| 1.0 – 2.5 | 🟡 Mostly guessing at deadlines |
| 2.5 – 4.0 | 🟢 Reasonable confidence |
| > 4.0 | 🟢 High confidence (many explicit dates found) |

---

## 3. Grading Accuracy Metrics

These measure how well the system determines whether past predictions came true.

---

### Grading Coverage (Past-Due)
**What it is:** Of predictions whose deadline has passed (129), what fraction have actually been graded?

**Your value:** 0.310 — only 31% of eligible predictions have been graded. The other 69% are eligible but ungraded (likely due to processing constraints or errors).

| Range | Interpretation |
|-------|---------------|
| < 0.3 | 🔴 Most eligible predictions are ungraded |
| 0.3 – 0.7 | 🟡 Partial coverage |
| > 0.7 | 🟢 Good coverage |

---

### Grading Distribution (TRUE / FALSE / PARTIALLY_TRUE)
**What it is:** The breakdown of how graded predictions turned out.

**Your values:** 35% TRUE, 35% FALSE, 30% PARTIALLY_TRUE — roughly balanced.

There's no "correct" range for this since it depends on the nature of the predictions. What to watch for:
- An extreme skew (e.g., 90% TRUE) may indicate the system is biased toward confirming predictions.
- Grading error rate of 0% is ideal.

---

### Grading Error Rate
**What it is:** Fraction of grading attempts that returned an error instead of a label.

**Your value:** 0.000 — no grading errors.

| Range | Interpretation |
|-------|---------------|
| 0.0 | 🟢 Perfect |
| > 0.05 | 🟡 Worth investigating |
| > 0.2 | 🔴 Systemic reliability issue |

---

## 4. Model Agreement Analysis

These measure how much the three AI models (GPT, Claude, Gemini) agree with each other on prediction grades.

---

### Agreement Rate (Claude / Gemini)
**What it is:** Fraction of graded predictions where Claude/Gemini says "YES" (agrees with GPT's grade).

**Your values:** Claude 62.5%, Gemini 77.5%.

| Range | Interpretation |
|-------|---------------|
| < 50% | 🔴 Models frequently disagree — grades are unreliable |
| 50 – 70% | 🟡 Moderate agreement |
| 70 – 85% | 🟢 Good agreement |
| > 85% | 🟢 Very high agreement (verify it's not sycophancy) |

---

### Trilateral Consensus
**What it is:** Fraction of predictions where both Claude AND Gemini agree with GPT's grade.

**Your value:** 57.5% (23/40).

| Range | Interpretation |
|-------|---------------|
| < 40% | 🔴 Models are frequently at odds |
| 40 – 60% | 🟡 Moderate consensus |
| 60 – 75% | 🟢 Good three-way agreement |
| > 75% | 🟢 Strong consensus |

---

### Cohen's Kappa (κ)
**What it is:** A statistical measure of agreement between two raters that corrects for chance agreement. Unlike raw agreement rate, κ = 0 means agreement no better than random, and κ = 1 means perfect agreement.

**Your values:**
- Claude ↔ Gemini: κ = 0.318
- Claude ↔ GPT: κ = 0.000
- Gemini ↔ GPT: κ = 0.000

The 0.000 values for model vs. GPT likely reflect that GPT's label set (YES/NO/PARTIALLY) doesn't align well with Claude/Gemini's response mapping, or a data encoding issue.

| κ Range | Interpretation |
|---------|---------------|
| < 0.0 | 🔴 Worse than chance |
| 0.0 – 0.2 | 🔴 Slight agreement |
| 0.2 – 0.4 | 🟡 Fair agreement |
| 0.4 – 0.6 | 🟡 Moderate agreement |
| 0.6 – 0.8 | 🟢 Substantial agreement |
| > 0.8 | 🟢 Near-perfect agreement |

---

### Agreement by GPT Label
**What it is:** Does Claude/Gemini agree more with GPT on TRUE predictions vs. FALSE predictions?

**Your pattern:**
- TRUE predictions → Claude 78.6%, Gemini 85.7% agreement
- FALSE predictions → Claude 64.3%, Gemini 92.9% agreement
- PARTIALLY_TRUE → Claude 41.7%, Gemini 50.0% agreement

"PARTIALLY_TRUE" is the hardest category to agree on — this is expected since it's the most subjective.

---

## 5. System Reliability Metrics

These measure whether the pipeline is running cleanly and producing output consistently.

---

### Claude / Gemini Error Rate
**What it is:** Fraction of verification attempts that failed with an error.

**Your values:** Claude 12.5%, Gemini 0%.

| Range | Interpretation |
|-------|---------------|
| 0.0 | 🟢 Perfect reliability |
| < 0.05 | 🟢 Good |
| 0.05 – 0.15 | 🟡 Moderate — investigate API stability |
| > 0.15 | 🔴 Significant reliability issue |

---

### Claude / Gemini Review Coverage
**What it is:** Fraction of graded predictions that received a verification review from each model.

**Your values:** Claude 87.5%, Gemini 100%.

| Range | Interpretation |
|-------|---------------|
| < 0.7 | 🔴 Many predictions are not being reviewed |
| 0.7 – 0.9 | 🟡 Good but some gaps |
| > 0.9 | 🟢 Excellent coverage |

---

### Verifiability–Certainty Pearson r
**What it is:** Correlation between the verifiability score and certainty score assigned to each prediction. Are these two scores measuring the same thing, or different things?

**Your value:** 0.158 — very weak correlation, meaning the two scores are largely independent.

| Range | Interpretation |
|-------|---------------|
| > 0.8 | 🔴 Redundant — scores measure the same thing |
| 0.4 – 0.8 | 🟡 Related but distinct |
| < 0.4 | 🟢 Scores capture different dimensions (ideal) |

A low r here is actually good — it means verifiability and certainty are capturing different aspects of a prediction.

---

### Avg Predictions Per Article
**What it is:** On average, how many prediction candidates does the system extract per article?

**Your value:** 6.39

There's no universal "right" number — it depends on article type. But combined with your precision of 0.29, this means roughly 1–2 genuine predictions per article, buried in ~4–5 false positives.

---

## 🔑 Key Takeaways & Priorities

| Issue | Severity | What to Do |
|-------|----------|------------|
| Filter Precision = 0.162 | 🔴 High | Your validation/filter step is the biggest problem — tighten the criteria for what counts as a prediction |
| Deadline Confidence = 0.00 | 🔴 High | Data pipeline bug — the confidence score isn't being recorded in the column used for analysis |
| Grading Coverage = 0.310 | 🟡 Medium | Only 31% of eligible predictions are graded — scale up grading or investigate why 69% are skipped |
| Claude Error Rate = 12.5% | 🟡 Medium | Claude API is failing occasionally — add better retry logic or fallback |
| κ(Claude↔GPT) = 0.000 | 🟡 Medium | Likely a label mapping bug — check how Claude's YES/NO/PARTIALLY maps to GPT's TRUE/FALSE grades |
| Extraction F1 = 0.447 | 🟡 Medium | Acceptable for v1, but improving the few-shot examples should push this higher |

---

*Generated to accompany the Baseline Evaluation Report from `evaluation_framework.py`.*