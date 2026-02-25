# AI Prediction Pipeline — Baseline Evaluation Report

**Generated:** 2026-02-23T18:37:09.172328  
**Data Source:** Grading_pred_anls_enhanced_with_multinov5.xlsx

---

## Dataset Summary

| Metric | Value |
|--------|-------|
| total_rows | 532 |
| unique_articles | 85 |
| total_predictions | 530 |
| human_validated | 86 |
| graded_predictions | 40 |
| past_deadline | 129 |
| true_predictions | 14 |
| false_predictions | 14 |
| partially_true | 12 |


---

## 1. Prediction Extraction Metrics

The extraction stage identifies prediction candidates from raw article text.
Ground truth is defined by human annotation (Column E = 1).

| Metric | Exact Match | Soft Match (token F1 ≥ 0.5) |
|--------|------------|---------------------------|
| Precision | 0.288 | 0.288 |
| Recall    | 1.000    | 1.000    |
| F1        | 0.447        | 0.447        |

- Gold predictions (human-validated): **86**
- System predictions (in gold articles): **299**
- Macro per-article F1 (soft): **0.462** ± 0.211

### Score Calibration

| Group | Avg Verifiability | Avg Certainty |
|-------|------------------|---------------|
| Human-validated predictions | 3.50 | 3.40 |
| Non-validated predictions   | 2.86 | N/A |

---

## 2. Validation Quality Metrics

| Metric | Value |
|--------|-------|
| Total accepted by system | 530 |
| Human-confirmed valid | 86 |
| Filter Precision | 0.162 |
| Predictions with valid deadline | 530 |
| Predictions past deadline | 129 |
| Avg deadline confidence | 0.00 |

---

## 3. Grading Accuracy Metrics

| Label | Count | Rate |
|-------|-------|------|
| TRUE | 14 | 35.0% |
| FALSE | 14 | 35.0% |
| PARTIALLY_TRUE | 12 | 30.0% |
| ERROR | 0 | 0.0% |

- Grading coverage (past-due): **0.310**

### Score Calibration by Outcome

| Outcome | Avg Conf. | Avg Verifiability |
|---------|----------|-------------------|
| TRUE  | 0.00 | 3.64 |
| FALSE | 0.00 | 3.71 |

---

## 4. Model Agreement Analysis

| Model | YES | NO | PARTIALLY | Error | Agreement Rate |
|-------|-----|----|-----------|-------|----------------|
| Claude | 25 | 7 | 3 | 5 | 62.5% |
| Gemini | 31 | 8 | 1 | 0 | 77.5% |

**Trilateral consensus (both agree with GPT):** 23/40 (57.5%)

### Cohen's Kappa

| Pair | κ | Interpretation |
|------|---|----------------|
| Claude ↔ Gemini | 0.318 | Fair |
| Claude ↔ GPT | 0.000 | Fair |
| Gemini ↔ GPT | 0.000 | Fair |

### Agreement by GPT Label

| GPT Label | n | Claude YES | Gemini YES |
|-----------|---|------------|------------|
| TRUE | 14 | 78.6% | 85.7% |
| FALSE | 14 | 64.3% | 92.9% |
| PARTIALLY_TRUE | 12 | 41.7% | 50.0% |


---

## 5. System Reliability Metrics

| Metric | Value |
|--------|-------|
| Grading error rate | 0.000 |
| Claude error rate | 0.125 |
| Gemini error rate | 0.000 |
| Any-component error rate | 0.009 |
| Grading coverage (past-due) | 0.310 |
| Claude review coverage | 0.875 |
| Gemini review coverage | 1.000 |
| Deadline coverage | 1.000 |
| Avg deadline confidence | 0.00 |
| Verifiability–Certainty Pearson r | 0.158 |
| Avg predictions per article | 6.39 |

---

*Report generated automatically by `evaluation_framework.py`.*
