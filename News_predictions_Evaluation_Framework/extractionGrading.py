"""
==============================================================================
 Phase 1 Evaluation Framework — AI Prediction Pipeline
 Module: evaluation_framework.py
 Author: Durga Abhiram Gorle
 Description:
     Provides a structured, reproducible evaluation system that measures the
     performance of the prediction extraction, validation, deadline estimation,
     grading, and model-agreement components of the PredictionProcessor pipeline.

 Sections:
     1.  EvaluationDataset     — builds ground-truth dataset from human labels
     2.  ExtractionMetrics     — precision, recall, F1 for prediction extraction
     3.  ValidationMetrics     — quality of the validation / deadline filter
     4.  GradingMetrics        — GPT grading accuracy vs. human labels
     5.  ModelAgreementMetrics — inter-model agreement (Claude, Gemini, GPT)
     6.  ReliabilityMetrics    — system-level error and coverage rates
     7.  EvaluationEngine      — orchestrator that ties all metrics together
     8.  BaselineReport        — human-readable report generator
==============================================================================
"""

import pandas as pd
import numpy as np
import json
import re
import warnings
from pathlib import Path
from datetime import date, datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# 0.  SHARED UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def normalize_label(label: Any) -> str:
    """Normalise a grading / agreement label to a canonical uppercase string."""
    if pd.isna(label):
        return "MISSING"
    s = str(label).strip().upper()
    # map common variants
    mapping = {
        "PARTIALLY": "PARTIALLY_TRUE",
        "PARTIAL":   "PARTIALLY_TRUE",
        "PARTIALLY TRUE": "PARTIALLY_TRUE",
        "YES":   "YES",
        "NO":    "NO",
        "TRUE":  "TRUE",
        "FALSE": "FALSE",
        "PENDING": "PENDING",
        "ERROR":   "ERROR",
        "N/A":     "NA",
    }
    return mapping.get(s, s)


def token_overlap_f1(pred_text: str, gold_text: str) -> float:
    """
    Compute token-level F1 between two strings.
    Used for soft / fuzzy matching of prediction spans.
    """
    pred_tokens = set(pred_text.lower().split())
    gold_tokens = set(gold_text.lower().split())
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return num / den if den != 0 else default


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EVALUATION DATASET
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionRecord:
    """A single prediction with all associated labels and scores."""
    article_number:        int
    prediction_number:     int
    prediction_text:       str
    prediction_context:    str
    article_text:          str
    verifiability_score:   float
    certainty_score:       float
    deadline_estimate:     str
    deadline_reasoning:    str
    deadline_confidence:   float
    grading:               str          # GPT label
    grading_justification: str
    claude_agrees:         str
    claude_context:        str
    gemini_agrees:         str
    gemini_context:        str
    human_validated:       bool          # col E == 1.0
    is_graded:             bool          # Grading ∈ {TRUE, FALSE, PARTIALLY_TRUE}
    is_past_deadline:      bool          # deadline <= today


class EvaluationDataset:
    """
    Builds a structured ground-truth dataset from the human-graded Excel file.

    Key concepts
    ─────────────
    • human_validated  : Column E == 1  → the human confirmed this IS a valid prediction.
    • is_graded        : Grading ∈ {TRUE, FALSE, PARTIALLY_TRUE} → outcome is known.
    • is_past_deadline : Deadline date ≤ today → the prediction is checkable.

    The 'Unnamed: 4' column is the **human validation flag**.
    When it is 1.0 the human said "yes, this is a real prediction worth tracking."
    When it is 0 the human did not mark it (treat as not-explicitly-validated).
    """

    GRADED_LABELS = {"TRUE", "FALSE", "PARTIALLY_TRUE"}
    ERROR_LABELS  = {"ERROR", "PENDING", "MISSING", "N/A", "NA"}

    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.raw_df     = pd.read_excel(excel_path)
        self.records: List[PredictionRecord] = []
        self._build_records()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _is_past_deadline(self, deadline_str: str) -> bool:
        try:
            d = date.fromisoformat(str(deadline_str).strip())
            return d <= date.today()
        except Exception:
            return False

    def _build_records(self):
        today = date.today()
        for _, row in self.raw_df.iterrows():
            # Skip rows with no prediction number or explicit 0 / negative
            try:
                pred_num = int(row.get("Prediction_Number", 0))
            except (ValueError, TypeError):
                pred_num = 0

            grading  = normalize_label(row.get("Grading"))
            deadline = str(row.get("Deadline_Estimate", "")).strip()

            rec = PredictionRecord(
                article_number        = int(row.get("Article_Number", 0)),
                prediction_number     = pred_num,
                prediction_text       = str(row.get("Prediction", "")),
                prediction_context    = str(row.get("Prediction_Context", "")),
                article_text          = str(row.get("Article_Text", "")),
                verifiability_score   = float(row.get("Verifiability_Score", 0) or 0),
                certainty_score       = float(row.get("Certainty_Score", 0) or 0),
                deadline_estimate     = deadline,
                deadline_reasoning    = str(row.get("Deadline_Reasoning", "")),
                deadline_confidence   = float(row.get("Deadline_Confidence", 0) or 0),
                grading               = grading,
                grading_justification = str(row.get("Grading_Justification", "")),
                claude_agrees         = normalize_label(row.get("Claude_Agrees")),
                claude_context        = str(row.get("Claude_Additional_Context", "")),
                gemini_agrees         = normalize_label(row.get("Gemini_Agrees")    ),
                gemini_context        = str(row.get("Gemini_Additional_Context", "")),
                human_validated       = int(row.get("Human Grading") or 0) == 1.0,
                is_graded             = grading in self.GRADED_LABELS,
                is_past_deadline      = self._is_past_deadline(deadline),
            )
            self.records.append(rec)

    # ── public accessors ─────────────────────────────────────────────────────

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    @property
    def all_predictions(self) -> List[PredictionRecord]:
        return [r for r in self.records if r.prediction_number > 0]

    @property
    def human_validated_predictions(self) -> List[PredictionRecord]:
        return [r for r in self.all_predictions if r.human_validated]

    @property
    def graded_predictions(self) -> List[PredictionRecord]:
        return [r for r in self.all_predictions if r.is_graded]

    @property
    def past_due_predictions(self) -> List[PredictionRecord]:
        return [r for r in self.all_predictions if r.is_past_deadline]

    def summary(self) -> Dict[str, int]:
        return {
            "total_rows":              len(self.records),
            "unique_articles":         len({r.article_number for r in self.records}),
            "total_predictions":       len(self.all_predictions),
            "human_validated":         len(self.human_validated_predictions),
            "graded_predictions":      len(self.graded_predictions),
            "past_deadline":           len(self.past_due_predictions),
            "true_predictions":        sum(1 for r in self.graded_predictions if r.grading == "TRUE"),
            "false_predictions":       sum(1 for r in self.graded_predictions if r.grading == "FALSE"),
            "partially_true":          sum(1 for r in self.graded_predictions if r.grading == "PARTIALLY_TRUE"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EXTRACTION METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Results for prediction extraction evaluation."""
    # Per-article aggregates
    total_articles_evaluated:  int   = 0
    total_gold_predictions:    int   = 0   # Human-validated positives
    total_system_predictions:  int   = 0   # All system-extracted predictions
    true_positives:            int   = 0   # System pred matched to a gold pred
    false_positives:           int   = 0   # System pred with no gold match
    false_negatives:           int   = 0   # Gold pred not found by system

    # Aggregate metrics
    precision:   float = 0.0
    recall:      float = 0.0
    f1:          float = 0.0

    # Soft-match metrics (token-level F1 >= threshold)
    soft_precision: float = 0.0
    soft_recall:    float = 0.0
    soft_f1:        float = 0.0

    # Score correlations
    avg_verifiability_true_preds:  float = 0.0
    avg_verifiability_false_preds: float = 0.0
    avg_certainty_true_preds:      float = 0.0

    per_article_f1: List[float] = field(default_factory=list)


class ExtractionMetrics:
    """
    Evaluates prediction extraction performance.

    Ground truth (gold set):
        Predictions where human_validated == True
        → The human confirmed these are real, valid predictions.

    System output (predicted set):
        All predictions extracted by the pipeline (prediction_number > 0).

    Matching strategy:
        Exact  : prediction text appears verbatim in gold set (per article).
        Soft   : token-level F1 >= soft_threshold (default 0.5).

    Why per-article?
        Precision / recall are computed per article first, then macro-averaged.
        This avoids bias from high-density articles dominating the metric.
    """

    def __init__(self, dataset: EvaluationDataset, soft_threshold: float = 0.5):
        self.dataset = dataset
        self.soft_threshold = soft_threshold

    def _match_predictions(
        self,
        system_preds: List[str],
        gold_preds: List[str],
        soft: bool = False
    ) -> Tuple[int, int, int]:
        """
        Returns (TP, FP, FN) for one article.

        Greedy matching: each gold prediction can be matched at most once.
        """
        matched_gold = set()
        tp = 0

        for sp in system_preds:
            best_score = 0.0
            best_idx   = -1

            for gi, gp in enumerate(gold_preds):
                if gi in matched_gold:
                    continue
                if soft:
                    score = token_overlap_f1(sp, gp)
                else:
                    score = 1.0 if sp.strip().lower() == gp.strip().lower() else 0.0

                if score > best_score:
                    best_score = score
                    best_idx   = gi

            if best_score >= (self.soft_threshold if soft else 1.0):
                tp += 1
                matched_gold.add(best_idx)

        fp = len(system_preds) - tp
        fn = len(gold_preds)   - tp
        return tp, fp, fn

    def evaluate(self) -> ExtractionResult:
        result = ExtractionResult()

        # Build per-article gold and system prediction maps
        gold_map   = defaultdict(list)   # article_num → [prediction_texts]
        system_map = defaultdict(list)

        for r in self.dataset.all_predictions:
            system_map[r.article_number].append(r.prediction_text)
            if r.human_validated:
                gold_map[r.article_number].append(r.prediction_text)

        articles_with_gold = set(gold_map.keys())
        result.total_articles_evaluated = len(articles_with_gold)

        exact_tp = exact_fp = exact_fn = 0
        soft_tp  = soft_fp  = soft_fn  = 0
        per_article_f1 = []

        for article_num in articles_with_gold:
            gold   = gold_map[article_num]
            system = system_map.get(article_num, [])

            # Exact match
            e_tp, e_fp, e_fn = self._match_predictions(system, gold, soft=False)
            exact_tp += e_tp; exact_fp += e_fp; exact_fn += e_fn

            # Soft match
            s_tp, s_fp, s_fn = self._match_predictions(system, gold, soft=True)
            soft_tp += s_tp; soft_fp += s_fp; soft_fn += s_fn

            # Per-article F1 (soft)
            prec = safe_div(s_tp, s_tp + s_fp)
            rec  = safe_div(s_tp, s_tp + s_fn)
            f1   = safe_div(2 * prec * rec, prec + rec)
            per_article_f1.append(f1)

        result.total_gold_predictions   = sum(len(v) for v in gold_map.values())
        result.total_system_predictions = sum(len(v) for v in system_map.values()
                                               if any(a == k for k in articles_with_gold
                                                      for a in [k]))
        # Recalc correctly
        result.total_system_predictions = sum(
            len(system_map[a]) for a in articles_with_gold
        )
        result.true_positives  = exact_tp
        result.false_positives = exact_fp
        result.false_negatives = exact_fn

        result.precision = safe_div(exact_tp, exact_tp + exact_fp)
        result.recall    = safe_div(exact_tp, exact_tp + exact_fn)
        result.f1        = safe_div(
            2 * result.precision * result.recall,
            result.precision + result.recall
        )

        result.soft_precision = safe_div(soft_tp, soft_tp + soft_fp)
        result.soft_recall    = safe_div(soft_tp, soft_tp + soft_fn)
        result.soft_f1        = safe_div(
            2 * result.soft_precision * result.soft_recall,
            result.soft_precision + result.soft_recall
        )

        result.per_article_f1 = per_article_f1

        # Score analysis: do higher scores correlate with human validation?
        validated = [r for r in self.dataset.all_predictions if r.human_validated]
        rejected  = [r for r in self.dataset.all_predictions if not r.human_validated]

        result.avg_verifiability_true_preds  = (
            np.mean([r.verifiability_score for r in validated]) if validated else 0.0
        )
        result.avg_verifiability_false_preds = (
            np.mean([r.verifiability_score for r in rejected]) if rejected else 0.0
        )
        result.avg_certainty_true_preds = (
            np.mean([r.certainty_score for r in validated]) if validated else 0.0
        )

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 3.  VALIDATION QUALITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    """Results for the prediction validation / deadline-filter stage."""
    total_candidates:         int   = 0
    passed_validation:        int   = 0
    rejected_by_system:       int   = 0

    # Of the passed ones, how many were human-confirmed?
    human_confirmed_passed:   int   = 0
    human_rejected_passed:    int   = 0

    # Deadline quality
    predictions_with_date:    int   = 0
    predictions_past_due:     int   = 0
    avg_deadline_confidence:  float = 0.0

    # Score-threshold analysis
    high_verif_score_count:   int   = 0    # verifiability >= 3
    low_verif_score_count:    int   = 0

    # Filter precision: of all system-accepted preds, what % are human-valid?
    filter_precision:         float = 0.0
    # Filter recall: of all human-valid preds, what % did system accept?
    filter_recall:            float = 0.0
    filter_f1:                float = 0.0


class ValidationMetrics:
    """
    Evaluates the quality of the validation and filtering stage.

    The system's validation stage (estimate_deadline_with_retry) acts as a
    binary classifier:
        • Accept (is_prediction = YES, confidence >= 2) → prediction_number > 0
        • Reject (is_prediction = NO or low confidence) → dropped

    We measure:
        • Filter Precision: what fraction of accepted predictions are real?
        • Filter Recall: what fraction of real predictions were accepted?
        • Deadline confidence distribution
        • Verifiability score stratification
    """

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset

    def evaluate(self) -> ValidationResult:
        result = ValidationResult()
        all_preds = self.dataset.all_predictions

        result.total_candidates   = len(all_preds)
        result.passed_validation  = len(all_preds)   # all stored preds passed
        result.rejected_by_system = 0                # we don't have rejected logs

        # Filter quality vs human labels
        human_confirmed  = [r for r in all_preds if r.human_validated]
        human_not_conf   = [r for r in all_preds if not r.human_validated]

        result.human_confirmed_passed = len(human_confirmed)
        result.human_rejected_passed  = len(human_not_conf)

        # Deadline quality
        date_preds = [
            r for r in all_preds
            if r.deadline_estimate not in ("N/A", "Error", "Unknown", "UNKNOWN", "")
        ]
        result.predictions_with_date = len(date_preds)
        result.predictions_past_due  = len([r for r in date_preds if r.is_past_deadline])
        result.avg_deadline_confidence = np.mean(
            [r.deadline_confidence for r in date_preds]
        ) if date_preds else 0.0

        # Verifiability score stratification
        result.high_verif_score_count = sum(
            1 for r in all_preds if r.verifiability_score >= 3
        )
        result.low_verif_score_count = sum(
            1 for r in all_preds if r.verifiability_score < 3
        )

        # Filter precision and recall (using human validation as ground truth)
        tp = result.human_confirmed_passed
        fp = result.human_rejected_passed
        # FN = human-validated predictions that were NOT accepted — 
        #      here we assume they were in the raw article but not extracted.
        #      In this dataset the only gold we have is what passed the filter,
        #      so we treat FN = 0 for the filter stage (system saw all articles).
        fn = 0  # conservative lower bound

        result.filter_precision = safe_div(tp, tp + fp)
        result.filter_recall    = safe_div(tp, tp + fn) if fn > 0 else 1.0
        result.filter_f1        = safe_div(
            2 * result.filter_precision * result.filter_recall,
            result.filter_precision + result.filter_recall
        )

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRADING ACCURACY METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GradingResult:
    """Results for the GPT grading accuracy evaluation."""
    total_graded:          int   = 0
    true_count:            int   = 0
    false_count:           int   = 0
    partially_true_count:  int   = 0
    error_count:           int   = 0

    # Distribution as fractions
    true_rate:             float = 0.0
    false_rate:            float = 0.0
    partially_true_rate:   float = 0.0
    error_rate:            float = 0.0

    # Coverage: what fraction of past-due predictions were graded?
    past_due_total:        int   = 0
    past_due_graded:       int   = 0
    grading_coverage:      float = 0.0

    # Per-label counts and ratios for human-validated graded preds
    human_val_graded:      int   = 0

    # Simple accuracy vs a human majority vote (if available)
    human_vs_gpt_accuracy: float = 0.0   # proportion of exact matches

    # Confidence calibration: do higher-confidence preds get graded TRUE more?
    avg_conf_true_preds:   float = 0.0
    avg_conf_false_preds:  float = 0.0
    avg_verif_true_preds:  float = 0.0
    avg_verif_false_preds: float = 0.0

    label_distribution:    Dict[str, int] = field(default_factory=dict)


class GradingMetrics:
    """
    Evaluates the outcome grading stage.

    GPT grades each past-due prediction as TRUE / FALSE / PARTIALLY_TRUE.
    We assess:
      • Label distribution
      • Coverage rate (% of past-due preds actually graded)
      • Agreement with human labels where both exist
      • Score calibration (do harder preds get lower confidence?)
    """

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset

    def evaluate(self) -> GradingResult:
        result   = GradingResult()
        graded   = self.dataset.graded_predictions
        past_due = self.dataset.past_due_predictions

        result.total_graded    = len(graded)
        result.past_due_total  = len(past_due)
        result.past_due_graded = len([r for r in past_due if r.is_graded])
        result.grading_coverage = safe_div(result.past_due_graded, result.past_due_total)

        # Label counts
        labels = [r.grading for r in graded]
        result.true_count           = labels.count("TRUE")
        result.false_count          = labels.count("FALSE")
        result.partially_true_count = labels.count("PARTIALLY_TRUE")
        result.error_count          = sum(
            1 for r in self.dataset.all_predictions
            if r.grading in ("ERROR", "MISSING")
        )
        result.label_distribution = {
            "TRUE": result.true_count,
            "FALSE": result.false_count,
            "PARTIALLY_TRUE": result.partially_true_count,
            "ERROR": result.error_count,
        }

        n = result.total_graded
        result.true_rate           = safe_div(result.true_count, n)
        result.false_rate          = safe_div(result.false_count, n)
        result.partially_true_rate = safe_div(result.partially_true_count, n)
        result.error_rate          = safe_div(result.error_count, n + result.error_count)

        # Human vs GPT accuracy (only rows that are both human-validated & graded)
        human_graded = [r for r in graded if r.human_validated]
        result.human_val_graded = len(human_graded)
        # We don't have a separate human grading column; only validation flag.
        # Where Unnamed:4 == 1 AND grading exists, human said "this is a real pred"
        # but didn't provide an outcome label. We note this gap.
        result.human_vs_gpt_accuracy = -1.0   # Not computable (no human outcome label)

        # Calibration
        true_preds  = [r for r in graded if r.grading == "TRUE"]
        false_preds = [r for r in graded if r.grading == "FALSE"]

        result.avg_conf_true_preds  = (
            np.mean([r.deadline_confidence for r in true_preds]) if true_preds else 0.0
        )
        result.avg_conf_false_preds = (
            np.mean([r.deadline_confidence for r in false_preds]) if false_preds else 0.0
        )
        result.avg_verif_true_preds = (
            np.mean([r.verifiability_score for r in true_preds]) if true_preds else 0.0
        )
        result.avg_verif_false_preds = (
            np.mean([r.verifiability_score for r in false_preds]) if false_preds else 0.0
        )

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL AGREEMENT METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgreementResult:
    """Inter-model agreement analysis."""
    total_graded:              int   = 0

    # Claude agreement with GPT
    claude_yes:                int   = 0
    claude_no:                 int   = 0
    claude_partially:          int   = 0
    claude_error:              int   = 0
    claude_agreement_rate:     float = 0.0

    # Gemini agreement with GPT
    gemini_yes:                int   = 0
    gemini_no:                 int   = 0
    gemini_partially:          int   = 0
    gemini_error:              int   = 0
    gemini_agreement_rate:     float = 0.0

    # Trilateral consensus
    all_agree:                 int   = 0    # Claude=YES & Gemini=YES
    all_disagree:              int   = 0    # Claude=NO  & Gemini=NO
    split:                     int   = 0    # mixed
    consensus_rate:            float = 0.0

    # Agreement by GPT label
    agreement_by_label:        Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Cohen's kappa (Claude-GPT, Gemini-GPT)
    cohen_kappa_claude_gpt:    float = 0.0
    cohen_kappa_gemini_gpt:    float = 0.0
    cohen_kappa_claude_gemini: float = 0.0


class ModelAgreementMetrics:
    """
    Measures inter-model agreement among GPT, Claude, and Gemini.

    Agreement signal:
        Claude_Agrees / Gemini_Agrees ∈ {YES, NO, PARTIALLY}
        YES → the verifier agrees with GPT's grading
        NO  → the verifier disagrees with GPT's grading

    Metrics computed:
        • Agreement rate per model
        • Trilateral consensus rate
        • Agreement rate stratified by GPT label
        • Cohen's Kappa (treating YES/NO/PARTIALLY as classes)
    """

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset

    def _cohen_kappa(
        self,
        labels_a: List[str],
        labels_b: List[str],
        all_labels: List[str]
    ) -> float:
        """
        Compute Cohen's kappa for two lists of categorical labels.
        κ = (Po - Pe) / (1 - Pe)
        """
        if len(labels_a) != len(labels_b) or len(labels_a) == 0:
            return 0.0

        n = len(labels_a)
        # Observed agreement
        po = sum(a == b for a, b in zip(labels_a, labels_b)) / n

        # Expected agreement
        pe = 0.0
        label_set = set(all_labels)
        for label in label_set:
            pa = labels_a.count(label) / n
            pb = labels_b.count(label) / n
            pe += pa * pb

        if 1 - pe == 0:
            return 1.0
        return (po - pe) / (1 - pe)

    def evaluate(self) -> AgreementResult:
        result  = AgreementResult()
        graded  = self.dataset.graded_predictions
        result.total_graded = len(graded)

        if result.total_graded == 0:
            return result

        # Claude counts
        result.claude_yes       = sum(1 for r in graded if r.claude_agrees == "YES")
        result.claude_no        = sum(1 for r in graded if r.claude_agrees == "NO")
        result.claude_partially = sum(1 for r in graded if r.claude_agrees == "PARTIALLY_TRUE")
        result.claude_error     = sum(
            1 for r in graded if r.claude_agrees in ("ERROR", "MISSING", "PENDING")
        )
        result.claude_agreement_rate = safe_div(result.claude_yes, result.total_graded)

        # Gemini counts
        result.gemini_yes       = sum(1 for r in graded if r.gemini_agrees == "YES")
        result.gemini_no        = sum(1 for r in graded if r.gemini_agrees == "NO")
        result.gemini_partially = sum(1 for r in graded if r.gemini_agrees == "PARTIALLY_TRUE")
        result.gemini_error     = sum(
            1 for r in graded if r.gemini_agrees in ("ERROR", "MISSING", "PENDING")
        )
        result.gemini_agreement_rate = safe_div(result.gemini_yes, result.total_graded)

        # Trilateral consensus
        result.all_agree    = sum(
            1 for r in graded
            if r.claude_agrees == "YES" and r.gemini_agrees == "YES"
        )
        result.all_disagree = sum(
            1 for r in graded
            if r.claude_agrees == "NO" and r.gemini_agrees == "NO"
        )
        result.split        = result.total_graded - result.all_agree - result.all_disagree
        result.consensus_rate = safe_div(result.all_agree, result.total_graded)

        # Agreement by GPT label
        for label in ("TRUE", "FALSE", "PARTIALLY_TRUE"):
            subset = [r for r in graded if r.grading == label]
            if not subset:
                continue
            c_yes = sum(1 for r in subset if r.claude_agrees == "YES")
            g_yes = sum(1 for r in subset if r.gemini_agrees == "YES")
            result.agreement_by_label[label] = {
                "n":               len(subset),
                "claude_yes_rate": safe_div(c_yes, len(subset)),
                "gemini_yes_rate": safe_div(g_yes, len(subset)),
            }

        # Cohen's kappa
        # Convert: YES → agreement, NO → disagreement, map to shared space
        gpt_as_verif  = ["YES"] * result.total_graded   # GPT's own answer = reference
        claude_labels = [r.claude_agrees for r in graded]
        gemini_labels = [r.gemini_agrees for r in graded]

        # Normalise for kappa (keep YES / NO / PARTIALLY_TRUE)
        valid_labels = {"YES", "NO", "PARTIALLY_TRUE"}
        paired_cg = [
            (c, g) for c, g in zip(claude_labels, gemini_labels)
            if c in valid_labels and g in valid_labels
        ]
        if paired_cg:
            cl = [p[0] for p in paired_cg]
            gl = [p[1] for p in paired_cg]
            all_l = list(valid_labels)
            result.cohen_kappa_claude_gemini = self._cohen_kappa(cl, gl, all_l)

        # Claude vs "GPT answer" as YES (binary: did it agree?)
        paired_cgpt = [
            ("YES", r.claude_agrees) for r in graded if r.claude_agrees in valid_labels
        ]
        if paired_cgpt:
            result.cohen_kappa_claude_gpt = self._cohen_kappa(
                [p[0] for p in paired_cgpt],
                [p[1] for p in paired_cgpt],
                list(valid_labels)
            )

        paired_ggpt = [
            ("YES", r.gemini_agrees) for r in graded if r.gemini_agrees in valid_labels
        ]
        if paired_ggpt:
            result.cohen_kappa_gemini_gpt = self._cohen_kappa(
                [p[0] for p in paired_ggpt],
                [p[1] for p in paired_ggpt],
                list(valid_labels)
            )

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 6.  SYSTEM RELIABILITY METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReliabilityResult:
    """System-level reliability and coverage statistics."""
    total_predictions:        int   = 0

    # Error rates
    grading_error_rate:       float = 0.0
    claude_error_rate:        float = 0.0
    gemini_error_rate:        float = 0.0
    any_error_rate:           float = 0.0

    # Coverage rates
    grading_coverage_rate:    float = 0.0   # % of past-due preds with a grading
    claude_coverage_rate:     float = 0.0   # % of graded preds that got Claude review
    gemini_coverage_rate:     float = 0.0   # % of graded preds that got Gemini review

    # Deadline estimation quality
    deadline_coverage:        float = 0.0   # % of preds with a valid date
    avg_deadline_confidence:  float = 0.0

    # Score consistency (are scores internally consistent?)
    verif_certainty_corr:     float = 0.0   # Pearson r between verif & certainty

    # Article-level throughput
    articles_with_zero_preds: int   = 0
    avg_preds_per_article:    float = 0.0
    max_preds_per_article:    int   = 0


class ReliabilityMetrics:
    """
    Measures system-level reliability: error rates, coverage, and consistency.

    Reliability ≠ accuracy. A reliable system:
        1. Returns a result (not an error) for every input
        2. Produces consistent scores across similar inputs
        3. Has high coverage (grades all past-due predictions)
    """

    def __init__(self, dataset: EvaluationDataset):
        self.dataset = dataset

    def evaluate(self) -> ReliabilityResult:
        result = ReliabilityResult()
        all_p  = self.dataset.all_predictions
        graded = self.dataset.graded_predictions
        result.total_predictions = len(all_p)

        if not all_p:
            return result

        # Error rates
        grading_errors = sum(
            1 for r in all_p if r.grading in ("ERROR", "MISSING")
        )
        claude_errors  = sum(
            1 for r in graded if r.claude_agrees in ("ERROR", "MISSING")
        )
        gemini_errors  = sum(
            1 for r in graded if r.gemini_agrees in ("ERROR", "MISSING")
        )
        any_errors     = sum(
            1 for r in all_p
            if r.grading in ("ERROR", "MISSING")
               or (r.is_graded and r.claude_agrees in ("ERROR", "MISSING"))
               or (r.is_graded and r.gemini_agrees in ("ERROR", "MISSING"))
        )

        result.grading_error_rate = safe_div(grading_errors, len(all_p))
        result.claude_error_rate  = safe_div(claude_errors, len(graded)) if graded else 0.0
        result.gemini_error_rate  = safe_div(gemini_errors, len(graded)) if graded else 0.0
        result.any_error_rate     = safe_div(any_errors, len(all_p))

        # Coverage
        past_due = self.dataset.past_due_predictions
        result.grading_coverage_rate = safe_div(
            sum(1 for r in past_due if r.is_graded), len(past_due)
        ) if past_due else 0.0

        if graded:
            result.claude_coverage_rate = safe_div(
                sum(1 for r in graded if r.claude_agrees not in ("PENDING", "MISSING", "ERROR")),
                len(graded)
            )
            result.gemini_coverage_rate = safe_div(
                sum(1 for r in graded if r.gemini_agrees not in ("PENDING", "MISSING", "ERROR")),
                len(graded)
            )

        # Deadline estimation
        date_preds = [
            r for r in all_p
            if r.deadline_estimate not in ("N/A", "Error", "Unknown", "UNKNOWN", "")
        ]
        result.deadline_coverage     = safe_div(len(date_preds), len(all_p))
        result.avg_deadline_confidence = (
            np.mean([r.deadline_confidence for r in date_preds]) if date_preds else 0.0
        )

        # Score consistency (Pearson correlation)
        verif  = [r.verifiability_score for r in all_p]
        cert   = [r.certainty_score     for r in all_p]
        if len(verif) > 2 and np.std(verif) > 0 and np.std(cert) > 0:
            result.verif_certainty_corr = float(np.corrcoef(verif, cert)[0, 1])

        # Article throughput
        from collections import Counter
        article_counts = Counter(r.article_number for r in all_p)
        unique_articles = set(r.article_number for r in self.dataset.records)
        result.articles_with_zero_preds = len(
            unique_articles - set(article_counts.keys())
        )
        counts = list(article_counts.values())
        result.avg_preds_per_article = float(np.mean(counts)) if counts else 0.0
        result.max_preds_per_article = int(max(counts)) if counts else 0

        return result


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION ENGINE (ORCHESTRATOR)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    """Complete evaluation report — all metric groups in one structure."""
    generated_at:     str
    data_source:      str
    dataset_summary:  Dict[str, int]
    extraction:       ExtractionResult
    validation:       ValidationResult
    grading:          GradingResult
    agreement:        AgreementResult
    reliability:      ReliabilityResult

    def to_dict(self) -> dict:
        return {
            "generated_at":    self.generated_at,
            "data_source":     self.data_source,
            "dataset_summary": self.dataset_summary,
            "extraction":      asdict(self.extraction),
            "validation":      asdict(self.validation),
            "grading":         asdict(self.grading),
            "agreement":       asdict(self.agreement),
            "reliability":     asdict(self.reliability),
        }

    def to_json(self, path: str = "eval_report.json"):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        print(f"✅ Report saved to {path}")


class EvaluationEngine:
    """
    Orchestrates all evaluation components and produces a unified report.

    Usage:
        engine = EvaluationEngine("Grading_pred_anls_enhanced_with_multinov5.xlsx")
        report = engine.run()
        engine.print_report(report)
        report.to_json("baseline_report.json")
    """

    def __init__(self, excel_path: str):
        print(f"📂 Loading dataset from: {excel_path}")
        self.dataset = EvaluationDataset(excel_path)
        print(f"✅ Dataset loaded: {self.dataset.summary()}")

    def run(self) -> EvaluationReport:
        print("\n🔬 Running evaluation pipeline...")

        print("  [1/5] Extraction metrics...")
        extraction = ExtractionMetrics(self.dataset).evaluate()

        print("  [2/5] Validation metrics...")
        validation = ValidationMetrics(self.dataset).evaluate()

        print("  [3/5] Grading metrics...")
        grading = GradingMetrics(self.dataset).evaluate()

        print("  [4/5] Model agreement metrics...")
        agreement = ModelAgreementMetrics(self.dataset).evaluate()

        print("  [5/5] Reliability metrics...")
        reliability = ReliabilityMetrics(self.dataset).evaluate()

        print("✅ Evaluation complete.\n")

        return EvaluationReport(
            generated_at    = datetime.now().isoformat(),
            data_source     = self.dataset.excel_path,
            dataset_summary = self.dataset.summary(),
            extraction      = extraction,
            validation      = validation,
            grading         = grading,
            agreement       = agreement,
            reliability     = reliability,
        )

    def print_report(self, report: EvaluationReport):
        """Pretty-print the evaluation report to stdout."""
        sep = "=" * 70

        print(sep)
        print("  AI PREDICTION PIPELINE — BASELINE EVALUATION REPORT")
        print(f"  Generated: {report.generated_at}")
        print(f"  Source:    {report.data_source}")
        print(sep)

        # ── Dataset Summary ──────────────────────────────────────────────────
        print("\n📊 DATASET SUMMARY")
        print("-" * 40)
        for k, v in report.dataset_summary.items():
            print(f"  {k:<35} {v:>6}")

        # ── Extraction ───────────────────────────────────────────────────────
        e = report.extraction
        print("\n\n🔍 1. PREDICTION EXTRACTION METRICS")
        print("-" * 40)
        print(f"  Articles evaluated (have gold preds)   {e.total_articles_evaluated:>6}")
        print(f"  Gold predictions (human-validated)     {e.total_gold_predictions:>6}")
        print(f"  System predictions (in those articles) {e.total_system_predictions:>6}")
        print()
        print("  ── Exact-Match ──")
        print(f"  True Positives                         {e.true_positives:>6}")
        print(f"  False Positives                        {e.false_positives:>6}")
        print(f"  False Negatives                        {e.false_negatives:>6}")
        print(f"  Precision                              {e.precision:>7.3f}")
        print(f"  Recall                                 {e.recall:>7.3f}")
        print(f"  F1 Score                               {e.f1:>7.3f}")
        print()
        print("  ── Soft-Match (token F1 ≥ 0.50) ──")
        print(f"  Soft Precision                         {e.soft_precision:>7.3f}")
        print(f"  Soft Recall                            {e.soft_recall:>7.3f}")
        print(f"  Soft F1                                {e.soft_f1:>7.3f}")
        print()
        print("  ── Score Calibration ──")
        print(f"  Avg Verifiability (human-validated)    {e.avg_verifiability_true_preds:>7.2f}")
        print(f"  Avg Verifiability (not validated)      {e.avg_verifiability_false_preds:>7.2f}")
        print(f"  Avg Certainty     (human-validated)    {e.avg_certainty_true_preds:>7.2f}")
        if e.per_article_f1:
            print(f"  Macro-avg per-article F1 (soft)        {np.mean(e.per_article_f1):>7.3f}")
            print(f"  Std dev per-article F1                 {np.std(e.per_article_f1):>7.3f}")

        # ── Validation ───────────────────────────────────────────────────────
        v = report.validation
        print("\n\n✅ 2. VALIDATION QUALITY METRICS")
        print("-" * 40)
        print(f"  Total predictions accepted by system   {v.total_candidates:>6}")
        print(f"  Human-confirmed as valid               {v.human_confirmed_passed:>6}")
        print(f"  Not explicitly human-confirmed         {v.human_rejected_passed:>6}")
        print(f"  Filter Precision (human/total)         {v.filter_precision:>7.3f}")
        print(f"  Predictions with valid deadline date   {v.predictions_with_date:>6}")
        print(f"  Predictions past deadline              {v.predictions_past_due:>6}")
        print(f"  Avg Deadline Confidence                {v.avg_deadline_confidence:>7.2f}")
        print(f"  High Verifiability (≥3) predictions    {v.high_verif_score_count:>6}")
        print(f"  Low  Verifiability (<3) predictions    {v.low_verif_score_count:>6}")

        # ── Grading ──────────────────────────────────────────────────────────
        g = report.grading
        print("\n\n📋 3. GRADING ACCURACY METRICS")
        print("-" * 40)
        print(f"  Total past-due predictions             {g.past_due_total:>6}")
        print(f"  Past-due predictions graded            {g.past_due_graded:>6}")
        print(f"  Grading coverage rate                  {g.grading_coverage:>7.3f}")
        print()
        print("  ── Label Distribution ──")
        print(f"  TRUE                                   {g.true_count:>6}  ({g.true_rate:.1%})")
        print(f"  FALSE                                  {g.false_count:>6}  ({g.false_rate:.1%})")
        print(f"  PARTIALLY_TRUE                         {g.partially_true_count:>6}  ({g.partially_true_rate:.1%})")
        print(f"  ERROR                                  {g.error_count:>6}  ({g.error_rate:.1%})")
        print()
        print("  ── Score Calibration (by label) ──")
        print(f"  Avg deadline confidence — TRUE         {g.avg_conf_true_preds:>7.2f}")
        print(f"  Avg deadline confidence — FALSE        {g.avg_conf_false_preds:>7.2f}")
        print(f"  Avg verifiability       — TRUE         {g.avg_verif_true_preds:>7.2f}")
        print(f"  Avg verifiability       — FALSE        {g.avg_verif_false_preds:>7.2f}")
        if g.human_vs_gpt_accuracy >= 0:
            print(f"  Human vs GPT accuracy                  {g.human_vs_gpt_accuracy:>7.3f}")
        else:
            print("  Human vs GPT accuracy                  N/A (no human outcome labels)")

        # ── Model Agreement ──────────────────────────────────────────────────
        a = report.agreement
        print("\n\n🤝 4. MODEL AGREEMENT ANALYSIS")
        print("-" * 40)
        print(f"  Graded predictions                     {a.total_graded:>6}")
        print()
        print("  ── Claude vs GPT ──")
        print(f"  Claude agrees (YES)                    {a.claude_yes:>6}  ({a.claude_agreement_rate:.1%})")
        print(f"  Claude disagrees (NO)                  {a.claude_no:>6}")
        print(f"  Claude partially agrees                {a.claude_partially:>6}")
        print(f"  Claude errors / missing                {a.claude_error:>6}")
        print()
        print("  ── Gemini vs GPT ──")
        print(f"  Gemini agrees (YES)                    {a.gemini_yes:>6}  ({a.gemini_agreement_rate:.1%})")
        print(f"  Gemini disagrees (NO)                  {a.gemini_no:>6}")
        print(f"  Gemini partially agrees                {a.gemini_partially:>6}")
        print(f"  Gemini errors / missing                {a.gemini_error:>6}")
        print()
        print("  ── Trilateral Consensus ──")
        print(f"  All agree (Claude=YES & Gemini=YES)    {a.all_agree:>6}  ({a.consensus_rate:.1%})")
        print(f"  All disagree (both NO)                 {a.all_disagree:>6}")
        print(f"  Split verdict                          {a.split:>6}")
        print()
        print("  ── Cohen's Kappa ──")
        print(f"  Claude ↔ Gemini                        {a.cohen_kappa_claude_gemini:>7.3f}")
        print(f"  Claude ↔ GPT (agreement as class)      {a.cohen_kappa_claude_gpt:>7.3f}")
        print(f"  Gemini ↔ GPT (agreement as class)      {a.cohen_kappa_gemini_gpt:>7.3f}")

        if a.agreement_by_label:
            print()
            print("  ── Agreement Rate by GPT Label ──")
            for label, stats in a.agreement_by_label.items():
                print(f"  {label:<18} n={stats['n']:<4}  "
                      f"Claude={stats['claude_yes_rate']:.1%}  "
                      f"Gemini={stats['gemini_yes_rate']:.1%}")

        # ── Reliability ──────────────────────────────────────────────────────
        r = report.reliability
        print("\n\n🔒 5. SYSTEM RELIABILITY METRICS")
        print("-" * 40)
        print(f"  Grading error rate                     {r.grading_error_rate:>7.3f}")
        print(f"  Claude verification error rate         {r.claude_error_rate:>7.3f}")
        print(f"  Gemini verification error rate         {r.gemini_error_rate:>7.3f}")
        print(f"  Any-component error rate               {r.any_error_rate:>7.3f}")
        print()
        print(f"  Grading coverage (past-due)            {r.grading_coverage_rate:>7.3f}")
        print(f"  Claude review coverage                 {r.claude_coverage_rate:>7.3f}")
        print(f"  Gemini review coverage                 {r.gemini_coverage_rate:>7.3f}")
        print()
        print(f"  Deadline coverage (% with valid date)  {r.deadline_coverage:>7.3f}")
        print(f"  Avg deadline confidence                {r.avg_deadline_confidence:>7.2f}")
        print(f"  Verifiability–Certainty Pearson r      {r.verif_certainty_corr:>7.3f}")
        print()
        print(f"  Articles with 0 predictions            {r.articles_with_zero_preds:>6}")
        print(f"  Avg predictions per article            {r.avg_preds_per_article:>7.2f}")
        print(f"  Max predictions per article            {r.max_preds_per_article:>6}")

        print(f"\n{sep}")
        print("  END OF REPORT")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 8.  BASELINE REPORT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

class BaselineReportGenerator:
    """
    Generates a human-readable, file-backed baseline report.

    Outputs:
        • Console report (via EvaluationEngine.print_report)
        • JSON machine-readable report
        • Excel summary workbook with per-metric tabs
        • Markdown report for academic documentation
    """

    def __init__(self, engine: EvaluationEngine):
        self.engine = engine

    def generate_all(self, output_prefix: str = "baseline"):
        report = self.engine.run()
        self.engine.print_report(report)

        # JSON
        json_path = f"{output_prefix}_eval_report.json"
        report.to_json(json_path)

        # Excel
        excel_path = f"{output_prefix}_eval_report.xlsx"
        self._to_excel(report, excel_path)

        # Markdown
        md_path = f"{output_prefix}_eval_report.md"
        self._to_markdown(report, md_path)

        return report, json_path, excel_path, md_path

    def _to_excel(self, report: EvaluationReport, path: str):
        """Write a multi-sheet Excel workbook."""
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # Sheet 1: Summary
            summary_rows = [
                ("Generated At", report.generated_at),
                ("Data Source",  report.data_source),
                *[(k, v) for k, v in report.dataset_summary.items()],
            ]
            pd.DataFrame(summary_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="Summary", index=False
            )

            # Sheet 2: Extraction
            e = report.extraction
            ext_rows = [
                ("Articles Evaluated",          e.total_articles_evaluated),
                ("Gold Predictions",            e.total_gold_predictions),
                ("System Predictions",          e.total_system_predictions),
                ("True Positives (exact)",      e.true_positives),
                ("False Positives (exact)",     e.false_positives),
                ("False Negatives (exact)",     e.false_negatives),
                ("Precision (exact)",           round(e.precision, 4)),
                ("Recall (exact)",              round(e.recall, 4)),
                ("F1 (exact)",                  round(e.f1, 4)),
                ("Precision (soft)",            round(e.soft_precision, 4)),
                ("Recall (soft)",               round(e.soft_recall, 4)),
                ("F1 (soft)",                   round(e.soft_f1, 4)),
                ("Avg Verif (human-validated)", round(e.avg_verifiability_true_preds, 3)),
                ("Avg Verif (not validated)",   round(e.avg_verifiability_false_preds, 3)),
                ("Macro per-article F1 (soft)", round(float(np.mean(e.per_article_f1)), 4) if e.per_article_f1 else "N/A"),
            ]
            pd.DataFrame(ext_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="1_Extraction", index=False
            )

            # Sheet 3: Validation
            v = report.validation
            val_rows = [
                ("Total Accepted Predictions",   v.total_candidates),
                ("Human-Confirmed Valid",         v.human_confirmed_passed),
                ("Not Human-Confirmed",           v.human_rejected_passed),
                ("Filter Precision",              round(v.filter_precision, 4)),
                ("Predictions With Date",         v.predictions_with_date),
                ("Predictions Past Deadline",     v.predictions_past_due),
                ("Avg Deadline Confidence",       round(v.avg_deadline_confidence, 3)),
                ("High Verif Score (≥3)",         v.high_verif_score_count),
                ("Low Verif Score (<3)",          v.low_verif_score_count),
            ]
            pd.DataFrame(val_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="2_Validation", index=False
            )

            # Sheet 4: Grading
            g = report.grading
            grad_rows = [
                ("Past-Due Total",               g.past_due_total),
                ("Past-Due Graded",              g.past_due_graded),
                ("Grading Coverage",             round(g.grading_coverage, 4)),
                ("TRUE count",                   g.true_count),
                ("FALSE count",                  g.false_count),
                ("PARTIALLY_TRUE count",         g.partially_true_count),
                ("ERROR count",                  g.error_count),
                ("TRUE rate",                    round(g.true_rate, 4)),
                ("FALSE rate",                   round(g.false_rate, 4)),
                ("PARTIALLY_TRUE rate",          round(g.partially_true_rate, 4)),
                ("Avg Conf (TRUE)",              round(g.avg_conf_true_preds, 3)),
                ("Avg Conf (FALSE)",             round(g.avg_conf_false_preds, 3)),
                ("Avg Verif (TRUE)",             round(g.avg_verif_true_preds, 3)),
                ("Avg Verif (FALSE)",            round(g.avg_verif_false_preds, 3)),
            ]
            pd.DataFrame(grad_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="3_Grading", index=False
            )

            # Sheet 5: Agreement
            a = report.agreement
            agr_rows = [
                ("Total Graded",                 a.total_graded),
                ("Claude YES",                   a.claude_yes),
                ("Claude NO",                    a.claude_no),
                ("Claude PARTIALLY",             a.claude_partially),
                ("Claude Error",                 a.claude_error),
                ("Claude Agreement Rate",        round(a.claude_agreement_rate, 4)),
                ("Gemini YES",                   a.gemini_yes),
                ("Gemini NO",                    a.gemini_no),
                ("Gemini PARTIALLY",             a.gemini_partially),
                ("Gemini Error",                 a.gemini_error),
                ("Gemini Agreement Rate",        round(a.gemini_agreement_rate, 4)),
                ("All Agree (Claude+Gemini YES)", a.all_agree),
                ("All Disagree (both NO)",       a.all_disagree),
                ("Split Verdict",                a.split),
                ("Consensus Rate",               round(a.consensus_rate, 4)),
                ("Kappa: Claude↔Gemini",         round(a.cohen_kappa_claude_gemini, 4)),
                ("Kappa: Claude↔GPT",            round(a.cohen_kappa_claude_gpt, 4)),
                ("Kappa: Gemini↔GPT",            round(a.cohen_kappa_gemini_gpt, 4)),
            ]
            pd.DataFrame(agr_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="4_ModelAgreement", index=False
            )

            # Agreement by label
            if a.agreement_by_label:
                label_rows = [
                    (label, stats["n"],
                     round(stats["claude_yes_rate"], 4),
                     round(stats["gemini_yes_rate"], 4))
                    for label, stats in a.agreement_by_label.items()
                ]
                pd.DataFrame(
                    label_rows,
                    columns=["GPT Label", "Count", "Claude YES Rate", "Gemini YES Rate"]
                ).to_excel(writer, sheet_name="4b_AgreementByLabel", index=False)

            # Sheet 6: Reliability
            r = report.reliability
            rel_rows = [
                ("Total Predictions",            r.total_predictions),
                ("Grading Error Rate",           round(r.grading_error_rate, 4)),
                ("Claude Error Rate",            round(r.claude_error_rate, 4)),
                ("Gemini Error Rate",            round(r.gemini_error_rate, 4)),
                ("Any Error Rate",               round(r.any_error_rate, 4)),
                ("Grading Coverage (past-due)",  round(r.grading_coverage_rate, 4)),
                ("Claude Review Coverage",       round(r.claude_coverage_rate, 4)),
                ("Gemini Review Coverage",       round(r.gemini_coverage_rate, 4)),
                ("Deadline Coverage",            round(r.deadline_coverage, 4)),
                ("Avg Deadline Confidence",      round(r.avg_deadline_confidence, 3)),
                ("Verif–Certainty Pearson r",    round(r.verif_certainty_corr, 4)),
                ("Articles with 0 Predictions", r.articles_with_zero_preds),
                ("Avg Predictions per Article",  round(r.avg_preds_per_article, 2)),
                ("Max Predictions per Article",  r.max_preds_per_article),
            ]
            pd.DataFrame(rel_rows, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="5_Reliability", index=False
            )

        print(f"✅ Excel report saved to {path}")

    def _to_markdown(self, report: EvaluationReport, path: str):
        """Generate a Markdown report suitable for academic papers."""
        e = report.extraction
        v = report.validation
        g = report.grading
        a = report.agreement
        r = report.reliability

        md = f"""# AI Prediction Pipeline — Baseline Evaluation Report

**Generated:** {report.generated_at}  
**Data Source:** {report.data_source}

---

## Dataset Summary

| Metric | Value |
|--------|-------|
{"".join(f"| {k} | {val} |" + chr(10) for k, val in report.dataset_summary.items())}

---

## 1. Prediction Extraction Metrics

The extraction stage identifies prediction candidates from raw article text.
Ground truth is defined by human annotation (Column E = 1).

| Metric | Exact Match | Soft Match (token F1 ≥ 0.5) |
|--------|------------|---------------------------|
| Precision | {e.precision:.3f} | {e.soft_precision:.3f} |
| Recall    | {e.recall:.3f}    | {e.soft_recall:.3f}    |
| F1        | {e.f1:.3f}        | {e.soft_f1:.3f}        |

- Gold predictions (human-validated): **{e.total_gold_predictions}**
- System predictions (in gold articles): **{e.total_system_predictions}**
- Macro per-article F1 (soft): **{np.mean(e.per_article_f1):.3f}** ± {np.std(e.per_article_f1):.3f}

### Score Calibration

| Group | Avg Verifiability | Avg Certainty |
|-------|------------------|---------------|
| Human-validated predictions | {e.avg_verifiability_true_preds:.2f} | {e.avg_certainty_true_preds:.2f} |
| Non-validated predictions   | {e.avg_verifiability_false_preds:.2f} | N/A |

---

## 2. Validation Quality Metrics

| Metric | Value |
|--------|-------|
| Total accepted by system | {v.total_candidates} |
| Human-confirmed valid | {v.human_confirmed_passed} |
| Filter Precision | {v.filter_precision:.3f} |
| Predictions with valid deadline | {v.predictions_with_date} |
| Predictions past deadline | {v.predictions_past_due} |
| Avg deadline confidence | {v.avg_deadline_confidence:.2f} |

---

## 3. Grading Accuracy Metrics

| Label | Count | Rate |
|-------|-------|------|
| TRUE | {g.true_count} | {g.true_rate:.1%} |
| FALSE | {g.false_count} | {g.false_rate:.1%} |
| PARTIALLY_TRUE | {g.partially_true_count} | {g.partially_true_rate:.1%} |
| ERROR | {g.error_count} | {g.error_rate:.1%} |

- Grading coverage (past-due): **{g.grading_coverage:.3f}**

### Score Calibration by Outcome

| Outcome | Avg Conf. | Avg Verifiability |
|---------|----------|-------------------|
| TRUE  | {g.avg_conf_true_preds:.2f} | {g.avg_verif_true_preds:.2f} |
| FALSE | {g.avg_conf_false_preds:.2f} | {g.avg_verif_false_preds:.2f} |

---

## 4. Model Agreement Analysis

| Model | YES | NO | PARTIALLY | Error | Agreement Rate |
|-------|-----|----|-----------|-------|----------------|
| Claude | {a.claude_yes} | {a.claude_no} | {a.claude_partially} | {a.claude_error} | {a.claude_agreement_rate:.1%} |
| Gemini | {a.gemini_yes} | {a.gemini_no} | {a.gemini_partially} | {a.gemini_error} | {a.gemini_agreement_rate:.1%} |

**Trilateral consensus (both agree with GPT):** {a.all_agree}/{a.total_graded} ({a.consensus_rate:.1%})

### Cohen's Kappa

| Pair | κ | Interpretation |
|------|---|----------------|
| Claude ↔ Gemini | {a.cohen_kappa_claude_gemini:.3f} | {'Substantial' if a.cohen_kappa_claude_gemini > 0.6 else 'Moderate' if a.cohen_kappa_claude_gemini > 0.4 else 'Fair'} |
| Claude ↔ GPT | {a.cohen_kappa_claude_gpt:.3f} | {'Substantial' if a.cohen_kappa_claude_gpt > 0.6 else 'Moderate' if a.cohen_kappa_claude_gpt > 0.4 else 'Fair'} |
| Gemini ↔ GPT | {a.cohen_kappa_gemini_gpt:.3f} | {'Substantial' if a.cohen_kappa_gemini_gpt > 0.6 else 'Moderate' if a.cohen_kappa_gemini_gpt > 0.4 else 'Fair'} |

### Agreement by GPT Label

| GPT Label | n | Claude YES | Gemini YES |
|-----------|---|------------|------------|
{"".join(f"| {label} | {stats['n']} | {stats['claude_yes_rate']:.1%} | {stats['gemini_yes_rate']:.1%} |" + chr(10) for label, stats in a.agreement_by_label.items())}

---

## 5. System Reliability Metrics

| Metric | Value |
|--------|-------|
| Grading error rate | {r.grading_error_rate:.3f} |
| Claude error rate | {r.claude_error_rate:.3f} |
| Gemini error rate | {r.gemini_error_rate:.3f} |
| Any-component error rate | {r.any_error_rate:.3f} |
| Grading coverage (past-due) | {r.grading_coverage_rate:.3f} |
| Claude review coverage | {r.claude_coverage_rate:.3f} |
| Gemini review coverage | {r.gemini_coverage_rate:.3f} |
| Deadline coverage | {r.deadline_coverage:.3f} |
| Avg deadline confidence | {r.avg_deadline_confidence:.2f} |
| Verifiability–Certainty Pearson r | {r.verif_certainty_corr:.3f} |
| Avg predictions per article | {r.avg_preds_per_article:.2f} |

---

*Report generated automatically by `evaluation_framework.py`.*
"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"✅ Markdown report saved to {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline_evaluation(excel_path: str, output_prefix: str = "baseline"):
    """
    One-call entry point for the full baseline evaluation.

    Args:
        excel_path:     Path to the graded Excel file.
        output_prefix:  Prefix for all output files (JSON, Excel, Markdown).

    Returns:
        EvaluationReport object with all metrics.
    """
    engine    = EvaluationEngine(excel_path)
    generator = BaselineReportGenerator(engine)
    report, json_p, excel_p, md_p = generator.generate_all(output_prefix)

    print(f"\n📁 Output files:")
    print(f"   JSON    → {json_p}")
    print(f"   Excel   → {excel_p}")
    print(f"   Markdown→ {md_p}")

    return report


if __name__ == "__main__":
    report = run_baseline_evaluation("Grading_pred_anls_enhanced_with_multinov5.xlsx", output_prefix="run2")