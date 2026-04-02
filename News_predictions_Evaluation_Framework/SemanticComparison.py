"""
==============================================================================
 Semantic Comparison Module — AI Prediction Pipeline
 Module: semantic_comparison.py

 PURPOSE:
     Evaluate whether a new model run improved over a baseline, using the
     human-annotated gold set as the reference — without requiring new
     human grading.

 HOW IT WORKS:
     1. Gold Set      → predictions where Human Grading == 1 (old file)
     2. System Output → predictions from any new model run (new file)
     3. Embedding     → encode all predictions via OpenAI embeddings API
     4. Matching      → cosine similarity, greedy per-article matching
     5. Metrics       → precision, recall, F1 at multiple thresholds
     6. Reporting     → Excel + JSON + console output

 REUSABLE:
     Pass any new output Excel file to compare() and get a full report.
     Works for any number of model variants — no human grading needed again.

 USAGE:
     from semantic_comparison import SemanticComparator
     comp = SemanticComparator(gold_file="old_graded.xlsx", api_key="...",
                               base_url="https://api.ai.it.ufl.edu")
     report = comp.compare(new_file="new_run.xlsx", run_label="Finetuned v1")
     comp.save_report(report, output_prefix="comparison_v1")
==============================================================================
"""

import os
import json
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

EMBED_MODEL      = "nomic-embed-text-v1.5"   # cheap, fast, strong
EMBED_BATCH_SIZE = 100                         # texts per API call
THRESHOLDS       = [0.70, 0.75, 0.80, 0.85, 0.90]  # all reported
DEFAULT_THRESHOLD = 0.80                       # primary threshold


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionEntry:
    article_number:      int
    prediction_number:   int
    prediction_text:     str
    verifiability_score: float
    certainty_score:     float
    deadline_estimate:   str
    grading:             str
    claude_agrees:       str
    gemini_agrees:       str
    human_validated:     bool    # only meaningful in gold set
    embedding:           Optional[List[float]] = field(default=None, repr=False)


@dataclass
class MatchedPair:
    """A gold prediction matched to a system prediction."""
    article_number:       int
    gold_text:            str
    system_text:          str
    similarity:           float
    gold_verif:           float
    system_verif:         float
    gold_certainty:       float
    system_certainty:     float
    gold_grading:         str
    system_grading:       str


@dataclass
class ThresholdMetrics:
    threshold:     float
    true_positives:  int   = 0
    false_positives: int   = 0
    false_negatives: int   = 0
    precision:       float = 0.0
    recall:          float = 0.0
    f1:              float = 0.0


@dataclass
class ComparisonReport:
    """Full comparison report between gold and one system run."""
    run_label:              str
    generated_at:           str
    gold_file:              str
    system_file:            str

    # Dataset counts
    gold_articles:          int = 0
    gold_predictions:       int = 0
    system_articles:        int = 0
    system_predictions:     int = 0
    shared_articles:        int = 0
    gold_in_shared:         int = 0
    system_in_shared:       int = 0

    # Extraction ratio (vs gold)
    gold_avg_per_article:   float = 0.0
    system_avg_per_article: float = 0.0
    over_extraction_ratio:  float = 0.0

    # Semantic metrics at multiple thresholds
    threshold_metrics:      List[ThresholdMetrics] = field(default_factory=list)

    # Primary threshold metrics (DEFAULT_THRESHOLD)
    primary_threshold:      float = DEFAULT_THRESHOLD
    primary_precision:      float = 0.0
    primary_recall:         float = 0.0
    primary_f1:             float = 0.0
    primary_tp:             int   = 0
    primary_fp:             int   = 0
    primary_fn:             int   = 0

    # Score calibration (matched pairs at primary threshold)
    matched_pairs:          List[MatchedPair] = field(default_factory=list)
    avg_similarity_matched: float = 0.0
    gold_avg_verif:         float = 0.0
    system_avg_verif:       float = 0.0
    verif_score_delta:      float = 0.0    # system - gold (positive = better)
    gold_avg_certainty:     float = 0.0
    system_avg_certainty:   float = 0.0
    certainty_score_delta:  float = 0.0

    # Per-article breakdown
    per_article_results:    List[Dict] = field(default_factory=list)

    # Summary verdict
    verdict:                str = ""


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Handles all OpenAI embedding API calls with batching and retry.
    Uses nomic-embed-text-v1.5: 1536 dims, fast, cheap.
    """

    def __init__(self, client: OpenAI):
        self.client = client
        self._cache: Dict[str, List[float]] = {}

    def embed_batch(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        """Embed a list of texts, returning one vector per text."""
        results = []
        batches = [
            texts[i:i + EMBED_BATCH_SIZE]
            for i in range(0, len(texts), EMBED_BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            # Check cache
            uncached_indices = []
            uncached_texts   = []
            cached_vectors   = {}

            for i, text in enumerate(batch):
                key = text.strip().lower()[:500]
                if key in self._cache:
                    cached_vectors[i] = self._cache[key]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)

            # Fetch uncached
            if uncached_texts:
                for attempt in range(retries):
                    try:
                        response = self.client.embeddings.create(
                            model=EMBED_MODEL,
                            input=uncached_texts
                        )
                        for j, embedding_obj in enumerate(response.data):
                            vec = embedding_obj.embedding
                            orig_idx = uncached_indices[j]
                            cached_vectors[orig_idx] = vec
                            key = uncached_texts[j].strip().lower()[:500]
                            self._cache[key] = vec
                        break
                    except Exception as e:
                        print(f"    ⚠ Embedding attempt {attempt+1} failed: {e}")
                        if attempt < retries - 1:
                            time.sleep(2 ** attempt)
                        else:
                            # Fallback: zero vectors
                            for idx in uncached_indices:
                                cached_vectors[idx] = [0.0] * 1536

            # Reconstruct in order
            for i in range(len(batch)):
                results.append(cached_vectors.get(i, [0.0] * 1536))

            if batch_idx % 5 == 0 and len(batches) > 1:
                print(f"    Embedded batch {batch_idx+1}/{len(batches)}...")

        return results

    def embed_entries(self, entries: List[PredictionEntry]) -> List[PredictionEntry]:
        """Embed all prediction entries in-place."""
        texts = [e.prediction_text for e in entries]
        print(f"  🔢 Embedding {len(texts)} predictions...")
        vectors = self.embed_batch(texts)
        for entry, vec in zip(entries, vectors):
            entry.embedding = vec
        return entries

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# ─────────────────────────────────────────────────────────────────────────────
# MATCHER
# ─────────────────────────────────────────────────────────────────────────────

class SemanticMatcher:
    """
    Greedy per-article matcher.

    For each article, matches system predictions to gold predictions
    using cosine similarity on embeddings.

    Matching rules:
        - Matching is done per article (cross-article matches are invalid)
        - Each gold prediction can be matched at most once (greedy, highest sim first)
        - Each system prediction can be matched at most once
        - A match is accepted if similarity >= threshold
    """

    def match_article(
        self,
        gold_entries:   List[PredictionEntry],
        system_entries: List[PredictionEntry],
        threshold:      float
    ) -> Tuple[List[MatchedPair], int, int, int]:
        """
        Returns (matched_pairs, TP, FP, FN) for one article.
        """
        if not gold_entries or not system_entries:
            return [], 0, len(system_entries), len(gold_entries)

        # Build similarity matrix
        sim_matrix = np.zeros((len(gold_entries), len(system_entries)))
        for gi, gold in enumerate(gold_entries):
            for si, sys in enumerate(system_entries):
                if gold.embedding and sys.embedding:
                    sim_matrix[gi, si] = EmbeddingEngine.cosine_similarity(
                        gold.embedding, sys.embedding
                    )

        matched_gold   = set()
        matched_system = set()
        pairs          = []

        # Greedy: sort all (gi, si) pairs by sim descending
        all_pairs = sorted(
            [(sim_matrix[gi, si], gi, si)
             for gi in range(len(gold_entries))
             for si in range(len(system_entries))],
            reverse=True
        )

        for sim, gi, si in all_pairs:
            if sim < threshold:
                break
            if gi in matched_gold or si in matched_system:
                continue

            matched_gold.add(gi)
            matched_system.add(si)

            gold = gold_entries[gi]
            sys  = system_entries[si]

            pairs.append(MatchedPair(
                article_number   = gold.article_number,
                gold_text        = gold.prediction_text,
                system_text      = sys.prediction_text,
                similarity       = round(sim, 4),
                gold_verif       = gold.verifiability_score,
                system_verif     = sys.verifiability_score,
                gold_certainty   = gold.certainty_score,
                system_certainty = sys.certainty_score,
                gold_grading     = gold.grading,
                system_grading   = sys.grading,
            ))

        tp = len(pairs)
        fp = len(system_entries) - len(matched_system)
        fn = len(gold_entries)   - len(matched_gold)

        return pairs, tp, fp, fn


# ─────────────────────────────────────────────────────────────────────────────
# MAIN COMPARATOR
# ─────────────────────────────────────────────────────────────────────────────

class SemanticComparator:
    """
    Main comparison class.

    Instantiate once with the gold file + API credentials.
    Call compare() for each new system run you want to evaluate.
    """

    def __init__(
        self,
        gold_file: str,
        api_key:   Optional[str] = None,
        base_url:  Optional[str] = None
    ):
        self.gold_file = gold_file
        api_key  = api_key  or os.getenv("navigator_api")
        base_url = base_url or "https://api.ai.it.ufl.edu"

        self.client  = OpenAI(api_key=api_key, base_url=base_url)
        self.embedder = EmbeddingEngine(self.client)
        self.matcher  = SemanticMatcher()

        print(f"📂 Loading gold set from: {gold_file}")
        self.gold_df   = pd.read_excel(gold_file)
        self.gold_entries = self._load_gold_entries()
        print(f"✅ Gold set: {len(self.gold_entries)} human-validated predictions "
              f"across {len({e.article_number for e in self.gold_entries})} articles")

        # Pre-embed gold set (done once, reused for all comparisons)
        print("🔢 Pre-embedding gold set...")
        self.gold_entries = self.embedder.embed_entries(self.gold_entries)
        print("✅ Gold embeddings ready.\n")

    def _load_gold_entries(self) -> List[PredictionEntry]:
        """Load human-validated predictions from the gold file."""
        entries = []

        # Detect human grading column
        human_col = None
        for col in self.gold_df.columns:
            if "human" in col.lower() or col == "Unnamed: 4":
                human_col = col
                break

        if human_col is None:
            raise ValueError(
                "Cannot find Human Grading column in gold file. "
                "Expected column named 'Human Grading' or 'Unnamed: 4'."
            )

        validated = self.gold_df[self.gold_df[human_col] == 1]

        for _, row in validated.iterrows():
            try:
                pred_num = int(row.get("Prediction_Number", 0))
                if pred_num <= 0:
                    continue
                entries.append(PredictionEntry(
                    article_number      = int(row.get("Article_Number", 0)),
                    prediction_number   = pred_num,
                    prediction_text     = str(row.get("Prediction", "")).strip(),
                    verifiability_score = float(row.get("Verifiability_Score") or 0),
                    certainty_score     = float(row.get("Certainty_Score") or 0),
                    deadline_estimate   = str(row.get("Deadline_Estimate", "")),
                    grading             = str(row.get("Grading", "Pending")),
                    claude_agrees       = str(row.get("Claude_Agrees", "Pending")),
                    gemini_agrees       = str(row.get("Gemini_Agrees", "Pending")),
                    human_validated     = True,
                ))
            except Exception as e:
                print(f"  ⚠ Skipping gold row: {e}")

        return entries

    def _load_system_entries(self, system_file: str) -> List[PredictionEntry]:
        """Load all predictions from a system output file."""
        df = pd.read_excel(system_file)
        entries = []

        for _, row in df.iterrows():
            try:
                pred_num = int(row.get("Prediction_Number", 0))
                if pred_num <= 0:
                    continue
                pred_text = str(row.get("Prediction", "")).strip()
                if not pred_text or pred_text.lower() in ("no predictions found", "nan"):
                    continue
                entries.append(PredictionEntry(
                    article_number      = int(row.get("Article_Number", 0)),
                    prediction_number   = pred_num,
                    prediction_text     = pred_text,
                    verifiability_score = float(row.get("Verifiability_Score") or 0),
                    certainty_score     = float(row.get("Certainty_Score") or 0),
                    deadline_estimate   = str(row.get("Deadline_Estimate", "")),
                    grading             = str(row.get("Grading", "Pending")),
                    claude_agrees       = str(row.get("Claude_Agrees", "Pending")),
                    gemini_agrees       = str(row.get("Gemini_Agrees", "Pending")),
                    human_validated     = False,
                ))
            except Exception as e:
                print(f"  ⚠ Skipping system row: {e}")

        return entries

    def compare(
        self,
        system_file: str,
        run_label:   str = "System Run",
        thresholds:  List[float] = THRESHOLDS
    ) -> ComparisonReport:
        """
        Compare a system output file against the gold set.

        Args:
            system_file : Path to the new model's output Excel file.
            run_label   : Human-readable label for this run (e.g. "Finetuned v1").
            thresholds  : List of similarity thresholds to evaluate at.

        Returns:
            ComparisonReport with all metrics.
        """
        print(f"\n{'='*60}")
        print(f"🔬 Comparing: {run_label}")
        print(f"{'='*60}")

        report = ComparisonReport(
            run_label    = run_label,
            generated_at = datetime.now().isoformat(),
            gold_file    = self.gold_file,
            system_file  = system_file,
        )

        # ── Load system entries ──────────────────────────────────────────────
        print(f"📂 Loading system file: {system_file}")
        system_entries = self._load_system_entries(system_file)
        print(f"✅ Loaded {len(system_entries)} system predictions")

        # ── Embed system entries ─────────────────────────────────────────────
        system_entries = self.embedder.embed_entries(system_entries)

        # ── Build article maps ───────────────────────────────────────────────
        gold_by_article   = defaultdict(list)
        system_by_article = defaultdict(list)

        for e in self.gold_entries:
            gold_by_article[e.article_number].append(e)
        for e in system_entries:
            system_by_article[e.article_number].append(e)

        shared_articles = set(gold_by_article.keys()) & set(system_by_article.keys())

        # ── Dataset counts ───────────────────────────────────────────────────
        report.gold_articles          = len(gold_by_article)
        report.gold_predictions       = len(self.gold_entries)
        report.system_articles        = len(system_by_article)
        report.system_predictions     = len(system_entries)
        report.shared_articles        = len(shared_articles)
        report.gold_in_shared         = sum(
            len(gold_by_article[a]) for a in shared_articles
        )
        report.system_in_shared       = sum(
            len(system_by_article[a]) for a in shared_articles
        )
        report.gold_avg_per_article   = (
            report.gold_in_shared / len(shared_articles) if shared_articles else 0
        )
        report.system_avg_per_article = (
            report.system_in_shared / len(shared_articles) if shared_articles else 0
        )
        report.over_extraction_ratio  = (
            report.system_avg_per_article / report.gold_avg_per_article
            if report.gold_avg_per_article > 0 else 0
        )

        # ── Multi-threshold evaluation ───────────────────────────────────────
        print(f"\n⚙  Running matching at {len(thresholds)} thresholds "
              f"across {len(shared_articles)} shared articles...")

        all_pairs_by_threshold = {}

        for threshold in thresholds:
            total_tp = total_fp = total_fn = 0
            all_pairs = []

            for article_num in sorted(shared_articles):
                gold_art   = gold_by_article[article_num]
                system_art = system_by_article[article_num]

                pairs, tp, fp, fn = self.matcher.match_article(
                    gold_art, system_art, threshold
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_pairs.extend(pairs)

            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec)   if (prec + rec) > 0 else 0

            tm = ThresholdMetrics(
                threshold       = threshold,
                true_positives  = total_tp,
                false_positives = total_fp,
                false_negatives = total_fn,
                precision       = round(prec, 4),
                recall          = round(rec, 4),
                f1              = round(f1, 4),
            )
            report.threshold_metrics.append(tm)
            all_pairs_by_threshold[threshold] = all_pairs

            print(f"  τ={threshold:.2f} → P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
                  f"(TP={total_tp} FP={total_fp} FN={total_fn})")

        # ── Primary threshold ────────────────────────────────────────────────
        primary = next(
            tm for tm in report.threshold_metrics
            if tm.threshold == DEFAULT_THRESHOLD
        )
        report.primary_threshold = DEFAULT_THRESHOLD
        report.primary_precision = primary.precision
        report.primary_recall    = primary.recall
        report.primary_f1        = primary.f1
        report.primary_tp        = primary.true_positives
        report.primary_fp        = primary.false_positives
        report.primary_fn        = primary.false_negatives

        # ── Score calibration (matched pairs at primary threshold) ───────────
        matched = all_pairs_by_threshold[DEFAULT_THRESHOLD]
        report.matched_pairs = matched

        if matched:
            sims           = [p.similarity      for p in matched]
            gold_verifs    = [p.gold_verif       for p in matched]
            sys_verifs     = [p.system_verif     for p in matched]
            gold_certs     = [p.gold_certainty   for p in matched]
            sys_certs      = [p.system_certainty for p in matched]

            report.avg_similarity_matched = round(float(np.mean(sims)), 4)
            report.gold_avg_verif         = round(float(np.mean(gold_verifs)), 3)
            report.system_avg_verif       = round(float(np.mean(sys_verifs)), 3)
            report.verif_score_delta      = round(
                report.system_avg_verif - report.gold_avg_verif, 3
            )
            report.gold_avg_certainty     = round(float(np.mean(gold_certs)), 3)
            report.system_avg_certainty   = round(float(np.mean(sys_certs)), 3)
            report.certainty_score_delta  = round(
                report.system_avg_certainty - report.gold_avg_certainty, 3
            )

        # ── Per-article breakdown ────────────────────────────────────────────
        for article_num in sorted(shared_articles):
            gold_art   = gold_by_article[article_num]
            system_art = system_by_article[article_num]

            _, tp, fp, fn = self.matcher.match_article(
                gold_art, system_art, DEFAULT_THRESHOLD
            )
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0

            report.per_article_results.append({
                "article_number":      article_num,
                "gold_count":          len(gold_art),
                "system_count":        len(system_art),
                "true_positives":      tp,
                "false_positives":     fp,
                "false_negatives":     fn,
                "precision":           round(prec, 4),
                "recall":              round(rec, 4),
                "f1":                  round(f1, 4),
            })

        # ── Verdict ──────────────────────────────────────────────────────────
        report.verdict = self._generate_verdict(report)

        return report

    def _generate_verdict(self, report: ComparisonReport) -> str:
        """Generate a human-readable verdict on improvement."""
        lines = []

        # Over-extraction
        if report.over_extraction_ratio <= 1.5:
            lines.append("✅ Extraction count is close to human benchmark "
                         f"({report.over_extraction_ratio:.1f}x over-extraction).")
        elif report.over_extraction_ratio <= 2.5:
            lines.append("⚠ Moderate over-extraction "
                         f"({report.over_extraction_ratio:.1f}x vs gold).")
        else:
            lines.append("❌ High over-extraction "
                         f"({report.over_extraction_ratio:.1f}x vs gold) — "
                         "model is finding too many spurious predictions.")

        # Recall
        if report.primary_recall >= 0.70:
            lines.append(f"✅ Strong recall ({report.primary_recall:.1%}) — "
                         "model finds most human-validated predictions.")
        elif report.primary_recall >= 0.50:
            lines.append(f"⚠ Moderate recall ({report.primary_recall:.1%}) — "
                         "model misses some human-validated predictions.")
        else:
            lines.append(f"❌ Low recall ({report.primary_recall:.1%}) — "
                         "model misses many human-validated predictions.")

        # Precision
        if report.primary_precision >= 0.50:
            lines.append(f"✅ Good precision ({report.primary_precision:.1%}) — "
                         "most system predictions align with gold set.")
        elif report.primary_precision >= 0.30:
            lines.append(f"⚠ Low precision ({report.primary_precision:.1%}) — "
                         "many system predictions have no gold equivalent.")
        else:
            lines.append(f"❌ Very low precision ({report.primary_precision:.1%}) — "
                         "system is generating many unvalidated predictions.")

        # Score delta
        delta = report.verif_score_delta
        if delta > 0.2:
            lines.append(f"✅ Verifiability scores improved by +{delta:.2f} on matched predictions.")
        elif delta < -0.2:
            lines.append(f"⚠ Verifiability scores dropped by {delta:.2f} on matched predictions.")

        return " | ".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT
    # ─────────────────────────────────────────────────────────────────────────

    def print_report(self, report: ComparisonReport):
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"  SEMANTIC COMPARISON REPORT — {report.run_label}")
        print(f"  Generated: {report.generated_at}")
        print(sep)

        print("\n📊 DATASET OVERVIEW")
        print("-" * 40)
        print(f"  Gold predictions (human-validated)  {report.gold_predictions:>6}")
        print(f"  Gold articles                        {report.gold_articles:>6}")
        print(f"  System predictions                   {report.system_predictions:>6}")
        print(f"  System articles                      {report.system_articles:>6}")
        print(f"  Shared articles (evaluated)          {report.shared_articles:>6}")
        print(f"  Gold preds in shared articles        {report.gold_in_shared:>6}")
        print(f"  System preds in shared articles      {report.system_in_shared:>6}")
        print()
        print(f"  Gold avg preds/article               {report.gold_avg_per_article:>7.2f}")
        print(f"  System avg preds/article             {report.system_avg_per_article:>7.2f}")
        print(f"  Over-extraction ratio                {report.over_extraction_ratio:>7.2f}x")

        print("\n\n🎯 SEMANTIC EXTRACTION METRICS")
        print("-" * 40)
        print(f"  {'Threshold':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} "
              f"{'TP':>6} {'FP':>6} {'FN':>6}")
        print("  " + "-" * 58)
        for tm in report.threshold_metrics:
            marker = " ◀ PRIMARY" if tm.threshold == report.primary_threshold else ""
            print(f"  τ={tm.threshold:.2f}       {tm.precision:>10.3f} {tm.recall:>10.3f} "
                  f"{tm.f1:>10.3f} {tm.true_positives:>6} {tm.false_positives:>6} "
                  f"{tm.false_negatives:>6}{marker}")

        print("\n\n📐 SCORE CALIBRATION (Matched Pairs at Primary Threshold)")
        print("-" * 40)
        print(f"  Matched pairs                        {len(report.matched_pairs):>6}")
        print(f"  Avg similarity of matched pairs      {report.avg_similarity_matched:>7.4f}")
        print()
        print(f"  {'Metric':<30} {'Gold':>8} {'System':>8} {'Delta':>8}")
        print("  " + "-" * 58)
        print(f"  {'Avg Verifiability Score':<30} {report.gold_avg_verif:>8.3f} "
              f"{report.system_avg_verif:>8.3f} "
              f"{report.verif_score_delta:>+8.3f}")
        print(f"  {'Avg Certainty Score':<30} {report.gold_avg_certainty:>8.3f} "
              f"{report.system_avg_certainty:>8.3f} "
              f"{report.certainty_score_delta:>+8.3f}")

        print("\n\n📋 PER-ARTICLE BREAKDOWN (Primary Threshold)")
        print("-" * 65)
        print(f"  {'Article':>8} {'Gold':>6} {'Sys':>6} {'TP':>5} "
              f"{'FP':>5} {'FN':>5} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("  " + "-" * 63)
        for row in report.per_article_results:
            print(f"  {row['article_number']:>8} {row['gold_count']:>6} "
                  f"{row['system_count']:>6} {row['true_positives']:>5} "
                  f"{row['false_positives']:>5} {row['false_negatives']:>5} "
                  f"{row['precision']:>7.3f} {row['recall']:>7.3f} {row['f1']:>7.3f}")

        print(f"\n\n🏁 VERDICT")
        print("-" * 40)
        for line in report.verdict.split(" | "):
            print(f"  {line}")

        print(f"\n{sep}\n")

    def save_report(self, report: ComparisonReport, output_prefix: str = "comparison"):
        """Save report as JSON + Excel."""
        # ── JSON ─────────────────────────────────────────────────────────────
        json_path = f"{output_prefix}_semantic_report.json"
        report_dict = asdict(report)
        # Remove raw embeddings from matched pairs (too large for JSON)
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"✅ JSON saved: {json_path}")

        # ── Excel ─────────────────────────────────────────────────────────────
        excel_path = f"{output_prefix}_semantic_report.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

            # Sheet 1: Summary
            summary = [
                ("Run Label",                report.run_label),
                ("Generated At",             report.generated_at),
                ("Gold File",                report.gold_file),
                ("System File",              report.system_file),
                ("Gold Predictions",         report.gold_predictions),
                ("Gold Articles",            report.gold_articles),
                ("System Predictions",       report.system_predictions),
                ("System Articles",          report.system_articles),
                ("Shared Articles",          report.shared_articles),
                ("Gold in Shared Articles",  report.gold_in_shared),
                ("System in Shared Articles",report.system_in_shared),
                ("Gold Avg per Article",     report.gold_avg_per_article),
                ("System Avg per Article",   report.system_avg_per_article),
                ("Over-extraction Ratio",    report.over_extraction_ratio),
                ("Primary Threshold",        report.primary_threshold),
                ("Primary Precision",        report.primary_precision),
                ("Primary Recall",           report.primary_recall),
                ("Primary F1",               report.primary_f1),
                ("Primary TP",               report.primary_tp),
                ("Primary FP",               report.primary_fp),
                ("Primary FN",               report.primary_fn),
                ("Matched Pairs",            len(report.matched_pairs)),
                ("Avg Similarity (matched)", report.avg_similarity_matched),
                ("Gold Avg Verifiability",   report.gold_avg_verif),
                ("System Avg Verifiability", report.system_avg_verif),
                ("Verifiability Delta",      report.verif_score_delta),
                ("Gold Avg Certainty",       report.gold_avg_certainty),
                ("System Avg Certainty",     report.system_avg_certainty),
                ("Certainty Delta",          report.certainty_score_delta),
            ]
            pd.DataFrame(summary, columns=["Metric", "Value"]).to_excel(
                writer, sheet_name="Summary", index=False
            )

            # Sheet 2: Threshold sweep
            thresh_rows = [
                (tm.threshold, tm.precision, tm.recall, tm.f1,
                 tm.true_positives, tm.false_positives, tm.false_negatives)
                for tm in report.threshold_metrics
            ]
            pd.DataFrame(thresh_rows, columns=[
                "Threshold", "Precision", "Recall", "F1",
                "TP", "FP", "FN"
            ]).to_excel(writer, sheet_name="Threshold_Sweep", index=False)

            # Sheet 3: Per-article
            pd.DataFrame(report.per_article_results).to_excel(
                writer, sheet_name="Per_Article", index=False
            )

            # Sheet 4: Matched pairs
            if report.matched_pairs:
                pairs_rows = [
                    {
                        "Article":         p.article_number,
                        "Similarity":      p.similarity,
                        "Gold Prediction": p.gold_text,
                        "System Prediction": p.system_text,
                        "Gold Verif":      p.gold_verif,
                        "System Verif":    p.system_verif,
                        "Gold Certainty":  p.gold_certainty,
                        "System Certainty":p.system_certainty,
                        "Gold Grading":    p.gold_grading,
                        "System Grading":  p.system_grading,
                    }
                    for p in report.matched_pairs
                ]
                pd.DataFrame(pairs_rows).to_excel(
                    writer, sheet_name="Matched_Pairs", index=False
                )

            # Sheet 5: Verdict
            pd.DataFrame(
                [{"Verdict": line} for line in report.verdict.split(" | ")]
            ).to_excel(writer, sheet_name="Verdict", index=False)

        print(f"✅ Excel saved: {excel_path}")
        return json_path, excel_path


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-RUN COMPARISON TABLE
# ─────────────────────────────────────────────────────────────────────────────

def compare_multiple_runs(
    gold_file:    str,
    runs:         List[Dict],   # [{"file": "...", "label": "..."}, ...]
    output_prefix: str = "multi_run",
    api_key:      Optional[str] = None,
    base_url:     Optional[str] = None,
) -> pd.DataFrame:
    """
    Compare multiple system runs against the same gold set.
    Produces a single comparison table — ideal for your paper.

    Args:
        gold_file:     Path to human-graded Excel file.
        runs:          List of dicts with 'file' and 'label' keys.
        output_prefix: Prefix for output files.

    Returns:
        DataFrame with one row per run, all key metrics as columns.

    Example:
        compare_multiple_runs(
            gold_file = "old_graded.xlsx",
            runs = [
                {"file": "baseline_run.xlsx",  "label": "Baseline (no finetuning)"},
                {"file": "finetuned_v1.xlsx",  "label": "Finetuned v1"},
                {"file": "finetuned_v2.xlsx",  "label": "Finetuned v2"},
            ],
            output_prefix = "paper_comparison"
        )
    """
    comparator = SemanticComparator(
        gold_file=gold_file, api_key=api_key, base_url=base_url
    )

    summary_rows = []
    all_reports  = []

    for run in runs:
        report = comparator.compare(
            system_file = run["file"],
            run_label   = run["label"]
        )
        comparator.print_report(report)
        comparator.save_report(
            report,
            output_prefix=f"{output_prefix}_{run['label'].replace(' ', '_')}"
        )
        all_reports.append(report)

        summary_rows.append({
            "Run":                    report.run_label,
            "System Preds":           report.system_predictions,
            "Avg Preds/Article":      round(report.system_avg_per_article, 2),
            "Over-extraction Ratio":  round(report.over_extraction_ratio, 2),
            "Precision (τ=0.80)":     report.primary_precision,
            "Recall (τ=0.80)":        report.primary_recall,
            "F1 (τ=0.80)":            report.primary_f1,
            "Precision (τ=0.75)":     next(t.precision for t in report.threshold_metrics if t.threshold == 0.75),
            "Recall (τ=0.75)":        next(t.recall    for t in report.threshold_metrics if t.threshold == 0.75),
            "F1 (τ=0.75)":            next(t.f1        for t in report.threshold_metrics if t.threshold == 0.75),
            "Matched Pairs":          len(report.matched_pairs),
            "Avg Similarity":         report.avg_similarity_matched,
            "Verif Delta":            report.verif_score_delta,
            "Certainty Delta":        report.certainty_score_delta,
        })

    df = pd.DataFrame(summary_rows)

    # Save combined comparison table
    combined_path = f"{output_prefix}_combined_comparison.xlsx"
    df.to_excel(combined_path, index=False)
    print(f"\n✅ Combined comparison table saved: {combined_path}")

    print("\n📊 COMBINED RUN COMPARISON")
    print(df.to_string(index=False))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run a direct comparison between old and new files.
    Uses environment variable 'navigator_api' for the API key.
    """
    GOLD_FILE   = "Grading_pred_anls_enhanced_with_multinov5.xlsx"
    SYSTEM_FILE = "prediction_anls_nov5_two_stage_extraction.xlsx"

    comparator = SemanticComparator(gold_file=GOLD_FILE)

    report = comparator.compare(
        system_file = SYSTEM_FILE,
        run_label   = "Finetuned with two stage"
    )

    comparator.print_report(report)
    comparator.save_report(report, output_prefix="finetuned_two_stage_comparison")