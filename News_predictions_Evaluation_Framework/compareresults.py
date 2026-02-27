from SemanticComparison import compare_multiple_runs

compare_multiple_runs(
    gold_file = "Grading_pred_anls_enhanced_with_multinov5.xlsx",
    runs = [
        {"file": "pred_anls_enhanced_with_multinov5.xlsx",    "label": "Baseline"},
        {"file": "pred_anls_with_learned_grading40.xlsx","label": "Finetuned with 40shot grading"},
        {"file": "pred_anls_with_ALL_learned_grading_.xlsx","label": "Finetuned with all grading"},
        {"file": "pred_anls_nov5_backtracking.xlsx",   "label": "backtracking"},
    ],
    output_prefix = "paper_comparison"
)