import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from datetime import date
import time
import traceback
import concurrent.futures
from threading import Lock
import re
import random

load_dotenv()

keymain = os.getenv("navigator_api")
client = OpenAI(api_key=keymain, base_url="https://api.ai.it.ufl.edu")

file_lock = Lock()

# ─────────────────────────────────────────────────────────────────────────────
# CORE IDEA:
#   Stage 1 — Liberal Extractor   (gpt-5): "pull everything that COULD be a prediction"
#   Stage 2 — Strict Filter       (gpt-4o): "would a journalist fact-check this in 6 months?"
#
#   Few-shot examples are trimmed to ~12 BOUNDARY cases (hard positives + hard negatives)
#   NOT all 450. The model needs to learn the decision edge, not the average.
# ─────────────────────────────────────────────────────────────────────────────

class PredictionProcessor:
    def __init__(self, checkpoint_file="processing_checkpoint.xlsx", backup_interval=5):
        self.checkpoint_file = checkpoint_file
        self.backup_interval = backup_interval
        self.processed_count = 0
        self.graded_examples = []

    # ── Utilities ─────────────────────────────────────────────────────────────

    def safe_json_parse(self, text, default_value):
        try:
            text = text.strip()
            if not text:
                return default_value
            text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group()
            text = re.sub(r'\s+', ' ', text)
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e} | raw: {text[:200]}")
            # Fallback regex extraction
            try:
                predictions_match = re.search(r'"predictions"\s*:\s*(\[.*?\])', text, re.DOTALL)
                if predictions_match:
                    return {"predictions": json.loads(predictions_match.group(1))}
                verdict_match = re.search(r'"verdict"\s*:\s*"([^"]+)"', text)
                reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
                if verdict_match:
                    return {
                        "verdict": verdict_match.group(1),
                        "reason": reason_match.group(1) if reason_match else ""
                    }
            except:
                pass
            return default_value
        except Exception as e:
            print(f"Unexpected JSON error: {e}")
            return default_value

    def extract_context_around_prediction(self, article_text, prediction_text, context_lines=2):
        try:
            sentences = re.split(r'[.!?]+', article_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            idx = -1
            for i, s in enumerate(sentences):
                if prediction_text.lower() in s.lower():
                    idx = i
                    break
            if idx == -1:
                for i, s in enumerate(sentences):
                    pred_words = set(prediction_text.lower().split())
                    sent_words = set(s.lower().split())
                    if len(pred_words.intersection(sent_words)) >= len(pred_words) * 0.6:
                        idx = i
                        break
            if idx == -1:
                return prediction_text
            start = max(0, idx - context_lines)
            end = min(len(sentences), idx + context_lines + 1)
            return '. '.join(sentences[start:end]) + '.'
        except Exception as e:
            print(f"Context extraction error: {e}")
            return prediction_text

    def clean_text_for_excel(self, text):
        if not isinstance(text, str):
            return text
        cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        cleaned = cleaned.replace('"', '""')
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        if len(cleaned) > 32767:
            cleaned = cleaned[:32760] + "..."
        return cleaned

    def save_checkpoint(self, output_data, output_file_path):
        try:
            with file_lock:
                cleaned_data = []
                for record in output_data:
                    cleaned_record = {
                        k: self.clean_text_for_excel(v) if isinstance(v, str) else v
                        for k, v in record.items()
                    }
                    cleaned_data.append(cleaned_record)

                checkpoint_df = pd.DataFrame(cleaned_data)
                try:
                    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                        checkpoint_df.to_excel(writer, sheet_name='Predictions_Analysis', index=False)
                        self.format_excel_columns(writer.sheets['Predictions_Analysis'])
                except Exception as excel_err:
                    print(f"⚠ Excel save failed, falling back to CSV: {excel_err}")
                    csv_file = output_file_path.replace('.xlsx', '_backup.csv')
                    checkpoint_df.to_csv(csv_file, index=False, encoding='utf-8')

                try:
                    checkpoint_df.to_excel(f"backup_{self.checkpoint_file}", index=False)
                except Exception as backup_err:
                    print(f"⚠ Backup failed: {backup_err}")

                print(f"✅ Checkpoint saved: {len(output_data)} records")
        except Exception as e:
            print(f"⚠ Checkpoint save error: {e}")
            try:
                import pickle
                with open(output_file_path.replace('.xlsx', '_emergency.pkl'), 'wb') as f:
                    pickle.dump(output_data, f)
                print("🆘 Emergency pickle saved")
            except:
                print("💥 Complete save failure")

    def format_excel_columns(self, worksheet):
        column_widths = {
            'A': 15, 'B': 60, 'C': 15, 'D': 60, 'E': 15, 'F': 15,
            'G': 15, 'H': 40, 'I': 15, 'J': 50, 'K': 20, 'L': 60,
            'M': 20, 'N': 60, 'O': 20, 'P': 60
        }
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width

    # ── Few-shot Example Builder ───────────────────────────────────────────────

    def build_boundary_examples(self, graded_examples, n=12):
        """
        Pick ~n boundary examples: articles that have BOTH validated predictions
        AND rejected predictions sitting side by side. These teach the decision edge,
        not just the average pattern.
        Falls back to any available examples if boundary ones aren't available.
        """
        boundary = [ex for ex in graded_examples
                    if ex.get('rejected_predictions') and ex.get('validated_predictions')]

        # If no boundary examples, use whatever we have
        if not boundary:
            print(f"  ⚠ No boundary examples found, using {min(n, len(graded_examples))} available examples")
            boundary = graded_examples

        selected = random.sample(boundary, min(n, len(boundary)))

        shots = ""
        for i, ex in enumerate(selected, 1):
            shots += f"\n--- EXAMPLE {i} ---\n"
            shots += f"ARTICLE SNIPPET:\n{ex['article_text'][:600]}...\n\n"
            shots += "✅ VALID PREDICTIONS (include these):\n"
            for p in ex['validated_predictions'][:3]:
                shots += f"  • {p}\n"
            if ex.get('rejected_predictions'):
                shots += "❌ FALSE POSITIVES (do NOT include these):\n"
                for p in ex['rejected_predictions'][:3]:
                    shots += f"  • {p}\n"
            shots += "\n"
        return shots

    # ── Stage 1: Liberal Extractor ─────────────────────────────────────────────

    def stage1_liberal_extract(self, article_text, few_shot_examples, max_retries=3):
        """
        Cast a wide net. Pull EVERYTHING that smells like a forward-looking claim or predicition.
        We deliberately over-extract here — Stage 2 will filter.
        """
        system_prompt = f"""You are a prediction extraction engine. Your job is to find EVERY statement
in the article that makes a claim about a future state of the world or looks like a prediction.

Cast a WIDE net. When in doubt, include it. Stage 2 will filter.

Capture statements with language like:
- will, would, shall, is expected to, is projected to, is forecast to
- could, may, might (when tied to a specific outcome)
- is set to, is poised to, is on track to, is likely to
- analysts/experts/officials predict/forecast/expect
- at risk of, faces the prospect of, threatens to

{few_shot_examples}

Return ONLY valid JSON:
{{"predictions": [{{"prediction": "exact or near-exact text from article", "signal_phrase": "the trigger word/phrase (e.g. is expected to)", "has_named_entity": true/false, "has_measurable_outcome": true/false}}]}}
"""
        user_prompt = f"Extract all forward-looking claims from this article:\n\n{article_text}"

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-5",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content.strip()
                result = self.safe_json_parse(content, {"predictions": []})
                if not isinstance(result.get("predictions"), list):
                    raise ValueError("Invalid format")
                return result.get("predictions", [])
            except Exception as e:
                print(f"  ⚠ Stage 1 attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return []

    # ── Stage 2: Strict Journalist Filter ─────────────────────────────────────

    def stage2_journalist_filter(self, prediction_text, article_context, has_named_entity, has_measurable_outcome, max_retries=3):
        """
        The single anchoring question: Would a journalist at a major outlet
        specifically assign someone to fact-check THIS statement in future?

        This forces the model to require:
          - A specific named entity (who/what)
          - A measurable outcome (what will happen)
          - Implicit or explicit timeframe (when)

        All three must be present or inferable. Generic intent, rhetorical claims,
        and vague speculation fail this test.
        """
        system_prompt = """You are a strict fact-check editor at a major newspaper.

A junior reporter hands you this statement extracted from a news article.
You must decide: would you ASSIGN someone to specifically fact-check this statement
in future? 

To say YES, the statement must have ALL THREE:
1. A specific subject (named person, company, country, institution — not vague "officials" or "experts")
2. A measurable, verifiable outcome (not just intent, hope, or concern)
3. A timeframe — explicit ("by Q3 2025") or strongly implied by context

Automatic NO if:
- It's a statement of intent or plan, not a predicted outcome ("the company plans to...")
- The subject is unnamed/generic ("analysts say", "some economists think")
- It's purely conditional with no anchor ("if X happens, Y could occur")  
- It's a rhetorical device, question, or narrative framing
- It already happened (past tense reframing)
- It's a fear/concern not tied to a specific forecast

CRITICAL: Respond ONLY with valid JSON:
{"verdict": "YES" | "NO", "reason": "one sentence explanation"}
"""
        user_prompt = f"""Statement: "{prediction_text}"
Article context: "{article_context}"
Has named entity: {has_named_entity}
Has measurable outcome: {has_measurable_outcome}

Would you assign a journalist to fact-check this in 6 months?"""

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content.strip()
                result = self.safe_json_parse(content, {"verdict": "NO", "reason": "parse error"})
                return result
            except Exception as e:
                print(f"  ⚠ Stage 2 attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        return {"verdict": "NO", "reason": f"Failed after {max_retries} attempts"}

    # ── Article Processor ──────────────────────────────────────────────────────

    def process_single_article(self, article_data):
        i, article, publication_date = article_data

        try:
            print(f"📄 Processing article {i}...")

            # Build boundary examples fresh each call (randomized selection)
            few_shot_examples = self.build_boundary_examples(
                getattr(self, 'graded_examples', []), n=12
            )

            # Stage 1: Liberal extraction
            candidates = self.stage1_liberal_extract(article, few_shot_examples)
            print(f"  Stage 1: {len(candidates)} candidates extracted")

            article_results = []
            passed_filter = 0

            for j, cand in enumerate(candidates):
                try:
                    prediction_text = cand.get('prediction', '').strip()
                    if not prediction_text:
                        continue

                    # Get context for this prediction
                    context = self.extract_context_around_prediction(article, prediction_text)

                    # Stage 2: Strict journalist filter
                    filter_result = self.stage2_journalist_filter(
                        prediction_text,
                        context,
                        cand.get('has_named_entity', False),
                        cand.get('has_measurable_outcome', False)
                    )

                    verdict = filter_result.get('verdict', 'NO')
                    reason = filter_result.get('reason', '')

                    if verdict != 'YES':
                        print(f"    ❌ FILTERED [{j+1}]: {reason}")
                        continue

                    passed_filter += 1
                    print(f"    ✅ PASSED [{j+1}]: {prediction_text[:80]}...")

                    article_results.append({
                        'Article_Number': i,
                        'Article_Text': article,
                        'Prediction_Number': passed_filter,
                        'Prediction': prediction_text,
                        'Prediction_Context': context,
                        'Verifiability_Score': 1 if cand.get('has_measurable_outcome') else 0,
                        'Certainty_Score': 1 if cand.get('has_named_entity') else 0,
                        'Signal_Phrase': cand.get('signal_phrase', ''),
                        'Filter_Reason': reason,
                        # ── Placeholder columns to keep eval framework intact ──
                        'Deadline_Estimate': 'PENDING',
                        'Deadline_Reasoning': 'Not estimated in extraction-only run',
                        'Deadline_Confidence': 0,
                        'Grading': 'PENDING',
                        'Grading_Justification': 'Grading not run in this mode',
                        'Claude_Agrees': 'PENDING',
                        'Claude_Additional_Context': 'Not run',
                        'Gemini_Agrees': 'PENDING',
                        'Gemini_Additional_Context': 'Not run'
                    })

                except Exception as e:
                    print(f"  ❌ Error on candidate {j+1}: {e}")
                    continue

            print(f"  Stage 2: {passed_filter}/{len(candidates)} passed filter")

            if not article_results:
                article_results.append({
                    'Article_Number': i,
                    'Article_Text': article,
                    'Prediction_Number': 0,
                    'Prediction': 'No predictions found',
                    'Prediction_Context': 'N/A',
                    'Verifiability_Score': 0,
                    'Certainty_Score': 0,
                    'Signal_Phrase': 'N/A',
                    'Filter_Reason': 'No candidates passed the journalist filter',
                    'Deadline_Estimate': 'N/A',
                    'Deadline_Reasoning': 'No predictions',
                    'Deadline_Confidence': 0,
                    'Grading': 'N/A',
                    'Grading_Justification': 'No predictions found',
                    'Claude_Agrees': 'N/A',
                    'Claude_Additional_Context': 'N/A',
                    'Gemini_Agrees': 'N/A',
                    'Gemini_Additional_Context': 'N/A'
                })

            return article_results

        except Exception as e:
            print(f"❌ Critical error on article {i}: {e}")
            traceback.print_exc()
            return [{
                'Article_Number': i,
                'Article_Text': f"Error: {str(e)}",
                'Prediction_Number': -1,
                'Prediction': 'Processing failed',
                'Prediction_Context': 'Error',
                'Verifiability_Score': 0,
                'Certainty_Score': 0,
                'Signal_Phrase': 'Error',
                'Filter_Reason': str(e),
                'Deadline_Estimate': 'Error',
                'Deadline_Reasoning': str(e),
                'Deadline_Confidence': 0,
                'Grading': 'Error',
                'Grading_Justification': 'Article processing failed',
                'Claude_Agrees': 'Error',
                'Claude_Additional_Context': 'Error',
                'Gemini_Agrees': 'Error',
                'Gemini_Additional_Context': 'Error'
            }]

    # ── Main Pipeline ──────────────────────────────────────────────────────────

    def process_articles_from_sheet(self, input_file_path, output_file_path, publication_date="2025-11-05"):
        try:
            df = pd.read_excel(input_file_path)
            articles = df.iloc[:100, 4].dropna().tolist()
            print(f"📚 Found {len(articles)} articles to process")

            article_data = [(i, article, publication_date) for i, article in enumerate(articles, 1)]
            output_data = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_article = {
                    executor.submit(self.process_single_article, data): data[0]
                    for data in article_data
                }

                for future in concurrent.futures.as_completed(future_to_article):
                    article_num = future_to_article[future]
                    try:
                        results = future.result(timeout=120)
                        output_data.extend(results)
                        self.processed_count += 1

                        if self.processed_count % self.backup_interval == 0:
                            self.save_checkpoint(output_data, output_file_path)

                    except concurrent.futures.TimeoutError:
                        print(f"⏰ Timeout for article {article_num}")
                        output_data.append({
                            'Article_Number': article_num,
                            'Article_Text': 'Timeout',
                            'Prediction_Number': -1,
                            'Prediction': 'Timeout error',
                            'Prediction_Context': 'Timeout',
                            'Verifiability_Score': 0,
                            'Certainty_Score': 0,
                            'Signal_Phrase': 'Timeout',
                            'Filter_Reason': 'Timeout',
                            'Deadline_Estimate': 'Error',
                            'Deadline_Reasoning': 'Timeout',
                            'Deadline_Confidence': 0,
                            'Grading': 'Error',
                            'Grading_Justification': 'Timeout',
                            'Claude_Agrees': 'Error',
                            'Claude_Additional_Context': 'Timeout',
                            'Gemini_Agrees': 'Error',
                            'Gemini_Additional_Context': 'Timeout'
                        })
                    except Exception as e:
                        print(f"❌ Future error for article {article_num}: {e}")

            self.save_checkpoint(output_data, output_file_path)
            print(f"🎉 Done! {len(output_data)} records saved")
            return pd.DataFrame(output_data)

        except Exception as e:
            print(f"💥 Critical pipeline error: {e}")
            traceback.print_exc()
            return None

    # ── Example Loader ─────────────────────────────────────────────────────────

    def load_graded_examples_full_context(self, graded_file_path, num_examples=455):
        """
        Load examples. Prioritize articles that have BOTH validated AND rejected
        predictions — these are the boundary cases that actually teach the model.
        """
        try:
            graded_df = pd.read_excel(graded_file_path)
            print(f"Columns: {graded_df.columns.tolist()}")
            print(f"Total rows: {len(graded_df)}, Validated: {len(graded_df[graded_df.iloc[:, 4] == 1])}")

            validated = graded_df[graded_df.iloc[:, 4] == 1].copy()
            rejected_all = graded_df[graded_df.iloc[:, 4] == 0].copy()

            article_groups = validated.groupby('Article_Number')
            examples = []

            for article_num, group in list(article_groups)[:num_examples]:
                article_text = group.iloc[0]['Article_Text']
                validated_predictions = group['Prediction'].tolist()
                article_rejected = rejected_all[
                    rejected_all['Article_Number'] == article_num
                ]['Prediction'].tolist()

                examples.append({
                    'article_text': article_text,
                    'validated_predictions': validated_predictions,
                    'rejected_predictions': article_rejected[:3],  # cap at 3 per article
                    'count': len(validated_predictions)
                })

            print(f"  Boundary examples (have both valid+rejected): "
                  f"{sum(1 for e in examples if e.get('rejected_predictions'))}")

            return examples, rejected_all

        except Exception as e:
            print(f"Error loading graded examples: {e}")
            return [], pd.DataFrame()


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    processor = PredictionProcessor()

    # ── Configure these paths ──
    input_file       = "news_articlesNov5.xlsx"
    graded_file      = "Grading_pred_anls_enhanced_with_multinov5.xlsx"
    output_file      = "prediction_anls_nov5_two_stage_extraction.xlsx"
    publication_date = "2025-11-05"

    try:
        print("📖 Loading graded examples...")
        graded_examples, rejected_examples = processor.load_graded_examples_full_context(
            graded_file, num_examples=50  # Load 50, but only 12 boundary ones used per call
        )
        processor.graded_examples = graded_examples
        processor.rejected_examples = rejected_examples
        print(f"✅ Loaded {len(graded_examples)} examples "
              f"({sum(1 for e in graded_examples if e.get('rejected_predictions'))} boundary examples)")

        print("🚀 Starting two-stage extraction...")
        result_df = processor.process_articles_from_sheet(input_file, output_file, publication_date)

        if result_df is not None:
            processor.save_checkpoint(result_df.to_dict('records'), output_file)

            print("\n📈 EXTRACTION SUMMARY 📈")
            print(f"Total articles processed : {result_df['Article_Number'].nunique()}")

            valid = result_df[result_df['Prediction_Number'] > 0]
            print(f"Total predictions kept   : {len(valid)}")

            articles_with_preds = valid['Article_Number'].nunique()
            articles_total = result_df['Article_Number'].nunique()
            print(f"Articles with ≥1 pred    : {articles_with_preds}/{articles_total}")

            if len(valid) > 0:
                avg_per_article = len(valid) / articles_with_preds
                print(f"Avg predictions/article  : {avg_per_article:.1f}")

            # Stage filter stats
            print(f"\nStage 1 (liberal) → Stage 2 (journalist filter) funnel logged per article above.")
            print("Check 'Filter_Reason' column in output for rejection explanations.")

        print("\n✅ Extraction complete!")

    except Exception as e:
        print(f"💥 Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()