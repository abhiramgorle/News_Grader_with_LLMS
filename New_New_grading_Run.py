import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from datetime import date
import time
import traceback
from pathlib import Path
import concurrent.futures
from threading import Lock
import re

load_dotenv()

keymain = os.getenv("navigator_api")
# claude_key = os.getenv("ANTHROPIC_API_KEY")  # Add this to your .env file
client = OpenAI(api_key=keymain, base_url="https://api.ai.it.ufl.edu")
# claude_client = anthropic.Anthropic(api_key=claude_key)

# Global lock for thread-safe file operations
file_lock = Lock()

class PredictionProcessor:
    def __init__(self, checkpoint_file="processing_checkpoint.xlsx", backup_interval=5):
        self.checkpoint_file = checkpoint_file
        self.backup_interval = backup_interval
        self.processed_count = 0
        
    def safe_json_parse(self, text, default_value):
        """Safely parse JSON with fallback handling"""
        try:
            # Clean the text first
            text = text.strip()
            if not text:
                return default_value            
            # Remove control characters that can break JSON parsing
            text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
            
            # Try to find JSON in the text if it's embedded in other content
            import re
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            if match:
                text = match.group()
            
            # Additional cleaning for common JSON issues
            text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            # Fix multiple spaces
            text = re.sub(r'\s+', ' ', text)
                
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw text: {text[:200]}...")
            # Try to extract just the key-value pairs we need
            try:
                # For Claude responses
                if 'claude_agrees' in text.lower():
                    agrees_match = re.search(r'"claude_agrees":\s*"([^"]+)"', text, re.IGNORECASE)
                    context_match = re.search(r'"claude_additional_context":\s*"([^"]+)"', text, re.IGNORECASE)
                    if agrees_match:
                        return {
                            "claude_agrees": agrees_match.group(1),
                            "claude_additional_context": context_match.group(1) if context_match else "Parsing error"
                        }
                # For Gemini responses
                elif 'gemini_agrees' in text.lower():
                    agrees_match = re.search(r'"gemini_agrees":\s*"([^"]+)"', text, re.IGNORECASE)
                    context_match = re.search(r'"gemini_additional_context":\s*"([^"]+)"', text, re.IGNORECASE)
                    if agrees_match:
                        return {
                            "gemini_agrees": agrees_match.group(1),
                            "gemini_additional_context": context_match.group(1) if context_match else "Parsing error"
                        }
                # For grading responses
                elif 'grading' in text.lower():
                    grading_match = re.search(r'"grading":\s*"([^"]+)"', text, re.IGNORECASE)
                    justification_match = re.search(r'"grading_justification":\s*"([^"]+)"', text, re.IGNORECASE)
                    if grading_match:
                        return {
                            "grading": grading_match.group(1),
                            "grading_justification": justification_match.group(1) if justification_match else "Parsing error"
                        }
            except:
                pass
            return default_value
        except Exception as e:
            print(f"Unexpected error parsing JSON: {e}")
            return default_value

    def extract_context_around_prediction(self, article_text, prediction_text, context_lines=2):
        """Extract context around a prediction for better verification"""
        try:
            # Split article into sentences
            sentences = re.split(r'[.!?]+', article_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Find the sentence containing the prediction
            prediction_sentence_idx = -1
            for i, sentence in enumerate(sentences):
                if prediction_text.lower() in sentence.lower():
                    prediction_sentence_idx = i
                    break
            
            if prediction_sentence_idx == -1:
                # If exact match not found, try partial matching
                for i, sentence in enumerate(sentences):
                    # Check if significant portion of prediction is in sentence
                    pred_words = set(prediction_text.lower().split())
                    sent_words = set(sentence.lower().split())
                    overlap = len(pred_words.intersection(sent_words))
                    if overlap >= len(pred_words) * 0.6:  # 60% overlap threshold
                        prediction_sentence_idx = i
                        break
            
            if prediction_sentence_idx == -1:
                return prediction_text  # Return original if no context found
            
            # Extract context
            start_idx = max(0, prediction_sentence_idx - context_lines)
            end_idx = min(len(sentences), prediction_sentence_idx + context_lines + 1)
            
            context_sentences = sentences[start_idx:end_idx]
            context = '. '.join(context_sentences) + '.'
            
            return context
            
        except Exception as e:
            print(f"Error extracting context: {e}")
            return prediction_text

    def clean_text_for_excel(self, text):
        """Clean text to make it safe for Excel"""
        if not isinstance(text, str):
            return text
        
        # Remove or replace characters that Excel doesn't like
        # Excel has issues with certain Unicode control characters
        cleaned = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
        
        # Replace problematic quotes that might cause Excel issues
        cleaned = cleaned.replace('"', '""')  # Escape quotes for Excel
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        
        # Truncate if too long (Excel has cell limits)
        if len(cleaned) > 32767:  # Excel's character limit per cell
            cleaned = cleaned[:32760] + "..."
            
        return cleaned

    def save_checkpoint(self, output_data, output_file_path):
        """Save current progress to prevent data loss with enhanced error handling"""
        try:
            with file_lock:
                # Clean all text data before creating DataFrame
                cleaned_data = []
                for record in output_data:
                    cleaned_record = {}
                    for key, value in record.items():
                        if isinstance(value, str):
                            cleaned_record[key] = self.clean_text_for_excel(value)
                        else:
                            cleaned_record[key] = value
                    cleaned_data.append(cleaned_record)
                
                checkpoint_df = pd.DataFrame(cleaned_data)
                
                # Save main file with error handling
                try:
                    with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                        checkpoint_df.to_excel(writer, sheet_name='Predictions_Analysis', index=False)
                        self.format_excel_columns(writer.sheets['Predictions_Analysis'])
                except Exception as excel_error:
                    print(f"⚠ Excel save failed, trying CSV backup: {excel_error}")
                    # Fallback to CSV if Excel fails
                    csv_file = output_file_path.replace('.xlsx', '_backup.csv')
                    checkpoint_df.to_csv(csv_file, index=False, encoding='utf-8')
                    print(f"✅ Saved as CSV backup: {csv_file}")
                
                # Save backup checkpoint
                try:
                    backup_file = f"backup_{self.checkpoint_file}"
                    checkpoint_df.to_excel(backup_file, index=False)
                except Exception as backup_error:
                    print(f"⚠ Backup save failed: {backup_error}")
                    # Try CSV backup
                    csv_backup = backup_file.replace('.xlsx', '.csv')
                    checkpoint_df.to_csv(csv_backup, index=False, encoding='utf-8')
                
                print(f"✅ Checkpoint saved: {len(output_data)} records")
                
        except Exception as e:
            print(f"⚠ Error saving checkpoint: {e}")
            # Last resort: save as pickle for debugging
            try:
                import pickle
                pickle_file = output_file_path.replace('.xlsx', '_emergency.pkl')
                with open(pickle_file, 'wb') as f:
                    pickle.dump(output_data, f)
                print(f"🆘 Emergency pickle save: {pickle_file}")
            except:
                print("💥 Complete save failure - data may be lost")

    def format_excel_columns(self, worksheet):
        """Format Excel columns for better readability"""
        column_widths = {
            'A': 15, 'B': 60, 'C': 15, 'D': 60, 'E': 15, 'F': 15, 
            'G': 15, 'H': 40, 'I': 15, 'J': 50, 'K': 20, 'L': 60,
            'M': 20, 'N': 60, 'O': 20, 'P': 60  # Added for new columns
        }
        for col, width in column_widths.items():
            worksheet.column_dimensions[col].width = width

    def extract_predictions_with_retry(self, article_text, max_retries=3):
        """Extract predictions with retry logic and better error handling"""
        system_prompt = """
        You are a News Prediction Extractor, an expert AI designed to analyze news articles and extract forward-looking statements.

        🔍 DETECTION SCOPE:
        Identify statements that meet any of these criteria:
        - **Explicit Predictions**: "is expected to," "will," "is projected to," "analysts forecast"
        - **Implicit Predictions**: "could lead to," "may result in," "poses a risk of," "signals a trend"
        - **Quoted Forecasts**: Predictions cited from experts, reports, or institutions
        - **Future-Tense Declarations**: "will be banned," "will receive," "will increase"
        - **Contingent Speculation**: "if rates continue to rise, housing demand could plummet"

        ⛔ Do not extract:
        - Commentary on past events unless explicitly tied to future consequences
        - Generic opinions or vague guesses
        - Statements lacking temporal direction
        - Hypothetical scenarios without a clear predictive element
        - facts or historical data
        - Rhetorical questions or speculative musings without a clear prediction
        - fears are not predictions unless tied to a specific forecast

        For each prediction, provide:
        1. The exact prediction statement (clear and standalone)
        2. Verifiability Score (1-5): How measurable the claim is (5 = highly specific and measurable, 1 = vague)
        3. Certainty Score (1-5): How confident the speaker appears (5 = very confident/definitive, 1 = tentative)

        CRITICAL: Respond ONLY with valid JSON in this exact format:
        {"predictions": [{"prediction": "exact text here", "verifiability_score": 3, "certainty_score": 5}]}
        If no predictions are found, return {"predictions": []}
        """
        
        user_prompt = f"""Extract all predictions(No restriction on number) from the following article:

        {article_text}""" 
        
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
                
                # Validate the result structure
                if not isinstance(result.get("predictions"), list):
                    raise ValueError("Invalid predictions format")
                    
                return result.get("predictions", [])
                
            except Exception as e:
                print(f"⚠ Attempt {attempt + 1} failed for prediction extraction: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ All attempts failed for prediction extraction")
                    return []

    def estimate_deadline_with_retry(self, prediction_text, article_date, max_retries=3):
        """Estimate deadline with retry logic"""
        system_prompt = """
        You are a Prediction Analyzer. Your task is to estimate a verification deadline for a given prediction.

        DEADLINE ESTIMATION:
        - Identify explicit timeframes in the prediction ("by 2025", "next year", "in five years")
        - Use common-sense reasoning for implicit timelines (tech releases: 1-2 years, political terms: 4 years)
        - Your goal is to determine the first reasonable date to check if the prediction came true

        CRITICAL: Respond ONLY with valid JSON:
        {
            "deadline_estimate": "YYYY-MM-DD",
            "deadline_reasoning": "Brief explanation of how you determined this deadline"
        }
        """
        
        user_prompt = f"""Estimate the verification deadline for this prediction.
        
        Article Publication Date: {article_date}
        Prediction: "{prediction_text}"
        """
        
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
                result = self.safe_json_parse(content, {
                    "deadline_estimate": "Unknown",
                    "deadline_reasoning": "Error in processing"
                })
                
                return result
                
            except Exception as e:
                print(f"⚠ Attempt {attempt + 1} failed for deadline estimation: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "deadline_estimate": "Error",
                        "deadline_reasoning": f"Failed after {max_retries} attempts: {str(e)}"
                    }

    def grade_prediction_with_retry(self, prediction_text, prediction_context, max_retries=3):
        """Grade prediction with retry logic"""
        system_prompt = """
        You are a fact-checker. Given a prediction whose deadline has passed, use your own knowledge and reasoning to grade its accuracy.

        - "TRUE": The prediction clearly came to pass
        - "FALSE": The prediction clearly did not come to pass  
        - "PARTIALLY_TRUE": The outcome was mixed or partially fulfilled

        CRITICAL: Respond ONLY with valid JSON:
        {
            "grading": "TRUE" | "FALSE" | "PARTIALLY_TRUE",
            "grading_justification": "few-line summary of what happened, based on your knowledge"
        }
        """
        
        user_prompt = f'Grade this prediction using your own knowledge.\n\nPrediction: "{prediction_text}" \n\nContext: "{prediction_context}"'
        
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
                result = self.safe_json_parse(content, {
                    "grading": "Error",
                    "grading_justification": "Error in processing"
                })
                
                return result
                
            except Exception as e:
                print(f"⚠ Attempt {attempt + 1} failed for grading: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "grading": "Error", 
                        "grading_justification": f"Failed after {max_retries} attempts: {str(e)}"
                    }

    def claude_verification_with_retry(self,prediction_text,prediction_context, gpt_grading, gpt_justification, max_retries=3):
        """Use Claude to verify GPT's grading with retry logic and context"""
        system_prompt = """
        You are a secondary fact-checker reviewing another AI's prediction grading. Your task is to:
        1. Assess whether you agree with the given grading (YES/NO/PARTIALLY)
        2. Provide additional context, evidence, or corrections if you have more knowledge in that prediction area
        3. Be objective and thorough in your analysis

        IMPORTANT: Keep responses concise and avoid special characters, quotes, or control characters.

        Respond with valid JSON only:
        {
            "claude_agrees": "YES" | "NO" | "PARTIALLY",
            "claude_additional_context": "Additional information, evidence, or corrections you can provide"
        }
        """
        
        user_prompt = f"""
        Please review this prediction grading with the provided context:

        prediction: "{prediction_text}"
        Context: "{prediction_context}"
        GPT Grading: {gpt_grading}
        GPT Justification: {gpt_justification}

        Do you agree with this grading? What additional context can you provide?
        """
        
        for attempt in range(max_retries):
            try:
                # Fixed: Using the correct model name and API structure
                response = client.chat.completions.create(
                    model="claude-4-sonnet", 
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                result = self.safe_json_parse(content, {
                    "claude_agrees": "Error",
                    "claude_additional_context": "Error in processing"
                })
                
                return result
                
            except Exception as e:
                print(f"⚠ Claude attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "claude_agrees": "Error",
                        "claude_additional_context": f"Failed after {max_retries} attempts: {str(e)}"
                    }

    def gemini_verification_with_retry(self,prediction_text, prediction_context, gpt_grading, gpt_justification, max_retries=3):
        """Use Gemini to verify GPT's grading with retry logic and context"""
        system_prompt = """
        You are a third fact-checker reviewing another AI's prediction grading. Your task is to:
        1. Assess whether you agree with the given grading (YES/NO/PARTIALLY)
        2. Provide additional context, evidence, or corrections if you have more knowledge in that prediction area
        3. Be objective and thorough in your analysis
        4. Compare your assessment with the provided context

        IMPORTANT: Keep responses concise and avoid special characters, quotes, or control characters.

        Respond with valid JSON only:
        {
            "gemini_agrees": "YES" | "NO" | "PARTIALLY",
            "gemini_additional_context": "Additional information, evidence, or corrections you can provide"
        }
        """
        
        user_prompt = f"""
        Please review this prediction grading with the provided context:
        prediction: "{prediction_text}"
        Context: "{prediction_context}"
        GPT Grading: {gpt_grading}
        GPT Justification: {gpt_justification}

        Do you agree with this grading? What additional context can you provide?
        """
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gemini-2.5-pro",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                result = self.safe_json_parse(content, {
                    "gemini_agrees": "Error",
                    "gemini_additional_context": "Error in processing"
                })
                
                return result
                
            except Exception as e:
                print(f"⚠ Gemini attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "gemini_agrees": "Error",
                        "gemini_additional_context": f"Failed after {max_retries} attempts: {str(e)}"
                    }

    def process_single_article(self, article_data):
        """Process a single article with comprehensive error handling"""
        i, article, publication_date = article_data
        
        try:
            print(f"📄 Processing article {i}...")
            
            # Extract predictions with retry
            predictions = self.extract_predictions_with_retry(article)
            print(f"  ✅ Found {len(predictions)} predictions")
            
            article_results = []
            
            if predictions:
                for j, pred in enumerate(predictions):
                    try:
                        prediction_text = pred.get('prediction', '')
                        if not prediction_text:
                            continue
                            
                        print(f"    🔍 Analyzing prediction {j+1}...")
                        
                        # Extract context around prediction
                        prediction_context = self.extract_context_around_prediction(
                            article, prediction_text
                        )
                        
                        # Estimate deadline
                        deadline_info = self.estimate_deadline_with_retry(
                            prediction_text, publication_date
                        )
                        
                        result = {
                            'Article_Number': i,
                            'Article_Text': article,
                            'Prediction_Number': j + 1,
                            'Prediction': prediction_text,
                            'Prediction_Context': prediction_context,  # New field
                            'Verifiability_Score': pred.get('verifiability_score', 0),
                            'Certainty_Score': pred.get('certainty_score', 0),
                            'Deadline_Estimate': deadline_info.get('deadline_estimate', 'Unknown'),
                            'Deadline_Reasoning': deadline_info.get('deadline_reasoning', 'Not provided'),
                            'Grading': 'Pending',
                            'Grading_Justification': 'Deadline not yet reached',
                            'Claude_Agrees': 'Pending',
                            'Claude_Additional_Context': 'Not yet reviewed',
                            'Gemini_Agrees': 'Pending',  # New field
                            'Gemini_Additional_Context': 'Not yet reviewed'  # New field
                        }
                        
                        article_results.append(result)
                        
                    except Exception as e:
                        print(f"    ❌ Error processing prediction {j+1}: {e}")
                        # Continue with next prediction instead of failing entirely
                        continue
            else:
                # No predictions found
                article_results.append({
                    'Article_Number': i,
                    'Article_Text': article,
                    'Prediction_Number': 0,
                    'Prediction': 'No predictions found',
                    'Prediction_Context': 'N/A',
                    'Verifiability_Score': 0,
                    'Certainty_Score': 0,
                    'Deadline_Estimate': 'N/A',
                    'Deadline_Reasoning': 'No predictions to evaluate',
                    'Grading': 'N/A',
                    'Grading_Justification': 'No predictions found',
                    'Claude_Agrees': 'N/A',
                    'Claude_Additional_Context': 'No predictions to review',
                    'Gemini_Agrees': 'N/A',
                    'Gemini_Additional_Context': 'No predictions to review'
                })
            
            return article_results
            
        except Exception as e:
            print(f"❌ Critical error processing article {i}: {e}")
            traceback.print_exc()
            
            # Return error record instead of failing completely
            return [{
                'Article_Number': i,
                'Article_Text': f"Error processing article: {str(e)}",
                'Prediction_Number': -1,
                'Prediction': 'Processing failed',
                'Prediction_Context': 'Error in processing',
                'Verifiability_Score': 0,
                'Certainty_Score': 0,
                'Deadline_Estimate': 'Error',
                'Deadline_Reasoning': f'Processing error: {str(e)}',
                'Grading': 'Error',
                'Grading_Justification': 'Article processing failed',
                'Claude_Agrees': 'Error',
                'Claude_Additional_Context': 'Could not process article',
                'Gemini_Agrees': 'Error',
                'Gemini_Additional_Context': 'Could not process article'
            }]

    def process_articles_from_sheet(self, input_file_path, output_file_path, publication_date="2025-11-05"):
        """Enhanced processing with comprehensive failsafes and parallel processing"""
        try:
            # Read input file
            df = pd.read_excel(input_file_path)
            articles = df.iloc[:92:, 4].dropna().tolist()
            print(f"📚 Found {len(articles)} articles to process")
            
            # Prepare for parallel processing
            article_data = [(i, article, publication_date) for i, article in enumerate(articles, 1)]
            output_data = []
            
            # Process articles with controlled parallelism
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_article = {
                    executor.submit(self.process_single_article, data): data[0] 
                    for data in article_data
                }
                
                for future in concurrent.futures.as_completed(future_to_article):
                    article_num = future_to_article[future]
                    try:
                        results = future.result(timeout=60)  # 1 minute timeout per article
                        output_data.extend(results)
                        
                        self.processed_count += 1
                        
                        # Save checkpoint every N articles
                        if self.processed_count % self.backup_interval == 0:
                            self.save_checkpoint(output_data, output_file_path)
                            
                    except concurrent.futures.TimeoutError:
                        print(f"⏰ Timeout for article {article_num}")
                        output_data.append({
                            'Article_Number': article_num,
                            'Article_Text': 'Timeout during processing',
                            'Prediction_Number': -1,
                            'Prediction': 'Timeout error',
                            'Prediction_Context': 'Timeout during processing',
                            'Verifiability_Score': 0,
                            'Certainty_Score': 0,
                            'Deadline_Estimate': 'Error',
                            'Deadline_Reasoning': 'Processing timeout',
                            'Grading': 'Error',
                            'Grading_Justification': 'Processing timeout',
                            'Claude_Agrees': 'Error',
                            'Claude_Additional_Context': 'Processing timeout',
                            'Gemini_Agrees': 'Error',
                            'Gemini_Additional_Context': 'Processing timeout'
                        })
                    except Exception as e:
                        print(f"❌ Future error for article {article_num}: {e}")
            
            # Final save
            self.save_checkpoint(output_data, output_file_path)
            
            print(f"🎉 Processing complete! {len(output_data)} records saved")
            return pd.DataFrame(output_data)
            
        except Exception as e:
            print(f"💥 Critical error in main processing: {e}")
            traceback.print_exc()
            return None

    def grade_past_due_predictions(self, df):
        """Enhanced grading with Claude and Gemini verification"""
        try:
            current_date = date.today()
            past_due_indices = []
            
            for idx, row in df.iterrows():
                if (row['Prediction_Number'] > 0 and 
                    row['Deadline_Estimate'] not in ['N/A', 'Error', 'Unknown']):
                    try:
                        deadline = date.fromisoformat(row['Deadline_Estimate'])
                        if deadline <= current_date:
                            past_due_indices.append(idx)
                    except:
                        continue
            
            print(f"🎯 Found {len(past_due_indices)} predictions past their deadline")
            
            for idx in past_due_indices:
                try:
                    prediction_text = df.at[idx, 'Prediction']
                    prediction_context = df.at[idx, 'Prediction_Context']
                    print(f"  📊 Grading prediction {idx}...")
                    
                    # Grade with GPT
                    gpt_result = self.grade_prediction_with_retry(prediction_text,prediction_context)
                    df.at[idx, 'Grading'] = gpt_result.get('grading', 'Error')
                    df.at[idx, 'Grading_Justification'] = gpt_result.get('grading_justification', 'No justification')
                    
                    #Verify with Claude (with context)
                    if gpt_result.get('grading') != 'Error':
                        print(f"    🤖 Getting Claude verification...")
                        claude_result = self.claude_verification_with_retry(
                            prediction_text,
                            prediction_context,  # Using context instead of just prediction
                            gpt_result.get('grading'),
                            gpt_result.get('grading_justification')
                        )
                        df.at[idx, 'Claude_Agrees'] = claude_result.get('claude_agrees', 'Error')
                        df.at[idx, 'Claude_Additional_Context'] = claude_result.get('claude_additional_context', 'No additional context')
                        
                        # Verify with Gemini (with context)
                        print(f"    🔮 Getting Gemini verification...")
                        gemini_result = self.gemini_verification_with_retry(
                            prediction_text,
                            prediction_context,  # Using context instead of just prediction
                            gpt_result.get('grading'),
                            gpt_result.get('grading_justification')
                        )
                        df.at[idx, 'Gemini_Agrees'] = gemini_result.get('gemini_agrees', 'Error')
                        df.at[idx, 'Gemini_Additional_Context'] = gemini_result.get('gemini_additional_context', 'No additional context')
                    
                    # Save progress periodically
                    if idx % 5 == 0:
                        self.save_checkpoint(df.to_dict('records'), "temp_grading_checkpoint.xlsx")
                        
                except Exception as e:
                    print(f"❌ Error grading prediction at index {idx}: {e}")
                    df.at[idx, 'Grading'] = 'Error'
                    df.at[idx, 'Grading_Justification'] = f'Grading error: {str(e)}'
                    df.at[idx, 'Claude_Agrees'] = 'Error'
                    df.at[idx, 'Claude_Additional_Context'] = f'Verification error: {str(e)}'
                    df.at[idx, 'Gemini_Agrees'] = 'Error'
                    df.at[idx, 'Gemini_Additional_Context'] = f'Verification error: {str(e)}'
                    continue
            
            return df
            
        except Exception as e:
            print(f"💥 Critical error in grading process: {e}")
            traceback.print_exc()
            return df


def main():
    processor = PredictionProcessor()
    
    # File paths
    input_file = "news_articlesNov5.xlsx"
    output_file = "pred_anls_enhanced_with_multinov5.xlsx"
    publication_date = "2025-11-05"
    
    try:
        # Process articles
        print("🚀 Starting enhanced processing with context and multi-model verification...")
        result_df = processor.process_articles_from_sheet(input_file, output_file, publication_date)
        
        if result_df is not None:
            # Grade past due predictions
            print("🎯 Starting grading process...")
            result_df = processor.grade_past_due_predictions(result_df)
            
            # Final save
            processor.save_checkpoint(result_df.to_dict('records'), output_file)
            
            # Summary statistics
            print("\n📈 ENHANCED SUMMARY 📈")
            print(f"Total articles processed: {result_df['Article_Number'].nunique()}")
            total_predictions = len(result_df[result_df['Prediction_Number'] > 0])
            print(f"Total predictions found: {total_predictions}")
            
            if total_predictions > 0:
                valid_scores = result_df[result_df['Verifiability_Score'] > 0]
                if not valid_scores.empty:
                    print(f"Average verifiability score: {valid_scores['Verifiability_Score'].mean():.2f}")
                    print(f"Average certainty score: {valid_scores['Certainty_Score'].mean():.2f}")
                
                # Grading statistics
                graded = result_df[result_df['Grading'].isin(['TRUE', 'FALSE', 'PARTIALLY_TRUE'])]
                if not graded.empty:
                    print(f"Predictions graded: {len(graded)}")
                    print(f"Claude agreements: {len(graded[graded['Claude_Agrees'] == 'YES'])}")
                    print(f"Claude disagreements: {len(graded[graded['Claude_Agrees'] == 'NO'])}")
                    print(f"Gemini agreements: {len(graded[graded['Gemini_Agrees'] == 'YES'])}")
                    print(f"Gemini disagreements: {len(graded[graded['Gemini_Agrees'] == 'NO'])}")
                    
                    # Model agreement analysis
                    both_agree = len(graded[(graded['Claude_Agrees'] == 'YES') & (graded['Gemini_Agrees'] == 'YES')])
                    both_disagree = len(graded[(graded['Claude_Agrees'] == 'NO') & (graded['Gemini_Agrees'] == 'NO')])
                    print(f"Both models agree with GPT: {both_agree}")
                    print(f"Both models disagree with GPT: {both_disagree}")
        
        print("✅ Process completed successfully!")
        
    except Exception as e:
        print(f"💥 Fatal error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()