
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
from datetime import date
load_dotenv()

keymain = os.getenv("navigator_api")
client = OpenAI(api_key=keymain, base_url="https://api.ai.it.ufl.edu")


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import re


def extract_predictions_from_article(article_text):
    """
    Extract forward-looking predictions from an article with scoring.
    Uses JSON mode for more reliable parsing.
    """
    # Compress article if too large
    # if len(article_text) > 12000:
    #     print("Large article detected, compressing...")
    #     article_text = compress_article_with_langchain(article_text, max_tokens=3000)
    
    system_prompt = """
    You are a News Prediction Extractor, an expert AI designed to analyze news articles and extract forward-looking statements. Your role is to identify and clearly state these predictive statements without rephrasing them.

    🔍 DETECTION SCOPE:
    Identify statements that meet any of these criteria:
    - **Explicit Predictions**: "is expected to," "will," "is projected to," "analysts forecast"
    - **Implicit Predictions**: "could lead to," "may result in," "poses a risk of," "signals a trend"
    - **Quoted Forecasts**: Predictions cited from experts, reports, or institutions
    - **Future-Tense Declarations**: "will be banned," "will receive," "will increase"
    - **Contingent Speculation**: "if rates continue to rise, housing demand could plummet"

    ❌ Do not extract:
    - Commentary on past events unless explicitly tied to future consequences
    - Generic opinions or vague guesses
    - Statements lacking temporal direction

    For each prediction, provide:
    1. The exact prediction statement (clear and standalone)
    2. Verifiability Score (1-5): How measurable the claim is (5 = highly specific and measurable, 1 = vague)
    3. Certainty Score (1-5): How confident the speaker appears (5 = very confident/definitive, 1 = tentative)

    Respond with a JSON object in this exact format:
    {"predictions": [{"prediction": "exact text here", "verifiability_score": 3, "certainty_score": 5}]}
    If no predictions are found, return {"predictions": []}
    """
    
    user_prompt = f"""
    Extract all forward-looking predictions from the following article. Focus on substantial predictions with significant consequences.

    Article:
    {article_text}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}  # Forces JSON output
        )
        
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        return result.get("predictions", [])
        
    except Exception as e:
        print(f"Error extracting predictions: {e}")
        return []

def estimate_prediction_deadline(prediction_text, article_date="2025-09-15"):
    """
    Estimate when a prediction should be checked for accuracy.
    This function only needs the prediction text, not the full article.
    """
    system_prompt = """
    You are a Prediction Analyzer. Your task is to estimate a verification deadline for a given prediction.

    DEADLINE ESTIMATION:
    - Identify explicit timeframes in the prediction ("by 2025", "next year", "in five years")
    - Use common-sense reasoning for implicit timelines (tech releases: 1-2 years, political terms: 4 years)
    - Your goal is to determine the first reasonable date to check if the prediction came true

    Respond with only a JSON object:
    {
        "deadline_estimate": "YYYY-MM-DD",
        "deadline_reasoning": "Brief explanation of how you determined this deadline"
    }
    """
    
    user_prompt = f"""
    Estimate the verification deadline for this prediction.
    
    Article Publication Date: {article_date}
    Prediction: "{prediction_text}"
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content.strip()
        return json.loads(content)
        
    except Exception as e:
        print(f"Error estimating deadline: {e}")
        return {
            "deadline_estimate": "Error",
            "deadline_reasoning": f"API Error: {e}"
        }

def grade_prediction_with_gpt_knowledge(prediction_text):
    """
    Grade a prediction using GPT-5's own knowledge and reasoning.
    This should only be called AFTER the prediction's deadline has passed.
    """
    system_prompt = """
    You are a fact-checker. Given a prediction whose deadline has passed, use your own knowledge and reasoning to grade its accuracy.

    - "TRUE": The prediction clearly came to pass
    - "FALSE": The prediction clearly did not come to pass  
    - "PARTIALLY_TRUE": The outcome was mixed or partially fulfilled

    Respond with only a JSON object:
    {
        "grading": "TRUE" | "FALSE" | "PARTIALLY_TRUE",
        "grading_justification": "One-line summary of what happened, based on your knowledge"
    }
    """
    
    user_prompt = f"""
    Grade this prediction using your own knowledge.

    Prediction: "{prediction_text}"
    """
    
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
        return json.loads(content)
        
    except Exception as e:
        print(f"Error grading prediction: {e}")
        return {"grading": "Error", "grading_justification": f"API Error: {e}"}
def process_articles_from_sheet(input_file_path, output_file_path, publication_date="2025-09-15"):
    """
    Enhanced processing function that extracts predictions AND estimates deadlines.
    Uses the new two-step workflow for cost efficiency.
    """
    try:
        # Read the input Excel file
        df = pd.read_excel(input_file_path)
        
        # Extract articles from column D (rows 82-120, which is index 81-119 in pandas)
        articles = df.iloc[1:100, 3].dropna().tolist()  # Column D is index 3
        
        print(f"Found {len(articles)} articles to process")
        
        # Prepare output data
        output_data = []
        
        for i, article in enumerate(articles, 1):
            print(f"Processing article {i}/{len(articles)}...")
            
            # Step 1: Extract predictions with scores
            predictions = extract_predictions_from_article(article)
            print(f"  Found {len(predictions)} predictions")
            
            if predictions:
                for j, pred in enumerate(predictions):
                    prediction_text = pred.get('prediction', '')
                    print(f"    Analyzing deadline for prediction {j+1}...")
                    
                    # Step 2: Estimate deadline for each prediction
                    deadline_info = estimate_prediction_deadline(prediction_text, publication_date)
                    
                    # Combine all information
                    output_data.append({
                        'Article_Number': i,
                        'Article_Text': article,
                        'Prediction_Number': j + 1,
                        'Prediction': prediction_text,
                        'Verifiability_Score': pred.get('verifiability_score', 0),
                        'Certainty_Score': pred.get('certainty_score', 0),
                        'Deadline_Estimate': deadline_info.get('deadline_estimate', 'Unknown'),
                        'Deadline_Reasoning': deadline_info.get('deadline_reasoning', 'Not provided'),
                        'Grading': 'Pending',  # Will be filled by future grading process
                        'Grading_Justification': 'Deadline not yet reached'
                    })
            else:
                # If no predictions found, still record the article
                output_data.append({
                    'Article_Number': i,
                    'Article_Text': article,
                    'Prediction_Number': 0,
                    'Prediction': 'No predictions found',
                    'Verifiability_Score': 0,
                    'Certainty_Score': 0,
                    'Deadline_Estimate': 'N/A',
                    'Deadline_Reasoning': 'No predictions to evaluate',
                    'Grading': 'N/A',
                    'Grading_Justification': 'No predictions found'
                })
        
        # Create DataFrame and save to Excel
        output_df = pd.DataFrame(output_data)
        
        # Save to Excel with formatting
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='Predictions_Analysis', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Predictions_Analysis']
            
            # Adjust column widths
            worksheet.column_dimensions['A'].width = 15  # Article_Number
            worksheet.column_dimensions['B'].width = 60  # Article_Text
            worksheet.column_dimensions['C'].width = 15  # Prediction_Number
            worksheet.column_dimensions['D'].width = 60  # Prediction
            worksheet.column_dimensions['E'].width = 15  # Verifiability_Score
            worksheet.column_dimensions['F'].width = 15  # Certainty_Score
            worksheet.column_dimensions['G'].width = 15  # Deadline_Estimate
            worksheet.column_dimensions['H'].width = 40  # Deadline_Reasoning
            worksheet.column_dimensions['I'].width = 15  # Grading
            worksheet.column_dimensions['J'].width = 50  # Grading_Justification
        
        print(f"Analysis complete! Results saved to {output_file_path}")
        total_predictions = len([x for x in output_data if x['Prediction_Number'] > 0])
        print(f"Processed {len(articles)} articles and found {total_predictions} predictions")
        
        return output_df
        
    except Exception as e:
        print(f"Error processing articles: {e}")
        return None

def grade_past_due_predictions(df, evidence_source="manual"):
    """
    Separate function to grade predictions whose deadlines have passed.
    In a real system, this would be run as a scheduled job with external research.
    """
    current_date = date.today()
    past_due_predictions = []
    
    for idx, row in df.iterrows():
        if row['Prediction_Number'] > 0 and row['Deadline_Estimate'] != 'N/A':
            try:
                deadline = date.fromisoformat(row['Deadline_Estimate'])
                if deadline <= current_date:
                    past_due_predictions.append(idx)
            except:
                continue  # Skip invalid dates
    
    print(f"Found {len(past_due_predictions)} predictions past their deadline")
    
    # In a real implementation, you would:
    # 1. Use a search API to gather evidence about each prediction
    # 2. Call grade_prediction_with_evidence() with that evidence
    # 3. Update the DataFrame with the grading results
    
    # For demonstration, we'll just mark them as ready for grading
    for idx in past_due_predictions:
        prediction_text = df.at[idx, 'Prediction']
        result = grade_prediction_with_gpt_knowledge(prediction_text)
        df.at[idx, 'Grading'] = result.get('grading', 'Error')
        df.at[idx, 'Grading_Justification'] = result.get('grading_justification', 'No justification')
    
    return df

def main():
    # File paths - modify these according to your setup
    input_file = "Scraped_news_20202.xlsx"  # Replace with your input file path
    output_file = "predictions_analysis_enhanced1.xlsx"  # Output file name
    publication_date = "2020-01-01"  # Adjust based on your data
    
    # Process the articles with the enhanced workflow
    result_df = process_articles_from_sheet(input_file, output_file, publication_date)
    
    if result_df is not None:
        # Check for predictions that are past their deadline
        result_df = grade_past_due_predictions(result_df)
        
        # Save the updated DataFrame
        result_df.to_excel(output_file, sheet_name='Predictions_Analysis', index=False)
        
        # Display summary statistics
        print("\n=== ENHANCED SUMMARY ===")
        print(f"Total articles processed: {result_df['Article_Number'].nunique()}")
        total_predictions = len(result_df[result_df['Prediction_Number'] > 0])
        print(f"Total predictions found: {total_predictions}")
        
        if total_predictions > 0:
            print(f"Average verifiability score: {result_df[result_df['Verifiability_Score'] > 0]['Verifiability_Score'].mean():.2f}")
            print(f"Average certainty score: {result_df[result_df['Certainty_Score'] > 0]['Certainty_Score'].mean():.2f}")
            
            # Count predictions by grading status
            ready_for_grading = len(result_df[result_df['Grading'] == 'Ready for grading'])
            print(f"Predictions ready for grading: {ready_for_grading}")
            print(f"Predictions still pending: {total_predictions - ready_for_grading}")

if __name__ == "__main__":
    main()