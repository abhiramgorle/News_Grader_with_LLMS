# Install required packages:
# pip install langchain tiktoken pandas openpyxl

import pandas as pd
import json
from datetime import datetime
from openai import OpenAI
import time
from dotenv import load_dotenv
import os

load_dotenv()
keymain = os.getenv("navigator_api")
client = OpenAI(api_key=keymain,base_url="https://api.ai.it.ufl.edu")


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import re


def openai_analyze_prediction_deadline_and_grading(article_text, prediction_text):
    """
    Analyze a prediction to estimate deadline and grade if past due date
    """
    system_prompt = """
    You are a Prediction Analyzer. Your task is to evaluate the accuracy of a prediction using the following steps:

1. DEADLINE ESTIMATION (When should we check if the prediction came true?):
    - Identify any explicit timeframes in the prediction text (e.g., "by 2025", "next year", "in five years").
    - Use contextual clues in the article (e.g., election cycles, fiscal years, quarterly earnings reports).
    - Apply common-sense reasoning if the timeline is implicit (e.g., for tech releases, trials, events).
    - If unclear, default to a reasonable window of 1–2 years depending on prediction type.

2. GRADING THE PREDICTION (ONLY IF corpus date > estimated deadline):
    - Research what *actually happened* by the deadline.
    - Determine whether the prediction turned out to be:
        - "TRUE": The prediction clearly came to pass by the deadline.
        - "FALSE": The prediction did not come to pass at all by the deadline.
        - "PARTIALLY_TRUE": The outcome was mixed or partially fulfilled.
    - Provide a **one line justification** using reliable evidence (news, official data, events).

3. FORMAT the output using this exact JSON schema and DO NOT include any explanations or extra text, don't even write json:
{
    "deadline_estimate": "YYYY-MM-DD",
    "deadline_reasoning": "Explain how you determined the best time to evaluate the prediction.",
    "grading_applicable": true/false,  # Only true if corpus date > deadline
    "grading": "TRUE" / "FALSE" / "PARTIALLY_TRUE" / null,
    "grading_justification": "What happened by the deadline that supports your grading, or null if grading not applicable"
}
    """
    
    user_prompt = f"""
    Please analyze the following prediction to:
1. Estimate the **best time (deadline)** to check if it turned out to be true.
2. If that date has already passed (compared to the article’s date), **grade the prediction** accordingly.
3. Provide a brief but clear **justification** for your grading.
    Article Text: {article_text}
    
    Prediction: {prediction_text}
    
    """
    
    try:
        # Using OpenAI API (adjust model as needed)
        response = client.chat.completions.create(
            model="llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5
        )
        
        result = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            analysis = json.loads(result)
            return analysis
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {result}")
            return {
                "deadline_estimate": "Unknown",
                "deadline_reasoning": "Failed to parse response",
                "grading_applicable": False,
                "grading": None,
                "grading_justification": None
            }
            
    except Exception as e:
        print(f"Error in API call: {e}")
        return {
            "deadline_estimate": "Error",
            "deadline_reasoning": f"API Error: {e}",
            "grading_applicable": False,
            "grading": None,
            "grading_justification": None
        }

def process_predictions_from_sheet(input_file, output_file):
    """
    Process predictions from Excel sheet with article text in column B and predictions in column D
    """
    try:
        # Read the Excel file
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
        
        # Identify columns (assuming B=1, D=3 in 0-indexed)
        article_col = df.columns[1]  # Column B
        prediction_col = df.columns[3]  # Column D
        
        print(f"Article column: {article_col}")
        print(f"Prediction column: {prediction_col}")
        
        output_data = []
        
        for i, row in df.iterrows():
            article_text = str(row[article_col]) if pd.notna(row[article_col]) else ""
            prediction_text = str(row[prediction_col]) if pd.notna(row[prediction_col]) else ""
            
            # Skip if no prediction or if prediction is "No predictions found"
            if not prediction_text or prediction_text.strip() == "" or "No predictions found" in prediction_text:
                print(f"Row {i+1}: Skipping - no valid prediction")
                continue
            
            print(f"Processing row {i+1}: {prediction_text[:100]}...")
            print(f"Article text: {article_text[:10]}...")
            
            # Analyze the prediction
            analysis = openai_analyze_prediction_deadline_and_grading(
                article_text, prediction_text
            )
            
            # Add to output data
            output_data.append({
                'Row_Number': i + 1,
                'Article_Text': article_text,
                'Prediction': prediction_text,
                'Deadline_Estimate': analysis.get('deadline_estimate', 'Unknown'),
                'Deadline_Reasoning': analysis.get('deadline_reasoning', ''),
                'Grading_Applicable': analysis.get('grading_applicable', False),
                'Grading': analysis.get('grading', ''),
                'Grading_Justification': analysis.get('grading_justification', '')
            })
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        # Create output DataFrame
        result_df = pd.DataFrame(output_data)
        
        # Save to Excel
        result_df.to_excel(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        return result_df
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    # File paths - modify these according to your setup
    input_file = "AI predictions.xlsx"  # Replace with your input file path
    output_file = "deadline_grading_analysis.xlsx"  # Output file name
    corpus_date = "2024-12-31"  # The date when your corpus was collected
    
    # Set your OpenAI API key
    keymain = os.getenv("navigator_api")
    client = OpenAI(api_key=keymain,base_url="https://api.ai.it.ufl.edu")
    # Process the predictions
    result_df = process_predictions_from_sheet(input_file, output_file)
    
    if result_df is not None:
        # Display summary statistics
        print("\n=== SUMMARY ===")
        print(f"Total rows processed: {len(result_df)}")
        print(f"Predictions with deadlines: {len(result_df[result_df['Deadline_Estimate'] != 'Unknown'])}")
        print(f"Predictions eligible for grading: {len(result_df[result_df['Grading_Applicable'] == True])}")
        
        # Grading breakdown
        if len(result_df[result_df['Grading_Applicable'] == True]) > 0:
            graded_df = result_df[result_df['Grading_Applicable'] == True]
            print("\nGrading Results:")
            print(graded_df['Grading'].value_counts())

if __name__ == "__main__":
    main()