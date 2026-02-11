from google import genai
from google.genai import types
import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
import os

load_dotenv()
keymain = os.getenv("navigator_api")
client = OpenAI(api_key=keymain,base_url="https://api.ai.it.ufl.edu")

keysub = os.getenv("gemini_api")
gemini_api = keysub
gemini_client = genai.Client(api_key=gemini_api)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import tiktoken
import re

def compress_article_with_langchain(article_text, max_tokens=3000):
    """
    Compress large articles using LangChain text splitter and OpenAI summarization
    """
    try:
        # Initialize tokenizer to count tokens
        encoding = tiktoken.encoding_for_model("llama-3.3-70b-instruct")
        current_tokens = len(encoding.encode(article_text))
        
        # If article is already small enough, return as is
        if current_tokens <= max_tokens:
            return article_text
        
        print(f"Article has {current_tokens} tokens, compressing to ~{max_tokens} tokens...")
        
        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_text(article_text)
        
        # Summarize each chunk while preserving predictions
        compressed_chunks = []
        for i, chunk in enumerate(chunks):
            summary_prompt = f"""
            Summarize this article section while preserving ALL forward-looking statements, predictions, forecasts, and future-oriented content. Keep the exact wording of any predictions or future claims.
            
            Focus on:
            - Maintaining all predictive statements verbatim
            - Preserving key context and facts
            - Keeping important quotes about future events
            - Retaining numerical forecasts and timelines
            
            Section {i+1}:
            {chunk}
            
            Provide a concise summary that retains all predictive content:
            """
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            compressed_chunks.append(response.choices[0].message.content)
        
        # Combine compressed chunks
        compressed_article = "\n\n".join(compressed_chunks)
        
        # Final compression if still too long
        final_tokens = len(encoding.encode(compressed_article))
        if final_tokens > max_tokens:
            final_prompt = f"""
            Further compress this article while preserving ALL predictions and forward-looking statements exactly as written:
            
            {compressed_article}
            
            Create a {max_tokens}-token summary that maintains all predictive content.
            """
            
            response = client.chat.completions.create(
                model="llama-3.3-70b-instruct",
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            compressed_article = response.choices[0].message.content
        
        print(f"Compressed from {current_tokens} to {len(encoding.encode(compressed_article))} tokens")
        return compressed_article
        
    except Exception as e:
        print(f"Error compressing article: {e}")
        return article_text[:15000]  # Fallback truncation


def openai_extract_predictions_with_scores(article_text):
    # if len(article_text) > 12000:  # Adjust threshold as needed
    #     print("Large article detected, compressing...")
    #     article_text = compress_article_with_langchain(article_text, max_tokens=3000)
    system_prompt = """
    You are a News Prediction Extractor, an expert AI assistant designed to analyze news articles and extract forward-looking statements—predictions about future events, trends, or outcomes. These may be made by the author or quoted from others.

Your role is to identify and clearly state these predictive statements, do not rephrase so that users can gain accurate insights into potential future developments.

🔍 Detection Scope:
Identify forward-looking statements that meet any of the following criteria:
- **Explicit Predictions**: Phrases such as "is expected to," "will," "is projected to," "analysts forecast," or "the report predicts."
- **Implicit Predictions**: Statements that imply future outcomes through logical inference, such as "could lead to," "may result in," "poses a risk of," or "signals a trend."
- **Quoted Forecasts**: Predictions cited from individuals, reports, experts, or institutions—even if speculative—as long as they are contextually future-oriented.
- **Future-Tense Declarations**: Use of definitive future tense ("will be banned," "will receive," "will increase") suggesting inevitable or planned developments.
- **Contingent Speculation**: Statements conditional on other events ("if rates continue to rise, housing demand could plummet").

❌ Do not extract:
- Commentary on past events, unless explicitly tied to future consequences.
- Generic opinions or vague guesses ("I hope things improve").
- Statements lacking temporal direction or relevance to broader outcomes.

🧠 Interpretation Strategy:
- Focus on the **intent** of the statement: does it communicate or imply something about the future?
- Clarify **who made the prediction** (author, quoted expert, etc.) if relevant to credibility.
- Prioritize **predictions with significant societal, economic, political, or technological consequences**.

    Your role is to identify and clearly state these predictive statements, do not rephrase, and provide confidence scores for each prediction.

    For each prediction, provide:
    1. The prediction statement (clear and standalone)
    2. Verifiability Score (1-5): How verifiable/measurable the claim is (5 = highly specific and measurable, 1 = vague and unverifiable)
    3. Certainty Score (1-5): How certain the speaker appears to be about the prediction (5 = very confident/definitive, 1 = tentative/uncertain)

    Format your response as JSON with this structure:
    {
        "predictions": [
            {
                "prediction": "prediction text here",
                "verifiability_score": 3,
                "certainty_score": 5
            }
        ]
    }

    If no predictions are found, return {"predictions": []}
    """
    
    user_prompt = f"""
    You are a news prediction extractor. Your task is to identify all forward-looking statements—explicit or implied predictions about future events, trends, or outcomes—in the article below and score them.

Extract predictions that fall into any of these categories:
- **Explicit Forecasts**: Statements using future-oriented language like "will," "is expected to," or "is projected to."
- **Implicit Predictions**: Suggestions of future outcomes even if not stated directly, such as "could trigger," "may result in," or "signals a shift."
- **Quoted Predictions**: Any forecast or expectation cited from experts, analysts, reports, or official sources.
- **Absolute Declarations**: Statements using future tense to assert planned or inevitable outcomes ("will receive sanctions," "will be implemented").
- **Conditional Speculation**: Statements predicting outcomes based on other events ("If inflation rises, interest rates will likely follow").

    Article:
    {article_text}

    Guidelines:
    - Focus on substantial predictions, not minor speculations
    - Avoid vague commentary or past-only statements unless they directly support a future-looking insight.
    - Rephrase predictions clearly and concisely
    - Provide realistic scores based on the language used
    - Limit to the 10 most significant predictions if there are many
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        # Check if content is empty
        if not content:
            print("OpenAI API returned empty content.")
            return {"predictions": []}
        # Try to find the first valid JSON object in the response
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Sometimes the model returns extra text before/after JSON, try to extract JSON substring
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    result = json.loads(match.group(0))
                except Exception:
                    print("Could not parse extracted JSON substring.")
                    result = {"predictions": []}
            else:
                print("No JSON object found in response.")
                result = {"predictions": []}
        return result
    except Exception as e:
        print(f"Error with OpenAI API: {e}")
        return {"predictions": []}

def process_articles_from_sheet(input_file_path, output_file_path):
    """
    Process articles from Excel sheet and create analysis output
    """
    try:
        # Read the input Excel file
        df = pd.read_excel(input_file_path)
        
        # Extract articles from column D (rows 2-71, which is index 1-70 in pandas)
        articles = df.iloc[81:120, 3].dropna().tolist()  # Column D is index 3
        
        print(f"Found {len(articles)} articles to process")
        
        # Prepare output data
        output_data = []
        
        for i, article in enumerate(articles, 1):
            print(f"Processing article {i}/{len(articles)}...")
            
            # Extract predictions with scores
            result = openai_extract_predictions_with_scores(article)
            predictions = result.get("predictions", [])
            print(f"  Found {predictions} predictions")
            
            if predictions:
                for j, pred in enumerate(predictions):
                    output_data.append({
                        'Article_Number': i,
                        'Article_Text': article,
                        'Prediction_Number': j + 1,
                        'Prediction': pred.get('prediction', ''),
                        'Verifiability_Score': pred.get('verifiability_score', 0),
                        'Certainty_Score': pred.get('certainty_score', 0)
                    })
            else:
                # If no predictions found, still record the article
                output_data.append({
                    'Article_Number': i,
                    'Article_Text': article,
                    'Prediction_Number': 0,
                    'Prediction': 'No predictions found',
                    'Verifiability_Score': 0,
                    'Certainty_Score': 0
                })
        
        # Create DataFrame and save to Excel
        output_df = pd.DataFrame(output_data)
        
        # Save to Excel with formatting
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='Predictions_Analysis12', index=False)
            
            # Get the workbook and worksheet
            workbook = writer.book
            worksheet = writer.sheets['Predictions_Analysis']
            
            # Adjust column widths
            worksheet.column_dimensions['A'].width = 15  # Article_Number
            worksheet.column_dimensions['B'].width = 80  # Article_Text
            worksheet.column_dimensions['C'].width = 20  # Prediction_Number
            worksheet.column_dimensions['D'].width = 80  # Prediction
            worksheet.column_dimensions['E'].width = 20  # Verifiability_Score
            worksheet.column_dimensions['F'].width = 15  # Certainty_Score
        
        print(f"Analysis complete! Results saved to {output_file_path}")
        print(f"Processed {len(articles)} articles and found {len([x for x in output_data if x['Prediction_Number'] > 0])} predictions")
        
        return output_df
        
    except Exception as e:
        print(f"Error processing articles: {e}")
        return None

def main():
    # File paths - modify these according to your setup
    input_file = "Scraped_news_20201.xlsx"  # Replace with your input file path
    output_file = "predictions_analysis12.xlsx"  # Output file name
    
    # Process the articles
    result_df = process_articles_from_sheet(input_file, output_file)
    
    if result_df is not None:
        # Display summary statistics
        print("\n=== SUMMARY ===")
        print(f"Total articles processed: {result_df['Article_Number'].nunique()}")
        print(f"Total predictions found: {len(result_df[result_df['Prediction_Number'] > 0])}")
        print(f"Average verifiability score: {result_df[result_df['Verifiability_Score'] > 0]['Verifiability_Score'].mean():.2f}")
        print(f"Average certainty score: {result_df[result_df['Certainty_Score'] > 0]['Certainty_Score'].mean():.2f}")

if __name__ == "__main__":
    main()