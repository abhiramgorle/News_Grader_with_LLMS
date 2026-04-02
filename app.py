import pandas as pd
from openai import OpenAI
import json
import streamlit as st
from dotenv import load_dotenv
import os
import re
from datetime import date
from newspaper import Article

load_dotenv()

keymain = os.getenv("navigator_api")
client = OpenAI(api_key=keymain, base_url="https://api.ai.it.ufl.edu")


def safe_json_parse(text, default_value):
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
    except Exception:
        return default_value


def extract_predictions(article_text, primary_model="gpt-5"):
    system_prompt = """
You are a News Prediction Extractor. Extract forward-looking statements from news articles.

Extract statements that are:
- Explicit Predictions: "is expected to," "will," "is projected to," "analysts forecast"
- Implicit Predictions: "could lead to," "may result in," "poses a risk of"
- Quoted Forecasts: Predictions from experts, reports, or institutions
- Future-Tense Declarations: "will be banned," "will receive," "will increase"
- Contingent Speculation: "if X, then Y will happen"

DO NOT extract:
- Past events or historical commentary
- Generic opinions without future implications
- Vague speculation without a clear directional claim
- Rhetorical questions

When in doubt, include the prediction. It is better to over-extract than miss valid ones.

Respond ONLY with valid JSON:
{"predictions": [{"prediction": "exact text here", "verifiability_score": 3, "certainty_score": 5}]}
"""
    try:
        response = client.chat.completions.create(
            model=primary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract all valid predictions from this article:\n\n{article_text}"}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        result = safe_json_parse(content, {"predictions": []})
        return result.get("predictions", [])
    except Exception as e:
        st.error(f"Extraction error: {e}")
        return []


def validate_and_estimate_deadline(prediction_text, article_date):
    system_prompt = """
You are a Prediction Analyzer. Determine if this is a real verifiable prediction and estimate when it can be verified.

A REAL prediction must make a specific claim about a future state of the world and be verifiable.

Respond ONLY with valid JSON:
{
  "is_prediction": "YES" | "NO",
  "deadline_estimate": "YYYY-MM-DD" | "UNKNOWN",
  "deadline_confidence": 1-5,
  "rejection_reason": "brief explanation if NO, otherwise empty string"
}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Article Date: {article_date}\nStatement: "{prediction_text}"\n\nIs this a real, verifiable prediction? If yes, when can we check it?'}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        return safe_json_parse(content, {
            "is_prediction": "NO", "deadline_estimate": "UNKNOWN",
            "deadline_confidence": 1, "rejection_reason": "Parse error"
        })
    except Exception as e:
        return {"is_prediction": "NO", "deadline_estimate": "UNKNOWN", "deadline_confidence": 1, "rejection_reason": str(e)}


def grade_prediction(prediction_text, primary_model="gpt-5"):
    system_prompt = """
You are a fact-checker. Grade the accuracy of this prediction using your knowledge.

- "TRUE": The prediction clearly came to pass
- "FALSE": The prediction clearly did not come to pass
- "PARTIALLY_TRUE": The outcome was mixed or partially fulfilled
- "UNVERIFIABLE": Cannot determine with available knowledge

Respond ONLY with valid JSON:
{
    "grading": "TRUE" | "FALSE" | "PARTIALLY_TRUE" | "UNVERIFIABLE",
    "grading_justification": "brief summary of what happened based on your knowledge"
}
"""
    try:
        response = client.chat.completions.create(
            model=primary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Grade this prediction using your own knowledge.\n\nPrediction: "{prediction_text}"'}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        return safe_json_parse(content, {"grading": "Error", "grading_justification": "Error"})
    except Exception as e:
        return {"grading": "Error", "grading_justification": str(e)}


def claude_verify(prediction_text, grading, justification):
    system_prompt = """
You are a secondary fact-checker reviewing another AI's prediction grading.
Assess whether you agree and provide additional context or corrections.

Respond with valid JSON only:
{
    "claude_agrees": "YES" | "NO" | "PARTIALLY",
    "claude_additional_context": "additional information, evidence, or corrections"
}
"""
    try:
        response = client.chat.completions.create(
            model="claude-4-sonnet",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Prediction: "{prediction_text}"\nGrading: {grading}\nJustification: {justification}\n\nDo you agree with this grading?'}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        return safe_json_parse(content, {"claude_agrees": "Error", "claude_additional_context": "Error"})
    except Exception as e:
        return {"claude_agrees": "Error", "claude_additional_context": str(e)}


def gemini_verify(prediction_text, grading, justification):
    system_prompt = """
You are a third fact-checker reviewing another AI's prediction grading.
Assess whether you agree and provide additional context or corrections.

Respond with valid JSON only:
{
    "gemini_agrees": "YES" | "NO" | "PARTIALLY",
    "gemini_additional_context": "additional information, evidence, or corrections"
}
"""
    try:
        response = client.chat.completions.create(
            model="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Prediction: "{prediction_text}"\nGrading: {grading}\nJustification: {justification}\n\nDo you agree with this grading?'}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        return safe_json_parse(content, {"gemini_agrees": "Error", "gemini_additional_context": "Error"})
    except Exception as e:
        return {"gemini_agrees": "Error", "gemini_additional_context": str(e)}


def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return ""


def process_article(article_text, year_of_article, primary_model):
    article_date = f"{year_of_article}-01-01"
    current_date = date.today()
    results = []

    with st.spinner("Extracting predictions..."):
        predictions = extract_predictions(article_text, primary_model)

    if not predictions:
        st.warning("No predictions found in this article.")
        return pd.DataFrame()

    st.info(f"Found {len(predictions)} potential predictions. Validating(through Backtracking) and grading...")
    progress_bar = st.progress(0)

    for i, pred in enumerate(predictions):
        prediction_text = pred.get('prediction', '').strip()
        if not prediction_text:
            progress_bar.progress((i + 1) / len(predictions))
            continue

        deadline_info = validate_and_estimate_deadline(prediction_text, article_date)

        if deadline_info.get('is_prediction') == 'NO':
            progress_bar.progress((i + 1) / len(predictions))
            continue

        deadline_str = deadline_info.get('deadline_estimate', 'UNKNOWN')
        result = {
            'Prediction': prediction_text,
            'Deadline': deadline_str,
            'Grading': 'Pending',
            'Grading_Justification': 'Deadline not yet reached',
            'Claude_Agrees': 'N/A',
            'Claude_Context': 'N/A',
            'Gemini_Agrees': 'N/A',
            'Gemini_Context': 'N/A',
        }

        # Grade if deadline has passed
        should_grade = False
        if deadline_str not in ['UNKNOWN', 'N/A', 'Error']:
            try:
                if date.fromisoformat(deadline_str) <= current_date:
                    should_grade = True
            except ValueError:
                should_grade = True
        else:
            should_grade = True  # Grade unknowns too — model can still assess

        if should_grade:
            gpt_result = grade_prediction(prediction_text, primary_model)
            result['Grading'] = gpt_result.get('grading', 'Error')
            result['Grading_Justification'] = gpt_result.get('grading_justification', '')

            if result['Grading'] not in ['Error', None]:
                claude_result = claude_verify(prediction_text, result['Grading'], result['Grading_Justification'])
                result['Claude_Agrees'] = claude_result.get('claude_agrees', 'Error')
                result['Claude_Context'] = claude_result.get('claude_additional_context', '')

                gemini_result = gemini_verify(prediction_text, result['Grading'], result['Grading_Justification'])
                result['Gemini_Agrees'] = gemini_result.get('gemini_agrees', 'Error')
                result['Gemini_Context'] = gemini_result.get('gemini_additional_context', '')

        results.append(result)
        progress_bar.progress((i + 1) / len(predictions))

    return pd.DataFrame(results)


def main():
    st.title("Grade Your News Article")
    st.write("This is a small prototype for building LLMs that can 'grade the news' by")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;identifying verifiable predictions that have been made in the past and then checking whether they panned out.")

    input_type = st.radio("Choose input method", ["Paste Article Text", "Provide Article URL"])
    news_article = ""
    url = ""

    if input_type == "Paste Article Text":
        news_article = st.text_area("Paste the news article here", height=200)
    else:
        url = st.text_input("Enter the URL of the news article")
        if st.button("Extract Text from URL") and url:
            with st.spinner("Extracting text from URL..."):
                extracted_text = extract_text_from_url(url)
                if extracted_text:
                    st.success("Text extracted successfully!")
                    st.text_area("Extracted Article Text", value=extracted_text, height=200)
                    news_article = extracted_text

    type_of_llm = st.selectbox("Select type of LLM you want to use", ["OpenAI GPT-4", "Google Gemini"])
    year_of_article = st.selectbox("Select the year of the article", ["2026","2025","2024", "2023", "2022", "2021", "2020"])

    primary_model = "gpt-5" if type_of_llm == "OpenAI GPT-4" else "gemini-2.5-pro"

    if st.button("Get the Predictions"):
        if not news_article and input_type == "Provide Article URL" and url:
            news_article = extract_text_from_url(url)

        if not news_article:
            st.error("Please provide an article text or URL first.")
            return

        result_df = process_article(news_article, year_of_article, primary_model)

        if not result_df.empty:
            st.write("### Predictions Analysis")
            st.dataframe(result_df, use_container_width=True)


if __name__ == "__main__":
    main()
