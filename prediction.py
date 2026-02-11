
from google import genai
from google.genai import types
import pandas as pd
from openai import OpenAI
import json
import streamlit as st
from dotenv import load_dotenv
import os
from newspaper import Article


load_dotenv()
keymain = os.getenv("OpenAI_api_key")
client = OpenAI(api_key=keymain)

keysub = os.getenv("gemini_api")
gemini_api = keysub
gemini_client = genai.Client(api_key=gemini_api)
def openai_extract_predictions(article_text):
  system_prompt="""
You are a News Prediction Extractor, an expert AI assistant designed to analyze news articles and extract forward-looking statements—predictions about future events, trends, or outcomes. These may be made by the author or quoted from others.

Your role is to identify and clearly rephrase these predictive statements so that users can gain accurate insights into potential future developments.

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

📋 Output Format:
Summarize each extracted prediction as a **concise, standalone sentence** or two, using paraphrasing when needed for clarity. Direct quotes are acceptable only when attribution strengthens credibility or precision.

If more than 10 predictions are identified, extract only the 5 most impactful ones, noting how many others were omitted.

If none are found, clearly respond with:
"No predictions or forward-looking statements were identified in the article."

🧭 Your mission is to deliver accurate, relevant, and well-structured insights into future-oriented content to help users anticipate trends and decisions based on news narratives.

"""
  
  prompt = f"""
    You are a news prediction extractor. Your task is to identify all forward-looking statements—explicit or implied predictions about future events, trends, or outcomes—in the article below.

Extract predictions that fall into any of these categories:
- **Explicit Forecasts**: Statements using future-oriented language like "will," "is expected to," or "is projected to."
- **Implicit Predictions**: Suggestions of future outcomes even if not stated directly, such as "could trigger," "may result in," or "signals a shift."
- **Quoted Predictions**: Any forecast or expectation cited from experts, analysts, reports, or official sources.
- **Absolute Declarations**: Statements using future tense to assert planned or inevitable outcomes ("will receive sanctions," "will be implemented").
- **Conditional Speculation**: Statements predicting outcomes based on other events ("If inflation rises, interest rates will likely follow").

Instructions:
- Rephrase each prediction as a standalone, clearly worded sentence.
- Avoid vague commentary or past-only statements unless they directly support a future-looking insight.
- If the article contains more than 10 forward-looking statements, extract only the 5 most significant ones and note how many were omitted.
- If no predictions are found, clearly state: "No predictions or forward-looking statements were identified in the article."

Article:
{article_text}

Return your output as a JSON:
s
  "predictions": [
    "Prediction 1 here...",
    "Prediction 2 here...",
    ...
  ]

    """
  response = client.chat.completions.create(
        model="gpt-4.1-nano",
        response_format={"type": "json_object"},
         messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
        temperature=0.3
    )
  # return response.choices[0].message.content
  return json.loads(response.choices[0].message.content)


def parse_markdown_json_block(text):
    """
    Extract and parse JSON from a Markdown-style ```json code block.
    """
    if text.startswith("```json"):
        # Remove the starting and ending triple backticks
        text = text.strip().removeprefix("```json").removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        return None

def gemini_extract_predictions(article_text):
  prompt = f"""
    You are a news prediction extractor specializing in identifying future-oriented insights from textual content.

Your task is to extract all forward-looking statements (predictions about future events, trends, or outcomes) from the following article. Focus on statements that speculate or forecast developments, excluding vague opinions or past events. Each prediction should be rephrased as a clear, standalone sentence, avoiding direct quotes unless necessary for clarity. If no predictions are found, state so explicitly.

Article:
{article_text}

Additional Guidelines:
- Prioritize predictions related to significant societal, economic, or technological impacts.
- Ignore minor or speculative guesses lacking substantiation (e.g., "it might rain tomorrow" unless tied to a broader trend).
- Limit each prediction summary to 1-2 sentences for brevity.
- If the article contains more than 10 predictions, list only the 5 most impactful ones, with a note on how many others were omitted.

Format in json as:
predictions : ..
and just give it as json, no other text, just json
    """


  response = gemini_client.models.generate_content(
      model="gemini-2.5-flash-preview-05-20",
      contents=prompt,
      config=types.GenerateContentConfig(
          temperature=0.3,
      )
  )
  jsosn = parse_markdown_json_block(response.text)
  return jsosn



def extract_text_from_url(url):
    """
    Extracts text content from a given URL using the newspaper3k library.
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        st.error(f"Error extracting text from URL: {e}")
        return ""
 
def main():
    st.title("Grade Your News Article")
    st.write("This is a small prototype for building LLMs that can “grade the news” by ")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;(a) identifying logical fallacies contained in individual news stories, and ")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;(b) identifying verifiable predictions that have been made in the past and then checking whether they panned out.")

    input_type = st.radio("Choose input method", ["Paste Article Text", "Provide Article URL"])
    news_article = ""
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
    year_of_article = st.selectbox("Select the year of the article", ["2024","2023", "2022", "2021", "2020"])

    if type_of_llm == "OpenAI GPT-4":
        try :
            if st.button("Get the Predictions"):
                with st.spinner("Processing..."):
                    
                    analz = []
                    news_article =  extract_text_from_url(url) if input_type == "Provide Article URL" else news_article
                    jsosn = openai_extract_predictions(news_article)
                    st.json(jsosn)
                    
                    
        except Exception as e:
            st.error(f"Error grading the news article: {e}")
    elif type_of_llm == "Google Gemini":
        
        try :

            if st.button("Get the Predictions"):
                with st.spinner("Processing..."):
                    
                    analz = []
                    news_article =  extract_text_from_url(url) if input_type == "Provide Article URL" else news_article
                    jsosn = gemini_extract_predictions(news_article)
                    
                    st.json(jsosn)
        except Exception as e:
            st.error(f"Error grading the news article: {e}")

if __name__ == "__main__":
    main()
