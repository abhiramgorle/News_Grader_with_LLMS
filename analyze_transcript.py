import os
import json
import pandas as pd
from openai import OpenAI
import PyPDF2
from typing import List, Dict
import re

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Simple sentence splitting - you can enhance this with nltk if needed
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def get_context(sentences: List[str], index: int, context_range: int = 1) -> str:
    """Get context sentences around the main sentence."""
    start_idx = max(0, index - context_range)
    end_idx = min(len(sentences), index + context_range + 1)

    context_sentences = []
    for i in range(start_idx, end_idx):
        if i == index:
            context_sentences.append(f"[MAIN] {sentences[i]}")
        else:
            context_sentences.append(sentences[i])

    return "\n".join(context_sentences)

def analyze_sentence_for_fallacy(client: OpenAI, sentence: str, context: str, fallacy_types: str) -> Dict:
    """Analyze a sentence for logical fallacies using LLM."""

    system_prompt = """You are an expert in logical reasoning and rhetoric analysis. Your task is to analyze sentences for logical fallacies.

You will be provided with:
1. A list of logical fallacy types and their definitions
2. A sentence to analyze (marked with [MAIN])
3. Context sentences (one before and one after)

Your job is to:
- Identify if the sentence contains a logical fallacy
- If yes, determine which type(s) of fallacy it contains
- Provide clear reasoning for why it's classified as that fallacy
- If no fallacy is present, return "none" as the category

Return your analysis in JSON format with the following structure:
{
    "has_fallacy": true/false,
    "fallacy_category": "name of fallacy or 'none'",
    "reasoning": "detailed explanation of why this is classified as the identified fallacy",
    "confidence": "high/medium/low"
}

Be precise and only identify clear instances of logical fallacies. Do not over-interpret or force a classification."""

    user_prompt = f"""LOGICAL FALLACY TYPES:
{fallacy_types}

CONTEXT (sentence to analyze marked with [MAIN]):
{context}

Analyze the [MAIN] sentence for logical fallacies."""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error analyzing sentence: {e}")
        return {
            "has_fallacy": False,
            "fallacy_category": "error",
            "reasoning": f"Error during analysis: {str(e)}",
            "confidence": "low"
        }

def batch_analyze_sentences(client: OpenAI, sentences: List[str], fallacy_types: str,
                           batch_size: int = 5) -> List[Dict]:
    """Analyze multiple sentences in batches for efficiency."""

    system_prompt = """You are an expert in logical reasoning and rhetoric analysis. Your task is to analyze multiple sentences for logical fallacies.

You will be provided with:
1. A list of logical fallacy types and their definitions
2. Multiple sentences to analyze, each with context

Your job is to analyze EACH sentence and identify if it contains a logical fallacy.

Return your analysis in JSON format with the following structure:
{
    "analyses": [
        {
            "sentence_index": 0,
            "has_fallacy": true/false,
            "fallacy_category": "name of fallacy or 'none'",
            "reasoning": "detailed explanation",
            "confidence": "high/medium/low"
        },
        ...
    ]
}

Be precise and only identify clear instances of logical fallacies."""

    results = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]

        # Prepare batch context
        batch_context = []
        for idx, sent_idx in enumerate(range(i, min(i + batch_size, len(sentences)))):
            context = get_context(sentences, sent_idx)
            batch_context.append(f"SENTENCE {idx}:\n{context}\n")

        user_prompt = f"""LOGICAL FALLACY TYPES:
{fallacy_types}

SENTENCES TO ANALYZE:
{"".join(batch_context)}

Analyze each sentence marked with [MAIN] for logical fallacies."""

        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content.strip()
            batch_results = json.loads(content)

            # Map results back to actual sentence indices
            for analysis in batch_results.get("analyses", []):
                batch_idx = analysis.get("sentence_index", 0)
                actual_idx = i + batch_idx
                if actual_idx < len(sentences):
                    results.append({
                        "sentence": sentences[actual_idx],
                        "context": get_context(sentences, actual_idx),
                        **{k: v for k, v in analysis.items() if k != "sentence_index"}
                    })
        except Exception as e:
            print(f"Error in batch {i}-{i+batch_size}: {e}")
            # Fallback to individual analysis for this batch
            for sent_idx in range(i, min(i + batch_size, len(sentences))):
                context = get_context(sentences, sent_idx)
                result = analyze_sentence_for_fallacy(client, sentences[sent_idx], context, fallacy_types)
                results.append({
                    "sentence": sentences[sent_idx],
                    "context": context,
                    **result
                })

    return results

def main():
    # Setup
    keymain = "sk-3wm8Eo8OwrJ3DNuo2T3hRw"
    if not keymain:
        raise ValueError("navigator_api environment variable not set")

    client = OpenAI(api_key=keymain, base_url="https://api.ai.it.ufl.edu")

    # Paths
    pdf_path = r"d:\ML\LLMS\logicalFallacy\Tuckcarlson_DarrylCooper.pdf"
    fallacy_file = r"d:\ML\LLMS\logicalFallacy\logicalfallacy.txt"
    output_file = r"d:\ML\LLMS\logicalFallacy\fallacy_analysis_results1.xlsx"

    print("Loading logical fallacy definitions...")
    with open(fallacy_file, 'r', encoding='utf-8') as f:
        fallacy_types = f.read()

    print("Extracting text from PDF...")
    transcript_text = extract_text_from_pdf(pdf_path)

    print("Splitting into sentences...")
    sentences = split_into_sentences(transcript_text)
    print(f"Found {len(sentences)} sentences to analyze")

    # Filter out very short sentences (likely noise)
    sentences = [s for s in sentences if len(s.split()) > 3]
    print(f"Analyzing {len(sentences)} sentences after filtering")

    print("\nAnalyzing sentences for logical fallacies...")
    print("This may take a while depending on the transcript length...")

    # Batch analysis (more efficient)
    all_results = batch_analyze_sentences(client, sentences, fallacy_types, batch_size=5)

    # Filter to only include sentences with fallacies
    fallacy_results = [r for r in all_results if r.get("has_fallacy", False)]

    print(f"\nFound {len(fallacy_results)} sentences with potential logical fallacies")

    # Prepare data for Excel
    excel_data = []
    for result in fallacy_results:
        excel_data.append({
            "Sentence": result["sentence"],
            "Fallacy Category": result["fallacy_category"],
            "Context": result["context"],
            "Reasoning": result["reasoning"],
            "Confidence": result.get("confidence", "medium")
        })

    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_data)

    # Save with formatting
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Fallacy Analysis', index=False)

        # Auto-adjust column widths
        worksheet = writer.sheets['Fallacy Analysis']
        for idx, col in enumerate(df.columns):
            max_length = max(
                df[col].astype(str).apply(len).max(),
                len(col)
            )
            # Cap at 100 for readability
            max_length = min(max_length, 100)
            worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2

    print(f"\nAnalysis complete! Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"Total sentences analyzed: {len(sentences)}")
    print(f"Fallacies found: {len(fallacy_results)}")
    print(f"\nFallacy breakdown:")
    fallacy_counts = df['Fallacy Category'].value_counts()
    for fallacy, count in fallacy_counts.items():
        print(f"  {fallacy}: {count}")

if __name__ == "__main__":
    main()
