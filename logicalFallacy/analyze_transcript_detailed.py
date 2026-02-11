import os
import json
import pandas as pd
from openai import OpenAI
import PyPDF2
from typing import List, Dict
import re
from tqdm import tqdm

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
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def get_context(sentences: List[str], index: int, context_range: int = 1) -> Dict[str, str]:
    """Get context sentences around the main sentence."""
    context = {
        "sentence_before": "",
        "main_sentence": sentences[index],
        "sentence_after": "",
        "full_context": ""
    }

    if index > 0:
        context["sentence_before"] = sentences[index - 1]

    if index < len(sentences) - 1:
        context["sentence_after"] = sentences[index + 1]

    # Build full context
    context_parts = []
    if context["sentence_before"]:
        context_parts.append(f"[BEFORE] {context['sentence_before']}")
    context_parts.append(f"[MAIN] {context['main_sentence']}")
    if context["sentence_after"]:
        context_parts.append(f"[AFTER] {context['sentence_after']}")

    context["full_context"] = "\n".join(context_parts)

    return context

def analyze_sentence_for_fallacy(client: OpenAI, sentence: str, context: str, fallacy_types: str) -> Dict:
    """Analyze a sentence for logical fallacies using LLM."""

    system_prompt = """You are an expert in logical reasoning and rhetoric analysis. Your task is to analyze sentences for logical fallacies.

You will be provided with:
1. A comprehensive list of logical fallacy types and their definitions
2. A sentence to analyze (marked with [MAIN])
3. Context sentences (one before and one after the main sentence)

Your job is to:
- Carefully analyze the [MAIN] sentence for any logical fallacies
- Consider the context to understand the argument being made
- If a fallacy is present, identify which specific type it is
- Provide clear, detailed reasoning for your classification
- If NO fallacy is present, return "none" as the category

IMPORTANT GUIDELINES:
- Only identify CLEAR instances of logical fallacies
- Do not over-interpret or force a classification
- The sentence must actually commit the fallacy, not just discuss it
- Consider that not every statement is an argument or contains a fallacy
- Be conservative in your classifications

Return your analysis in JSON format:
{
    "has_fallacy": true/false,
    "fallacy_category": "exact name from the provided list or 'none'",
    "reasoning": "detailed explanation of why this is (or isn't) the identified fallacy",
    "confidence": "high/medium/low",
    "is_argumentative": true/false
}"""

    user_prompt = f"""LOGICAL FALLACY TYPES AND DEFINITIONS:
{fallacy_types}

---

SENTENCE TO ANALYZE WITH CONTEXT:
{context}

---

Analyze the [MAIN] sentence for logical fallacies. Remember:
- Only classify as a fallacy if it clearly matches the definition
- Consider the context to understand the full argument
- Return "none" if no fallacy is present
- Be specific about which fallacy type it is"""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,  # Lower temperature for more consistent analysis
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        print(f"\nError analyzing sentence: {e}")
        return {
            "has_fallacy": False,
            "fallacy_category": "error",
            "reasoning": f"Error during analysis: {str(e)}",
            "confidence": "low",
            "is_argumentative": False
        }

def main():
    # Setup
    keymain = os.getenv("navigator_api")
    if not keymain:
        raise ValueError("navigator_api environment variable not set")

    client = OpenAI(api_key=keymain, base_url="https://api.ai.it.ufl.edu")

    # Paths
    pdf_path = r"d:\ML\LLMS\logicalFallacy\JRE-Rogan-Malone-Transcript.pdf"
    fallacy_file = r"d:\ML\LLMS\logicalFallacy\logicalfallacy.txt"
    output_file = r"d:\ML\LLMS\logicalFallacy\fallacy_analysis_detailed.xlsx"
    checkpoint_file = r"d:\ML\LLMS\logicalFallacy\analysis_checkpoint.json"

    print("=" * 70)
    print("LOGICAL FALLACY ANALYSIS - DETAILED MODE")
    print("=" * 70)

    print("\n1. Loading logical fallacy definitions...")
    with open(fallacy_file, 'r', encoding='utf-8') as f:
        fallacy_types = f.read()
    print("   ✓ Loaded fallacy definitions")

    print("\n2. Extracting text from PDF...")
    transcript_text = extract_text_from_pdf(pdf_path)
    print(f"   ✓ Extracted {len(transcript_text)} characters")

    print("\n3. Splitting into sentences...")
    sentences = split_into_sentences(transcript_text)
    print(f"   ✓ Found {len(sentences)} sentences")

    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 5]
    print(f"   ✓ {len(sentences)} sentences after filtering (min 5 words)")

    # Check for checkpoint
    start_idx = 0
    all_results = []

    if os.path.exists(checkpoint_file):
        print("\n4. Found checkpoint file. Resuming from last position...")
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get('last_index', 0) + 1
            all_results = checkpoint.get('results', [])
        print(f"   ✓ Resuming from sentence {start_idx}")
    else:
        print("\n4. Starting fresh analysis...")

    print(f"\n5. Analyzing sentences {start_idx} to {len(sentences)}...")
    print("   (This may take a while. Progress is saved every 10 sentences)")
    print("-" * 70)

    # Analyze with progress bar
    for idx in tqdm(range(start_idx, len(sentences)), desc="Analyzing"):
        sentence = sentences[idx]
        context_data = get_context(sentences, idx)

        # Analyze the sentence
        result = analyze_sentence_for_fallacy(
            client,
            sentence,
            context_data["full_context"],
            fallacy_types
        )

        # Add sentence and context to result
        result_entry = {
            "index": idx,
            "sentence": sentence,
            "sentence_before": context_data["sentence_before"],
            "sentence_after": context_data["sentence_after"],
            "full_context": context_data["full_context"],
            **result
        }

        all_results.append(result_entry)

        # Save checkpoint every 10 sentences
        if (idx + 1) % 10 == 0:
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'last_index': idx,
                    'results': all_results
                }, f, indent=2)

    print("\n" + "-" * 70)
    print("✓ Analysis complete!")

    # Filter to only include sentences with fallacies
    fallacy_results = [r for r in all_results if r.get("has_fallacy", False)]

    print(f"\n6. Preparing Excel output...")
    print(f"   Total sentences analyzed: {len(all_results)}")
    print(f"   Fallacies found: {len(fallacy_results)}")

    # Prepare data for Excel
    excel_data = []
    for result in fallacy_results:
        excel_data.append({
            "Sentence": result["sentence"],
            "Fallacy Category": result["fallacy_category"],
            "Context (1 before, main, 1 after)": result["full_context"],
            "Reasoning": result["reasoning"],
            "Confidence": result.get("confidence", "medium"),
            "Is Argumentative": result.get("is_argumentative", "N/A")
        })

    # Create DataFrame and save to Excel
    if excel_data:
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
                # Cap at 80 for readability
                max_length = min(max_length, 80)
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2

        print(f"\n7. Results saved to: {output_file}")

        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"\nFallacy type breakdown:")
        fallacy_counts = df['Fallacy Category'].value_counts()
        for fallacy, count in fallacy_counts.items():
            print(f"  • {fallacy}: {count}")

        print(f"\nConfidence levels:")
        confidence_counts = df['Confidence'].value_counts()
        for conf, count in confidence_counts.items():
            print(f"  • {conf}: {count}")
    else:
        print("\nNo fallacies found in the transcript.")

    # Clean up checkpoint file
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("\n✓ Checkpoint file cleaned up")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
