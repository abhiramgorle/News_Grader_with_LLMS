import pandas as pd
from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import time
import traceback
from pathlib import Path
import concurrent.futures
from threading import Lock
import re

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# API Key and Endpoint
# Make sure your .env file has: navigator_api="your_api_key_here"
NAVIGATOR_API_KEY = os.getenv("navigator_api")
NAVIGATOR_BASE_URL = "https://api.ai.it.ufl.edu"

# File Paths
INPUT_FILE = "Op-Ed_Articles.xlsx"  # Your input file with articles
OUTPUT_FILE = "journalism_markers_analysis.xlsx"
CHECKPOINT_FILE = "journalism_analysis_checkpoint.xlsx"

# Processing Settings
MAX_WORKERS = 3  # Number of articles to process in parallel
BACKUP_INTERVAL = 5  # Save progress every 5 articles
REQUEST_TIMEOUT = 180 # Timeout in seconds for a single article analysis

# --- MAIN SCRIPT ---

# Global lock for thread-safe file operations
file_lock = Lock()
client = OpenAI(api_key=NAVIGATOR_API_KEY, base_url=NAVIGATOR_BASE_URL)

class JournalismMarkerProcessor:
    """
    Processes articles to find signals of high-quality journalism
    based on a predefined set of principles.
    """
    def __init__(self, checkpoint_file=CHECKPOINT_FILE, backup_interval=BACKUP_INTERVAL):
        self.checkpoint_file = checkpoint_file
        self.backup_interval = backup_interval
        self.processed_count = 0

    def safe_json_parse(self, text, default_value):
        """Safely parse JSON from model output, with robust fallbacks."""
        try:
            # Clean text before parsing
            text = text.strip()
            if not text:
                return default_value

            # Find JSON embedded in other text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group()
            
            return json.loads(text)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw text causing error: {text[:300]}...")
            return default_value
        except Exception as e:
            print(f"Unexpected error parsing JSON: {e}")
            return default_value

    def clean_text_for_excel(self, text):
        """Cleans text to prevent issues when saving to Excel."""
        if not isinstance(text, str):
            return text
        
        # Excel has a character limit per cell
        if len(text) > 32767:
            text = text[:32760] + "..."
            
        # Escape quotes and remove control characters
        cleaned = text.replace('"', '""')
        cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\t\n\r')
        return cleaned

    def save_checkpoint(self, output_data, output_file_path):
        """Saves current progress to an Excel file with error handling."""
        with file_lock:
            try:
                # Clean all text data before creating the DataFrame
                cleaned_data = []
                for record in output_data:
                    cleaned_record = {key: self.clean_text_for_excel(value) for key, value in record.items()}
                    cleaned_data.append(cleaned_record)
                
                if not cleaned_data:
                    print("No data to save.")
                    return

                checkpoint_df = pd.DataFrame(cleaned_data)
                
                # Save to Excel
                with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                    checkpoint_df.to_excel(writer, sheet_name='Journalism_Markers', index=False)
                    self.format_excel_columns(writer.sheets['Journalism_Markers'])
                
                print(f"✅ Checkpoint saved: {len(output_data)} records to {output_file_path}")

            except Exception as e:
                print(f"⚠ Error saving checkpoint: {e}")
                traceback.print_exc()

    def format_excel_columns(self, worksheet):
        """Sets column widths for better readability in the output Excel file."""
        column_widths = {
            'A': 15,  # Article_Number
            'B': 50,  # Marker_Category
            'C': 80,  # Extracted_Text
            'D': 80,  # Reasoning
            'E': 100, # Article_Text
        }
        for col, width in column_widths.items():
            if col in worksheet.column_dimensions:
                worksheet.column_dimensions[col].width = width

    def extract_journalism_markers(self, article_text, max_retries=3):
        """
        Calls the AI model to extract journalism markers based on the provided principles.
        """
        system_prompt = """
You are an expert analyst tasked with identifying signals of high-quality journalism within news articles. Your goal is to detect and extract specific passages that exemplify one of the nine principles defined below.

For each article, you must respond ONLY with a valid JSON object in the following format:
{
  "journalism_markers": [
    {
      "marker_category": "Name of the Category",
      "extracted_text": "The exact sentence or short passage from the article that exemplifies the principle.",
      "reasoning": "A brief explanation of why this text fits the specified category, referencing its definition."
    }
  ]
}

If no markers are found, return an empty list: {"journalism_markers": []}

---
### The 9 Principles of High-Quality Journalism

1.  **Steel-manning**: The author constructively summarizes the strongest possible version of an opposing argument, using its own core principles and terminology accurately and without disparagement, before presenting their own critique.
    *   Example: "Proponents of a wealth tax argue it is not merely a tool for revenue, but a necessary corrective for democratic erosion. Their strongest case rests on the premise that extreme concentrations of capital inevitably translate into disproportionate political power..."

2.  **Caveats & Self-Skepticism**: The author explicitly identifies and acknowledges limitations, weaknesses, or boundary conditions of their own argument, evidence, or expertise.
    *   Example: "My analysis suggests this AI model will significantly accelerate scientific discovery. However, this prediction is based on its performance on benchmarked tasks; real-world scientific breakthroughs involve serendipity that may not be captured by these metrics."

3.  **Nuance Introduction**: The author identifies a prevailing binary or overly simplistic framing of a debate and introduces a more complex spectrum of possibilities, a critical distinction, or a neglected contextual factor.
    *   Example: "The discussion on remote work is often framed as a binary: full productivity versus lost collaboration. This misses a critical third variable: task differentiation."

4.  **Quantified Uncertainty**: The author uses explicit, quantified language to express the degree of confidence in a claim (e.g., probabilities, confidence intervals).
    *   Example: "Based on current leading indicators, I estimate a 65% probability of a mild recession beginning within the next 12 months."

5.  **Openness to Reconsideration**: The author explicitly states that they have changed, or are actively reconsidering, a previously stated position on a specific issue, providing a reason.
    *   Example: "I have long been a skeptic of congestion pricing... However, the empirical data from London and Stockholm... has caused me to seriously reconsider my opposition."

6.  **Research Synthesis**: The author fairly summarizes the state of empirical evidence on a specific, contested question, accurately representing key findings from multiple sides.
    *   Example: "The question of whether sugar-sweetened beverage taxes reduce consumption has produced mixed evidence. A 2019 meta-analysis... found an average sales reduction of 10%... However, a 2021 study... highlighted significant cross-border shopping..."

7.  **Adversarial Engagement**: The author directly addresses a specific counter-argument from a named opponent, school of thought, or publication, and responds to its substance point-by-point.
    *   Example: "In her recent essay, Dr. Elena Vance argues that direct air capture is a dangerous distraction... However, her calculation on energy use relies on 2020 technology. Recent pilot plants... have shown a 40% reduction in energy requirements..."

8.  **Argumentative Specificity**: The author grounds claims in concrete details, precise data, named entities, or specific logical steps.
    *   Example: "The claim that Country X is expanding its influence... is supported by three concrete developments... the completion of the naval facility in Port Y, the signing of a security pact with Nation Z, and a 300% increase in development loans..."

9.  **Source Transparency**: The author provides accessible references (hyperlinks, citations) that allow a reader to verify data or quotes.
    *   Example: "According to the Federal Reserve's Q3 2023 Financial Stability Report (see page 12), delinquency rates... have risen to 4.5%..."
---
"""
        user_prompt = f"Based on the principles above, analyze this article and extract all journalism markers:\n\n{article_text}"

        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",  # Using a powerful model for this nuanced task
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                content = response.choices[0].message.content.strip()
                result = self.safe_json_parse(content, {"journalism_markers": []})
                
                # Validate the structure of the response
                if isinstance(result.get("journalism_markers"), list):
                    return result.get("journalism_markers", [])
                else:
                    raise ValueError("Invalid response format: 'journalism_markers' is not a list.")

            except Exception as e:
                print(f"⚠ Attempt {attempt + 1} failed for marker extraction: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ Failed to extract markers after {max_retries} attempts.")
                    return []

    def process_single_article(self, article_data):
        """Processes one article, extracts markers, and returns a list of results."""
        article_num, article_text = article_data
        
        try:
            print(f"📄 Processing article {article_num}...")
            markers = self.extract_journalism_markers(article_text)
            print(f"  ✅ Found {len(markers)} potential markers in article {article_num}.")

            article_results = []
            if markers:
                for marker in markers:
                    result = {
                        'Article_Number': article_num,
                        'Marker_Category': marker.get('marker_category', 'Unknown'),
                        'Extracted_Text': marker.get('extracted_text', 'Extraction Failed'),
                        'Reasoning': marker.get('reasoning', 'No reasoning provided.'),
                        'Article_Text': article_text,
                    }
                    article_results.append(result)
            else:
                # If no markers are found, still create a record for the article
                article_results.append({
                    'Article_Number': article_num,
                    'Marker_Category': 'No Markers Found',
                    'Extracted_Text': 'N/A',
                    'Reasoning': 'The model did not identify any of the nine journalism markers in this article.',
                    'Article_Text': article_text,
                })
            return article_results

        except Exception as e:
            print(f"❌ Critical error processing article {article_num}: {e}")
            traceback.print_exc()
            # Return an error record
            return [{
                'Article_Number': article_num,
                'Marker_Category': 'Processing Error',
                'Extracted_Text': str(e),
                'Reasoning': traceback.format_exc(),
                'Article_Text': article_text,
            }]

    def run_processing(self, input_file_path, output_file_path):
        """Main function to read articles, process them in parallel, and save results."""
        try:
            df = pd.read_excel(input_file_path)
            # Assuming articles are in the 5th column (index 4), as in your original script.
            # Adjust the column index if necessary.
            articles = df.iloc[:, 1].dropna().tolist()
            print(f"📚 Found {len(articles)} articles to process from {input_file_path}.")

            article_data_list = [(i, article) for i, article in enumerate(articles, 1)]
            all_results = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_article_num = {
                    executor.submit(self.process_single_article, data): data[0]
                    for data in article_data_list
                }

                for future in concurrent.futures.as_completed(future_to_article_num):
                    article_num = future_to_article_num[future]
                    try:
                        results = future.result(timeout=REQUEST_TIMEOUT)
                        all_results.extend(results)
                        
                        self.processed_count += 1
                        if self.processed_count % self.backup_interval == 0:
                            self.save_checkpoint(all_results, CHECKPOINT_FILE)

                    except concurrent.futures.TimeoutError:
                        print(f"⏰ Timeout processing article {article_num}. It will be marked as an error.")
                        all_results.append({
                            'Article_Number': article_num,
                            'Marker_Category': 'Timeout Error',
                            'Extracted_Text': 'Processing exceeded the timeout limit.',
                            'Reasoning': f'The analysis took longer than {REQUEST_TIMEOUT} seconds.',
                            'Article_Text': 'N/A',
                        })
                    except Exception as e:
                        print(f"❌ Error processing future for article {article_num}: {e}")

            print("\n🎉 Processing complete!")
            # Final save of all results
            self.save_checkpoint(all_results, output_file_path)
            
            # Final Summary
            final_df = pd.DataFrame(all_results)
            marker_counts = final_df['Marker_Category'].value_counts()
            print("\n--- 📈 Final Summary ---")
            print(f"Total articles processed: {len(articles)}")
            print(f"Total markers found: {len(final_df[final_df['Marker_Category'] != 'No Markers Found'])}")
            print("\nDistribution of markers found:")
            print(marker_counts)
            print("-------------------------\n")


        except FileNotFoundError:
            print(f"💥 FATAL ERROR: Input file not found at '{input_file_path}'")
        except Exception as e:
            print(f"💥 A fatal error occurred in the main processing run: {e}")
            traceback.print_exc()


def main():
    print("🚀 Starting High-Quality Journalism Marker Detector...")
    processor = JournalismMarkerProcessor()
    processor.run_processing(INPUT_FILE, OUTPUT_FILE)
    print("✅ Process finished successfully.")

if __name__ == "__main__":
    main()
