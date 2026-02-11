# Logical Fallacy Analysis Scripts

This directory contains scripts to analyze the JRE-Rogan-Malone transcript for logical fallacies.

## Files

1. **analyze_transcript.py** - Batch processing version (faster, processes 5 sentences at a time)
2. **analyze_transcript_detailed.py** - Detailed version (slower but more thorough, with checkpointing)

## Requirements

Install required packages:
```bash
pip install openai pandas PyPDF2 openpyxl tqdm
```

## Setup

Make sure your `navigator_api` environment variable is set:
```bash
# Windows
set navigator_api=your_api_key_here

# Linux/Mac
export navigator_api=your_api_key_here
```

## Usage

### Option 1: Batch Analysis (Faster)
```bash
python analyze_transcript.py
```
- Processes 5 sentences at a time
- Faster overall execution
- Good for initial analysis

### Option 2: Detailed Analysis (More Thorough)
```bash
python analyze_transcript_detailed.py
```
- Processes one sentence at a time
- More thorough analysis
- **Checkpoint feature**: Saves progress every 10 sentences
- Can resume if interrupted
- Better progress tracking with tqdm

## Output

Both scripts create an Excel file with the following columns:

| Column | Description |
|--------|-------------|
| **Sentence** | The main sentence containing the fallacy |
| **Fallacy Category** | Type of logical fallacy (e.g., "ad hominem", "strawman") |
| **Context** | 1 sentence before, main sentence, 1 sentence after |
| **Reasoning** | Detailed explanation of why it's classified as that fallacy |
| **Confidence** | LLM's confidence level (high/medium/low) |

## Output Files

- `fallacy_analysis_results.xlsx` - Output from batch version
- `fallacy_analysis_detailed.xlsx` - Output from detailed version
- `analysis_checkpoint.json` - Checkpoint file (auto-deleted when complete)

## How It Works

1. **Extract PDF**: Extracts text from the JRE transcript PDF
2. **Sentence Splitting**: Breaks text into individual sentences
3. **Filtering**: Removes very short sentences (< 5 words)
4. **Context Gathering**: Gets 1 sentence before and after each main sentence
5. **LLM Analysis**: Uses GPT-5 to analyze each sentence for fallacies
6. **Excel Export**: Creates formatted Excel file with results

## System Prompts

The scripts use carefully crafted prompts that:
- Provide all 24 fallacy type definitions
- Request conservative classification (avoid over-interpretation)
- Ask for detailed reasoning
- Return structured JSON responses
- Consider context for better accuracy

## Fallacy Types Detected

The analysis looks for these 24 logical fallacies:
- Strawman
- False Cause
- Appeal to Emotion
- The Fallacy Fallacy
- Slippery Slope
- Ad Hominem
- Tu Quoque
- Personal Incredulity
- Special Pleading
- Loaded Question
- Burden of Proof
- Ambiguity
- The Gambler's Fallacy
- Bandwagon
- Appeal to Authority
- Composition/Division
- No True Scotsman
- Genetic
- Black-or-White
- Begging the Question
- Appeal to Nature
- Anecdotal
- The Texas Sharpshooter
- Middle Ground

## Tips

- The detailed version is recommended for thorough analysis
- Use the batch version if you want quick results
- The checkpoint feature in detailed version is useful for long transcripts
- Review low-confidence results manually
- Consider the context when interpreting results
