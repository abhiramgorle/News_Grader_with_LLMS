import streamlit as st
import openai
import json
import re
from typing import Dict, List, Any
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class NewsAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the analyzer with OpenAI API key"""
        self.client = openai.OpenAI(api_key="sk-DvmrslgloHiPLxdtkrT8aA", base_url="https://api.ai.it.ufl.edu")
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using NLTK"""
        return sent_tokenize(text)
    
    def analyze_article(self, article_text: str) -> Dict[str, Any]:
        """Analyze a news article and identify different types of statements"""
        
        sentences = self.split_into_sentences(article_text)
        
        prompt = f"""
        Please analyze the following news article sentences and identify different types of statements and rhetorical elements. 

        For each sentence, identify if it contains any of the following elements:
        1. **Prediction** - Statements about future events or outcomes
        2. **Ad Hominem** - Personal attacks on individuals rather than their arguments
        3. **Fact** - Verifiable, objective statements
        4. **Opinion** - Subjective viewpoints or interpretations
        5. **Speculation** - Unconfirmed theories or guesswork
        6. **Appeal to Authority** - Using someone's credentials/status to support an argument
        7. **Emotional Appeal** - Language designed to evoke emotional responses
        8. **Bias** - Language that shows clear bias or slant
        9. **Logical Fallacy** - Other logical errors (strawman, false dichotomy, etc.)
        10. **Evidence** - References to data, studies, or sources
        11. **Neutral** - Objective, balanced reporting

        Provide your analysis in this exact JSON format:
        {{
            "sentences": [
                {{
                    "text": "exact sentence text here",
                    "primary_category": "single main category",
                    "secondary_categories": ["additional categories if any"],
                    "explanation": "brief explanation",
                    "confidence": "high/medium/low"
                }}
            ]
        }}

        Sentences to analyze:
        {json.dumps(sentences)}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-instruct",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert in media literacy and rhetorical analysis. Analyze each sentence objectively and provide consistent JSON output."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Clean and parse JSON
            if "```json" in analysis_text:
                start = analysis_text.find("```json") + 7
                end = analysis_text.find("```", start)
                json_content = analysis_text[start:end].strip()
            else:
                json_content = analysis_text
            
            try:
                return json.loads(json_content)
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse API response: {e}")
                return {"error": "JSON parsing failed", "raw": analysis_text}
                
        except Exception as e:
            st.error(f"API call failed: {str(e)}")
            return {"error": str(e)}

def get_category_color(category: str) -> str:
    """Return color for each category"""
    color_map = {
        "Prediction": "#FF6B6B",      # Red
        "Ad Hominem": "#FF4757",      # Dark Red
        "Fact": "#2ED573",            # Green
        "Opinion": "#FFA726",         # Orange
        "Speculation": "#9C88FF",     # Purple
        "Appeal to Authority": "#45B7D1", # Blue
        "Emotional Appeal": "#F8B500", # Yellow
        "Bias": "#E74C3C",           # Crimson
        "Logical Fallacy": "#8B0000", # Dark Red
        "Evidence": "#2ECC71",        # Emerald
        "Neutral": "#95A5A6"          # Gray
    }
    return color_map.get(category, "#95A5A6")

def create_highlighted_text(sentences_data: List[Dict]) -> str:
    """Create HTML with highlighted sentences and hover tooltips"""
    
    html_parts = []
    
    # Add CSS for tooltips
    css = """
    <style>
    .tooltip {
        position: relative;
        display: inline;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 300px;
        background-color: #333;
        color: white;
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1000;
        bottom: 125%;
        left: 50%;
        margin-left: -150px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.3);
    }
    
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    .sentence {
        padding: 2px 4px;
        border-radius: 3px;
        margin: 1px;
        line-height: 1.6;
    }
    </style>
    """
    
    html_parts.append(css)
    html_parts.append('<div style="font-family: Arial, sans-serif; line-height: 1.8; font-size: 16px;">')
    
    for i, sentence_data in enumerate(sentences_data):
        sentence = sentence_data['text']
        primary_category = sentence_data['primary_category']
        secondary_categories = sentence_data.get('secondary_categories', [])
        explanation = sentence_data['explanation']
        confidence = sentence_data.get('confidence', 'medium')
        
        color = get_category_color(primary_category)
        
        # Create tooltip content
        all_categories = [primary_category] + secondary_categories
        tooltip_content = f"""
        <strong>Primary:</strong> {primary_category}<br/>
        <strong>Secondary:</strong> {', '.join(secondary_categories) if secondary_categories else 'None'}<br/>
        <strong>Confidence:</strong> {confidence}<br/>
        <strong>Analysis:</strong> {explanation}
        """
        
        # Create highlighted sentence with tooltip
        sentence_html = f"""
        <span class="tooltip sentence" style="background-color: {color}; background-color: {color}33;">
            {sentence}
            <span class="tooltiptext">{tooltip_content}</span>
        </span>
        """
        
        html_parts.append(sentence_html)
        html_parts.append(" ")  # Space between sentences
    
    html_parts.append('</div>')
    
    return ''.join(html_parts)

def create_legend() -> str:
    """Create a color legend for categories"""
    categories = [
        "Prediction", "Ad Hominem", "Fact", "Opinion", "Speculation",
        "Appeal to Authority", "Emotional Appeal", "Bias", "Logical Fallacy",
        "Evidence", "Neutral"
    ]
    
    legend_html = '<div style="margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px;">'
    legend_html += '<h4>Color Legend:</h4>'
    legend_html += '<div style="display: flex; flex-wrap: wrap; gap: 10px;">'
    
    for category in categories:
        color = get_category_color(category)
        legend_html += f'''
        <span style="background-color: {color}33; border: 1px solid {color}; 
                     padding: 5px 10px; border-radius: 15px; font-size: 12px;">
            {category}
        </span>
        '''
    
    legend_html += '</div></div>'
    return legend_html

def main():
    st.set_page_config(
        page_title="News Article Analyzer",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 News Article Analyzer")
    st.markdown("*Analyze news articles for rhetorical elements, bias, and logical fallacies*")
    api_key = "fasfasfas"
    # Sidebar for API key
    # with st.sidebar:
    #     st.header("Settings")
        
        
    #     # if not api_key:
    #     #     st.warning("Please enter your OpenAI API key to use the analyzer.")
        
    #     st.markdown("---")
    #     st.markdown("### How to use:")
    #     st.markdown("1. Enter your OpenAI API key")
    #     st.markdown("2. Paste your news article in the text box")
    #     st.markdown("3. Click 'Analyze Article'")
    #     st.markdown("4. Hover over highlighted sentences to see analysis")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Input Article")
        article_text = st.text_area(
            "Paste your news article here:",
            height=400,
            placeholder="Paste the full text of the news article you want to analyze..."
        )
        
        analyze_button = st.button("🔍 Analyze Article", type="primary", disabled=not api_key)
    
    with col2:
        st.header("Analysis Results")
        
        if analyze_button and api_key and article_text.strip():
            with st.spinner("Analyzing article... This may take a moment."):
                analyzer = NewsAnalyzer(api_key)
                analysis = analyzer.analyze_article(article_text)
                
                if "error" in analysis:
                    st.error(f"Analysis failed: {analysis['error']}")
                    if "raw" in analysis:
                        st.text(analysis["raw"])
                else:
                    # Store analysis in session state
                    st.session_state.analysis = analysis
        
        # Display results if available
        if hasattr(st.session_state, 'analysis') and 'sentences' in st.session_state.analysis:
            analysis = st.session_state.analysis
            
            # Show legend
            st.markdown(create_legend(), unsafe_allow_html=True)
            
            # Show highlighted text
            st.subheader("Analyzed Article")
            st.markdown("*Hover over highlighted sentences to see detailed analysis*")
            
            highlighted_html = create_highlighted_text(analysis['sentences'])
            st.markdown(highlighted_html, unsafe_allow_html=True)
            
            # Summary statistics
            st.markdown("---")
            st.subheader("📊 Analysis Summary")
            
            categories = {}
            for sentence in analysis['sentences']:
                primary = sentence['primary_category']
                categories[primary] = categories.get(primary, 0) + 1
                for secondary in sentence.get('secondary_categories', []):
                    categories[secondary] = categories.get(secondary, 0) + 1
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("Total Sentences", len(analysis['sentences']))
            
            with col_stats2:
                most_common = max(categories.items(), key=lambda x: x[1]) if categories else ("None", 0)
                st.metric("Most Common Category", f"{most_common[0]} ({most_common[1]})")
            
            with col_stats3:
                fact_count = categories.get('Fact', 0)
                opinion_count = categories.get('Opinion', 0)
                if fact_count + opinion_count > 0:
                    fact_ratio = fact_count / (fact_count + opinion_count) * 100
                    st.metric("Fact vs Opinion Ratio", f"{fact_ratio:.1f}% Facts")
                else:
                    st.metric("Fact vs Opinion Ratio", "N/A")
            
            # Category breakdown
            st.subheader("Category Breakdown")
            if categories:
                for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                    color = get_category_color(category)
                    st.markdown(
                        f'<span style="background-color: {color}33; border: 1px solid {color}; '
                        f'padding: 2px 8px; border-radius: 10px; margin-right: 10px;">'
                        f'{category}: {count}</span>', 
                        unsafe_allow_html=True
                    )
                    st.write("")  # Add space
        
        elif not api_key:
            st.info("Please enter your OpenAI API key in the sidebar to start analyzing articles.")
        elif not article_text.strip():
            st.info("Please paste a news article in the text box and click 'Analyze Article'.")

if __name__ == "__main__":
    main()



