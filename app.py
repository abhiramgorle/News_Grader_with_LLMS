
from google import genai
from google.genai import types
import pandas as pd
from openai import OpenAI
import json
import streamlit as st
from dotenv import load_dotenv
import os

gemini_api = "AIzaSyDKdCmbgQGJyg2lKBjQWJUShef1uj_5ss8"

load_dotenv()
keymain = os.getenv("OpenAI_api_key")
client = OpenAI(api_key=keymain)

keysub = os.getenv("gemini_api")
gemini_api = keysub
gemini_client = genai.Client(api_key=gemini_api)
def openai_extract_predictions(article_text):
  system_prompt="""You are a News Prediction Extractor, an expert AI assistant designed to analyze textual content from news articles and extract forward-looking statements (predictions about future events, trends, or outcomes). Your primary goal is to identify and summarize speculative or forecasting statements, presenting them in a clear, standalone format for users seeking insights into potential future developments.

Core Guidelines:
1. **Definition of Forward-Looking Statements**: Focus on statements that predict or speculate about future events, such as economic trends, technological advancements, societal shifts, or policy changes. Exclude vague opinions, past events, or unsubstantiated guesses unless tied to a broader, impactful trend.
2. **Clarity and Independence**: Rephrase each prediction as a concise, standalone sentence or two, ensuring it can be understood without additional context from the article. Use direct quotes only when necessary for precision or credibility.
3. **Relevance and Prioritization**: Prioritize predictions with significant societal, economic, or technological impact. Ignore minor or overly speculative statements (e.g., "it might rain tomorrow" unless part of a larger climate trend).
4. **Brevity and Focus**: Limit each prediction summary to 1-2 sentences. If an article contains more than 10 predictions, extract only the 5 most impactful ones, noting how many others were omitted.
5. **Edge Cases**: If no forward-looking statements are found in the provided text, explicitly state, "No predictions or forward-looking statements were identified in the article."

Output Format:
For each task, structure your response as follows:
- Prediction 1: [Clear statement of the prediction]
- Prediction 2: [Clear statement of the prediction]
- ...
If applicable: [Note on omitted predictions or absence of forward-looking statements]

Tone and Style:
- Maintain a neutral, professional tone, focusing on factual summarization without adding personal opinions or speculation beyond what is in the text.
- Ensure responses are concise, logical, and user-friendly, catering to individuals who may use these predictions for decision-making or research.

Adaptability:
- Adjust to varying article lengths and complexities while adhering to the above guidelines.
- If a user provides specific instructions or additional context (e.g., focusing on a particular industry or theme), incorporate those preferences into your analysis while maintaining the core format and quality standards.

Your mission is to deliver accurate, relevant, and well-structured insights into future-oriented content, helping users anticipate trends and outcomes based on news narratives."""
  
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

def openai_verify_prediction(prediction, year_of_article=2023):
    system_prompt = """"You are a News Prediction Evaluator, an expert AI assistant specializing in assessing the accuracy of past news predictions based on current knowledge. Your primary goal is to evaluate whether a given prediction, made at a specified past date, has come true, partially come true, or not come true as of the evaluation date (defaulting to June 2025 unless otherwise specified). Your assessments must be clear, evidence-based, and structured for users seeking reliable analyses of predictive accuracy.

Core Guidelines:
1. **Outcome Categories**: Classify the prediction's accuracy as:
   - Yes: The prediction is fully accurate based on known events or data.
   - No: The prediction is completely inaccurate or did not occur as stated.
   - Partially: Some elements of the prediction are accurate, while others are not.
2. **Justification Standards**: Provide a detailed explanation for the chosen outcome, citing relevant facts, events, or developments up to the evaluation date. If information is incomplete or unavailable, note this limitation explicitly.
3. **Evaluation Criteria**: Consider the prediction's scope (what it claims), timeframe (when it was expected to occur), and original context (circumstances at the time of prediction). If the prediction is vague or ambiguous, state this and evaluate based on the most reasonable interpretation.
4. **Edge Cases**: If a prediction cannot be assessed due to lack of data or other constraints, explain why and provide a tentative outcome if possible, with caveats noted.

Output Format:
For each evaluation, structure your response as:
- Outcome: [Yes/No/Partially]
- Justification: [Detailed explanation]

Tone and Style:
- Maintain a neutral, analytical tone, focusing on factual analysis without personal opinions or speculation beyond the evidence.
- Ensure responses are concise yet comprehensive, balancing detail with clarity for users who may use these evaluations for research or decision-making.

Adaptability:
- Adjust to user-provided evaluation dates or specific instructions (e.g., focusing on particular aspects of a prediction) while adhering to the core guidelines.
- If a user provides additional context or sources for evaluation, incorporate them into your justification while maintaining objectivity.

Your mission is to deliver accurate, well-reasoned, and structured evaluations of news predictions, helping users understand the reliability of past forecasts in light of current realities.
"""
    prompt = f"""
    You are a News Prediction Evaluator tasked with assessing the accuracy of predictions made in {year_of_article}, as of June 2025.

Your task is to evaluate the given prediction and determine whether it came true, partially came true, or did not come true. Provide a clear outcome and a detailed justification based on evidence or reasoning.

Guidelines:
- Outcome must be one of: Yes (fully accurate), No (completely inaccurate), or Partially (some elements accurate, others not).
- Justification should explain the reasoning behind the outcome, citing relevant facts, events, or developments up to June 2025.
- Consider the prediction's scope, timeframe, and original context when evaluating.
- If the prediction is vague or ambiguous, note this in your justification and assess based on the most reasonable interpretation.
- If information is unavailable or incomplete as of June 2025, state this limitation in your analysis.

Format your response in json as follows:
prediction1:
- Outcome: [Yes/No/Partially]
- Justification: [Detailed explanation]
prediction2 : ...

Prediction:
"{prediction}"
    """
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
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

def gemini_verify_predictions(prediction, year_of_article=2023):
  prompt = f"""
    You are a News Prediction Evaluator tasked with assessing the accuracy of predictions made in {year_of_article}, as of June 2025.

Your task is to evaluate the given prediction and determine whether it came true, partially came true, or did not come true. Provide a clear outcome and a detailed justification based on evidence or reasoning.

Guidelines:
- Outcome must be one of: Yes (fully accurate), No (completely inaccurate), or Partially (some elements accurate, others not).
- Justification should explain the reasoning behind the outcome, citing relevant facts, events, or developments up to June 2025.
- Consider the prediction's scope, timeframe, and original context when evaluating.
- If the prediction is vague or ambiguous, note this in your justification and assess based on the most reasonable interpretation.
- If information is unavailable or incomplete as of June 2025, state this limitation in your analysis.

Format your response in json as follows:
prediction1:
- Outcome: [Yes/No/Partially]
- Justification: [Detailed explanation]
prediction2 : ...

Prediction:
"{prediction}"
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

def detect_fallacies(article_text):
    
    prompt = f"""
    You are a Logic and Argumentation Expert tasked with analyzing news articles to identify logical fallacies.

Your task is to carefully read the provided article and identify any logical fallacies present. For each fallacy you find:
- Name the fallacy using standard terminology (e.g., strawman, ad hominem, false dilemma).
- Quote or paraphrase the specific text causing the fallacy.
- Provide a clear explanation of why this text constitutes a logical fallacy.

Only include actual logical fallacies, not opinions or subjective interpretations.

Additional Guidelines:
- Focus on common logical fallacies relevant to argumentation and reasoning.
- Avoid labeling statements as fallacies if they are merely opinions or unsupported claims.

Article:
{article_text}

    Output Format is json in :
    1. Fallacy: [Name]
       Text: "[Quoted or paraphrased section]"
       Explanation: [Why it's a fallacy]
    """
    response = client.chat.completions.create(
       model="gpt-4.1-nano",
        response_format={"type": "json_object"},
         messages=[
                {"role": "user", "content": prompt}
            ],
        temperature=0.3
    )
    # return response.choices[0].message.content
    return json.loads(response.choices[0].message.content)

def grade_news(article_text, fallacies, predictions_analysis):
  prompt = f"""You are an expert in grading news articles by analyzing logical fallacies and evaluating the accuracy of verifiable predictions made in the past.

Your task is to carefully read the provided article {article_text}, review the identified fallacies {fallacies}, and assess the analysis of verifiable predictions {predictions_analysis}. Based on these inputs, assign a grade to the article on a scale of 1 to 5, where 1 indicates poor quality (numerous fallacies or inaccurate predictions) and 5 indicates high quality (minimal fallacies and accurate predictions).

Additional Guidelines:
- Weigh logical fallacies and prediction accuracy equally unless otherwise specified in the inputs.
- Consider the severity and frequency of fallacies: major fallacies (e.g., strawman, false dilemma) or multiple instances lower the grade more than minor or isolated flaws.
- For predictions, assess whether they panned out as described in the analysis; a higher proportion of accurate predictions contributes to a higher grade.
- Only base your grade on the provided inputs; do not incorporate external information or assumptions.
- If the inputs are incomplete or unclear (e.g., missing analysis or article text), note this limitation and provide a tentative grade with an explanation.

Output Format in JSON:
  Grade: [number in range of 1-5],
  Reasoning: [Brief explanation of the grade based on fallacies and predictions]

"""
  response = client.chat.completions.create(
       model="gpt-4.1-nano",
        response_format={"type": "json_object"},
         messages=[
                {"role": "user", "content": prompt}
            ],
        temperature=0.3
    )
    # return response.choices[0].message.content
  return json.loads(response.choices[0].message.content)

def gemini_grade_news(article_text, fallacies, predictions_analysis):
  prompt = f"""You are an expert in grading news articles by analyzing logical fallacies and evaluating the accuracy of verifiable predictions made in the past.

Your task is to carefully read the provided article {article_text}, review the identified fallacies {fallacies}, and assess the analysis of verifiable predictions {predictions_analysis}. Based on these inputs, assign a grade to the article on a scale of 1 to 5, where 1 indicates poor quality (numerous fallacies or inaccurate predictions) and 5 indicates high quality (minimal fallacies and accurate predictions).

Additional Guidelines:
- Weigh logical fallacies and prediction accuracy equally unless otherwise specified in the inputs.
- Consider the severity and frequency of fallacies: major fallacies (e.g., strawman, false dilemma) or multiple instances lower the grade more than minor or isolated flaws.
- For predictions, assess whether they panned out as described in the analysis; a higher proportion of accurate predictions contributes to a higher grade.
- Only base your grade on the provided inputs; do not incorporate external information or assumptions.
- If the inputs are incomplete or unclear (e.g., missing analysis or article text), note this limitation and provide a tentative grade with an explanation.

Output Format in JSON:
  Grade: [number in range of 1-5],
  Reasoning: [Brief explanation of the grade based on fallacies and predictions]

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

articles = ["""Economic worries could cost Biden some of his 2020 supporters, Reuters/Ipsos poll shows
By Jason Lange and Andrea Shalal
August 4, 20236:07 AM EDTUpdated 2 years ago



U.S. President Joe Biden delivers remarks on the economy at Auburn Manufacturing, in Auburn
U.S. President Joe Biden delivers remarks on the economy at Auburn Manufacturing, a company that produces heat- and fire-resistant fabrics for a range of industrial uses in the U.S. and abroad, in Auburn, Maine, U.S. July 28, 2023. REUTERS/Jonathan Ernst/File Photo Purchase Licensing Rights, opens new tab
WASHINGTON, Aug 4 (Reuters) - Many Americans who voted for U.S. President Joe Biden in 2020 say they believe the economy has faired poorly under his stewardship and that they might not vote for him in the 2024 election, according to a new Reuters/Ipsos poll.
Biden, a Democrat who in 2020 defeated former Republican President Donald Trump, could be on track for a rematch next year against his old foe, who leads the Republican nomination contest and was due in court Thursday to face a third criminal indictment.
The Reuters Tariff Watch newsletter is your daily guide to the latest global trade and tariff news. Sign up here.

Poll respondents were asked which of the two they would vote for "if the election for president were held today," and 19% of Biden's 2020 voters participating in the poll said they either weren't sure or would pick someone other than Biden or Trump. Six percent of Biden's 2020 voters picked Trump.
Forty-two percent of Biden's 2020 voters in the poll said the economy was "worse" than it was in 2020, compared to 33% who said it was "better" and 24% who said it was "about the same."
The poll results underscore a dynamic White House officials are fighting to reverse with visits this month to towns across the country to raise awareness of Biden's efforts to help the economy. Biden himself will visit Arizona, New Mexico and Utah from Aug. 7-10 to hail how his Inflation Reduction Act law will boost manufacturing and fight climate change.

About half the respondents in the poll who voted for Biden in 2020 said they have heard little or nothing of his major policy initiatives to reduce inflation or boost spending on infrastructure.
Most of Biden's 2020 voters in the poll - 78% - said they approved of his performance as president, well above his 40% approval rating among all respondents.
But a critical division exists among Biden's 2020 voters.
Among those who don't approve of his performance, respondents were roughly twice as likely to say inflation was America's biggest problem, compared with the 2020 Biden voters who said the president is doing a good job. Those who approve of Biden put relatively more emphasis on the dangers of political extremism.
"The difference is the focus on inflation and the cost of living," said Chris Jackson, a public opinion researcher at polling firm Ipsos, which conducts polls for Reuters and other media organizations. He added there was still time for Biden to court his wayward supporters.

US-China trade talks going well, says Lutnick

"They are not off the board. They materially need to be feeling better and they have to think Biden is doing something," Jackson said.
The poll results point to a lack of enthusiasm for Biden and highlights the risks that Democrats might not turn out as strongly at the polls next year and that independents who voted Democratic in 2020 could vote Republican in 2024.
U.S. households have struggled with the most severe inflation in decades under Biden, though the rate of price increases has recently fallen sharply. Economic growth has been modest, though the unemployment rate has fallen to its lowest levels in decades.
Biden's campaign did not respond to a request for comment.
One administration official said measures of consumer sentiment pointed to increased optimism, as has previous Reuters/Ipsos polling showing that people expecting their own finances to improve outnumber those who expect them to deteriorate.

Another administration official, however, said last week that the White House was keenly aware that officials must do a better job of selling Biden's policy successes.
When poll respondents were asked to describe how much Biden and his administration have invested in the U.S. economy, 35% of Biden's 2020 voters selected the option "not enough." Some 53% picked "about the right amount," 3% said "too much" and the rest didn't know.
The poll was conducted online nationwide, gathering responses from 2,009 U.S. adults with a credibility interval of about three percentage points."""
,"""
Exclusive news, data and analytics for financial market professionals
World

WASHINGTON, Aug 16 (Reuters) - Blame it on economic theory not matching reality, groupthink among forecasters, or political partisanship by opponents of the Biden administration, but a year ago much of the U.S. was convinced the country was in a recession, or would be soon.
The first two quarters of 2022 had seen U.S. economic output contract at a 1.6% annual rate from January through March and at a 0.6% annual rate from April through June. By one common, though not technically accurate, definition the country had already entered a downturn.
Get a look at the day ahead in U.S. and global markets with the Morning Bid U.S. newsletter. Sign up here.
Advertisement · Scroll to continue


Why wouldn't it? The Federal Reserve was quickly cranking interest rates higher, housing investment seemed to be buckling, and the conventional wisdom was that other industries, consumer spending, and the job market would all tumble as well.
"A number of forces have coincided to slow economic momentum more rapidly than we previously expected," Michael Gapen, chief U.S. economist at Bank of America, said in a July 2022 analysis. "We now forecast a mild recession in the U.S. economy this year ... In addition to fading of prior fiscal support ... inflation shocks have eaten into real spending power of households more forcefully than we forecasted previously and financial conditions have tightened noticeably as the Fed shifted its tone toward more rapid increases in its policy rate."
Advertisement · Scroll to continue
Fast forward a year, and the unemployment rate at 3.5% in July is actually lower than the point where many analysts expected it to begin rising, consumers continue to spend, and many professional economic forecasts have followed Gapen in a course correction.
Reuters polls of economists over the past year showed the risk of a recession one year out rising from 25% in April 2022, the month after the first rate hike of the Fed's current tightening cycle, to 65% in October. The most recent read: 55%.
"Incoming data has made us reassess our prior view" of a coming recession that had already been pushed into 2024, Gapen wrote earlier this month. "We revise our outlook in favor of a 'soft landing' where growth falls below trend in 2024, but remains positive throughout."
Reuters Graphics
The recession revisionists include the Fed's own staff, who followed their models to steadily downgrade the outlook for the U.S., moving from increased concerns about "downside risk" as of last autumn, to citing recession as a "plausible" outcome as of last December, and then projecting as of the Fed's March 2023 meeting that recession would begin this year.

With the failure of California-based Silicon Valley Bank expected to put an extra constraint on bank credit, "the staff's projection ... included a mild recession starting later this year, with a recovery over the subsequent two years," the minutes of the Fed's March 21-22 meeting showed.
In May and June, the Fed staff projections "continued to assume" the U.S. economy would be in recession by the end of the year.
The more dour outlook disappeared at the July 25-26 meeting, according to newly released minutes.
"The staff no longer judged that the economy would enter a mild recession toward the end of the year," the minutes said, though the staff still felt the economy would slow to a growth rate below its long-run potential in 2024 and 2025, with inflation falling and risks "tilted to the downside."
Fed policymakers' projections, which are issued on a quarterly basis and are independent of the staff view, never showed GDP contracting on an annual basis.
The U.S. Federal Reserve building is pictured in Washington, March 18, 2008. REUTERS/Jason Reed/File Photo Purchase Licensing Rights, opens new tab
'CHUGGING ALONG'
What made the difference between an in-the-moment recession that many thought was underway last year to growth that has surprised to the upside?
The forecast miss wasn't even really close: By the third quarter of last year, growth had rebounded to a rapid 3.2% annual rate, and has remained at 2% or above since then, higher than the 1.8% the Fed considers as the economy's underlying potential. An Atlanta Fed GDP "nowcast" puts output growth for the current July-September period at a startling 5.8%, showing continued strong momentum from consumption and a surprising bounce in industrial production and home starts.
Reuters Graphics
A big part of the story is the staying power of U.S. consumers, who have continued "chugging along" and spending more than expected, as Omair Sharif, president of Inflation Insights, puts it.
Spending has shifted from the goods-gorging purchases seen at the start of the coronavirus pandemic to the hot services economy that exploded this summer in billion-dollar movie runs and music concerts.
But the dollar amounts keep growing regardless of what's in the basket, leaving economists to steadily push back the date when the "excess savings" of the pandemic era will run dry, or puzzle over whether low unemployment, ongoing strong hiring and labor "hoarding" by companies, along with rising earnings, have trumped any anxiety over the outlook.
But it isn't just that.
It may be that high interest rates don't work the same way in an economy that spends more on less rate-sensitive services, and where businesses have continued to borrow and invest more than many economists anticipated - perhaps to capitalize on regulatory shifts aimed at encouraging technology and green energy projects.
A surge in local government spending also gave an unexpected boost to growth as localities put pandemic-era funds to work on a delayed basis.
Can it last?
One risk is if inflation resurges alongside a tighter-than-expected economy, and Fed policy needs to become even stricter and induce the inflation-killing downturn that officials still hope to avoid.
But the odds of that may be falling.
"We've been wavering for a while on whether to shift to the 'soft-landing' camp, but no longer," noted Sal Guatieri, a senior economist at BMO Capital Markets, in reference to the Fed's hopes of lowering inflation without provoking a recession.
"Broad strength" in the U.S. economy, he said, "convinced us that the U.S. economy is more durable than expected ... Not only is it not slowing further, it might be picking up."
""",
"""Many CEOs, investors and economists had penciled in 2023 as the year when a recession would hit the American economy.

The thinking was that the US economy would grind to a halt because the Federal Reserve was effectively slamming the brakes to squash inflation. Businesses would lay off workers and inflation-weary Americans would slash spending.

But the case for a 2023 US recession is crumbling for a simple reason: America’s jobs market is way too strong.

Hiring unexpectedly accelerated again last month, with employers adding an impressive 339,000 jobs in May. Not only is that more than any major forecaster expected, but it’s more jobs than the US economy added in any single month in 2019, a very strong year for the jobs market.

“This economy is incredibly resilient, despite all the slings and arrows – despite the banking crisis, rate hikes, the debt ceiling,” Mark Zandi, chief economist at Moody’s Analytics, told CNN in a phone interview on Friday.

Zandi is growing more confident that 2023 won’t be the year when a downturn will begin.

“For this year, given these jobs numbers, it’s hard to see a recession. Increasingly, the odds of a recession this year are fading,” Zandi said. “A lot of economists who have called for a recession are now in the uncomfortable position of pushing back the start date.”

Although it’s possible, things would have to deteriorate very quickly in the economy, and the jobs market specifically, for a downturn to start this year.

“We’re running out of time for a 2023 recession,” Justin Wolfers, an economics professor at the University of Michigan, told CNN. “We’ve never had a recession when the labor market was running this hot. In fact, it would be absurd to use r-word at a time when we’re creating jobs at this rate.”

Not only did nonfarm payrolls soar by 339,000 jobs in May, but the government revised the prior two months of job growth significantly higher, too. Now the Bureau of Labor Statistics says payrolls increased by 217,000 jobs in March and 294,000 in April.

That’s miles away from the dark predictions issued not long ago. Last fall, Bank of America warned payrolls would begin shrinking in early 2023, translating to the loss of about 175,000 jobs a month during the first quarter followed by job losses through much of the year.

Conflicting signals
Some companies are indeed cutting jobs, especially in the tech and media industries.

The number of announced job cuts has quadrupled so far this year, according to Challenger, Gray & Christmas. But the economic indicators suggest many people who are laid off are quickly getting rehired.

Friday’s jobs report did offer some conflicting signals, especially in the household survey, which economists put less weight on because it tends to be noisier.

The household survey showed the unemployment rate, which had been tied for a 53-year low, jumped by 0.3 percentage points – the most since April 2020 – as employment fell sharply.

Yet Wolfers noted the three-month moving average for the unemployment remains extremely low at 3.5%. He described the jobs market as “really freaking good” and said the latest report further disputes the notion that the US economy is already in recession – a belief many Americans have. (In a May CNN poll, 76% of respondents described the economy as in poor shape).

“We are not in a recession. People have been telling us we’re in a recession for the last two years. They’ve been wrong each and every day,” Wolfers said. “Employment has grown gangbusters. The data is crystal clear on this. There is no recession.”

What could change
Of course, it’s possible that something happens to change that story in the coming months. And there is a significant risk of a recession in the medium-term as well as growing evidence that consumers are feeling real financial pain following two years of high inflation.

Dollar General slashed its forecast for the year and warned customers are being forced to “rely more on food banks, savings and credit cards.” Macy’s blamed slowing customer demand for cutting its own forecast. Federal Reserve researchers have found that auto loan delinquencies are rising, surpassing pre-Covid levels.

The other problem is the Fed’s war on inflation is hitting the economy with a lag. That means the full effect of the most aggressive interest rate hikes in four decades may not have been felt yet. This raises the risk the Fed overdoes it – or already has.

Zandi sees a one in three chance of a recession this year, but that rises to “uncomfortably high” odds of 50/50 in 2024.

Still, there is nothing about the latest jobs reports that signal an ongoing or imminent recession.

“As long as the economy continues to produce above 200,000 jobs per month this economy simply is not going to slip into recession,” Joe Brusuelas, chief economist at RSM, wrote in a report.

Morgan Stanley seems to agree, telling clients that the May jobs report “continues to point to a soft landing for the economy,” a Fed term for raising rates without triggering a recession.

Wolfers, the University of Michigan professor, said the risk of a hard landing “looks quite remote.”

If anything, the hot jobs market keeps alive a no-landing scenario: The economy grows so rapidly that the Fed has to slam the brakes even harder, risking a recession. But that would take time to play out, making it a problem for 2024.
""",
"""Is China's economy a 'ticking time bomb'?
30 August 2023

Share

Save
Nick Marsh
Asia business correspondent
Getty Images A worker welds at a temperature control equipment manufacturing enterprise in Qingzhou Economic Development Zone, East China's Shandong province.Getty Images
China's post-Covid recovery has been slow
The past six months has brought a stream of bad news for China's economy: slow growth, record youth unemployment, low foreign investment, weak exports and currency, and a property sector in crisis.

US President Joe Biden described the world's second-largest economy as "a ticking time bomb", predicting growing discontent in the country.

China's leader Xi Jinping hit back, defending the "strong resilience, tremendous potential and great vitality" of the economy.

So who is right - Mr Biden or Mr Xi? As is often the case, the answer probably lies somewhere in between.

While the economy is unlikely to implode any time soon, China faces huge, deep-rooted challenges.


A property crisis and poorer households
Central to China's economic problems is its property market. Until recently, real estate accounted for a third of its entire wealth.

"This made no sense. No sense at all," says Antonio Fatas, professor of economics at the business school INSEAD in Singapore.

For two decades, the sector boomed as developers rode a wave of privatisation. But crisis struck in 2020. A global pandemic and a shrinking population at home are not good ingredients for a programme of relentless housebuilding.

The government, fearing a US-style 2008 meltdown, then put limits on how much developers could borrow. Soon they owed billions they could not pay back.

Now demand for houses has slumped and property prices have plunged. This has made Chinese homeowners - emerging from three years of tough coronavirus restrictions - poorer.


"In China, property is effectively your savings," says Alicia Garcia-Herrero, chief Asia economist at wealth management firm Natixis. "Until recently, it seemed better than putting your money into the crazy stock market or a bank account with low interest rates"

It means that, unlike in Western countries, there has been no post-pandemic spending boom or major economic bounce back.

Getty Images An unfinished five-star hotel is seen in Huai 'an City, Anhui Province, China, 20 February 2023.Getty Images
China's crisis-hit property market is weighing on the world's second-largest economy
"There was this notion that Chinese people would spend like crazy after zero-Covid," Ms Garcia-Herrero says. "They'd travel, go to Paris, buy the Eiffel Tower. But actually they knew their savings were getting hammered by the fall in house prices, so they've decided to keep hold of what cash they have."

Not only has this situation made households feel poorer, it has worsened the debt problems faced by the country's local governments.


It is estimated that more than a third of their multi-billion dollar revenues come from selling land to developers, which are now in crisis.

According to some economists, it will take years for this property pain to subside.

A flawed economic model
The property crisis also highlights problems in the way China's economy functions.

The country's astonishing growth in the past 30 years was propelled by building: everything from roads, bridges and train lines to factories, airports and houses. It is the responsibility of local governments to carry this out.

However, some economists argue this approach is starting to run out of road, figuratively and literally.


One of the more bizarre examples of China's addiction to building can be found in Yunnan province, near the border with Myanmar. This year, officials there bafflingly confirmed they would go ahead with plans to build a new multi-million dollar Covid-19 quarantine facility.

Heavily indebted local governments are under so much pressure that this year some were reportedly found to be selling land to themselves to fund building programmes.

The bottom line is that there is only so much China can build before it starts becoming a waste of money. The country needs to find another way of generating prosperity for its people.

"We're at an inflection point," Professor Fatas says. "The old model is not working, but in order to change focus you need serious structural and institutional reforms."

For example, he argues, if China wanted a financial sector to fire up its economy and rival the US or Europe, the government would first need to loosen regulation considerably, ceding large amounts of power to private interests.


In reality, the opposite has happened. The Chinese government has tightened its grip on the finance sector, scolded "westernised" bankers for their hedonism and cracked down on big technology firms like Alibaba.

One way this has been reflected is in youth unemployment. Across China, millions of well-educated graduates are struggling to find decent white-collar jobs in urban areas.

Getty Images China's exports fell sharply in JulyGetty Images
China's exports fell sharply in July
In July, figures showed a record 21.3% of jobseekers between the ages of 16 and 25 were out of work. The following month, officials announced they would stop publishing the figures.

According to Professor Fatas, it is testament to a "rigid, centralised economy" struggling to absorb such a high number of people into the labour force.


A top-down system is effective when you want to build a new bridge, but looks cumbersome when the bridge has already been built and people are still looking for work.

What will the government do now?
A change of economic direction requires a change of political ideology. Judging by the Chinese Communist Party's (CCP) tightening grip on life recently and President Xi's tightening grip on the CCP, this doesn't look likely. The leadership might argue it is not even necessary.

In some ways, China is a victim of its own success. The current rate of growth is only considered "slow" when you compare it with the staggeringly high numbers of previous years.

Since 1989, China has averaged a growth rate of around 9% per year. In 2023, that figure is predicted to be around 4.5%.

It is a big drop off, but still much higher than the economies of the US, the UK and most European countries. Some have argued that this suits China's leadership just fine.


Western economies tend to be powered by people spending, but Beijing is wary of this consumerist model. Not only is it deemed wasteful, it is also individualistic.

Empowering consumers to buy a new TV, subscribe to streaming services or go on holiday may help stimulate the economy, but it does little for China's national security or its competition with the US.

Essentially, Mr Xi wants growth, but not for the sake of it. This may be behind the recent boom in cutting-edge industries, such as semiconductors, artificial intelligence and green technology - all of which keep China globally competitive and make it less reliant on others.

This idea might also explain the government's limited response to the faltering economy. So far it has only tweaked around the edges - easing borrowing limits or shaving a fraction off interest rates - rather than pumping in large amounts of money.

Foreign investors in China are worried and want the government to take action quickly, but those in charge seem to be playing the long game.


They know that, on paper, China still has massive potential for more growth. It may be an economic powerhouse, but average annual income is still only $12,850. Almost 40% of people still live in rural areas.

Getty Images Chinese President Xi Jinping waves s at The Great Hall of People on October 23, 2022 in BeijingGetty Images
Xi Jinping secured a historic third term as the country's leader earlier this year
So on the one hand, not being tied to election cycles has allowed and will allow China the luxury of taking such a long-term view.

But on the other, many economists argue that an authoritarian political system is not compatible with the kind of flexible, open economy needed for living standards matching those in officially "high-income" countries.

There could be a danger that Mr Xi is prioritising ideology over effective governance, or control over pragmatism.


For most people, this is fine when the economy is doing well. But as China comes out of three years of zero-Covid, with many struggling to find a job and family homes plunging in value, it is a different story.

This takes us back to Mr Biden's "ticking time bomb" description, which suggests civil unrest or, even more seriously, some kind of dangerous foreign policy action in response to it.

At the moment, though, that is pure speculation. China has emerged from any number of crises in the past. But there is no doubt that the country's leadership is now facing a unique set of challenges.

"Are they worried about the current situation? Of course, they see the numbers," Professor Fatas says.

"Do they understand what needs to be done? I'm not sure. My guess is they're missing certain things that are fundamental for the future of China."
""",
"""Unemployment, GDP and inflation data will soon show us more about UK economy
9 September 2023

Share

Save
By Faisal Islam profile image
Faisal Islam
Economics editor•@faisalislam
Getty Images Office workers discuss somethingGetty Images
So far 2023 has seen some false dawns for the UK economy. The next few weeks' data are critical.

Recession has been avoided but growth has bumped along the bottom.

And even as inflation falls from the double-digit levels of a year ago, it has proven more stubborn and sticky, and spread to the service sector.

The ONS's recent huge revision of historical growth changes the picture of the immediate post-pandemic recovery, especially relative to other European countries.

But a broader reassessment of UK prospects may have to wait for news in the coming weeks.


Data released in September could show whether the crises of the past three years are being put firmly behind us.

Expectations within government are for the rollercoaster ride to continue for the next few weeks at least.

Unemployment might tick up again when new figures are released on Tuesday. However, the UK should finally return to a situation where earnings are growing by more than the rise in the cost of living too.

The economy (GDP) could also have shrunk a little in July - we'll find out on Wednesday.

Rising fuel prices in August are likely to lead to a blip in the latest inflation numbers, released the following Wednesday, according to both Chancellor Jeremy Hunt and Bank of England governor Andrew Bailey.


All of this will feed into the Bank of England's interest rate decision in a fortnight.

A rate rise had been expected, but recent hints have suggested the Bank may prefer to keep rates at current levels for longer.

Against this backdrop, the Office for Budget Responsibility (OBR) is plugging the latest data into its forecasts to be published in November, alongside the Autumn Statement.

On the face of it, higher wages are pushing up the tax take, meaning that this year's borrowing numbers are coming in less than originally forecast.

However, there is more red ink pouring into the projections. At the Budget forecast in March, the peak in Bank of England rates was expected to be 4.3%. It is already 5.25%.


Ten-year UK borrowing rates were forecast to be an average of 3.6% in March, and they reached 4.8% last month.

The OBR already stated at the Budget that a one percentage point rise in borrowing costs would increase borrowing by £20bn in 2027-28, "wiping out headroom" in its forecast.

When the OBR points out that the Treasury is not on course to meet its self-imposed constraints on borrowing, that can result in pressure for tax rises or spending cuts.

Right now the political conversation is about the opposite - pre-election tax cuts, or more spending on, for example, school repairs.

For the chancellor, this autumn should help settle Britain on a stable, steady economic trajectory.


It will not be spectacular, but it will be a world away from last year's shambles under his predecessor.

Inflation should continue to fall, down to 3% in a year's time. The UK will stay in a respectable middle lane of growth in the major G7 economies.

The Treasury's main medium-term policy focus will be acknowledging and trying to deal with the UK's relatively poor record on business investment.

The Budget contained a suite of measures designed to help ease the labour supply problem.

The Autumn Statement will be about this business investment challenge. The Treasury thinks it explains a quarter of the UK's productivity underperformance with other major economies.


The prize, if the UK was as productive as Germany, for example, would be an increase in GDP per head of £6,000.

But households are very much not out of the woods.

Even a declining headline rate of inflation, and rising average earnings, will not mask increasing pain as rising interest rates hit homeowners and renters.

The ONS consumer habits survey shows the bulk of people still spending more than usual on food shopping, buying less, and noticing less variety on the shelves.

Supermarkets notice hundreds of thousands of home meals, replacing eating out.


Banks notice mortgage holders who used to shop at the priciest of supermarkets switching to discount retailers.

By the end of the month the Bank of England could give a definitive steer that interest rates have peaked at 5.5%, albeit at the cost of their staying at such a level for the next year or so.

Industry is confident that high stocks of gas, and the ability to reduce demand, mean the whole of Europe should be resilient to any further energy market disruptions.

But the combination of some further stoppage in gas tanker trade and a very cold winter still has the capacity to create a nasty inflationary surprise in the new year.

A path to a more normal economic situation could emerge soon. The data about to be released should give some big clues.
"""]
#https://www.reuters.com/world/us/economic-worries-could-cost-biden-some-his-2020-supporters-reutersipsos-2023-08-04/
#https://www.reuters.com/markets/us/with-gallic-shrug-fed-bids-adieu-recession-that-wasnt-2023-08-16/
#https://www.cnn.com/2023/06/05/economy/recession-chances
#https://www.bbc.com/news/business-66636403
#https://www.bbc.com/news/business-66755407


#streamlit app
def main():
    st.title("Grade Your News Article")
    st.write("This is a small prototype for building LLMs that can “grade the news” by ")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;(a) identifying logical fallacies contained in individual news stories, and ")
    st.write("&nbsp;&nbsp;&nbsp;&nbsp;(b) identifying verifiable predictions that have been made in the past and then checking whether they panned out.")

    news_article = st.text_area("Paste the news article here", height=200)
    type_of_llm = st.selectbox("Select type of LLM you want to use", ["OpenAI GPT-4", "Google Gemini"])
    year_of_article = st.selectbox("Select the year of the article", ["2024","2023", "2022", "2021", "2020"])

    if type_of_llm == "OpenAI GPT-4":
        try :
            if st.button("grade the news article"):
                with st.spinner("Processing..."):
                    
                    analz = []
                    jsosn = openai_extract_predictions(news_article)
                    
                    somethi = openai_verify_prediction(jsosn['predictions'],year_of_article)
                    st.write("Predictions and Justifications done ✅ ")
                    fallacies_detected = detect_fallacies(news_article)
                    st.write("Fallacies Detected✅ ")
                    for j in range(max(len(jsosn["predictions"]), len(somethi))):
                        pred_text = jsosn["predictions"][j] if j < len(jsosn["predictions"]) else "(Missing prediction)"
                        pred_key = f'prediction{j+1}'

                        if pred_key in somethi:
                            outcome = somethi[pred_key].get('Outcome', 'N/A')
                            justification = somethi[pred_key].get('Justification', 'N/A')
                        else:
                            outcome = 'Missing'
                            justification = 'No evaluation provided'

                        analz.append({
                            "Prediction": pred_text,
                            "Outcome": outcome,
                            "Justification": justification
                        })
                    grade = grade_news(news_article,fallacies_detected["fallacies"],analz)
                    grade_num = grade["Grade"]
                    st.success(f"Grade: {grade_num} /5")
                    df = pd.DataFrame(analz, columns=["Prediction", "Outcome", "Justification"])
                    
                    st.write("Predictions and Outcomes:")
                    st.dataframe(df, use_container_width=False)
                    st.write("Fallacies Detected:")
                    st.dataframe(fallacies_detected["fallacies"], use_container_width=False)
        except Exception as e:
            st.error(f"Error grading the news article: {e}")
    elif type_of_llm == "Google Gemini":
        
        try :

            if st.button("grade the news article"):
                with st.spinner("Processing..."):
                    
                    analz = []
                    jsosn = gemini_extract_predictions(news_article)
                    
                    somethi = gemini_verify_predictions(jsosn['predictions'],year_of_article)
                    st.write("Predictions and Justifications done ✅ ")
                    fallacies_detected = detect_fallacies(news_article)
                    st.write("Fallacies Detected✅ ")
                    for j in range(max(len(jsosn["predictions"]), len(somethi))):
                        pred_text = jsosn["predictions"][j] if j < len(jsosn["predictions"]) else "(Missing prediction)"
                        pred_key = f'prediction{j+1}'

                        if pred_key in somethi:
                            outcome = somethi[pred_key].get('Outcome', 'N/A')
                            justification = somethi[pred_key].get('Justification', 'N/A')
                        else:
                            outcome = 'Missing'
                            justification = 'No evaluation provided'

                        analz.append({
                            "Prediction": pred_text,
                            "Outcome": outcome,
                            "Justification": justification
                        })
                    grade = gemini_grade_news(news_article,fallacies_detected["fallacies"],analz)
                    grade_num = grade["Grade"]
                    st.success(f"Grade: {grade_num} /5")
                    df = pd.DataFrame(analz, columns=["Prediction", "Outcome", "Justification"])
                    
                    st.write("Predictions and Outcomes:")
                    st.dataframe(df, use_container_width=False)
                    st.write("Fallacies Detected:")
                    st.dataframe(fallacies_detected["fallacies"], use_container_width=False)
        except Exception as e:
            st.error(f"Error grading the news article: {e}")

if __name__ == "__main__":
    main()
