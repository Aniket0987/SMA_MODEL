import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os

def train_model(data_path=None):
    # Download NLTK data
    nltk.download('vader_lexicon')

    # Initialize Sentiment Analyzer
    sia = SentimentIntensityAnalyzer()

    # Example input data with sentiment labels (use this if no data path provided)
    default_data = data = [
    {
            "title": "Sensex Drops on Weak Global Cues",
            "description": "Weak global cues led to a drop in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Falls Amid Currency Depreciation",
            "description": "Currency depreciation caused Nifty to fall.",
            "sentiment": "negative"
        },
        {
            "title": "Market Sentiment Dampens on Rising Inflation",
            "description": "Rising inflation dampened market sentiment.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Stocks Tumble on Poor Quarterly Earnings",
            "description": "Poor quarterly earnings led to a tumble in tech stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Oil Sector Underperforms Amid Geopolitical Tensions",
            "description": "Geopolitical tensions caused underperformance in the oil sector.",
            "sentiment": "negative"
        },
        {
            "title": "Banking Stocks Decline on Regulatory Concerns",
            "description": "Regulatory concerns led to a decline in banking stocks.",
            "sentiment": "negative"
        },
        {
            "title": "FII Outflows Cause Broad Market Sell-Off",
            "description": "Outflows from Foreign Institutional Investors caused a broad market sell-off.",
            "sentiment": "negative"
        },
        {
            "title": "Auto Sector Hit by Emission Standards",
            "description": "Stricter emission standards hit the auto sector hard.",
            "sentiment": "negative"
        },
        {
            "title": "Political Uncertainty Weighs on Market",
            "description": "Political uncertainty weighed heavily on market performance.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Slides on Weak Consumer Confidence",
            "description": "Weak consumer confidence led to a slide in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Pharma Stocks Decline on Regulatory Setbacks",
            "description": "Regulatory setbacks caused a decline in pharma stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Market Volatility Spikes on Trade Disputes",
            "description": "Trade disputes led to a spike in market volatility.",
            "sentiment": "negative"
        },
        {
            "title": "Real Estate Stocks Suffer Amid Construction Slowdown",
            "description": "A slowdown in construction activity hurt real estate stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Ends Lower on Commodity Market Weakness",
            "description": "Weakness in commodity markets led to a lower close for the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Rising Fuel Costs Drag Down Transportation Stocks",
            "description": "Increased fuel costs dragged down transportation stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Sector Dips on Data Breach Concerns",
            "description": "Concerns over data breaches caused a dip in the tech sector.",
            "sentiment": "negative"
        },
        {
            "title": "Metal Stocks Underperform Amid Falling Prices",
            "description": "Falling metal prices led to underperformance in the metal sector.",
            "sentiment": "negative"
        },
        {
            "title": "Market Dips on Interest Rate Hike Fears",
            "description": "Fears of an interest rate hike caused the market to dip.",
            "sentiment": "negative"
        },
        {
            "title": "Financial Sector Pressured by High NPA Levels",
            "description": "High levels of Non-Performing Assets pressured the financial sector.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Drops on Subdued Economic Data",
            "description": "Subdued economic data caused a drop in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Market Falls on Global Growth Concerns",
            "description": "Concerns over global economic growth led to a market fall.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Declines Amid Weakening Rupee",
            "description": "A weakening rupee led to a decline in Nifty.",
            "sentiment": "negative"
        },
        {
            "title": "Rising Raw Material Costs Hit Manufacturing Stocks",
            "description": "Manufacturing stocks were hit by rising raw material costs.",
            "sentiment": "negative"
        },
        {
            "title": "IT Stocks Drop on Weak Global Demand",
            "description": "Weak global demand caused a drop in IT stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Oil Prices Surge, Market Reacts Negatively",
            "description": "A surge in oil prices led to a negative market reaction.",
            "sentiment": "negative"
        },
        {
            "title": "Banking Sector Sees Sell-Off Amid Loan Concerns",
            "description": "Concerns over loan defaults led to a sell-off in the banking sector.",
            "sentiment": "negative"
        },
        {
            "title": "Foreign Fund Outflows Weaken Market",
            "description": "Outflows of foreign funds caused market weakness.",
            "sentiment": "negative"
        },
        {
            "title": "Auto Sector Struggles with Chip Shortages",
            "description": "Chip shortages caused significant struggles in the auto sector.",
            "sentiment": "negative"
        },
        {
            "title": "Political Instability Causes Market Decline",
            "description": "Market decline was caused by political instability.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Falls on Weak Export Performance",
            "description": "Weak export performance caused a fall in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Pharma Sector Underperforms on FDA Warnings",
            "description": "FDA warnings led to underperformance in the pharma sector.",
            "sentiment": "negative"
        },
        {
            "title": "Market Volatility Rises on Economic Uncertainty",
            "description": "Economic uncertainty led to a rise in market volatility.",
            "sentiment": "negative"
        },
        {
            "title": "Real Estate Stocks Decline on Weak Sales Data",
            "description": "Weak sales data caused a decline in real estate stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Ends Lower on Commodity Price Decline",
            "description": "A decline in commodity prices led to a lower close for the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Rising Fuel Prices Hurt Aviation Sector Stocks",
            "description": "Aviation sector stocks were hurt by rising fuel prices.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Stocks Dip on Cybersecurity Concerns",
            "description": "Cybersecurity concerns caused a dip in tech stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Metal Stocks Slump on Weak Global Demand",
            "description": "Weak global demand led to a slump in metal stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Market Dips on Worries Over Policy Tightening",
            "description": "Worries over policy tightening led to a market dip.",
            "sentiment": "negative"
        },
        {
            "title": "Financial Stocks Fall on Rising Bad Loans",
            "description": "Rising bad loans caused a fall in financial stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Drops Amid Global Market Turbulence",
            "description": "Global market turbulence caused a drop in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Hits Record High on Economic Optimism",
            "description": "Nifty hit a record high on rising economic optimism.",
            "sentiment": "positive"
        },
        {
            "title": "Market Soars on Positive Government Reforms",
            "description": "Positive government reforms led to a soaring market.",
            "sentiment": "positive"
        },
        {
            "title": "RBI's Policy Measures Fuel Market Rally",
            "description": "Policy measures by the RBI fueled a market rally.",
            "sentiment": "positive"
        },
        {
            "title": "Pharma Stocks Gain on New Drug Approvals",
            "description": "New drug approvals led to gains in pharma stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Market Continues Uptrend on Strong Corporate Results",
            "description": "Strong corporate results continued to support the market uptrend.",
            "sentiment": "positive"
        },
        {
            "title": "IT Sector Rises Amid Increased Tech Spending",
            "description": "Increased tech spending led to a rise in the IT sector.",
            "sentiment": "positive"
        },
        {
            "title": "Banking Stocks Rally on Improved Credit Growth",
            "description": "Improved credit growth led to a rally in banking stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Sector Sees Gains on New Model Launches",
            "description": "New model launches led to gains in the auto sector.",
            "sentiment": "positive"
        },
        {
            "title": "Renewables Sector Surges on Government Incentives",
            "description": "Government incentives led to a surge in the renewables sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Rallies on Upbeat GDP Forecasts",
            "description": "Upbeat GDP forecasts led to a market rally.",
            "sentiment": "positive"
        },
        {
            "title": "Consumer Goods Stocks Gain on Rising Disposable Income",
            "description": "Rising disposable income led to gains in consumer goods stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Infrastructure Stocks Climb on New Project Announcements",
            "description": "New project announcements led to gains in infrastructure stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Market Gains on Positive Business Sentiment",
            "description": "Positive business sentiment led to market gains.",
            "sentiment": "positive"
        },
        {
            "title": "Healthcare Stocks Rise on Innovative Treatments",
            "description": "Innovative treatments led to gains in healthcare stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Tech Stocks Gain on Strong Earnings Forecasts",
            "description": "Strong earnings forecasts led to gains in tech stocks.",
            "sentiment": "positive"
        },
        {
            "title": "FMCG Sector Benefits from Rural Market Expansion",
            "description": "Expansion in rural markets benefited the FMCG sector.",
            "sentiment": "positive"
        },
        {
            "title": "Real Estate Sector Gains on Lower Interest Rates",
            "description": "Lower interest rates led to gains in the real estate sector.",
            "sentiment": "positive"
        },
        {
            "title": "Positive Market Sentiment on Global Trade Deals",
            "description": "Global trade deals led to positive market sentiment.",
            "sentiment": "positive"
        },
        {
            "title": "Financial Stocks Strengthen on Solid Loan Growth",
            "description": "Solid loan growth led to strengthening in financial stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Energy Sector Rises on Increased Renewables Adoption",
            "description": "Increased adoption of renewable energy sources led to gains in the energy sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Ends Higher on Robust Domestic Growth",
            "description": "Robust domestic growth led to a higher market close.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Stocks Climb on Strong Export Orders",
            "description": "Strong export orders led to gains in auto stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Renewables Sector Gains on International Investments",
            "description": "International investments led to gains in the renewables sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Rises on Positive Industrial Production Data",
            "description": "Positive industrial production data led to market gains.",
            "sentiment": "positive"
        },
        {
            "title": "Consumer Goods Stocks Up on Improved Sales",
            "description": "Improved sales led to gains in consumer goods stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Infrastructure Sector Gains on Increased Government Spending",
            "description": "Increased government spending on infrastructure led to sector gains.",
            "sentiment": "positive"
        },
        {
            "title": "Market Climbs on Optimistic Economic Projections",
            "description": "Optimistic economic projections led to market gains.",
            "sentiment": "positive"
        },
        {
            "title": "Healthcare Stocks Surge on Successful Clinical Trials",
            "description": "Successful clinical trials led to a surge in healthcare stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Tech Sector Gains on Strong Investor Interest",
            "description": "Strong investor interest led to gains in the tech sector.",
            "sentiment": "positive"
        },
        {
            "title": "FMCG Stocks Benefit from Strong Brand Performance",
            "description": "Strong brand performance led to gains in FMCG stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Real Estate Sector Rallies on Increased Home Sales",
            "description": "Increased home sales led to a rally in the real estate sector.",
            "sentiment": "positive"
        },
        {
            "title": "Positive Market Sentiment on Infrastructure Development",
            "description": "Infrastructure development led to positive market sentiment.",
            "sentiment": "positive"
        },
        {
            "title": "Financial Stocks Gain on Improved Profit Margins",
            "description": "Improved profit margins led to gains in financial stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Energy Sector Benefits from Technological Advances",
            "description": "Technological advances benefited the energy sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Ends Higher on Increased Foreign Investment",
            "description": "Increased foreign investment led to a higher market close.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Sector Gains on Improved Consumer Confidence",
            "description": "Improved consumer confidence led to gains in the auto sector.",
            "sentiment": "positive"
        },
        {
            "title": "Renewables Sector Surges on Environmental Policies",
            "description": "Environmental policies led to a surge in the renewables sector.",
            "sentiment": "positive"
        },
    {
            "title": "Sensex Nosedives Amid Global Recession Fears",
            "description": "The Sensex nosedived as fears of a global recession mounted.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Declines on Weak Industrial Output Data",
            "description": "Weak industrial output data led to a decline in Nifty.",
            "sentiment": "negative"
        },
        {
            "title": "Market Sentiment Dampens on Rising Unemployment",
            "description": "Rising unemployment dampened overall market sentiment.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Stocks Fall as Data Privacy Concerns Mount",
            "description": "Tech stocks fell amid mounting data privacy concerns.",
            "sentiment": "negative"
        },
        {
            "title": "Oil Stocks Plunge Due to Supply Chain Disruptions",
            "description": "Supply chain disruptions caused oil stocks to plunge.",
            "sentiment": "negative"
        },
        {
            "title": "Banking Sector Faces Pressure from Regulatory Issues",
            "description": "Regulatory issues put pressure on the banking sector.",
            "sentiment": "negative"
        },
        {
            "title": "FII Outflows Intensify, Market Declines",
            "description": "Intensified FII outflows led to a broad market decline.",
            "sentiment": "negative"
        },
        {
            "title": "Auto Sector Struggles with Production Delays",
            "description": "Production delays caused significant struggles in the auto sector.",
            "sentiment": "negative"
        },
        {
            "title": "Market Reacts Negatively to Political Scandals",
            "description": "Political scandals led to a negative reaction in the market.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Slides on Disappointing Corporate Results",
            "description": "Disappointing corporate results caused the Sensex to slide.",
            "sentiment": "negative"
        },
        {
            "title": "Pharma Stocks Hit by Patent Litigation",
            "description": "Patent litigation led to a decline in pharma stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Market Volatility Increases Amid Trade Sanctions",
            "description": "Trade sanctions increased market volatility significantly.",
            "sentiment": "negative"
        },
        {
            "title": "Real Estate Stocks Decline on Sluggish Sales",
            "description": "Sluggish sales caused a decline in real estate stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Ends in Red as Commodity Prices Fall",
            "description": "Falling commodity prices led the Sensex to end in the red.",
            "sentiment": "negative"
        },
        {
            "title": "Rising Fuel Prices Hurt Transport Sector Stocks",
            "description": "Transport sector stocks were hurt by rising fuel prices.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Stocks Drop on Cybersecurity Breach Fears",
            "description": "Fears of cybersecurity breaches caused tech stocks to drop.",
            "sentiment": "negative"
        },
        {
            "title": "Metal Stocks Slump Due to Lower Demand",
            "description": "Lower demand caused a significant slump in metal stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Market Dips on Global Monetary Policy Tightening",
            "description": "Tightening global monetary policies led to a market dip.",
            "sentiment": "negative"
        },
        {
            "title": "Financial Sector Sees Sell-Off on Regulatory Risks",
            "description": "Regulatory risks led to a sell-off in the financial sector.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Sheds Points on Currency Fluctuations",
            "description": "Fluctuations in currency values caused the Sensex to shed points.",
            "sentiment": "negative"
        },
        {
            "title": "Market Decline Triggered by Global Trade Disruptions",
            "description": "Disruptions in global trade triggered a market decline.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Down as Corporate Earnings Miss Expectations",
            "description": "Nifty fell as corporate earnings missed expectations.",
            "sentiment": "negative"
        },
        {
            "title": "Rupee Weakness Leads to Market Losses",
            "description": "Weakness in the rupee led to significant market losses.",
            "sentiment": "negative"
        },
        {
            "title": "IT Stocks Fall Amid Tech Regulation Concerns",
            "description": "Concerns over tech regulations caused a fall in IT stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Oil Prices Spike, Market Reacts Negatively",
            "description": "A spike in oil prices led to a negative market reaction.",
            "sentiment": "negative"
        },
        {
            "title": "Banking Sector Under Strain from Bad Loans",
            "description": "Bad loans put the banking sector under significant strain.",
            "sentiment": "negative"
        },
        {
            "title": "Foreign Fund Outflows Drag Market Down",
            "description": "Outflows of foreign funds dragged the market down.",
            "sentiment": "negative"
        },
        {
            "title": "Auto Sector Faces Headwinds from Supply Shortages",
            "description": "Supply shortages caused headwinds for the auto sector.",
            "sentiment": "negative"
        },
        {
            "title": "Political Instability Weighs on Market Sentiment",
            "description": "Market sentiment was weighed down by political instability.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Dips on Poor Export Data",
            "description": "Poor export data caused a dip in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Pharma Sector Hit by Regulatory Hurdles",
            "description": "Regulatory hurdles caused a decline in the pharma sector.",
            "sentiment": "negative"
        },
        {
            "title": "Market Volatility Rises Amid Economic Uncertainty",
            "description": "Economic uncertainty led to a rise in market volatility.",
            "sentiment": "negative"
        },
        {
            "title": "Real Estate Stocks Struggle with Weak Demand",
            "description": "Weak demand caused real estate stocks to struggle.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Ends Lower on Commodity Price Volatility",
            "description": "Volatility in commodity prices led to a lower close for the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Transport Sector Stocks Fall on Fuel Price Hikes",
            "description": "Fuel price hikes caused a fall in transport sector stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Stocks Decline on Rising Regulatory Scrutiny",
            "description": "Rising regulatory scrutiny caused a decline in tech stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Metal Stocks Drop Amid Global Trade Uncertainty",
            "description": "Global trade uncertainty caused a drop in metal stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Market Dips on Concerns Over Monetary Tightening",
            "description": "Concerns over monetary tightening led to a market dip.",
            "sentiment": "negative"
        },
        {
            "title": "Financial Stocks Underperform Due to Loan Defaults",
            "description": "Loan defaults caused underperformance in financial stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Slides on Economic Growth Concerns",
            "description": "Concerns over economic growth caused the Sensex to slide.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Surges on Positive Economic Outlook",
            "description": "The Sensex surged on a positive economic outlook.",
            "sentiment": "positive"
        },
        {
            "title": "Nifty Climbs on Strong Corporate Earnings",
            "description": "Strong corporate earnings led to a climb in Nifty.",
            "sentiment": "positive"
        },
        {
            "title": "Market Gains on Government Policy Support",
            "description": "Supportive government policies led to market gains.",
            "sentiment": "positive"
        },
        {
            "title": "RBI's Rate Cut Spurs Market Rally",
            "description": "A rate cut by the RBI spurred a market rally.",
            "sentiment": "positive"
        },
        {
            "title": "Pharma Stocks Rally on Strong Growth Prospects",
            "description": "Strong growth prospects led to a rally in pharma stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Market Uptrend Continues on Robust Demand",
            "description": "Robust demand continued to support the market uptrend.",
            "sentiment": "positive"
        },
        {
            "title": "IT Stocks Shine Amid High Digital Adoption",
            "description": "High levels of digital adoption led to gains in IT stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Banking Stocks Rise on Improved Asset Quality",
            "description": "Improved asset quality led to gains in banking stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Sector Rebounds on Strong Sales",
            "description": "Strong sales led to a rebound in the auto sector.",
            "sentiment": "positive"
        },
        {
            "title": "Renewables Sector Gains on Increased Investment",
            "description": "Increased investment led to gains in the renewables sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Rises on Positive Export Data",
            "description": "Positive export data led to a rise in the market.",
            "sentiment": "positive"
        },
        {
            "title": "Consumer Goods Stocks Climb on Rising Demand",
            "description": "Rising demand led to gains in consumer goods stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Infrastructure Stocks Gain on Government Spending",
            "description": "Government spending on infrastructure projects led to sector gains.",
            "sentiment": "positive"
        },
        {
            "title": "Market Cheers on Optimistic Corporate Guidance",
            "description": "Optimistic corporate guidance was cheered by the market.",
            "sentiment": "positive"
        },
        {
            "title": "Healthcare Stocks Surge on Vaccine Efficacy Data",
            "description": "Positive vaccine efficacy data led to a surge in healthcare stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Tech Stocks Gain on Strong Global Trends",
            "description": "Strong global trends led to gains in tech stocks.",
            "sentiment": "positive"
        },
        {
            "title": "FMCG Sector Benefits from Urban Demand Growth",
            "description": "Urban demand growth benefited the FMCG sector.",
            "sentiment": "positive"
        },
        {
            "title": "Real Estate Sector Sees Gains on Recovery Hopes",
            "description": "Recovery hopes led to gains in the real estate sector.",
            "sentiment": "positive"
        },
        {
            "title": "Positive Earnings Reports Boost Market Sentiment",
            "description": "Positive earnings reports boosted overall market sentiment.",
            "sentiment": "positive"
        },
        {
            "title": "Financial Sector Strengthens on Solid Growth",
            "description": "Solid growth led to strengthening in the financial sector.",
            "sentiment": "positive"
        },
        {
            "title": "Energy Stocks Rise on Stable Crude Market",
            "description": "A stable crude oil market led to a rise in energy stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Market Ends Higher on Domestic Investment Boost",
            "description": "Boost in domestic investment led to a higher market close.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Stocks Climb on EV Market Growth",
            "description": "Growth in the electric vehicle market led to gains in auto stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Renewables Sector Flourishes on Policy Support",
            "description": "Supportive policies led to flourishing in the renewables sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Rallies on Strong Economic Data",
            "description": "Strong economic data led to a market rally.",
            "sentiment": "positive"
        },
        {
            "title": "Consumer Goods Stocks Up on Festive Season Sales",
            "description": "Strong festive season sales led to gains in consumer goods stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Infrastructure Sector Gains on New Project Approvals",
            "description": "New project approvals led to gains in the infrastructure sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Climbs on Optimistic Business Outlook",
            "description": "An optimistic business outlook led to market gains.",
            "sentiment": "positive"
        },
        {
            "title": "Healthcare Stocks Rise on Breakthrough Therapies",
            "description": "Breakthrough therapies led to gains in healthcare stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Tech Sector Gains from Strong Earnings Reports",
            "description": "Strong earnings reports led to gains in the tech sector.",
            "sentiment": "positive"
        },
        {
            "title": "FMCG Stocks Benefit from Strong Consumer Sentiment",
            "description": "Strong consumer sentiment led to gains in FMCG stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Real Estate Sector Rallies on Policy Reforms",
            "description": "Policy reforms led to a rally in the real estate sector.",
            "sentiment": "positive"
        },
        {
            "title": "Positive Market Sentiment on Trade Deal Hopes",
            "description": "Hopes for a trade deal led to positive market sentiment.",
            "sentiment": "positive"
        },
        {
            "title": "Financial Stocks Gain on Robust Banking Performance",
            "description": "Robust performance in the banking sector led to gains in financial stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Energy Sector Benefits from Renewable Investments",
            "description": "Investments in renewable energy benefited the energy sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Ends Higher on Global Market Optimism",
            "description": "Optimism in global markets led to a higher close for the market.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Sector Gains on Government Subsidies",
            "description": "Government subsidies led to gains in the auto sector.",
            "sentiment": "positive"
        },
    {
            "title": "Sensex Plunges as Trade War Concerns Escalate",
            "description": "Escalating trade war concerns led to a significant plunge in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Slips Amid Global Market Uncertainty",
            "description": "Global market uncertainty caused Nifty to slip.",
            "sentiment": "negative"
        },
        {
            "title": "Rupee Depreciation Spurs Market Sell-Off",
            "description": "Depreciation of the rupee against major currencies triggered a market sell-off.",
            "sentiment": "negative"
        },
        {
            "title": "Energy Stocks Fall Due to Regulatory Changes",
            "description": "Energy stocks fell following unfavorable regulatory changes.",
            "sentiment": "negative"
        },
        {
            "title": "Healthcare Sector Underperforms Amid Policy Shifts",
            "description": "The healthcare sector underperformed due to sudden policy shifts.",
            "sentiment": "negative"
        },
        {
            "title": "Market Dips as Crude Oil Prices Surge",
            "description": "A surge in crude oil prices led to a market dip.",
            "sentiment": "negative"
        },
        {
            "title": "Weak Consumer Demand Drags Down Retail Stocks",
            "description": "Weak consumer demand dragged down retail stocks.",
            "sentiment": "negative"
        },
        {
            "title": "US stock market LIVE: Dow Jones tumbles 1,600 points, S&P on track to post biggest single-day loss since 2020",
            "description": "Uncertainty surrounding the budget increased market volatility.",
            "sentiment": "negative"
        },
        {
            "title": "Auto Sector Hit by Emission Norms Changes",
            "description": "Changes in emission norms negatively impacted the auto sector.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Drops on Weak Manufacturing Data",
            "description": "Weak manufacturing data led to a drop in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Financial Sector Weighed Down by Policy Uncertainty",
            "description": "Policy uncertainty weighed heavily on the financial sector.",
            "sentiment": "negative"
        },
        {
            "title": "Telecom Stocks Decline Amid Pricing Wars",
            "description": "Telecom stocks declined amid intensifying pricing wars.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Falls on Global Economic Slowdown",
            "description": "Global economic slowdown fears caused Nifty to fall.",
            "sentiment": "negative"
        },
        {
            "title": "FMCG Sector Struggles with Input Cost Increases",
            "description": "The FMCG sector struggled as input costs increased.",
            "sentiment": "negative"
        },
        {
            "title": "Realty Stocks Dip on Rising Interest Rates",
            "description": "Rising interest rates caused a dip in realty stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Market Downturn Triggered by Political Instability",
            "description": "Political instability triggered a broad market downturn.",
            "sentiment": "negative"
        },
        {
            "title": "Metal Stocks Hit by Global Demand Concerns",
            "description": "Global demand concerns hit metal stocks hard.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Tumbles as Bond Yields Rise",
            "description": "A rise in bond yields caused the Sensex to tumble.",
            "sentiment": "negative"
        },
        {
            "title": "Market Sentiment Sours on Trade Deficit Worries",
            "description": "Trade deficit worries soured overall market sentiment.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Sector Falls Amid Weak Global Cues",
            "description": "The tech sector fell sharply amid weak global cues.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Rallies on Upbeat Economic Indicators",
            "description": "The Sensex rallied following the release of upbeat economic indicators.",
            "sentiment": "positive"
        },
        {
            "title": "Nifty Hits New High on Bullish Market Sentiment",
            "description": "Nifty hit a new high as market sentiment turned bullish.",
            "sentiment": "positive"
        },
        {
            "title": "Market Climbs on Strong Quarterly Results",
            "description": "Strong quarterly results from major companies led to a market climb.",
            "sentiment": "positive"
        },
        {
            "title": "RBI's Accommodative Stance Boosts Market Confidence",
            "description": "The RBI's accommodative stance boosted market confidence.",
            "sentiment": "positive"
        },
        {
            "title": "Pharma Stocks Surge on Robust Earnings",
            "description": "Pharma stocks surged on the back of robust earnings.",
            "sentiment": "positive"
        },
        {
            "title": "Market Uptrend Continues on Positive Monsoon Forecast",
            "description": "A positive monsoon forecast continued to support the market uptrend.",
            "sentiment": "positive"
        },
        {
            "title": "IT Stocks Shine Amid Growing Digital Adoption",
            "description": "IT stocks shone brightly amid growing digital adoption across sectors.",
            "sentiment": "positive"
        },
        {
            "title": "Banking Stocks Gain on Lower NPA Levels",
            "description": "Lower Non-Performing Assets levels led to gains in banking stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Sector Revives with Festive Season Demand",
            "description": "The auto sector saw a revival due to strong festive season demand.",
            "sentiment": "positive"
        },
        {
            "title": "Renewables Sector Booms on Government Initiatives",
            "description": "Government initiatives led to a boom in the renewables sector.",
            "sentiment": "positive"
        },
        {
            "title": "Market Rally Driven by Strong FDI Inflows",
            "description": "Strong Foreign Direct Investment inflows drove the market rally.",
            "sentiment": "positive"
        },
        {
            "title": "Consumer Goods Stocks Benefit from Rural Demand",
            "description": "Rural demand boosted consumer goods stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Infrastructure Sector Gains from Increased Spending",
            "description": "Increased spending on infrastructure projects led to sector gains.",
            "sentiment": "positive"
        },
        {
            "title": "Market Cheers Positive Corporate Outlook",
            "description": "A positive corporate outlook was cheered by the market.",
            "sentiment": "positive"
        },
        {
            "title": "Healthcare Stocks Rise on Vaccine Distribution Progress",
            "description": "Progress in vaccine distribution led to a rise in healthcare stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Tech Stocks Benefit from Strong Global Demand",
            "description": "Strong global demand boosted tech stocks.",
            "sentiment": "positive"
        },
        {
            "title": "FMCG Stocks Climb on Improving Rural Economy",
            "description": "An improving rural economy helped FMCG stocks climb.",
            "sentiment": "positive"
        },
        {
            "title": "Real Estate Sector Sees Uptick on Lower Interest Rates",
            "description": "Lower interest rates led to an uptick in the real estate sector.",
            "sentiment": "positive"
        },
        {
            "title": "Positive Global Sentiment Lifts Indian Market",
            "description": "Positive sentiment from global markets lifted the Indian market.",
            "sentiment": "positive"
        },
        {
            "title": "Financial Sector Gains on Robust Credit Growth",
            "description": "Robust credit growth led to gains in the financial sector.",
            "sentiment": "positive"
        },
        {
            "title": "Energy Stocks Rally on Stable Crude Prices",
            "description": "Stable crude oil prices led to a rally in energy stocks.",
            "sentiment": "positive"
        },
        {
            "title": "Market Ends Higher on Strong Domestic Investment",
            "description": "Strong domestic investment flows helped the market end higher.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Stocks Gain on Government's EV Push",
            "description": "Government incentives for electric vehicles led to gains in auto stocks.",
            "sentiment": "positive"
        },
    {
            "title": "Sensex Drops Amid Global Trade Tensions",
            "description": "The Sensex dropped sharply as escalating global trade tensions dampened investor sentiment.",
            "sentiment": "negative"
        },
        {
            "title": "Nifty Slumps Due to Rising Inflation Concerns",
            "description": "Nifty fell as rising inflation concerns led to a sell-off in the market.",
            "sentiment": "negative"
        },
        {
            "title": "Rupee Weakens Against Dollar, Market Reacts Negatively",
            "description": "The weakening of the rupee against the dollar has caused a negative reaction in the stock market.",
            "sentiment": "negative"
        },
        {
            "title": "IT Stocks Plunge Amid Global Chip Shortage",
            "description": "IT stocks saw a significant decline due to the ongoing global chip shortage.",
            "sentiment": "negative"
        },
        {
            "title": "Oil Price Hike Hits Market Hard",
            "description": "A sudden hike in oil prices led to a broad-based market decline.",
            "sentiment": "negative"
        },
        {
            "title": "Banking Sector Under Pressure Due to Rising NPAs",
            "description": "The banking sector is under pressure with rising Non-Performing Assets (NPAs).",
            "sentiment": "negative"
        },
        {
            "title": "FII Outflows Cause Market Slump",
            "description": "The market slumped as Foreign Institutional Investors (FIIs) pulled out funds.",
            "sentiment": "negative"
        },
        {
            "title": "Auto Sector Struggles Amid Semiconductor Shortage",
            "description": "The auto sector is struggling due to a shortage of semiconductors.",
            "sentiment": "negative"
        },
        {
            "title": "Market Reacts Negatively to New COVID-19 Variant",
            "description": "The discovery of a new COVID-19 variant has led to a market downturn.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Falls on Poor Corporate Earnings",
            "description": "Poor corporate earnings reports have led to a fall in the Sensex.",
            "sentiment": "negative"
        },
        {
            "title": "Pharma Stocks Tumble After Regulatory Crackdown",
            "description": "Pharma stocks tumbled following a regulatory crackdown.",
            "sentiment": "negative"
        },
        {
            "title": "Market Volatility Increases Amid Political Uncertainty",
            "description": "Political uncertainty has led to increased market volatility.",
            "sentiment": "negative"
        },
        {
            "title": "Real Estate Stocks Decline Due to Policy Changes",
            "description": "Policy changes have led to a decline in real estate stocks.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Ends in Red as Global Markets Weaken",
            "description": "The Sensex ended in red as global markets showed weakness.",
            "sentiment": "negative"
        },
        {
            "title": "Rising Fuel Prices Lead to Market Downturn",
            "description": "Rising fuel prices have caused a market downturn.",
            "sentiment": "negative"
        },
        {
            "title": "Tech Stocks Drop Amid Regulatory Concerns",
            "description": "Tech stocks dropped amid growing regulatory concerns.",
            "sentiment": "negative"
        },
        {
            "title": "Metal Stocks Underperform Due to Weak Demand",
            "description": "Metal stocks are underperforming due to weak demand.",
            "sentiment": "negative"
        },
        {
            "title": "Market Dips on Global Growth Concerns",
            "description": "Concerns over global economic growth have led to a market dip.",
            "sentiment": "negative"
        },
        {
            "title": "Financial Sector Faces Sell-Off Amid Interest Rate Hikes",
            "description": "The financial sector faced a sell-off as interest rates hiked.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Sheds Points on Geopolitical Tensions",
            "description": "Geopolitical tensions have caused the Sensex to shed points.",
            "sentiment": "negative"
        },
        {
            "title": "Sensex Hits Record High on Positive Economic Data",
            "description": "The Sensex reached a record high following the release of positive economic data.",
            "sentiment": "positive"
        },
        {
            "title": "Nifty Surges as Corporate Earnings Beat Expectations",
            "description": "Nifty surged as corporate earnings exceeded expectations.",
            "sentiment": "positive"
        },
        {
            "title": "Market Gains on Optimistic GDP Growth Forecast",
            "description": "The market gained on an optimistic GDP growth forecast.",
            "sentiment": "positive"
        },
        {
            "title": "RBI's Rate Cut Boosts Market Sentiment",
            "description": "A rate cut by the RBI has boosted market sentiment.",
            "sentiment": "positive"
        },
        {
            "title": "Tech Stocks Rally on Strong Performance",
            "description": "Tech stocks rallied on the back of strong performance reports.",
            "sentiment": "positive"
        },
        {
            "title": "Pharma Sector Leads Market Rally",
            "description": "The pharma sector led a market rally with robust growth.",
            "sentiment": "positive"
        },
        {
            "title": "Sensex Soars as FII Inflows Increase",
            "description": "The Sensex soared as Foreign Institutional Investors increased their inflows.",
            "sentiment": "positive"
        },
        {
            "title": "Auto Stocks Rise Amid Recovery Signs",
            "description": "Auto stocks rose as signs of recovery emerged in the sector.",
            "sentiment": "positive"
        },
        {
            "title": "Banking Sector Boosted by Policy Support",
            "description": "The banking sector was boosted by supportive policy measures.",
            "sentiment": "positive"
        },
        {
            "title": "Market Optimistic on Vaccine Rollout",
            "description": "The market showed optimism with the rollout of COVID-19 vaccines.",
            "sentiment": "positive"
        },
        {
            "title": "Renewable Energy Stocks Gain on Policy Incentives",
            "description": "Renewable energy stocks gained on the back of new policy incentives.",
            "sentiment": "positive"
        },
        {
            "title": "Consumer Goods Stocks Rise on Strong Demand",
            "description": "Consumer goods stocks rose due to strong demand.",
            "sentiment": "positive"
        },
        {
            "title": "Market Rebounds on Global Market Recovery",
            "description": "The market rebounded as global markets showed signs of recovery.",
            "sentiment": "positive"
        },
        {
            "title": "Sensex Climbs on Positive Manufacturing Data",
            "description": "The Sensex climbed following the release of positive manufacturing data.",
            "sentiment": "positive"
        },
        {
            "title": "IT Sector Shines Amid Digital Transformation",
            "description": "The IT sector shone as companies embraced digital transformation.",
            "sentiment": "positive"
        },
        {
            "title": "Infrastructure Stocks Rally on Government Spending",
            "description": "Infrastructure stocks rallied following increased government spending.",
            "sentiment": "positive"
        },
        {
            "title": "Sensex Gains on Positive Corporate Guidance",
            "description": "The Sensex gained as companies provided positive guidance for the coming quarters.",
            "sentiment": "positive"
        },
        {
            "title": "Retail Sector Sees Growth Amid E-commerce Boom",
            "description": "The retail sector saw growth driven by a boom in e-commerce.",
            "sentiment": "positive"
        },
        {
            "title": "Market Cheers as Trade Deficit Narrows",
            "description": "The market reacted positively as the trade deficit narrowed.",
            "sentiment": "positive"
        },
        {
            "title": "Positive Global Cues Lift Market Sentiment",
            "description": "Positive cues from global markets lifted investor sentiment.",
            "sentiment": "positive"
        },
        {
            "title": "Market Ends Higher on Strong Foreign Inflows",
            "description": "The market ended higher as strong foreign inflows boosted sentiment.",
            "sentiment": "positive"
        },
    {
        "title": "Sensex Drops Amid Global Trade Tensions",
        "description": "The Sensex dropped sharply as escalating global trade tensions dampened investor sentiment.",
        "sentiment": "negative"
    },
    {
        "title": "IT Stocks Fall on Declining Project Wins",
        "description": "IT stocks fell as declining project wins signaled a slowdown in new business acquisitions.",
        "sentiment": "negative"
    },
    {
        "title": "Auto Sector Struggles with Labor Strikes",
        "description": "The auto sector struggled as ongoing labor strikes disrupted production and delivery schedules.",
        "sentiment": "negative"
    },
    {
        "title": "Banking Shares Weaken on Regulatory Investigations",
        "description": "Banking shares weakened following news of regulatory investigations into alleged financial irregularities.",
        "sentiment": "negative"
    },
    {
        "title": "Pharma Sector Faces Headwinds from Patent Litigations",
        "description": "The pharma sector faced headwinds as patent litigations increased legal costs and uncertainty.",
        "sentiment": "negative"
    },
    {
        "title": "Global Markets Rally on Trade Deal Optimism",
        "description": "Optimism surrounding a new trade deal spurred global market rallies.",
        "sentiment": "positive"
    },
    {
        "title": "Stocks Decline Amid Renewed Recession Fears",
        "description": "Renewed fears of a global recession caused a market decline.",
        "sentiment": "negative"
    },
    {
        "title": "Nifty Ends in Green as Oil Prices Fall",
        "description": "A drop in oil prices boosted the Nifty index, ending the day in green.",
        "sentiment": "positive"
    },
    {
        "title": "Sensex Dips as Geopolitical Tensions Rise",
        "description": "Increasing geopolitical tensions led to a dip in the Sensex.",
        "sentiment": "negative"
    },
    {
        "title": "Pharma Sector Boosted by Positive Earnings Reports",
        "description": "Strong earnings reports from pharmaceutical companies boosted the sector.",
        "sentiment": "positive"
    },
    {
        "title": "Tech Stocks Slump Amid Semiconductor Shortages",
        "description": "A global shortage of semiconductors dragged tech stocks down.",
        "sentiment": "negative"
    },
    {
        "title": "Sensex Jumps on Government's Stimulus Plan",
        "description": "The Sensex surged after the government announced a major stimulus plan.",
        "sentiment": "positive"
    },
    {
        "title": "Nifty Falls as Bond Yields Climb",
        "description": "Climbing bond yields put pressure on Nifty, leading to a decline.",
        "sentiment": "negative"
    },
    {
        "title": "Banking Stocks Surge on RBI Policy Boost",
        "description": "RBI's new policy boosted banking stocks, leading to strong gains.",
        "sentiment": "positive"
    },
    {
        "title": "Rising COVID Cases Send Markets into Freefall",
        "description": "A surge in COVID cases caused panic selling, sending markets into a freefall.",
        "sentiment": "negative"
    },
    {
        "title": "Energy Stocks Rise as Crude Oil Prices Stabilize",
        "description": "Stabilizing crude oil prices led to gains in energy stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Metals Sector Hit Hard by Weak Commodity Prices",
        "description": "Weak global commodity prices led to a sharp decline in the metals sector.",
        "sentiment": "negative"
    },
    {
        "title": "Consumer Goods Stocks Rally on Strong Retail Sales",
        "description": "Better-than-expected retail sales figures caused a rally in consumer goods stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Market Declines as Investors React to Inflation Worries",
        "description": "Worries about rising inflation caused markets to decline.",
        "sentiment": "negative"
    },
    {
        "title": "Nifty Rises as Foreign Investors Return",
        "description": "A return of foreign investors into the market led to gains in the Nifty.",
        "sentiment": "positive"
    },
    {
        "title": "Auto Stocks Drop Amid Supply Chain Disruptions",
        "description": "Ongoing supply chain disruptions caused a sell-off in auto stocks.",
        "sentiment": "negative"
    },
    {
        "title": "IT Sector Soars as Tech Giants Report Record Profits",
        "description": "Record profits from tech giants boosted the IT sector.",
        "sentiment": "positive"
    },
    {
        "title": "Global Sell-Off Drags Domestic Markets Down",
        "description": "A global sell-off dragged down domestic markets, with heavy losses across sectors.",
        "sentiment": "negative"
    },
    {
        "title": "Healthcare Stocks Gain on Vaccine Rollout Progress",
        "description": "Progress in the global vaccine rollout boosted healthcare stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Financial Sector Struggles Amid Regulatory Uncertainty",
        "description": "Uncertainty around new financial regulations led to a slump in the sector.",
        "sentiment": "negative"
    },
    {
        "title": "Sensex Climbs as Inflation Fears Subside",
        "description": "Subdued inflation fears led to a climb in the Sensex.",
        "sentiment": "positive"
    },
    {
        "title": "Energy Sector Tumbles on OPEC Disagreements",
        "description": "Disagreements within OPEC caused a decline in energy stocks.",
        "sentiment": "negative"
    },
    {
        "title": "Infrastructure Stocks Rally on Government Spending Plans",
        "description": "New government spending plans caused a rally in infrastructure stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Global Market Jitters Push Nifty Lower",
        "description": "Nifty fell due to global market jitters and investor caution.",
        "sentiment": "negative"
    },
    {
        "title": "Tech Stocks Lead Market Recovery",
        "description": "A strong performance in tech stocks led to a broad market recovery.",
        "sentiment": "positive"
    },
    {
        "title": "Banking Stocks Hit by Rising Bad Loans",
        "description": "A rise in bad loans hit the banking sector, leading to widespread losses.",
        "sentiment": "negative"
    },
    {
        "title": "Midcap Stocks Rally as Sentiment Improves",
        "description": "Improved investor sentiment caused a rally in midcap stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Nifty Falls as Crude Oil Prices Spike",
        "description": "A spike in crude oil prices caused a decline in the Nifty.",
        "sentiment": "negative"
    },
    {
        "title": "Sensex Gains on Positive Trade Balance Data",
        "description": "Positive trade balance data led to gains in the Sensex.",
        "sentiment": "positive"
    },
    {
        "title": "Currency Volatility Leads to Broad Market Losses",
        "description": "Volatile currency movements led to broad-based market losses.",
        "sentiment": "negative"
    },
    {
        "title": "Real Estate Stocks Climb on Housing Demand Surge",
        "description": "A surge in housing demand caused a rise in real estate stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Investor Caution Leads to Market Decline",
        "description": "Widespread investor caution led to a broad market decline.",
        "sentiment": "negative"
    },
    {
        "title": "Retail Sector Outperforms on Strong Consumer Spending",
        "description": "Strong consumer spending led to outperformance in the retail sector.",
        "sentiment": "positive"
    },
    {
        "title": "Global Trade Uncertainty Weighs on Markets",
        "description": "Uncertainty in global trade negotiations weighed down markets.",
        "sentiment": "negative"
    },
    {
        "title": "Technology Sector Rallies on Innovation Announcements",
        "description": "New innovation announcements in the tech industry led to a rally in tech stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Markets Dip on Weak Manufacturing Data",
        "description": "Weaker-than-expected manufacturing data caused markets to dip.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Sensex, Nifty Strategy as US Fed Cuts Rate by 50 bps",
        "description": "The US Federal Reserve's decision to cut rates by 50 bps prompted new strategies for the Indian markets.",
        "sentiment": "positive"
    },
    {
        "title": "Blockbuster Friday | Nifty at 25,800, Sensex Gains 1,360 pts; All Sectors in the Green",
        "description": "A blockbuster Friday session saw Nifty rise to 25,800 and Sensex jump by 1,360 points, with all sectors posting gains.",
        "sentiment": "positive"
    },
    {
        "title": "Stock Market Today: Nifty Breaks 25,500 Resistance, Forms Long Bull Candle",
        "description": "Nifty broke through the 25,500 resistance level, forming a bullish pattern on the charts.",
        "sentiment": "positive"
    },
    {
        "title": "How to Trade on Monday: Sensex Hits 84K, Nifty Up 1%; Here's Why the Stock Market is Rising Today",
        "description": "Sensex hit a historic high of 84,000, and Nifty gained 1%, supported by positive global cues and strong domestic data.",
        "sentiment": "positive"
    },
    {
        "title": "Stock Market Sell-Off Continues Amid Global Growth Concerns",
        "description": "Concerns over global economic growth caused a broad sell-off in the stock market.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Sensex Slides 800 Points as Inflation Data Worries Investors",
        "description": "Higher-than-expected inflation data triggered an 800-point decline in the Sensex.",
        "sentiment": "negative"
    },
    {
        "title": "Nifty Closes Below 15,000 for the First Time in Months Amid Rising Bond Yields",
        "description": "Rising bond yields sent Nifty below the 15,000 mark, marking a significant decline after months of gains.",
        "sentiment": "negative"
    },
    {
        "title": "Markets Set for Volatile Week as Central Bank Meeting Looms",
        "description": "With a key central bank meeting approaching, analysts expect a volatile week for the markets.",
        "sentiment": "negative"
    },
    {
        "title": "Tech Stocks Surge as Investors Look to Growth Sectors",
        "description": "Investors flocked to tech stocks, driving the sector higher amid optimism around growth.",
        "sentiment": "positive"
    },
    {
        "title": "Stock Market Today: Nifty Crosses 30,000 as IT and Pharma Lead the Rally",
        "description": "IT and pharma sectors led the Nifty past the 30,000 mark in a strong market rally.",
        "sentiment": "positive"
    },
    {
        "title": "Auto Stocks Decline as Semiconductor Shortage Worsens",
        "description": "A worsening semiconductor shortage caused a sharp decline in auto stocks.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Nifty Inches Towards 26,000, Bulls in Control",
        "description": "Bulls pushed Nifty closer to the 26,000 mark in a steady upward momentum.",
        "sentiment": "positive"
    },
    {
        "title": "How to Trade on Friday: Sensex Slips 1,200 Points as Global Markets Weaken",
        "description": "Weakness in global markets triggered a sharp 1,200-point drop in the Sensex.",
        "sentiment": "negative"
    },
    {
        "title": "Nifty Crosses 27,000, Sensex Jumps 1,500 pts as Domestic Demand Picks Up",
        "description": "A surge in domestic demand pushed Nifty past 27,000 and Sensex up by 1,500 points.",
        "sentiment": "positive"
    },
    {
        "title": "Stock Market Today: Nifty Falls 2% on Profit Booking, Sensex Down 600 Points",
        "description": "Profit booking in key sectors led to a 2% decline in Nifty, with Sensex down by 600 points.",
        "sentiment": "negative"
    },
    {
        "title": "How to Trade on Monday: Nifty Breaks 28,000 as Market Sentiment Improves",
        "description": "Improved market sentiment lifted Nifty above the 28,000 level as bulls maintained control.",
        "sentiment": "positive"
    },
    {
        "title": "Sensex Falls 1,000 Points as Oil Prices Soar to New Highs",
        "description": "A sharp rise in oil prices caused a 1,000-point decline in the Sensex.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Sensex Surges 2,000 Points on Foreign Fund Inflows",
        "description": "A surge in foreign fund inflows pushed Sensex up by 2,000 points.",
        "sentiment": "positive"
    },
    {
        "title": "Markets Drop Sharply on Fears of New Regulatory Measures",
        "description": "Fear of new regulatory measures led to sharp drops in market indices.",
        "sentiment": "negative"
    },
    {
        "title": "How to Trade This Week: Nifty Closes Near All-Time High of 29,500",
        "description": "Nifty closed near its all-time high of 29,500, indicating bullish momentum going into the next week.",
        "sentiment": "positive"
    },
    {
        "title": "Sensex Tanks 2,500 Points Amid Global Economic Slowdown Concerns",
        "description": "Concerns over a global economic slowdown caused Sensex to drop by 2,500 points.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Sensex Gains 1,800 Points on Corporate Earnings Boost",
        "description": "Positive corporate earnings led to an 1,800-point gain in Sensex, pushing markets higher.",
        "sentiment": "positive"
    },
    {
        "title": "Nifty Drops Below 25,000 as Inflationary Pressures Mount",
        "description": "Mounting inflationary pressures sent Nifty below the critical 25,000 level.",
        "sentiment": "negative"
    },
    {
        "title": "How to Trade on Tuesday: Sensex Rises 2% on Banking Sector Strength",
        "description": "A strong performance in the banking sector pushed Sensex up by 2%, indicating further gains.",
        "sentiment": "positive"
    },
    {
        "title": "Sensex Slides 1,500 Points on Profit Booking, Global Cues Weak",
        "description": "Profit booking and weak global cues led to a 1,500-point decline in the Sensex.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Nifty Surges to 28,500 as Bulls Remain in Control",
        "description": "Nifty surged to 28,500 as market bulls continued to dominate trading sessions.",
        "sentiment": "positive"
    },
    {
        "title": "Markets Struggle as FII Outflows Continue, Nifty Down 1.5%",
        "description": "Foreign institutional investor (FII) outflows led to a 1.5% drop in Nifty.",
        "sentiment": "negative"
    },
    {
        "title": "How to Trade on Thursday: Nifty Surpasses 29,000 as Market Sentiment Remains Positive",
        "description": "Nifty surpassed the 29,000 mark as positive sentiment continued to drive the markets higher.",
        "sentiment": "positive"
    },
    {
        "title": "Nifty Slips Below 27,000 as Dollar Strengthens Against Rupee",
        "description": "A strengthening dollar sent Nifty below the 27,000 level, dragging markets down.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Sensex Gains 1,200 Points on Optimism Over Trade Talks",
        "description": "Optimism surrounding ongoing trade talks pushed Sensex up by 1,200 points.",
        "sentiment": "positive"
    },
    {
        "title": "Sensex Plummets 2,200 Points as Investor Confidence Wanes",
        "description": "Waning investor confidence triggered a sharp 2,200-point decline in the Sensex.",
        "sentiment": "negative"
    },
     {
        "title": "Sorry, the Fed Can’t Save Us from a Bear Market | Stock Market News",
        "description": "Even with interventions, the Fed may not be able to prevent a looming bear market, experts warn.",
        "sentiment": "negative"
    },
    {
        "title": "Wall Street Today: US Stocks Surge Stops as Investors Pause After Big US Fed Rate Cut",
        "description": "US stocks stopped surging as investors paused following a significant rate cut by the Federal Reserve.",
        "sentiment": "negative"
    },
    {
        "title": "Nike Gains 6.59% as US Stocks Show Resilience | Stock Market News",
        "description": "Nike led gains in the market, rising 6.59% amid broader resilience in US stocks.",
        "sentiment": "positive"
    },
    {
        "title": "Wall Street Today: US Stocks Jump After Fed Rate Cut, Big Tech Shares Rally",
        "description": "US stocks rallied following the Fed’s rate cut, with Big Tech shares leading the surge.",
        "sentiment": "positive"
    },
    {
        "title": "Market Highlights, Sept 17: Sensex Up 90pts, Nifty Holds 25,400; BHFL Hits Upper Band for 2nd Day",
        "description": "Sensex posted gains of 90 points, while BHFL hit the upper price band for the second consecutive day.",
        "sentiment": "positive"
    },
    {
        "title": "Nifty Breaks 26,000 Barrier as Global Markets Rally",
        "description": "Nifty surged past the 26,000 mark, buoyed by positive global market sentiment.",
        "sentiment": "positive"
    },
    {
        "title": "Stock Market Today: Sensex Falls 600 Points as Crude Oil Prices Surge",
        "description": "A spike in crude oil prices triggered a 600-point decline in the Sensex.",
        "sentiment": "negative"
    },
    {
        "title": "Tech Stocks Continue to Outperform as Nifty Climbs to 28,200",
        "description": "Tech stocks led the market rally, pushing Nifty up to 28,200.",
        "sentiment": "positive"
    },
    {
        "title": "Nifty Slides Below 25,000 as Inflation Fears Weigh on Sentiment",
        "description": "Concerns over rising inflation drove Nifty below the critical 25,000 mark.",
        "sentiment": "negative"
    },
    {
        "title": "Stock Market Today: Sensex Gains 1,200 Points on Strong Corporate Earnings",
        "description": "Strong corporate earnings results lifted Sensex by 1,200 points.",
        "sentiment": "positive"
    },
    {
        "title": "Global Markets Fall as Recession Fears Mount",
        "description": "Growing fears of a global recession sent markets lower across the board.",
        "sentiment": "negative"
    },
    {
        "title": "How to Trade on Friday: Sensex Loses 700 Points as Investors Turn Risk-Averse",
        "description": "Risk aversion among investors caused Sensex to drop by 700 points.",
        "sentiment": "negative"
    },
    {
        "title": "Indian Stock Market: Nifty Gains 500 Points as FII Inflows Continue",
        "description": "Nifty climbed 500 points, buoyed by continued foreign institutional investor inflows.",
        "sentiment": "positive"
    },
    {
        "title": "Wall Street Falls as Fed Officials Signal More Rate Hikes",
        "description": "Wall Street fell as comments from Fed officials hinted at further rate hikes.",
        "sentiment": "negative"
    },
    {
    "title": "Indices 50 Fail to Hold Intraday Record Levels, Nifty 50 Gives Up 26,200",
    "description": "Stock markets saw a pullback as the Nifty 50 failed to sustain intraday record levels, dropping below 26,200.",
    "sentiment": "negative"
    },

    {
        "title": "Sensex Jumps 1,500 Points as Government Unveils New Economic Reforms",
        "description": "The Indian government’s new economic reforms sparked a 1,500-point rise in the Sensex.",
        "sentiment": "positive"
    },
    {
        "title": "Nifty Falls 2% on Weak Asian Market Cues, Sensex Drops 1,200 Points",
        "description": "Weak cues from Asian markets led to a 2% fall in Nifty, with Sensex shedding 1,200 points.",
        "sentiment": "negative"
    },
    {
        "title": "Nifty Gains 1% After Better-Than-Expected Q2 Results",
        "description": "Positive Q2 earnings pushed Nifty up by 1%, indicating market optimism.",
        "sentiment": "positive"
    },
    {
        "title": "Sensex Down 800 Points as Rupee Weakens Against Dollar",
        "description": "A weakening rupee caused Sensex to drop by 800 points.",
        "sentiment": "negative"
    },
    {
        "title": "Sensex Rallies 1,800 Points as Financial Sector Posts Strong Q3 Results",
        "description": "The financial sector’s strong Q3 results led to an 1,800-point rally in Sensex.",
        "sentiment": "positive"
    },
    {
    "title": "Japan's Nikkei Plunges 4% on Stronger Yen After Ishiba Win",
    "description": "The Nikkei index dropped 4% as a stronger yen, following Ishiba's victory, weighed on market sentiment.",
    "sentiment": "negative"
    },

    {
        "title": "Nifty Closes 1.5% Lower as Foreign Outflows Accelerate",
        "description": "Accelerating foreign outflows caused Nifty to close 1.5% lower.",
        "sentiment": "negative"
    },
    {
    "title": "Stock Market Weekly Recap: SENSEX Hits Record High",
    "description": "SENSEX reached a new record high during the weekly trading session, marking a positive outlook for the markets.",
    "sentiment": "positive"
},
]

    # Load data or use default
    if data_path and os.path.exists(data_path):
        try:
            # Assuming CSV format with title, description, and sentiment columns
            df = pd.read_csv(data_path)
            data = df.to_dict('records')
            print(f"Loaded {len(data)} records from {data_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using default data instead.")
            data = default_data
    else:
        print("No data path provided or file not found. Using default data.")
        data = default_data

    # Preprocess text data
    def preprocess_data(data):
        texts = []
        labels = []
        vader_scores = []
        for item in data:
            text = item['title'] + " " + item['description']
            sentiment = item['sentiment']
            texts.append(text)
            labels.append(sentiment)
            vader_score = sia.polarity_scores(text)['compound']
            vader_scores.append(vader_score)
        return texts, labels, vader_scores

    texts, labels, vader_scores = preprocess_data(data)

    MAX_NUM_WORDS = 10000
    MAX_SEQUENCE_LENGTH = 100
    EMBEDDING_DIM = 100

    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data_matrix = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # sentiment labels to numerical values
    def convert_sentiments_to_labels(labels):
        label_map = {'positive': 1, 'negative': 0}
        return [label_map[label] for label in labels]

    labels = convert_sentiments_to_labels(labels)

    X_train, X_test, y_train, y_test, vader_train, vader_test = train_test_split(
        data_matrix, labels, vader_scores, test_size=0.2, random_state=42
    )

    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Define LSTM model with VADER score input
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='text_input')
    embedding_layer = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(text_input)
    lstm_layer = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(embedding_layer)

    vader_input = Input(shape=(1,), dtype='float32', name='vader_input')
    concatenated = Concatenate()([lstm_layer, vader_input])

    dense = Dense(64, activation='relu')(concatenated)
    output = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[text_input, vader_input], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # Train the model
    history = model.fit(
        [X_train, np.array(vader_train)],
        np.array(y_train),
        epochs=10,
        batch_size=16,
        validation_split=0.2,
        class_weight=class_weight_dict,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    score, accuracy = model.evaluate([X_test, np.array(vader_test)], np.array(y_test), verbose=1)
    print(f"Test loss: {score}")
    print(f"Test accuracy: {accuracy}")

    # Save model and tokenizer
    model.save('sentiment_model.h5')
    
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Model and tokenizer saved successfully!")
    
    return model, tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train sentiment analysis model')
    parser.add_argument('--data', type=str, help='Path to CSV data file', default=None)
    
    args = parser.parse_args()
    
    train_model(args.data)