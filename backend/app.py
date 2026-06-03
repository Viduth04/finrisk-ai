from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os
from google import genai
from dotenv import load_dotenv
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from xml.etree import ElementTree as ET
from html import unescape
from datetime import datetime, timezone

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini = None
if GEMINI_API_KEY:
    try:
        gemini = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        gemini = None

app = FastAPI(title="FinRisk AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
BASE = os.path.join(os.path.dirname(__file__), '..', 'models')
xgb_model = joblib.load(os.path.join(BASE, 'xgb_classifier.pkl'))
lgb_model  = joblib.load(os.path.join(BASE, 'lgb_regressor.pkl'))
scaler     = joblib.load(os.path.join(BASE, 'scaler.pkl'))
with open(os.path.join(BASE, 'feature_names.json')) as f:
    feature_names = json.load(f)

BUSINESS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/technologyNews",
]

NEWS_FALLBACK = [
    {
        "title": "Markets watch earnings and inflation signals as investors stay cautious",
        "link": "https://www.reuters.com/business/",
        "published": "Today",
        "summary": "A snapshot of the business climate while live headlines are unavailable.",
        "source": "Reuters Business",
    },
    {
        "title": "Currency swings and rate expectations remain key risk themes",
        "link": "https://www.reuters.com/business/",
        "published": "Today",
        "summary": "Useful context for the loan and risk decisions shown in FinRisk AI.",
        "source": "Reuters Business",
    },
    {
        "title": "Business leaders weigh growth, regulation and AI investment",
        "link": "https://www.reuters.com/business/",
        "published": "Today",
        "summary": "Broad market sentiment and policy shifts still shape credit and lending risk.",
        "source": "Reuters Business",
    },
]

class UserInput(BaseModel):
    age: int
    annual_income: float
    credit_score: int
    employment_status: str
    loan_amount: float
    loan_duration: int
    debt_to_income_ratio: float
    previous_loan_defaults: int
    bankruptcy_history: int
    savings_account_balance: float
    total_assets: float
    total_liabilities: float
    net_worth: float
    loan_purpose: str
    question: str

class ChatMessage(BaseModel):
    message: str
    risk_score: float = None
    loan_approved: int = None
    financial_context: dict = None

def _normalize_text(value):
    return unescape(" ".join((value or "").split())).strip()

def _parse_date(value):
    value = _normalize_text(value)
    return value or "Today"

def _fetch_url(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=8) as response:
        return response.read()

def _parse_feed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    items = []

    if root.tag.endswith("rss"):
        for item in root.findall("./channel/item"):
            title = _normalize_text(item.findtext("title"))
            link = _normalize_text(item.findtext("link"))
            summary = _normalize_text(item.findtext("description"))
            published = _parse_date(item.findtext("pubDate"))
            if title:
                items.append(
                    {
                        "title": title,
                        "link": link or "https://www.reuters.com/business/",
                        "published": published,
                        "summary": summary,
                        "source": "Reuters Business",
                    }
                )
    else:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall(".//atom:entry", ns):
            title = _normalize_text(entry.findtext("atom:title", default="", namespaces=ns))
            summary = _normalize_text(entry.findtext("atom:summary", default="", namespaces=ns))
            published = _parse_date(entry.findtext("atom:updated", default="", namespaces=ns))
            link_el = entry.find("atom:link[@rel='alternate']", ns)
            link = link_el.attrib.get("href") if link_el is not None else ""
            if title:
                items.append(
                    {
                        "title": title,
                        "link": link or "https://www.reuters.com/business/",
                        "published": published,
                        "summary": summary,
                        "source": "Reuters Business",
                    }
                )

    return items

def get_latest_business_news(limit=3):
    for feed_url in BUSINESS_FEEDS:
        try:
            xml_bytes = _fetch_url(feed_url)
            items = _parse_feed(xml_bytes)
            if items:
                return items[:limit]
        except (URLError, HTTPError, ET.ParseError, TimeoutError, ValueError):
            continue
        except Exception:
            continue
    return NEWS_FALLBACK[:limit]

def _gemini_fallback_main():
    return (
        "AI advice is temporarily unavailable. Please rotate your Gemini API key and try again. "
        "For now, use the risk score, loan decision, and business news panel as your guide."
    )

def _gemini_fallback_chat():
    return "AI advice is temporarily unavailable right now. Try again after rotating the Gemini API key."

def _generate_gemini_text(prompt: str):
    if gemini is None:
        return None
    try:
        response = gemini.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return getattr(response, "text", None)
    except Exception as e:
        message = str(e)
        if "PERMISSION_DENIED" in message or "reported as leaked" in message or "403" in message:
            return None
        return None

def build_feature_vector(data: UserInput):
    row = {col: 0 for col in feature_names}
    row['Age']                      = data.age
    row['AnnualIncome']             = data.annual_income
    row['CreditScore']              = data.credit_score
    row['LoanAmount']               = data.loan_amount
    row['LoanDuration']             = data.loan_duration
    row['DebtToIncomeRatio']        = data.debt_to_income_ratio
    row['PreviousLoanDefaults']     = data.previous_loan_defaults
    row['BankruptcyHistory']        = data.bankruptcy_history
    row['SavingsAccountBalance']    = data.savings_account_balance
    row['TotalAssets']              = data.total_assets
    row['TotalLiabilities']         = data.total_liabilities
    row['NetWorth']                 = data.net_worth
    row['MonthlyIncome']            = data.annual_income / 12
    row['SavingsToLoanRatio']       = data.savings_account_balance / (data.loan_amount + 1)
    row['AssetsToLiabilitiesRatio'] = data.total_assets / (data.total_liabilities + 1)
    row['IncomeToLoanRatio']        = data.annual_income / (data.loan_amount + 1)
    row['CreditScoreCategory']      = min(4, max(0, (data.credit_score - 580) // 60))
    emp_col = f'EmploymentStatus_{data.employment_status}'
    if emp_col in row: row[emp_col] = 1
    purpose_col = f'LoanPurpose_{data.loan_purpose}'
    if purpose_col in row: row[purpose_col] = 1
    return pd.DataFrame([row])

def generate_chart(risk_score, loan_approved, data: UserInput):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('#0f172a')

    ax1 = axes[0]
    ax1.set_facecolor('#0f172a')
    colors = ['#22c55e', '#eab308', '#ef4444']
    ax1.pie([40, 30, 30], colors=colors, startangle=180,
            counterclock=False, wedgeprops=dict(width=0.4))
    score_color = '#22c55e' if risk_score < 45 else '#eab308' if risk_score < 60 else '#ef4444'
    ax1.text(0, -0.1, f'{risk_score:.0f}', ha='center', va='center',
             fontsize=36, fontweight='bold', color=score_color)
    ax1.text(0, -0.45, 'Risk Score', ha='center', fontsize=12, color='white')
    label = 'LOW RISK' if risk_score < 45 else 'MEDIUM RISK' if risk_score < 60 else 'HIGH RISK'
    ax1.text(0, -0.7, label, ha='center', fontsize=11,
             color=score_color, fontweight='bold')
    ax1.set_title('Risk Score Gauge', color='white', fontsize=13, pad=10)

    ax2 = axes[1]
    ax2.set_facecolor('#0f172a')
    metrics = ['Credit Score', 'Debt Ratio', 'Savings Ratio', 'Income Ratio']
    values2 = [
        min(100, data.credit_score / 9),
        min(100, data.debt_to_income_ratio * 100),
        min(100, (data.savings_account_balance / (data.loan_amount + 1)) * 10),
        min(100, (data.annual_income / (data.loan_amount + 1)) * 10),
    ]
    bars = ax2.barh(metrics, values2,
                    color=['#3b82f6', '#ef4444', '#22c55e', '#a855f7'],
                    height=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_facecolor('#0f172a')
    ax2.tick_params(colors='white')
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color('#334155')
    for bar, val in zip(bars, values2):
        ax2.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f'{val:.0f}', va='center', color='white', fontsize=10)
    ax2.set_title('Key Financial Metrics', color='white', fontsize=13, pad=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                bbox_inches='tight', facecolor='#0f172a')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def ask_gemini(question, risk_score, loan_approved, data: UserInput):
    risk_level = "LOW" if risk_score < 45 else "MEDIUM" if risk_score < 60 else "HIGH"
    loan_text  = "APPROVED" if loan_approved == 1 else "REJECTED"

    prompt = f"""
You are FinRisk AI, an expert financial risk advisor specialized in Sri Lanka's economy.

A user has asked: "{question}"

Their financial profile:
- Age: {data.age}
- Annual Income: LKR {data.annual_income:,.0f}
- Credit Score: {data.credit_score}/900
- Employment: {data.employment_status}
- Loan Amount Requested: LKR {data.loan_amount:,.0f}
- Loan Duration: {data.loan_duration} months
- Loan Purpose: {data.loan_purpose}
- Debt-to-Income Ratio: {data.debt_to_income_ratio:.2f}
- Previous Loan Defaults: {data.previous_loan_defaults}
- Bankruptcy History: {"Yes" if data.bankruptcy_history else "No"}
- Savings Balance: LKR {data.savings_account_balance:,.0f}
- Total Assets: LKR {data.total_assets:,.0f}
- Total Liabilities: LKR {data.total_liabilities:,.0f}
- Net Worth: LKR {data.net_worth:,.0f}

ML Model Results:
- Risk Score: {risk_score:.1f}/100 ({risk_level} RISK)
- Loan Decision: {loan_text}

Sri Lanka Context:
- Consider current LKR currency volatility
- High inflation environment
- Consider local business risks (location, sector, regulations)
- Post-economic crisis recovery phase

Give a clear, friendly, personalized response in 4-6 sentences.
Include specific actionable advice relevant to Sri Lanka.
End with one concrete next step they should take.
"""
    response_text = _generate_gemini_text(prompt)
    return response_text or _gemini_fallback_main()

@app.get("/")
def root():
    return {"message": "FinRisk AI API is running ✅"}

@app.post("/predict")
def predict(data: UserInput):
    df = build_feature_vector(data)
    scaled = scaler.transform(df)

    risk_score    = lgb_model.predict(scaled)[0]
    loan_approved = xgb_model.predict(scaled)[0]

    try:
        gemini_response = ask_gemini(data.question, risk_score, loan_approved, data)
    except Exception as e:
        gemini_response = f"AI advice unavailable: {str(e)}"

    chart_b64  = generate_chart(risk_score, loan_approved, data)
    risk_level = "LOW" if risk_score < 45 else "MEDIUM" if risk_score < 60 else "HIGH"

    return {
        "risk_score":      round(float(risk_score), 1),
        "loan_approved":   int(loan_approved),
        "risk_level":      risk_level,
        "loan_decision":   "Loan Likely Approved" if loan_approved == 1 else "Loan Likely Rejected",
        "gemini_response": gemini_response,
        "chart":           chart_b64,
        "sri_lanka_note":  "Maintain at least 6 months of expenses in savings before committing to large loans in current Sri Lanka economic conditions."
    }

@app.post("/chat")
def chat(msg: ChatMessage):
    try:
        prompt = f"""
You are FinRisk AI, a friendly financial risk advisor for Sri Lanka.
User asks: "{msg.message}"
{"Their risk score is: " + str(msg.risk_score) if msg.risk_score else ""}
Give helpful, concise financial advice in 3-4 sentences relevant to Sri Lanka.
"""
        response_text = _generate_gemini_text(prompt)
        return {"response": response_text or _gemini_fallback_chat()}
    except Exception as e:
        return {"response": _gemini_fallback_chat()}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    return {
        "rows":    len(df),
        "columns": df.columns.tolist(),
        "preview": df.head(3).to_dict(orient='records')
    }

@app.get("/news")
def latest_business_news():
    return {
        "source": "Reuters Business",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "items": get_latest_business_news(),
    }
