# FinGPT: Corporate FX Exposure Management Use Cases

> Applying FinGPT to the real cost problem in corporate treasury: subsidiaries hedging gross
> FX exposure when they should be netting first — and the AI-powered plugin pattern that fixes it.
>
> Contributed by [Moiz Mujtaba](https://github.com/MoizMujtaba) — Director of Product Management,
> cross-border payments and FX risk platforms across 17 global markets.

---

## Who This Is For

**Primary Persona: FX Dealer / Relationship Manager at a payments company**
(Airwallex, Ebury, Wise, Wealthsimple, Corpay, Western Union Business Solutions, or similar)

| Dimension | Detail |
|---|---|
| **Job-to-be-done** | Grow revenue per client by deepening FX product utilisation |
| **Measured on** | Spread captured per client, client retention, wallet share |
| **Peak pain moment** | Month-end: corporate client calls asking why FX costs spiked again |
| **Root cause they rarely surface** | Client subsidiaries are hedging gross exposure — they never netted first |
| **Primary value delivered** | Business value: directly reduces client's FX translation losses — a measurable, reportable number the CFO cares about |
| **Secondary value** | Emotional: the dealer looks like a strategic advisor, not just a rate quoter |

---

## The Core Problem: FX Clutter Cost

Corporate subsidiaries in 3+ countries each manage their own payables and receivables
independently. Without visibility across entities, each subsidiary hedges its own gross
exposure. The result:

```
Without netting:
  Subsidiary A hedges: USD 500,000 long GBP
  Subsidiary B hedges: USD 480,000 short GBP
  Net company exposure: USD 20,000
  Actual hedging cost paid: on USD 980,000 ← this is FX clutter cost

With netting first:
  Net exposure: USD 20,000
  Hedging cost paid: on USD 20,000 ← 98% reduction
```

This is not a trading problem. It is a **visibility and consolidation problem** —
and it is where AI creates immediate, measurable business value.

---

## The Two-Layer Engine

This document describes a two-layer AI engine built on FinGPT that solves this problem
as an embeddable plugin for accounting software and treasury tools.

### Layer 1 — Multi-Entity FX Exposure Consolidator
Reads multi-source treasury data (Excel, CSV, Xero, QuickBooks, NetSuite, Sage),
identifies offsetting intercompany positions across subsidiaries and currencies,
calculates net exposure, and produces a netting schedule with projected cost savings.

### Layer 2 — FX Exposure Intelligence Layer
Takes post-netting residual exposure as input, layers in live FX rates and real-time
market sentiment, and generates plain-English hedging recommendations the dealer
can present directly to the CFO.

```
[Data Sources]
Excel / CSV / Xero / QuickBooks / NetSuite / Sage
         │
         ▼
[Layer 1: Multi-Entity FX Exposure Consolidator]
FinGPT-RAG reads files → extracts payables & receivables by entity/currency
Netting engine → identifies offsets → calculates net exposure
         │
         ▼
[Layer 2: FX Exposure Intelligence Layer]
OANDA API → live rates for residual exposure valuation
Serper AI → real-time FX news retrieval
FinGPT-Sentiment → sentiment scoring per currency pair
FinGPT-RAG → generates plain-English hedging recommendation
         │
         ▼
[Output — rendered in Gradio UI or embedded in accounting software]
• Netting schedule with savings estimate
• Residual exposure by currency pair
• Hedging recommendation: instrument, ratio, plain-English rationale
• FX cost reduction vs. gross hedging baseline (the number the CFO reports)
```

---

## Tech Stack

| Component | Role |
|---|---|
| **FinGPT-RAG** | File parsing, netting logic, recommendation generation |
| **FinGPT-Sentiment** | Currency pair sentiment scoring from live news |
| **Hugging Face Hub** | Model hosting — `FinGPT/fingpt-sentiment_llama2-13b_lora` |
| **Gradio** | Demo UI — dealer uploads file, sees output in browser, no setup required |
| **Serper AI** | Real-time FX news retrieval (Google News via API) for sentiment context |
| **OANDA API** | Live mid-market FX rates for exposure valuation and netting calculations |
| **Alpha Vantage** | Historical FX rate data for hedge ratio backtesting |
| **ECB SDMX API** | EUR reference rates (free, no key required) |
| **pandas + openpyxl** | Excel/CSV parsing before RAG ingestion |
| **Plaid / TrueLayer** | Optional: direct bank feed ingestion instead of manual upload |

---

## Reforge Value Matrix

| Value Type | What It Looks Like Here |
|---|---|
| **Functional** | Dealer uploads one file instead of manually consolidating 6 subsidiary spreadsheets |
| **Emotional** | Dealer walks into the CFO meeting with a cost reduction number, not just a rate sheet |
| **Business** | CFO sees FX translation losses reduced by 40–70% — reportable to board, auditable |
| **Social** | Dealer is now a strategic treasury advisor, not a commodity FX provider — harder to replace |

**The moment that matters:** Month-end. The CFO has just seen the FX line on the P&L.
The dealer who arrives with a netting analysis and a forward recommendation *before* the
CFO asks why costs are high — that dealer keeps the relationship. That dealer grows wallet share.

---

## Use Case 1: Netting + Hedging Recommendation (Core Flow)

### Prompt A — Multi-source File Extraction
```
Instruction: You are a corporate treasury analyst. Extract all multi-currency
intercompany payables and receivables from the data below.

Return a structured table with columns:
| Subsidiary | Counterparty Subsidiary | Currency | Amount | Direction | Due Date |

Input: [PASTE CONTENT FROM EXCEL / XERO / QUICKBOOKS / NETSUUITE / SAGE EXPORT]

Output:
```

### Prompt B — Netting Schedule Generation
```
Instruction: You are a corporate treasury analyst. Given the following intercompany
payment schedule, identify all netting opportunities across subsidiaries.

For each netting pair output:
1. Gross settlement amounts (both directions)
2. Net settlement amount and direction
3. Estimated FX conversion cost saving (use provided mid-market rates)
4. Recommended settlement date

Use OANDA mid-market rates: [RATES FROM API CALL]

Input: [STRUCTURED TABLE FROM PROMPT A]

Output:
```

### Prompt C — FX Exposure Intelligence (Post-Netting)
```
Instruction: You are an FX risk advisor presenting to a CFO with no derivatives
background. Based on the residual post-netting exposure below and current market
conditions, generate a hedging recommendation.

Your output must include:
1. Hedge or not (yes / monitor / no action)
2. Recommended instrument (Forward / Vanilla Option / Natural Hedge)
3. Suggested hedge ratio with plain-English rationale
4. One-paragraph CFO summary — no jargon

Residual exposure: [FROM NETTING OUTPUT]

Current market context (from Serper AI news retrieval):
[TOP 3 RELEVANT FX NEWS HEADLINES WITH DATES]

Current sentiment score (from FinGPT-Sentiment):
[SENTIMENT SCORE AND CONFIDENCE PER CURRENCY PAIR]

Output:
```

---

## Use Case 2: Dealer Demo Mode (Gradio UI)

The fastest way for a payments company PM to demo this to a CFO client:
a Gradio app the dealer opens on a laptop, uploads the client's treasury export,
and shows real output within 60 seconds.

### Gradio App Structure
```python
import gradio as gr
from fingpt_rag import extract_positions, generate_netting_schedule
from fingpt_sentiment import score_currency_sentiment
from oanda_api import get_live_rates
from serper_api import get_fx_news

def run_fx_analysis(file, currency_pairs):
    # Step 1: Extract positions from uploaded file
    positions = extract_positions(file)

    # Step 2: Get live rates from OANDA
    rates = get_live_rates(currency_pairs)

    # Step 3: Generate netting schedule
    netting = generate_netting_schedule(positions, rates)

    # Step 4: Get market context
    news = get_fx_news(currency_pairs)
    sentiment = score_currency_sentiment(news)

    # Step 5: Generate hedging recommendation
    recommendation = generate_recommendation(netting.residual, rates, sentiment)

    return netting.summary, recommendation

gr.Interface(
    fn=run_fx_analysis,
    inputs=[
        gr.File(label="Upload treasury file (Excel, CSV, Xero, QuickBooks, Sage)"),
        gr.CheckboxGroup(["EUR/USD", "GBP/USD", "USD/CAD", "USD/JPY"], label="Currency pairs")
    ],
    outputs=[
        gr.Dataframe(label="Netting Schedule + Savings"),
        gr.Textbox(label="Hedging Recommendation (CFO-ready)")
    ],
    title="Corporate FX Exposure Consolidator",
    description="Upload your multi-entity treasury file. Get a netting schedule and hedging recommendation in 60 seconds."
).launch()
```

**Deploy to Hugging Face Spaces** (free, shareable link):
```bash
huggingface-cli login
gradio deploy
```
The dealer sends the CFO a link. No installation. No platform adoption required.

---

## Use Case 3: Accounting Software Plugin (Embedded Pattern)

For payments company PMs who want to embed this inside their clients' existing tools
rather than as a standalone app.

### Xero / QuickBooks / NetSuite / Sage
```python
# Pull live payables/receivables directly — no manual upload
from xero_python import AccountingApi
from fingpt_rag import extract_positions_from_structured_data

# 1. Connect to accounting API
positions = AccountingApi.get_invoices(
    statuses=["AUTHORISED"],
    date_from="2026-02-01"
)

# 2. Run netting + FX analysis (same pipeline as Use Case 1)
analysis = run_fx_analysis(positions)

# 3. Push recommendation back into accounting software as a memo
AccountingApi.create_account_note(analysis.cfo_summary)
```

### Excel Add-in
- Expose the same pipeline as an Excel add-in via Office.js
- Dealer installs once; CFO runs the analysis from the ribbon
- Output lands in a new worksheet tab: netting schedule + recommendation side by side

---

## What to Build Next (Contribution Opportunities)

| Feature | Complexity | Impact |
|---|---|---|
| MT940 bank statement parser (for European corporates) | Medium | High |
| Multi-period netting (weekly/monthly cycle optimisation) | High | High |
| Hedge ratio backtester using Alpha Vantage historical data | Medium | Medium |
| AML flag integration — surface sanctioned counterparty exposure | High | High |
| SAP / Oracle ERP connector | High | High |

---

## References

- [FinGPT-RAG](fingpt/FinGPT_RAG/) — Retrieval Augmented Generation pipeline
- [FinGPT-Sentiment](fingpt/FinGPT_Sentiment_Analysis_v3/) — Financial sentiment analysis
- [Hugging Face Spaces](https://huggingface.co/spaces) — Free Gradio app hosting
- [OANDA API](https://developer.oanda.com/) — Live and historical FX rates
- [Serper AI](https://serper.dev/) — Real-time news retrieval API
- [Alpha Vantage](https://www.alphavantage.co/) — Historical FX data
- [ECB SDMX API](https://data.ecb.europa.eu/help/api/overview) — EUR reference rates (free)
- [ISO 20022](https://www.iso20022.org/) — Global payment messaging standard
- [AI4Finance Foundation](https://ai4finance.org/)
