# Data Source Suggestion: FinancialReports.eu for Financial LLM Training

## Overview

[FinancialReports.eu](https://financialreports.eu) provides API access to **14M+ filings** from data sources across 30+ countries. Its **Markdown conversion endpoint** returns LLM-ready text from annual reports, making it a rich source of financial text data for training and fine-tuning financial language models.

## Why This Fits FinGPT

FinGPT needs diverse, high-quality financial text for model training. FinancialReports.eu offers:

- **14M+ filings** — annual reports, interim reports, ESG disclosures, M&A announcements
- **Markdown endpoint** (`GET /filings/{id}/markdown/`) — returns clean, structured text from filing PDFs, ready for LLM consumption
- **Global & multilingual** — filings from 30+ countries, extending training data beyond US filings
- **33,000+ companies** with ISIN, LEI, and GICS classification for structured metadata
- **11 standardized filing categories** — enables category-specific fine-tuning (e.g., ESG reports, financial statements, M&A disclosures)
- **Temporal coverage** — historical filings for time-series analysis and temporal reasoning tasks

## Integration Approaches

### 1. Training Data Pipeline

Use the API to build a corpus of financial filings text:

```python
import requests

headers = {"X-API-Key": "your-api-key"}

# Fetch recent annual reports for European companies
resp = requests.get("https://api.financialreports.eu/filings/",
    headers=headers,
    params={
        "categories": "2",          # Financial Reporting
        "countries": "DE,FR,GB",    # Germany, France, UK
        "page_size": 100
    }
)
filings = resp.json()["results"]

# Extract Markdown text for each filing
for filing in filings:
    content = requests.get(
        f"https://api.financialreports.eu/filings/{filing['id']}/markdown/",
        headers=headers
    ).text
    # Feed into training pipeline...
```

### 2. MCP Server for Interactive Analysis

FinancialReports.eu offers an [MCP server integration](https://financialreports.eu) compatible with Claude.ai — enabling interactive financial analysis that can be used for evaluation and demonstration.

### 3. Python SDK

```bash
pip install financial-reports-generated-client
```

```python
from financial_reports_client import Client
from financial_reports_client.api.filings import filings_list, filings_markdown_retrieve

client = Client(base_url="https://api.financialreports.eu")
client = client.with_headers({"X-API-Key": "your-api-key"})

# List filings
filings = filings_list.sync(client=client, categories="2", countries="DE,FR,GB")

# Get Markdown content
content = filings_markdown_retrieve.sync(client=client, id=12345)
```

## API Details

| Property | Value |
|---|---|
| **Base URL** | `https://api.financialreports.eu` |
| **API Docs** | [docs.financialreports.eu](https://docs.financialreports.eu/) |
| **Authentication** | API key via `X-API-Key` header |
| **Python SDK** | `pip install financial-reports-generated-client` |
| **Rate Limiting** | Burst limit + monthly quota |
| **Companies** | 33,230+ |
| **Total Filings** | 14,135,359+ |
| **Coverage** | 30+ countries |
| **Countries** | 30+ |

## Use Cases for FinGPT

| Use Case | How FinancialReports.eu Helps |
|---|---|
| Financial NLP training | 14M+ filings as Markdown text corpus |
| Multilingual models | Filings in multiple European languages |
| ESG analysis | Dedicated ESG filing category |
| Filing summarization | Full annual reports for summarization tasks |
| Temporal reasoning | Historical filings with timestamps |
| Cross-market analysis | Standardized categories across 30+ countries |
