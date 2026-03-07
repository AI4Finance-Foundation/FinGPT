# Finogrid v1

**Non-Custodial Stablecoin Payout Orchestration Platform**

Finogrid is a B2B payout control plane that lets business clients initiate cross-border payouts
in USDT or USDC, with optional last-mile delivery to recipient wallets or local fiat through
regulated off-ramp partners.

---

## Architecture Overview

```
Client API → Ingress → Routing Engine → Compliance Gate → Execution Bridge → Reconciliation
                                                                   ↓
                                                    Bridge (conversion/orchestration)
                                                    Off-ramp partners (local fiat)
```

**AI Agents (off hot path):**
- Ops & Oversight Agent — powered by FinGPT Sentiment + Forecaster
- Audit & Governance Agent — powered by FinGPT RAG
- Process Improvement Agent — powered by FinGPT Forecaster + Sentiment
- Internal Support Agent — powered by FinGPT RAG
- Treasury Strategy Agent — powered by FinGPT Robo-Advisor

---

## v1 Launch Corridors

| Country | Asset | Last Mile |
|---------|-------|-----------|
| Brazil | USDT, USDC | Wallet, PIX fiat |
| Argentina | USDT, USDC | Wallet, Fiat |
| Vietnam | USDT, USDC | Wallet, Fiat |
| India | USDT, USDC | Wallet, UPI fiat |
| UAE | USDT, USDC | Wallet, Fiat |
| Indonesia | USDT, USDC | Wallet, Fiat |
| Philippines | USDT, USDC | Wallet, Fiat |
| Nigeria | USDT, USDC | Wallet, Mobile money fiat |

---

## Tech Stack

- **Backend**: Python / FastAPI (Cloud Run)
- **Database**: PostgreSQL / AlloyDB
- **Messaging**: Google Pub/Sub
- **AI Layer**: FinGPT (Llama-2 LoRA, Transformers, PEFT)
- **MCP**: Model Context Protocol servers for Bridge, KYT/AML, Identity
- **Infrastructure**: GCP (Cloud Run, BigQuery, Secret Manager, IAM)

---

## Project Structure

```
finogrid/
├── services/           # Hot-path deterministic services
│   ├── ingress_api/
│   ├── routing_engine/
│   ├── compliance_gate/
│   ├── partner_execution/
│   ├── reconciliation/
│   └── onboarding/
├── agents/             # Off-hot-path AI agents (FinGPT-powered)
│   ├── ops_oversight/
│   ├── audit_governance/
│   ├── process_improvement/
│   ├── internal_support/
│   └── treasury_strategy/
├── corridors/          # Per-country adapter + policy packs
├── mcp/                # MCP servers for partner APIs
├── fingpt_integration/ # FinGPT module adapters for Finogrid
├── database/           # Schemas, migrations, models
├── infrastructure/     # Terraform / GCP IaC
├── docs/               # Architecture docs, runbooks
└── tests/
```

---

## Design Principles

1. **Non-custodial**: Finogrid never holds private keys or pools client funds.
2. **Partner-led execution**: Bridge and regulated off-ramp partners handle money movement.
3. **Deterministic hot path**: Core transaction services are normal services, not free-form agents.
4. **Agents for leverage**: AI agents observe, audit, recommend — they do not release payouts.
5. **Corridor-aware**: Each country has its own adapter, policy, and exception rules.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in secrets
uvicorn services.ingress_api.main:app --reload
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `BRIDGE_API_KEY` | Bridge orchestration API key |
| `BRIDGE_API_URL` | Bridge API base URL |
| `KYT_API_KEY` | KYT/AML provider API key |
| `KYT_API_URL` | KYT/AML provider base URL |
| `KYB_API_KEY` | Identity/KYB provider API key |
| `DATABASE_URL` | PostgreSQL connection string |
| `PUBSUB_PROJECT_ID` | GCP project for Pub/Sub |
| `OPENAI_API_KEY` | For FinGPT data labeling (agents only) |
| `HUGGINGFACE_TOKEN` | For FinGPT model loading |

---

*Document Version: 1.0 | Date: March 7, 2026*
