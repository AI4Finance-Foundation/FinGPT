# Finogrid

**B2B Stablecoin Payout Orchestration + Agent-to-Agent (A2A) Micro-Transaction Platform**

Finogrid is a non-custodial B2B infrastructure platform with two interconnected layers:

- **v1 Payout Engine** — cross-border B2B payouts in USDT/USDC via Bridge + regulated corridor adapters
- **Agent Ledger** — A2A stablecoin micro-transactions with KYA, closed/open-loop wallets, x402, and Mandate-based access control

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  B2B CLIENTS                                                     │
│  POST /v1/batches              POST /v1/agent-accounts          │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────┐           ┌─────────────────────┐
│  Ingress API     │           │  Agent Ledger API   │
│  Port 8000       │           │  Port 8100          │
│  B2B payouts     │           │  A2A micropay       │
└──────┬───────────┘           └─────────┬───────────┘
       │ Pub/Sub                         │ DB + chain_watcher
       ▼                                 ▼
┌──────────────┐               ┌──────────────────────┐
│ Routing      │               │  Off-chain Ledger    │
│ Engine       │               │  (AlloyDB)           │
└──────┬───────┘               └─────────┬────────────┘
       │                                 │ sweeps
       ▼                                 ▼
┌──────────────┐               ┌──────────────────────┐
│ Compliance   │               │  Base L2 (USDC)      │
│ Gate         │               │  On-chain settlement │
└──────┬───────┘               └──────────────────────┘
       │
       ▼
┌──────────────────────┐       ┌──────────────────────┐
│ Partner Execution    │       │  Ops Console API     │
│ (Bridge + Plaid)     │       │  Port 8200           │
└──────────────────────┘       └──────────────────────┘

AI Agents (off hot path — FinGPT-powered):
  OpsOversight · AuditGovernance · ProcessImprovement
  InternalSupport · TreasuryStrategy
```

---

## v1 Launch Corridors

| Country | Rail | Last Mile | Notes |
|---------|------|-----------|-------|
| Brazil | PIX | Wallet + fiat | CPF/CNPJ validation |
| Nigeria | NIBSS | Wallet + mobile money | BVN ≥$100, TRC-20 USDT pref |
| India | UPI | Wallet + UPI fiat | VPA required |
| Argentina | CBU | Wallet + fiat | CUIT, $25k max |
| Vietnam | Napas/VietQR | Wallet + fiat | BIN code |
| UAE | IBAN/SWIFT | Wallet + fiat | $100k max (hub market) |
| Indonesia | BI-FAST | Wallet + fiat | NIK required |
| Philippines | InstaPay/PESONet | Wallet + fiat | GCash optional |

---

## Agent Ledger — A2A Micropay

Actor hierarchy:

```
AgentOwner (human / Client)
  └── Principal  ← authorising entity
        └── Mandate  ← scope · limits · corridors · approval mode
              └── AgentAccount  ← master agent; must complete KYA
                    └── AgentWallet  ← open-loop OR closed-loop
                          └── PaymentIntent  ← closed-loop only
                                └── MicroTransaction  ← off-chain → on-chain sweep
```

### KYA Levels

| Level | Daily Limit | Requirement |
|-------|------------|-------------|
| `unverified` | $0 | — |
| `pending` | $0 (top-up only) | KYA submitted |
| `basic` | $1/day | Owner identity + purpose confirmed |
| `enhanced` | $100/day | Enhanced validator token |

### Loop Types

| Type | Intent Required | Use Case |
|------|----------------|----------|
| `closed` | Yes — `payment_intent_id` mandatory | Auditable agent commerce |
| `open` | No | High-frequency micropay within limits |

### x402 Protocol

Resources can be payment-walled with HTTP 402. Client reads `PAYMENT-REQUIRED` header (base64 JSON requirement), attaches `PAYMENT-SIGNATURE` on retry, receives `PAYMENT-RESPONSE` settlement receipt. Follows Coinbase/Cloudflare x402 standard.

---

## Services

| Service | Port | Purpose |
|---------|------|---------|
| Ingress API | 8000 | B2B batch payout intake |
| Agent Ledger API | 8100 | A2A stablecoin micropay |
| Ops Console API | 8200 | Search, approvals, ledger explorer, mandate lifecycle |

---

## MCP Servers

| Server | Port | Purpose |
|--------|------|---------|
| bridge | 9001 | Bridge API (fiat↔stablecoin payout) |
| kyt_aml | 9002 | Chainalysis / Elliptic screening |
| identity | 9003 | KYB provider |
| wallet_factory | 9004 | EVM wallet registration + on-chain verification |
| kya_validator | 9005 | KYA validation (internal / Sardine / Persona) |
| plaid | 9006 | Bank connectivity (Plaid Link + ACH pull for collections) |

---

## TypeScript SDK

```bash
npm install @finogrid/agent-ledger-sdk
```

```typescript
import { FinogridClient } from "@finogrid/agent-ledger-sdk";

const finogrid = new FinogridClient({
  baseUrl: "https://agent-ledger.finogrid.io",
  apiKey: "fig_agent_...",
});

// Register agent
const agent = await finogrid.agents.create({ name: "my-agent", owner_client_id: "..." });

// KYA — submit and wait
await finogrid.kya.submit(agent.agent_account_id, { agent_purpose: "...", ... });
await finogrid.kya.pollUntil(agent.agent_account_id, "basic");

// Closed-loop wallet
const wallet = await finogrid.wallets.create(agent.agent_account_id, {
  wallet_address: "0x...", loop_type: "closed",
  max_per_tx_usdc: "0.10", max_daily_usdc: "1.00",
});

// Reserve intent → pay
const intent = await finogrid.paymentIntents.create({
  payer_wallet_id: wallet.wallet_id, amount_usdc: "0.05",
  intent_description: "Pay for AI inference", intent_category: "compute",
  expires_at: new Date(Date.now() + 300_000).toISOString(),
});
const tx = await finogrid.micropay.pay({
  idempotency_key: crypto.randomUUID(), payer_wallet_id: wallet.wallet_id,
  payee_address: "0xpayee...", amount_usdc: "0.05",
  payment_intent_id: intent.payment_intent_id,
});
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API services | Python 3.12 / FastAPI / Cloud Run |
| Database | AlloyDB (PostgreSQL) / SQLAlchemy 2.0 async |
| Messaging | GCP Pub/Sub |
| On-chain | Base L2 (native USDC, ~$0.007/tx, 2–10s confirmation) |
| AI (inference) | FinGPT Llama-2 LoRA + OpenAI / [MiniMax](https://platform.minimaxi.com/) fallback |
| SDK | TypeScript 5.3 (`@finogrid/agent-ledger-sdk`) |
| Infrastructure | GCP (Cloud Run, BigQuery, Secret Manager, IAM) |

---

## Project Structure

```
finogrid/
├── services/
│   ├── ingress_api/           # Port 8000 — B2B batch intake
│   ├── routing_engine/        # Pub/Sub worker — corridor routing
│   ├── compliance_gate/       # Pub/Sub worker — sanctions + KYT
│   ├── partner_execution/     # Pub/Sub worker — Bridge execution
│   ├── reconciliation/        # Scheduled — Bridge poll + reports
│   ├── agent_ledger_api/      # Port 8100 — A2A micropay
│   │   ├── routers/           # kya, wallets, payment_intents, micropay, topup
│   │   ├── middleware/        # x402 payment_required
│   │   ├── intent_sweeper.py  # Expired intent → release reserved balance
│   │   └── chain_watcher.py   # Base L2 deposit watcher + sweep
│   └── ops_console/           # Port 8200 — search, approvals, ledger explorer
│       └── routers/           # search, exceptions, approvals, ledger_explorer,
│                              # agents_explorer, mandates_console, corridors
├── database/
│   ├── models/                # base, client, batch, execution, audit, routing,
│   │                          # instruction, agent_ledger, mandate
│   └── migrations/            # 001_initial_schema, 002_agent_ledger, 003_mandate
├── corridors/                 # 8 country adapters (BR, NG, IN, AR, VN, AE, ID, PH)
├── mcp/                       # 6 MCP servers: bridge, kyt_aml, identity,
│                              # wallet_factory, kya_validator, plaid
├── agents/                    # 5 FinGPT-powered AI agents (off hot path)
├── fingpt_integration/        # Sentiment, Forecaster, RAG adapters
├── sdk/typescript/            # @finogrid/agent-ledger-sdk
├── docs/                      # architecture.md, dr-runbook.md, corridors/
└── tests/
    └── unit/                  # test_routing_engine, test_corridor_adapters,
                               # test_agent_ledger (35 tests)
```

---

## Design Principles

1. **Non-custodial** — Finogrid never holds private keys or pools client funds.
2. **Partner-led execution** — Bridge + regulated partners handle money movement.
3. **Deterministic hot path** — Core transaction services are standard services, not agents.
4. **Agents for leverage** — AI agents observe, audit, recommend; they never release payments.
5. **Mandate-first authority** — Every agent action traces to a valid Principal → Mandate.
6. **Closed-loop auditability** — Closed-loop wallets require declared intent before every payment; intents are never deleted.
7. **Additive architecture** — Agent Ledger is a zero-touch addition; all v1 models unchanged.
8. **Corridor-aware** — Each country has its own adapter, policy, and exception rules.

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env                    # fill in secrets

# v1 Payout API
uvicorn services.ingress_api.main:app --port 8000 --reload

# Agent Ledger API
uvicorn services.agent_ledger_api.main:app --port 8100 --reload

# Ops Console
uvicorn services.ops_console.main:app --port 8200 --reload

# Background workers
python -m services.agent_ledger_api.intent_sweeper &
python -m services.agent_ledger_api.chain_watcher &

# MCP servers
python -m mcp.bridge.server &
python -m mcp.wallet_factory.server &
python -m mcp.kya_validator.server &
python -m mcp.plaid.server &
```

---

## Key Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `DATABASE_URL` | all | PostgreSQL async connection string |
| `BRIDGE_API_KEY` | ingress, execution | Bridge orchestration API key |
| `KYT_API_KEY` | compliance | KYT/AML provider key |
| `BASE_RPC_URL` | agent ledger | Base L2 JSON-RPC endpoint |
| `AGENT_LEDGER_DEPOSIT_ADDRESS` | agent ledger | USDC deposit address on Base |
| `USDC_CONTRACT_ADDRESS_BASE` | chain watcher | Native USDC contract on Base |
| `KYA_ENABLED` | agent ledger | Toggle KYA gate (default: true) |
| `CHAIN_ENABLED` | chain watcher | Toggle on-chain verification |
| `PLAID_CLIENT_ID` | Plaid MCP | Plaid API credentials |
| `PLAID_SECRET` | Plaid MCP | Plaid secret |
| `KYA_VALIDATOR_BACKEND` | KYA MCP | internal \| sardine \| persona |
| `OPS_API_KEY` | ops console | Ops-level auth key |
| `FINGPT_LLM_PROVIDER` | agents | LLM provider: `openai` \| `minimax` \| `fingpt` |
| `OPENAI_API_KEY` | agents | FinGPT OpenAI fallback |
| `MINIMAX_API_KEY` | agents | [MiniMax](https://platform.minimaxi.com/) API key (when provider=minimax) |
| `PUBSUB_PROJECT_ID` | workers | GCP Pub/Sub project |

See `.env.example` for the full list.

---

*Document Version: 2.0 | Updated: March 8, 2026*
