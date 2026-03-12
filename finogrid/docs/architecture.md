# Finogrid — Technical Architecture

**Version:** 2.0 | **Updated:** March 8, 2026

---

## Platform Overview

| Layer | Description |
|-------|-------------|
| **v1 Payout Engine** | B2B cross-border payouts via Bridge + 8 corridor adapters. |
| **Agent Ledger** | A2A stablecoin micro-transactions on Base L2. KYA-gated, Mandate-controlled. |

---

## Flow 1 — B2B Payout

```
Client → Ingress API (8000) → Pub/Sub → Routing Engine
      → Compliance Gate → Partner Execution (Bridge MCP 9001)
      → Reconciliation → Client Webhook
```

Steps: Auth + corridor check → route decision → sanctions + KYT → Bridge submit → poll → callback

---

## Flow 2 — A2A Stablecoin Micropay

```
Setup:  AgentAccount → KYA → Mandate → AgentWallet → USDC top-up
Pay:    closed-loop: PaymentIntent → micropay (10 gates) → off-chain settle
        open-loop:  micropay (10 gates) → off-chain settle
Sweep:  chain_watcher → settled_offchain → settled_onchain (Base L2)
Expiry: intent_sweeper → expired intents → release reserved_balance
```

### Micropay Gates (in order)

| # | Gate | Code |
|---|------|------|
| 1 | KYA ≥ basic + token not expired | 403 |
| 2 | Idempotency replay | 200 |
| 3 | Wallet ownership + active | 404/400 |
| 4 | Closed-loop: intent reserved + not expired + amount match | 400/410 |
| 5 | Counterparty allowlist | 403 |
| 6 | Per-tx cap | 400 |
| 7 | Daily velocity (wallet + KYA level) | 400/403 |
| 8 | Wallet expiry + max_uses | 400 |
| 9 | Available balance | 402 |
| 10 | Settle off-chain + ledger entry | — |

---

## Flow 3 — Fiat Collections (Plaid)

```
Client → Plaid Link UI → exchange public_token → initiate ACH pull
       → Plaid webhook → credit prefund_balance_usdc
```

---

## Database Schema

```
v1:            clients → client_corridor_permissions
               batches → payout_tasks → payout_instructions → execution_events
               routing_profiles, compliance_profiles, audit_logs

migration 002: agent_accounts → agent_kya, agent_wallets → payment_intents
               micro_transactions, agent_ledger_entries

migration 003: principals → mandates → mandate_events
```

---

## Mandate Model

```
Principal → grants → Mandate → authorises → AgentAccount

Key fields:
  scope              payout|collect|topup|full|read_only
  approval_mode      auto|manual|threshold
  approval_threshold transactions ≥ N USDC → human approval queue
  allowed_corridors  []= all;  ["BR","NG"] = restricted
  allowed_chains     []= all;  ["base"] = Base only
  status lifecycle   draft → active → suspended|revoked|expired|superseded
  MandateEvent       append-only; every change logged
```

---

## Service Map

| Service | Port | Hot path |
|---------|------|---------|
| Ingress API | 8000 | Yes |
| Agent Ledger API | 8100 | Yes |
| Ops Console API | 8200 | No |
| Routing Engine | Pub/Sub | Yes |
| Compliance Gate | Pub/Sub | Yes |
| Partner Execution | Pub/Sub | Yes |
| Reconciliation | Scheduled | No |
| intent_sweeper | Cron | No |
| chain_watcher | Cron | No |

---

## MCP Server Map

| Server | Port | Key Tools |
|--------|------|-----------|
| bridge | 9001 | create_transfer, get_transfer, cancel_transfer |
| kyt_aml | 9002 | screen_address, screen_sanctions |
| identity | 9003 | submit_kyb, get_kyb_status |
| wallet_factory | 9004 | register_wallet, check_tx_confirmed, get_wallet_balance |
| kya_validator | 9005 | submit_kya, get_kya_status, verify_kya_token, renew_kya |
| plaid | 9006 | create_link_token, initiate_ach_pull, handle_webhook |

Swap any partner = deploy new MCP with same tool interface. Core services unchanged.

---

## Ops Console (port 8200)

| Endpoint | Purpose |
|----------|---------|
| GET /v1/ops/search?q= | Unified cross-entity search |
| GET /v1/ops/exceptions | Held tasks, KYA blocks, expired intents |
| GET/POST /v1/ops/approvals | Mandate threshold approval queue |
| GET /v1/ops/ledger | Ledger explorer (filterable) |
| GET /v1/ops/agents/{id} | Agent detail: KYA, wallets, txs, daily spend |
| POST /v1/ops/mandates/{id}/activate|suspend|resume|revoke | Mandate lifecycle |
| GET /v1/ops/corridors | Volume + error rates per corridor |

---

## AI Agents (off hot path)

| Agent | Schedule | Uses |
|-------|----------|------|
| OpsOversight | 15 min | FinGPT Sentiment + Forecaster |
| ProcessImprovement | Weekly | FinGPT Forecaster + FX data |
| AuditGovernance | On-demand | FinGPT RAG (ChromaDB) |
| InternalSupport | Always-on | FinGPT RAG |
| TreasuryStrategy | On-demand | FinGPT Forecaster |

Agents observe, report, recommend. They never release payments.

---

## Security

- No private keys stored
- Client + Agent API keys SHA-256 hashed; agent keys shown once
- Ops Console uses separate ops API key
- x402 nonce + timestamp TTL prevents replay
- KYA validator tokens have configurable expiry (default 365 days)
- Ledger entries, mandate events, audit_logs are append-only
- All traffic TLS; secrets in GCP Secret Manager

---

## Environments

| Env | Notes |
|-----|-------|
| Local | chain_enabled=false, kya_validator_backend=internal |
| Demo | Sandbox Bridge, Plaid sandbox |
| Production | chain_enabled=true, hardened config |
