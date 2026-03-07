# Finogrid v1 — Technical Architecture

**Version:** 1.0 | **Date:** March 7, 2026

---

## End-to-End Flow

```
Client API Call
    │
    ▼
[1] Ingress API (FastAPI / Cloud Run)
    • Auth (API key → client record)
    • Schema validation (Pydantic)
    • Corridor permission check
    • Persist batch + tasks as DRAFT
    • Publish → batch_events/batch_created
    │
    ▼ (Pub/Sub)
[2] Routing Engine (Worker / Cloud Run)
    • Load RoutingProfile + ComplianceProfile per corridor
    • Resolve asset (USDT/USDC), mode (wallet/fiat), partner
    • Validate amount limits
    • Update task → COMPLIANCE_CHECK
    • Publish → task_events/task_routed
    │
    ▼ (Pub/Sub)
[3] Compliance Gate (Worker / Cloud Run)
    • Sanctions screen (always)
    • KYT/AML screen (wallet delivery)
    • Apply corridor risk thresholds
    • PASS → task_events/task_cleared
    • HOLD → task_events/task_held (manual review queue)
    • FAIL → task permanently failed
    │
    ▼ (Pub/Sub — only PASS tasks)
[4] Partner Execution Service (Worker / Cloud Run)
    • Submit to Bridge via BridgeClient
    • Capture partner_tx_id
    • Handle retries (max 3, exponential backoff)
    • Publish → execution_status_events
    │
    ▼ (Scheduled + Pub/Sub callbacks)
[5] Reconciliation Service
    • Poll Bridge for executing tasks
    • Transition states → COMPLETED | FAILED
    • Update batch counters
    • Generate client reports (JSON/CSV)
    • Publish → audit_events
    │
    ▼ (Async / Scheduled)
[6] Client Webhook
    • POST batch/task status to client webhook_url
    • Retry on failure
```

---

## Service Boundaries

| Service | Transport | Runs on | Latency target |
|---------|-----------|---------|----------------|
| Ingress API | HTTP | Cloud Run | <200ms |
| Routing Engine | Pub/Sub | Cloud Run | <500ms per task |
| Compliance Gate | Pub/Sub | Cloud Run | <2s (KYT call) |
| Execution Bridge | Pub/Sub | Cloud Run | <5s (Bridge call) |
| Reconciliation | Scheduled | Cloud Run | Every 5 minutes |
| Agent layer | Scheduled | Cloud Run | Every 15min / 1hr |

---

## Database Schema (PostgreSQL / AlloyDB)

```
clients
  └─ client_corridor_permissions

batches
  └─ payout_tasks
       └─ payout_instructions
       └─ execution_events

routing_profiles          (one per corridor)
compliance_profiles       (one per corridor)
audit_logs                (append-only)
```

---

## Pub/Sub Topics

| Topic | Publisher | Consumer |
|-------|-----------|---------|
| `batch_events` | Ingress API | Routing Engine |
| `task_events` | Routing, Compliance | Compliance Gate, Execution |
| `execution_status_events` | Execution Bridge, Partner webhooks | Reconciliation |
| `audit_events` | Reconciliation, Compliance | Audit Agent |

---

## AI Agent Layer (off hot path)

All agents run on Cloud Scheduler — they observe, report, and recommend.
They do NOT release payouts or modify routing config.

```
Scheduled Agents:
  ├── OpsOversightAgent        (every 15 min) → FinGPT Sentiment + Forecaster
  ├── ProcessImprovementAgent  (weekly)        → FinGPT Sentiment + FX data
  ├── AuditGovernanceAgent     (on-demand)     → FinGPT RAG
  ├── InternalSupportAgent     (always-on)     → FinGPT RAG
  └── TreasuryStrategyAgent    (on-demand)     → FinGPT Forecaster
```

---

## Corridor Adapter Pattern

Each corridor is a self-contained adapter:

```python
corridors/
  ├── brazil/adapter.py     # PIX, CPF/CNPJ validation
  ├── nigeria/adapter.py    # NIBSS, BVN, TRC-20 USDT preference
  ├── india/adapter.py      # UPI/VPA
  ...
```

Adding a new corridor = create one adapter file, one RoutingProfile DB row,
one ComplianceProfile DB row. No changes to core services.

---

## MCP Server Pattern

Partner integrations are MCP servers:

```
finogrid/mcp/
  ├── bridge/server.py      port 9001  (Bridge API)
  ├── kyt_aml/server.py     port 9002  (Chainalysis/Elliptic)
  └── identity/server.py   port 9003  (KYB provider)
```

Swapping a partner = deploy a new MCP server. Core services unchanged.

---

## Security

- No private keys stored or generated
- All secrets in GCP Secret Manager
- Per-service IAM service accounts (least privilege)
- API keys hashed before storage
- All traffic TLS
- Audit log is append-only

---

## Environments

| Env | Branch | Data |
|-----|--------|------|
| Demo | `develop` | Synthetic, sandbox Bridge API |
| Production | `main` | Live clients, hardened config |
