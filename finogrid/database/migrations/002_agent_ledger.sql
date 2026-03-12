-- Finogrid Agent Ledger — Schema Migration 002
-- Additive only: zero modifications to v1 tables.
-- Apply after 001_initial_schema.sql.

-- ── AgentAccounts ─────────────────────────────────────────────────────────────
-- Master agent entity. No KYB required. Links to AgentOwner (Client).
-- kya_status is denormalized from agent_kya for fast compliance gate checks.

CREATE TABLE agent_accounts (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name                    VARCHAR(255) NOT NULL,
    owner_client_id         UUID REFERENCES clients(id) ON DELETE SET NULL,
    api_key_hash            VARCHAR(255) NOT NULL UNIQUE,
    status                  VARCHAR(32) NOT NULL DEFAULT 'active',
    kya_status              VARCHAR(32) NOT NULL DEFAULT 'unverified',
    chain                   VARCHAR(32) NOT NULL DEFAULT 'base',
    prefund_balance_usdc    NUMERIC(28,8) NOT NULL DEFAULT 0,
    reserved_balance_usdc   NUMERIC(28,8) NOT NULL DEFAULT 0,
    metadata                JSONB NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agent_accounts_owner_client ON agent_accounts(owner_client_id);
CREATE INDEX idx_agent_accounts_kya_status ON agent_accounts(kya_status);

-- ── AgentKYA ──────────────────────────────────────────────────────────────────
-- Know Your Agent record. One per AgentAccount.
-- Third-party validator issues a stamped token (validator_token) confirming
-- owner identity, agent purpose, and sanctions clearance.
-- KYA levels gate transaction limits:
--   basic    → ≤ $1/day aggregate outbound
--   enhanced → ≤ $100/day aggregate outbound

CREATE TABLE agent_kya (
    id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_account_id         UUID NOT NULL REFERENCES agent_accounts(id) ON DELETE CASCADE,

    -- Status (mirrors agent_accounts.kya_status, source of truth here)
    status                   VARCHAR(32) NOT NULL DEFAULT 'unverified',
    kya_level                VARCHAR(32) NOT NULL DEFAULT 'unverified',

    -- Third-party validator
    validator_name           VARCHAR(128),
    validator_ref            VARCHAR(255),
    validator_token          TEXT,                  -- encrypted JWT/stamp in prod
    validator_expires_at     TIMESTAMPTZ,
    validated_at             TIMESTAMPTZ,
    last_reviewed_at         TIMESTAMPTZ,

    -- Agent identity fields (captured at KYA submission)
    agent_purpose            TEXT,
    agent_owner_attestation  TEXT,
    declared_use_case        VARCHAR(64),
    -- Allowed values: data_retrieval | content_generation | trading_support | general

    submitted_at             TIMESTAMPTZ,
    metadata                 JSONB NOT NULL DEFAULT '{}',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_agent_kya_agent_account UNIQUE (agent_account_id)
);

-- ── AgentWallets ──────────────────────────────────────────────────────────────
-- Sub-wallet under a MasterAgent (AgentAccount).
-- loop_type is set by AgentOwner at creation and is IMMUTABLE.
-- Spending rules are enforced off-chain in the micropay router.

CREATE TABLE agent_wallets (
    id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_account_id         UUID NOT NULL REFERENCES agent_accounts(id) ON DELETE CASCADE,
    label                    VARCHAR(255) NOT NULL,
    wallet_address           VARCHAR(255) NOT NULL,
    chain                    VARCHAR(32) NOT NULL DEFAULT 'base',
    status                   VARCHAR(32) NOT NULL DEFAULT 'active',

    -- Loop type: owner decides at creation; immutable thereafter
    loop_type                VARCHAR(16) NOT NULL DEFAULT 'open',
    -- Constraint: open | closed

    -- Spending rules
    max_per_tx_usdc          NUMERIC(28,8) NOT NULL DEFAULT 0.10,
    max_daily_usdc           NUMERIC(28,8) NOT NULL DEFAULT 1.00,
    daily_spent_usdc         NUMERIC(28,8) NOT NULL DEFAULT 0,
    daily_reset_at           TIMESTAMPTZ,
    allowed_counterparties   JSONB NOT NULL DEFAULT '[]',
    -- Empty array = any payee permitted; non-empty = allowlist enforced
    expires_at               TIMESTAMPTZ,
    max_uses                 INTEGER,
    use_count                INTEGER NOT NULL DEFAULT 0,

    metadata                 JSONB NOT NULL DEFAULT '{}',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_agent_wallets_agent_account ON agent_wallets(agent_account_id);
CREATE INDEX idx_agent_wallets_wallet_address ON agent_wallets(wallet_address);
CREATE INDEX idx_agent_wallets_loop_type ON agent_wallets(loop_type);

-- ── PaymentIntents ────────────────────────────────────────────────────────────
-- Closed-loop payment intent. Created BEFORE the payment.
-- Mandatory when AgentWallet.loop_type = 'closed'.
--
-- Lifecycle:
--   reserved → consumed   (payment made; consumed_micro_tx_id populated)
--            → expired    (unused past expires_at; intent_sweeper triggers refund)
--            → refunded   (reserved amount returned; refund_tx_id populated)
--            → superseded (purpose changed; superseded_by_intent_id populated)
--
-- Records are NEVER deleted — superseded intents are preserved with a pointer
-- to the replacement intent.

CREATE TABLE payment_intents (
    id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    payer_wallet_id          UUID NOT NULL REFERENCES agent_wallets(id) ON DELETE CASCADE,
    amount_usdc              NUMERIC(28,8) NOT NULL,
    asset                    VARCHAR(16) NOT NULL DEFAULT 'USDC',
    intent_description       TEXT NOT NULL,
    intent_category          VARCHAR(32) NOT NULL DEFAULT 'other',
    -- Allowed: compute | data | agent_service | content | offramp | other
    status                   VARCHAR(32) NOT NULL DEFAULT 'reserved',
    -- Allowed: reserved | consumed | expired | refunded | superseded
    expires_at               TIMESTAMPTZ NOT NULL,

    -- Populated on state transitions
    consumed_micro_tx_id     UUID,           -- FK added below (forward reference)
    superseded_by_intent_id  UUID REFERENCES payment_intents(id),
    refund_tx_id             UUID,           -- FK added below (forward reference)
    audit_note               TEXT,

    metadata                 JSONB NOT NULL DEFAULT '{}',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_payment_intents_payer_wallet ON payment_intents(payer_wallet_id);
CREATE INDEX idx_payment_intents_status ON payment_intents(status);
CREATE INDEX idx_payment_intents_expires_at ON payment_intents(expires_at)
    WHERE status = 'reserved';

-- ── MicroTransactions ─────────────────────────────────────────────────────────
-- Single stablecoin micro-payment. NOT a batch.
-- Settled synchronously off-chain in PostgreSQL; swept on-chain by chain_watcher.
-- idempotency_key is caller-supplied and DB-unique to prevent double-spend.
-- loop_type mirrors payer wallet's loop_type at settlement time (immutable).

CREATE TABLE micro_transactions (
    id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    idempotency_key          VARCHAR(255) NOT NULL UNIQUE,

    -- Parties
    payer_wallet_id          UUID NOT NULL REFERENCES agent_wallets(id),
    payee_address            VARCHAR(255) NOT NULL,
    payee_wallet_id          UUID REFERENCES agent_wallets(id),

    -- Value
    amount_usdc              NUMERIC(28,8) NOT NULL,
    chain                    VARCHAR(32) NOT NULL DEFAULT 'base',
    loop_type                VARCHAR(16) NOT NULL,

    -- Closed-loop linkage
    payment_intent_id        UUID REFERENCES payment_intents(id),

    -- x402 linkage
    x402_payment_header      TEXT,
    x402_resource_url        VARCHAR(2048),

    -- Settlement
    status                   VARCHAR(32) NOT NULL DEFAULT 'pending',
    -- pending | settled_offchain | settled_onchain | failed | refunded
    on_chain_tx_hash         VARCHAR(255),
    on_chain_block           BIGINT,
    on_chain_confirmed_at    TIMESTAMPTZ,
    failure_reason           TEXT,
    settled_at               TIMESTAMPTZ,

    metadata                 JSONB NOT NULL DEFAULT '{}',
    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_micro_tx_payer_wallet ON micro_transactions(payer_wallet_id);
CREATE INDEX idx_micro_tx_status ON micro_transactions(status);
CREATE INDEX idx_micro_tx_payment_intent ON micro_transactions(payment_intent_id);
CREATE INDEX idx_micro_tx_on_chain_tx ON micro_transactions(on_chain_tx_hash)
    WHERE on_chain_tx_hash IS NOT NULL;

-- Add deferred foreign keys from payment_intents → micro_transactions
ALTER TABLE payment_intents
    ADD CONSTRAINT fk_pi_consumed_micro_tx
        FOREIGN KEY (consumed_micro_tx_id) REFERENCES micro_transactions(id) DEFERRABLE INITIALLY DEFERRED,
    ADD CONSTRAINT fk_pi_refund_tx
        FOREIGN KEY (refund_tx_id) REFERENCES micro_transactions(id) DEFERRABLE INITIALLY DEFERRED;

-- ── AgentLedgerEntries ────────────────────────────────────────────────────────
-- Append-only double-entry ledger for AgentAccount balances.
-- balance_after and reserved_balance_after are point-in-time snapshots
-- for audit purposes only. Live balances always come from agent_accounts.

CREATE TABLE agent_ledger_entries (
    id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_account_id         UUID NOT NULL REFERENCES agent_accounts(id),
    entry_type               VARCHAR(32) NOT NULL,
    -- credit | debit | refund | fee | intent_reserve | intent_release
    amount_usdc              NUMERIC(28,8) NOT NULL,
    balance_after            NUMERIC(28,8) NOT NULL,
    reserved_balance_after   NUMERIC(28,8) NOT NULL,

    micro_tx_id              UUID REFERENCES micro_transactions(id),
    payment_intent_id        UUID REFERENCES payment_intents(id),
    on_chain_tx_hash         VARCHAR(255),
    description              TEXT,

    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_ledger_agent_account ON agent_ledger_entries(agent_account_id);
CREATE INDEX idx_ledger_micro_tx ON agent_ledger_entries(micro_tx_id);
CREATE INDEX idx_ledger_payment_intent ON agent_ledger_entries(payment_intent_id);
