-- Migration 003: Mandate, Principal, and MandateEvent tables
-- Additive — zero changes to any v1 or agent_ledger tables.
--
-- Mandate is the core authorisation primitive:
--   Principal → grants → Mandate → authorises → AgentAccount
-- Every money movement must be traceable to a valid active Mandate.

BEGIN;

-- ── Enums ─────────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'mandate_status') THEN
        CREATE TYPE mandate_status AS ENUM (
            'draft', 'active', 'suspended', 'revoked', 'expired', 'superseded'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'mandate_scope') THEN
        CREATE TYPE mandate_scope AS ENUM (
            'payout', 'collect', 'topup', 'full', 'read_only'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'approval_mode') THEN
        CREATE TYPE approval_mode AS ENUM (
            'auto', 'manual', 'threshold'
        );
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'mandate_event_type') THEN
        CREATE TYPE mandate_event_type AS ENUM (
            'created', 'activated', 'suspended', 'resumed',
            'revoked', 'expired', 'superseded', 'limit_hit'
        );
    END IF;
END $$;

-- ── principals ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS principals (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(255) NOT NULL,
    email       VARCHAR(255),
    client_id   UUID REFERENCES clients(id) ON DELETE SET NULL,   -- link to v1 Client
    status      VARCHAR(50)  NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'suspended', 'terminated')),
    kyb_status  VARCHAR(50)  NOT NULL DEFAULT 'pending',
    metadata    JSONB,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_principals_client_id ON principals(client_id) WHERE client_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_principals_status ON principals(status);

-- ── mandates ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mandates (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    principal_id                UUID NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
    agent_account_id            UUID NOT NULL REFERENCES agent_accounts(id) ON DELETE RESTRICT,
    status                      mandate_status NOT NULL DEFAULT 'draft',
    scope                       mandate_scope  NOT NULL DEFAULT 'payout',
    approval_mode               approval_mode  NOT NULL DEFAULT 'auto',

    -- Amount limits (USDC)
    max_amount_per_tx_usdc      NUMERIC(28,8),
    max_daily_usdc              NUMERIC(28,8),
    max_monthly_usdc            NUMERIC(28,8),
    approval_threshold_usdc     NUMERIC(28,8),
    lifetime_cap_usdc           NUMERIC(28,8),
    lifetime_spent_usdc         NUMERIC(28,8) NOT NULL DEFAULT 0,

    -- Scope constraints (empty array or NULL = no restriction)
    allowed_corridors           JSONB DEFAULT '[]'::jsonb,
    allowed_assets              JSONB DEFAULT '[]'::jsonb,
    allowed_chains              JSONB DEFAULT '[]'::jsonb,
    allowed_counterparties      JSONB DEFAULT '[]'::jsonb,

    -- Validity
    activated_at                TIMESTAMPTZ,
    expires_at                  TIMESTAMPTZ,
    revoked_at                  TIMESTAMPTZ,
    revocation_reason           TEXT,

    -- Supersession chain
    superseded_by_mandate_id    UUID REFERENCES mandates(id) ON DELETE SET NULL,
    supersession_note           TEXT,

    -- Metadata
    description                 TEXT,
    metadata                    JSONB,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mandates_principal ON mandates(principal_id);
CREATE INDEX IF NOT EXISTS idx_mandates_agent ON mandates(agent_account_id);
CREATE INDEX IF NOT EXISTS idx_mandates_status ON mandates(status);
CREATE INDEX IF NOT EXISTS idx_mandates_active ON mandates(agent_account_id, status)
    WHERE status = 'active';

-- ── mandate_events ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS mandate_events (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    mandate_id       UUID NOT NULL REFERENCES mandates(id) ON DELETE CASCADE,
    event_type       mandate_event_type NOT NULL,
    actor            VARCHAR(255) NOT NULL,
    previous_status  VARCHAR(50),
    new_status       VARCHAR(50),
    note             TEXT,
    details          JSONB,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mandate_events_mandate ON mandate_events(mandate_id);
CREATE INDEX IF NOT EXISTS idx_mandate_events_type ON mandate_events(event_type);

-- ── Trigger: auto-update updated_at ──────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_updated_at_principals') THEN
        CREATE TRIGGER set_updated_at_principals
            BEFORE UPDATE ON principals
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_updated_at_mandates') THEN
        CREATE TRIGGER set_updated_at_mandates
            BEFORE UPDATE ON mandates
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

COMMIT;
