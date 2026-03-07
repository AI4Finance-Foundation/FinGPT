-- Finogrid v1 Initial Schema
-- Run against PostgreSQL / AlloyDB

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Clients ───────────────────────────────────────────────────────────────────
CREATE TABLE clients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    legal_name VARCHAR(512) NOT NULL,
    registration_country CHAR(2) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'pending_kyb',
    kyb_status VARCHAR(32) NOT NULL DEFAULT 'not_started',
    kyb_provider_ref VARCHAR(255),
    webhook_url VARCHAR(1024),
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE client_corridor_permissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id) ON DELETE CASCADE,
    corridor_code CHAR(2) NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    max_single_payout_usd INTEGER NOT NULL DEFAULT 10000,
    max_daily_volume_usd INTEGER NOT NULL DEFAULT 100000,
    allowed_assets JSONB NOT NULL DEFAULT '["USDT","USDC"]',
    allowed_modes JSONB NOT NULL DEFAULT '["wallet","fiat"]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(client_id, corridor_code)
);

-- ── Batches ───────────────────────────────────────────────────────────────────
CREATE TABLE batches (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    client_id UUID NOT NULL REFERENCES clients(id),
    reference VARCHAR(255) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'draft',
    total_amount_usd NUMERIC(18,6) NOT NULL,
    task_count INTEGER NOT NULL DEFAULT 0,
    completed_count INTEGER NOT NULL DEFAULT 0,
    failed_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE payout_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id UUID NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    corridor_code CHAR(2) NOT NULL,
    recipient_name VARCHAR(512) NOT NULL,
    recipient_ref VARCHAR(255) NOT NULL,
    amount_usd NUMERIC(18,6) NOT NULL,
    preferred_asset VARCHAR(16),
    preferred_mode VARCHAR(16),
    resolved_asset VARCHAR(16),
    resolved_mode VARCHAR(16),
    partner_route VARCHAR(128),
    fallback_route VARCHAR(128),
    status VARCHAR(32) NOT NULL DEFAULT 'pending',
    compliance_result JSONB,
    partner_tx_id VARCHAR(255),
    failure_reason TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    beneficiary_data JSONB NOT NULL DEFAULT '{}',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_payout_tasks_batch_id ON payout_tasks(batch_id);
CREATE INDEX idx_payout_tasks_status ON payout_tasks(status);
CREATE INDEX idx_payout_tasks_corridor ON payout_tasks(corridor_code);

-- ── Execution Events ──────────────────────────────────────────────────────────
CREATE TABLE execution_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id UUID NOT NULL REFERENCES payout_tasks(id) ON DELETE CASCADE,
    event_type VARCHAR(64) NOT NULL,
    partner VARCHAR(128),
    partner_ref VARCHAR(255),
    payload JSONB NOT NULL DEFAULT '{}',
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_execution_events_task_id ON execution_events(task_id);

-- ── Audit Logs (append-only) ──────────────────────────────────────────────────
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    actor_type VARCHAR(64) NOT NULL,
    actor_id VARCHAR(255),
    action VARCHAR(128) NOT NULL,
    resource_type VARCHAR(64),
    resource_id VARCHAR(255),
    corridor_code CHAR(2),
    client_id UUID,
    batch_id UUID,
    task_id UUID,
    before_state JSONB,
    after_state JSONB,
    detail TEXT,
    ip_address VARCHAR(45),
    request_id VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_logs_batch_id ON audit_logs(batch_id);
CREATE INDEX idx_audit_logs_client_id ON audit_logs(client_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);

-- ── Routing Profiles (per corridor) ──────────────────────────────────────────
CREATE TABLE routing_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    corridor_code CHAR(2) UNIQUE NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    usdt_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    usdc_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    wallet_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    fiat_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    wallet_partner VARCHAR(128) NOT NULL DEFAULT 'bridge',
    fiat_partner VARCHAR(128) NOT NULL,
    fallback_partner VARCHAR(128),
    min_amount_usd NUMERIC(18,2) NOT NULL DEFAULT 10.00,
    max_amount_usd NUMERIC(18,2) NOT NULL DEFAULT 50000.00,
    wallet_sla_minutes INTEGER NOT NULL DEFAULT 60,
    fiat_sla_minutes INTEGER NOT NULL DEFAULT 1440,
    required_beneficiary_fields JSONB NOT NULL DEFAULT '[]',
    max_retries INTEGER NOT NULL DEFAULT 3,
    retry_backoff_seconds INTEGER NOT NULL DEFAULT 60,
    extra_config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Compliance Profiles (per corridor) ───────────────────────────────────────
CREATE TABLE compliance_profiles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    corridor_code CHAR(2) UNIQUE NOT NULL,
    kyt_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    kyt_risk_threshold INTEGER NOT NULL DEFAULT 7,
    sanctions_screen_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    travel_rule_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    travel_rule_threshold_usd NUMERIC(18,2),
    originator_data_required BOOLEAN NOT NULL DEFAULT TRUE,
    beneficiary_data_required BOOLEAN NOT NULL DEFAULT TRUE,
    auto_hold_on_high_risk BOOLEAN NOT NULL DEFAULT TRUE,
    manual_review_threshold INTEGER NOT NULL DEFAULT 8,
    extra_rules JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Seed: v1 corridor profiles ────────────────────────────────────────────────
INSERT INTO routing_profiles (corridor_code, fiat_partner, wallet_sla_minutes, fiat_sla_minutes, required_beneficiary_fields) VALUES
    ('BR', 'bridge_pix',     60,   120,  '["pix_key","recipient_document"]'),
    ('AR', 'bridge_ar',      60,   2880, '["cbu_or_alias","cuit"]'),
    ('VN', 'bridge_vn',      60,   480,  '["bank_account_number","bank_bin_code"]'),
    ('IN', 'bridge_upi',     60,   240,  '["upi_id"]'),
    ('AE', 'bridge_ae',      60,   480,  '["iban","swift_bic"]'),
    ('ID', 'bridge_id',      60,   240,  '["bank_account_number","bank_code"]'),
    ('PH', 'bridge_ph',      60,   120,  '["bank_account_number","bank_code"]'),
    ('NG', 'bridge_ng',      60,   1440, '["bank_account_number","bank_code","bvn"]');

INSERT INTO compliance_profiles (corridor_code, kyt_risk_threshold) VALUES
    ('BR', 7), ('AR', 6), ('VN', 7), ('IN', 7),
    ('AE', 6), ('ID', 7), ('PH', 7), ('NG', 6);
