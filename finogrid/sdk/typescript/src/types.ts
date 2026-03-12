/**
 * Finogrid Agent Ledger SDK — Core Types
 */

// ── Enums ─────────────────────────────────────────────────────────────────────

export type KYAStatus = "unverified" | "pending" | "basic" | "enhanced";
export type LoopType = "open" | "closed";
export type IntentStatus = "reserved" | "consumed" | "expired" | "refunded" | "superseded";
export type IntentCategory = "compute" | "data" | "agent_service" | "content" | "offramp" | "other";
export type MicroTxStatus = "pending" | "settled_offchain" | "settled_onchain" | "failed" | "refunded";
export type MandateStatus = "draft" | "active" | "suspended" | "revoked" | "expired" | "superseded";
export type MandateScope = "payout" | "collect" | "topup" | "full" | "read_only";
export type ApprovalMode = "auto" | "manual" | "threshold";

// ── Agent Account ─────────────────────────────────────────────────────────────

export interface AgentAccountCreateRequest {
  name: string;
  owner_client_id: string;
  chain?: string;
}

export interface AgentAccountCreateResponse {
  agent_account_id: string;
  name: string;
  api_key: string;  // Returned only once at creation — store securely
  chain: string;
  kya_status: KYAStatus;
  message: string;
}

export interface AgentBalanceResponse {
  agent_account_id: string;
  prefund_balance_usdc: string;
  reserved_balance_usdc: string;
  available_balance_usdc: string;
  kya_status: KYAStatus;
  chain: string;
  recent_ledger: LedgerEntry[];
}

export interface LedgerEntry {
  entry_id: string;
  entry_type: string;
  amount_usdc: string;
  balance_after: string;
  reserved_balance_after: string;
  description: string;
  created_at: string;
}

// ── KYA ───────────────────────────────────────────────────────────────────────

export interface KYASubmitRequest {
  agent_purpose: string;
  declared_use_case: string;
  agent_owner_attestation: string;
  validator_name?: string;
}

export interface KYAStatusResponse {
  agent_account_id: string;
  kya_status: KYAStatus;
  validator_name: string | null;
  validator_ref: string | null;
  validator_token_present: boolean;
  validator_expires_at: string | null;
  validated_at: string | null;
  message: string;
}

// ── Wallets ───────────────────────────────────────────────────────────────────

export interface AgentWalletCreateRequest {
  wallet_address: string;
  loop_type: LoopType;
  label?: string;
  max_per_tx_usdc: string;
  max_daily_usdc: string;
  allowed_counterparties?: string[];
  expires_at?: string;
  max_uses?: number;
}

export interface AgentWalletCreateResponse {
  wallet_id: string;
  wallet_address: string;
  loop_type: LoopType;
  spending_rules: Record<string, unknown>;
}

export interface AgentWallet {
  wallet_id: string;
  label: string | null;
  wallet_address: string;
  loop_type: LoopType;
  status: string;
  max_per_tx_usdc: number;
  max_daily_usdc: number;
  daily_spent_usdc: number;
  use_count: number;
  max_uses: number | null;
  expires_at: string | null;
}

// ── Payment Intents ───────────────────────────────────────────────────────────

export interface PaymentIntentCreateRequest {
  payer_wallet_id: string;
  amount_usdc: string;
  intent_description: string;
  intent_category: IntentCategory;
  expires_at: string;  // ISO 8601
}

export interface PaymentIntentCreateResponse {
  payment_intent_id: string;
  payer_wallet_id: string;
  amount_usdc: string;
  intent_category: IntentCategory;
  expires_at: string;
  message?: string;
}

export interface PaymentIntentSupersede {
  new_amount_usdc: string;
  new_intent_description: string;
  new_intent_category: IntentCategory;
  new_expires_at: string;
  audit_note: string;  // Required — explains why intent changed
}

// ── Micropay ──────────────────────────────────────────────────────────────────

export interface MicroPayRequest {
  idempotency_key: string;
  payer_wallet_id: string;
  payee_address: string;
  amount_usdc: string;
  payment_intent_id?: string;  // Required for closed-loop wallets
  x402_payment_header?: string;
  x402_resource_url?: string;
  metadata?: Record<string, unknown>;
}

export interface MicroPayResponse {
  transaction_id: string;
  idempotency_key: string;
  status: MicroTxStatus;
  amount_usdc: string;
  loop_type: LoopType;
  payment_intent_id: string | null;
  payer_available_balance_after: string;
  settled_at: string;
  on_chain_tx_hash: string | null;
  message?: string;
}

// ── Top-up ────────────────────────────────────────────────────────────────────

export interface TopUpRequest {
  deposit_tx_hash: string;
}

export interface TopUpResponse {
  agent_account_id: string;
  deposit_tx_hash: string;
  status: "pending_confirmation" | "credited";
  message: string;
}

// ── Mandates ──────────────────────────────────────────────────────────────────

export interface MandateCreateRequest {
  principal_id: string;
  agent_account_id: string;
  scope: MandateScope;
  approval_mode: ApprovalMode;
  max_amount_per_tx_usdc?: string;
  max_daily_usdc?: string;
  max_monthly_usdc?: string;
  approval_threshold_usdc?: string;
  lifetime_cap_usdc?: string;
  allowed_corridors?: string[];
  allowed_assets?: string[];
  allowed_chains?: string[];
  allowed_counterparties?: string[];
  expires_at?: string;
  description?: string;
}

export interface MandateResponse {
  mandate_id: string;
  principal_id: string;
  agent_account_id: string;
  status: MandateStatus;
  scope: MandateScope;
  approval_mode: ApprovalMode;
  max_amount_per_tx_usdc: string | null;
  max_daily_usdc: string | null;
  approval_threshold_usdc: string | null;
  allowed_corridors: string[];
  allowed_chains: string[];
  activated_at: string | null;
  expires_at: string | null;
  created_at: string;
}

// ── x402 ─────────────────────────────────────────────────────────────────────

export interface X402PaymentRequirement {
  scheme: "x402";
  version: string;
  network: string;
  asset: string;
  payTo: string;
  maxAmountRequired: string;
  resource: string;
  description: string;
}

export interface X402PaymentSignature {
  network: string;
  asset: string;
  payTo: string;
  amount: string;
  nonce: string;
  timestamp: string;
  resource: string;
}

// ── SDK Config ────────────────────────────────────────────────────────────────

export interface FinogridClientConfig {
  baseUrl: string;
  apiKey: string;
  timeout?: number;   // ms, default 10000
  retries?: number;   // default 2
}
