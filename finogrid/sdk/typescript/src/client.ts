/**
 * Finogrid Agent Ledger SDK — HTTP Client
 *
 * Usage:
 *   const finogrid = new FinogridClient({
 *     baseUrl: "https://agent-ledger.finogrid.io",
 *     apiKey: "fig_agent_...",
 *   });
 *
 *   // Register agent
 *   const agent = await finogrid.agents.create({ name: "my-agent", owner_client_id: "..." });
 *
 *   // Submit KYA
 *   await finogrid.kya.submit(agent.agent_account_id, { agent_purpose: "...", ... });
 *
 *   // Create closed-loop wallet
 *   const wallet = await finogrid.wallets.create(agent.agent_account_id, {
 *     wallet_address: "0x...",
 *     loop_type: "closed",
 *     max_per_tx_usdc: "0.10",
 *     max_daily_usdc: "1.00",
 *   });
 *
 *   // Reserve intent
 *   const intent = await finogrid.paymentIntents.create({
 *     payer_wallet_id: wallet.wallet_id,
 *     amount_usdc: "0.05",
 *     intent_description: "Pay for AI inference call",
 *     intent_category: "compute",
 *     expires_at: new Date(Date.now() + 300_000).toISOString(),
 *   });
 *
 *   // Execute micropay
 *   const tx = await finogrid.micropay.pay({
 *     idempotency_key: crypto.randomUUID(),
 *     payer_wallet_id: wallet.wallet_id,
 *     payee_address: "0xpayee...",
 *     amount_usdc: "0.05",
 *     payment_intent_id: intent.payment_intent_id,
 *   });
 */
import axios, { AxiosInstance, AxiosRequestConfig } from "axios";
import {
  FinogridClientConfig,
  AgentAccountCreateRequest, AgentAccountCreateResponse,
  AgentBalanceResponse,
  KYASubmitRequest, KYAStatusResponse,
  AgentWalletCreateRequest, AgentWalletCreateResponse, AgentWallet,
  PaymentIntentCreateRequest, PaymentIntentCreateResponse, PaymentIntentSupersede,
  MicroPayRequest, MicroPayResponse,
  TopUpRequest, TopUpResponse,
  MandateCreateRequest, MandateResponse,
  X402PaymentRequirement, X402PaymentSignature,
} from "./types";

// ── Base HTTP client ──────────────────────────────────────────────────────────

class BaseClient {
  protected http: AxiosInstance;
  private retries: number;

  constructor(config: FinogridClientConfig) {
    this.retries = config.retries ?? 2;
    this.http = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout ?? 10_000,
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": config.apiKey,
      },
    });
  }

  protected async request<T>(config: AxiosRequestConfig, attempt = 0): Promise<T> {
    try {
      const response = await this.http.request<T>(config);
      return response.data;
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response) {
        // Parse x402 Payment Required
        if (err.response.status === 402) {
          const requirementHeader = err.response.headers["payment-required"];
          const requirement = requirementHeader
            ? (JSON.parse(Buffer.from(requirementHeader, "base64").toString()) as X402PaymentRequirement)
            : null;
          throw new X402PaymentRequiredError(requirement, err.response.data);
        }
        throw new FinogridApiError(
          err.response.status,
          err.response.data?.detail ?? "API error",
          err.response.data
        );
      }
      // Network error — retry with exponential backoff
      if (attempt < this.retries) {
        await sleep(200 * Math.pow(2, attempt));
        return this.request<T>(config, attempt + 1);
      }
      throw err;
    }
  }
}

// ── Resource clients ──────────────────────────────────────────────────────────

export class AgentsClient extends BaseClient {
  async create(req: AgentAccountCreateRequest): Promise<AgentAccountCreateResponse> {
    return this.request({ method: "POST", url: "/v1/agent-accounts", data: req });
  }

  async getBalance(agentAccountId: string): Promise<AgentBalanceResponse> {
    return this.request({ method: "GET", url: `/v1/agent-accounts/${agentAccountId}/balance` });
  }

  async topup(agentAccountId: string, req: TopUpRequest): Promise<TopUpResponse> {
    return this.request({ method: "POST", url: `/v1/agent-accounts/${agentAccountId}/topup`, data: req });
  }
}

export class KYAClient extends BaseClient {
  async submit(agentAccountId: string, req: KYASubmitRequest): Promise<KYAStatusResponse> {
    return this.request({ method: "POST", url: `/v1/agent-accounts/${agentAccountId}/kya`, data: req });
  }

  async getStatus(agentAccountId: string): Promise<KYAStatusResponse> {
    return this.request({ method: "GET", url: `/v1/agent-accounts/${agentAccountId}/kya` });
  }

  /** Poll until KYA reaches target status or timeout. */
  async pollUntil(
    agentAccountId: string,
    targetStatus: "basic" | "enhanced",
    opts: { intervalMs?: number; timeoutMs?: number } = {}
  ): Promise<KYAStatusResponse> {
    const interval = opts.intervalMs ?? 2000;
    const timeout = opts.timeoutMs ?? 60_000;
    const deadline = Date.now() + timeout;

    while (Date.now() < deadline) {
      const status = await this.getStatus(agentAccountId);
      const order = { unverified: 0, pending: 1, basic: 2, enhanced: 3 };
      if (order[status.kya_status] >= order[targetStatus]) return status;
      await sleep(interval);
    }
    throw new Error(`KYA did not reach '${targetStatus}' within ${timeout}ms`);
  }
}

export class WalletsClient extends BaseClient {
  async create(agentAccountId: string, req: AgentWalletCreateRequest): Promise<AgentWalletCreateResponse> {
    return this.request({ method: "POST", url: `/v1/agent-accounts/${agentAccountId}/wallets`, data: req });
  }

  async list(agentAccountId: string): Promise<{ agent_account_id: string; wallets: AgentWallet[] }> {
    return this.request({ method: "GET", url: `/v1/agent-accounts/${agentAccountId}/wallets` });
  }
}

export class PaymentIntentsClient extends BaseClient {
  async create(req: PaymentIntentCreateRequest): Promise<PaymentIntentCreateResponse> {
    return this.request({ method: "POST", url: "/v1/payment-intents", data: req });
  }

  async supersede(intentId: string, req: PaymentIntentSupersede): Promise<PaymentIntentCreateResponse> {
    return this.request({ method: "PATCH", url: `/v1/payment-intents/${intentId}`, data: req });
  }
}

export class MicropayClient extends BaseClient {
  async pay(req: MicroPayRequest): Promise<MicroPayResponse> {
    return this.request({ method: "POST", url: "/v1/micropay", data: req });
  }
}

export class MandatesClient extends BaseClient {
  async create(req: MandateCreateRequest): Promise<MandateResponse> {
    return this.request({ method: "POST", url: "/v1/mandates", data: req });
  }

  async get(mandateId: string): Promise<MandateResponse> {
    return this.request({ method: "GET", url: `/v1/mandates/${mandateId}` });
  }

  async revoke(mandateId: string, reason: string): Promise<MandateResponse> {
    return this.request({
      method: "POST",
      url: `/v1/mandates/${mandateId}/revoke`,
      data: { reason },
    });
  }
}

// ── x402 helpers ──────────────────────────────────────────────────────────────

export class X402Helper {
  /**
   * Build a PAYMENT-SIGNATURE header to pay for an x402 protected resource.
   * In production, this signature is a real on-chain USDC transfer receipt.
   * For testing: generates the signed structure with a nonce + timestamp.
   */
  static buildPaymentSignature(
    requirement: X402PaymentRequirement,
    opts: { amount?: string } = {}
  ): string {
    const sig: X402PaymentSignature = {
      network: requirement.network,
      asset: requirement.asset,
      payTo: requirement.payTo,
      amount: opts.amount ?? requirement.maxAmountRequired,
      nonce: crypto.randomUUID(),
      timestamp: String(Date.now() / 1000),
      resource: requirement.resource,
    };
    return Buffer.from(JSON.stringify(sig)).toString("base64");
  }

  /** Decode a PAYMENT-REQUIRED header from a 402 response. */
  static decodeRequirement(header: string): X402PaymentRequirement {
    return JSON.parse(Buffer.from(header, "base64").toString()) as X402PaymentRequirement;
  }
}

// ── Main SDK entry point ──────────────────────────────────────────────────────

export class FinogridClient {
  public readonly agents: AgentsClient;
  public readonly kya: KYAClient;
  public readonly wallets: WalletsClient;
  public readonly paymentIntents: PaymentIntentsClient;
  public readonly micropay: MicropayClient;
  public readonly mandates: MandatesClient;
  public readonly x402: typeof X402Helper;

  constructor(config: FinogridClientConfig) {
    this.agents = new AgentsClient(config);
    this.kya = new KYAClient(config);
    this.wallets = new WalletsClient(config);
    this.paymentIntents = new PaymentIntentsClient(config);
    this.micropay = new MicropayClient(config);
    this.mandates = new MandatesClient(config);
    this.x402 = X402Helper;
  }
}

// ── Error classes ─────────────────────────────────────────────────────────────

export class FinogridApiError extends Error {
  constructor(
    public readonly statusCode: number,
    message: string,
    public readonly body?: unknown
  ) {
    super(`Finogrid API ${statusCode}: ${message}`);
    this.name = "FinogridApiError";
  }
}

export class X402PaymentRequiredError extends Error {
  constructor(
    public readonly requirement: X402PaymentRequirement | null,
    public readonly body?: unknown
  ) {
    super("402 Payment Required — x402 payment needed to access resource");
    this.name = "X402PaymentRequiredError";
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
