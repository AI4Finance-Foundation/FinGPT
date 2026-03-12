/**
 * SDK unit tests — no network required
 */
import { X402Helper, FinogridApiError, X402PaymentRequiredError } from "./client";
import type { X402PaymentRequirement } from "./types";

describe("X402Helper", () => {
  const mockRequirement: X402PaymentRequirement = {
    scheme: "x402",
    version: "1",
    network: "base-mainnet",
    asset: "USDC",
    payTo: "0x0000000000000000000000000000000000000001",
    maxAmountRequired: "0.001",
    resource: "/v1/micropay",
    description: "Access fee",
  };

  it("builds a valid payment signature", () => {
    const sig = X402Helper.buildPaymentSignature(mockRequirement);
    expect(typeof sig).toBe("string");
    const decoded = JSON.parse(Buffer.from(sig, "base64").toString());
    expect(decoded.network).toBe("base-mainnet");
    expect(decoded.asset).toBe("USDC");
    expect(decoded.payTo).toBe(mockRequirement.payTo);
    expect(decoded.amount).toBe(mockRequirement.maxAmountRequired);
    expect(decoded.resource).toBe(mockRequirement.resource);
    expect(typeof decoded.nonce).toBe("string");
    expect(typeof decoded.timestamp).toBe("string");
  });

  it("uses custom amount when provided", () => {
    const sig = X402Helper.buildPaymentSignature(mockRequirement, { amount: "0.005" });
    const decoded = JSON.parse(Buffer.from(sig, "base64").toString());
    expect(decoded.amount).toBe("0.005");
  });

  it("decodes a PAYMENT-REQUIRED header correctly", () => {
    const encoded = Buffer.from(JSON.stringify(mockRequirement)).toString("base64");
    const decoded = X402Helper.decodeRequirement(encoded);
    expect(decoded.scheme).toBe("x402");
    expect(decoded.resource).toBe("/v1/micropay");
    expect(decoded.asset).toBe("USDC");
  });

  it("round-trips encode/decode without loss", () => {
    const encoded = Buffer.from(JSON.stringify(mockRequirement)).toString("base64");
    const decoded = X402Helper.decodeRequirement(encoded);
    expect(decoded).toEqual(mockRequirement);
  });
});

describe("FinogridApiError", () => {
  it("has correct name and statusCode", () => {
    const err = new FinogridApiError(404, "Not found", { detail: "Agent not found" });
    expect(err.name).toBe("FinogridApiError");
    expect(err.statusCode).toBe(404);
    expect(err.message).toContain("404");
    expect(err.message).toContain("Not found");
  });
});

describe("X402PaymentRequiredError", () => {
  it("carries the requirement object", () => {
    const mockRequirementEncoded: X402PaymentRequirement = {
      scheme: "x402",
      version: "1",
      network: "base-mainnet",
      asset: "USDC",
      payTo: "0xabc",
      maxAmountRequired: "0.001",
      resource: "/test",
      description: "test",
    };
    const err = new X402PaymentRequiredError(mockRequirementEncoded);
    expect(err.name).toBe("X402PaymentRequiredError");
    expect(err.requirement?.payTo).toBe("0xabc");
    expect(err.message).toContain("402");
  });

  it("handles null requirement gracefully", () => {
    const err = new X402PaymentRequiredError(null);
    expect(err.requirement).toBeNull();
  });
});
