# Nigeria (NG) Corridor Runbook

**Status:** v1 Launch Market | **Last updated:** March 7, 2026

---

## Why Nigeria (not "Africa")

Nigeria is modeled as its own corridor with its own adapter, partner map, and risk profile.
"Africa" is not a corridor. Different African markets have different rails, regulators,
risk profiles, and AML requirements. Nigeria is our v1 Africa entry point only.

---

## Key Facts

- Largest stablecoin market in Sub-Saharan Africa
- Top-tier global stablecoin adoption (Chainalysis reports)
- Strong remittance and cross-border freelancer use case
- Regulated by CBN (Central Bank of Nigeria) + SEC Nigeria
- USDT on TRON (TRC-20) is widely used due to low fees

---

## Payout Modes

| Mode | Rail | Partner | SLA |
|------|------|---------|-----|
| Wallet | USDT (TRC-20) or USDC (ETH) | Bridge | 60 min |
| Fiat NGN | NIBSS bank transfer | Bridge NG off-ramp | 24 hours |

---

## Required Beneficiary Fields

**Wallet delivery:**
- recipient_wallet (TRC-20 or ERC-20 address)

**Fiat NGN delivery:**
- bank_account_number (10-digit NUBAN)
- bank_code (CBN bank code)
- bvn (Bank Verification Number — required for amounts ≥ $100 per CBN AML rules)

---

## Compliance Rules

- KYT: enabled, threshold 6/10 (stricter than default)
- Sanctions: OFAC + UN + CBN watch lists
- BVN mandatory for fiat delivery ≥ $100
- Travel Rule: disabled in v1

---

## Exception Policy

- Max retries: 3
- On hard fail: cancel and notify client
- On compliance hold: escalate to manual review within 2 business hours

---

## Known Issues

- NGN fiat SLA can extend to 48h during CBN settlement windows (public holidays)
- TRC-20 network congestion can delay wallet delivery — monitor Tronscan
- Some Nigerian banks reject transfers from non-Nigerian originators — verify via partner before routing
