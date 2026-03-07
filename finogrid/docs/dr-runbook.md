# Finogrid Disaster Recovery Runbook

**Owner:** DevOps | **Updated:** March 7, 2026

---

## 1. Bridge API Outage

**Symptoms:** Execution events timing out; partner_tx_id not assigned; tasks stuck in EXECUTING.

**Steps:**
1. Check Bridge status page and Slack incident channel
2. Pause new batch processing: set `routing_profiles.enabled = false` for affected corridors
3. Tasks already submitted: wait for Bridge recovery; retries will resume automatically (max 3)
4. If Bridge outage > 2 hours: notify affected clients via webhook + email
5. On Bridge recovery: re-enable corridors, reconcile open tasks

---

## 2. KYT Provider Outage

**Symptoms:** Compliance gate failing; tasks stuck in COMPLIANCE_CHECK.

**Steps:**
1. Check KYT provider status
2. If outage < 30 min: queue tasks; auto-retry on recovery
3. If outage > 30 min: escalate to compliance lead — decide whether to:
   - Hold all tasks until provider recovers (safest)
   - Enable fallback screening (requires compliance approval)
4. Document all decisions in audit_logs with actor_type = "operator"

---

## 3. Database Outage (AlloyDB)

**Symptoms:** All services returning 500; DB connection pool exhausted.

**Steps:**
1. Check AlloyDB console — failover to read replica if primary down
2. Cloud Run services will auto-retry DB connections (pool_pre_ping=True)
3. If regional outage: trigger cross-region failover via AlloyDB console
4. Pub/Sub messages will queue during DB outage — they will be processed on recovery
5. After recovery: run reconciliation job manually for all EXECUTING tasks

---

## 4. Pub/Sub Topic Backlog

**Symptoms:** Tasks processing slowly; topic backlog > 10,000 messages.

**Steps:**
1. Scale up Cloud Run instances for the consumer service
2. Check for poison pill messages (repeated failures): purge with `gcloud pubsub subscriptions seek`
3. If backlog is due to DB slowness: address DB first, then Pub/Sub will clear

---

## 5. Runaway Agent

**Symptoms:** Agent sending unexpected webhooks or making unexpected API calls.

**Steps:**
1. Agents have read-only DB access and cannot release payouts — impact is limited
2. If agent is making unexpected external calls: revoke its service account in IAM
3. Review agent logs in Cloud Logging
4. Agents run on Cloud Scheduler — disable the schedule to stop execution

---

## 6. Stablecoin Depeg Event

**Symptoms:** OpsOversightAgent fires critical alert for USDT or USDC depeg > 50bps.

**Steps:**
1. Immediately pause new batch creation for affected asset (toggle routing_profiles)
2. Notify all active clients with pending batches
3. For in-flight tasks: let them complete (Bridge conversion already locked in)
4. Convene leadership within 1 hour to assess resumption timeline
5. Document in audit_logs

---

## Recovery Contacts

| System | Contact | Escalation |
|--------|---------|-----------|
| Bridge | account@bridge.xyz | Slack #bridge-oncall |
| KYT Provider | support@provider.com | Phone: +1-xxx |
| AlloyDB / GCP | GCP support ticket | GCP console |
| Internal oncall | PagerDuty rotation | #ops-alerts |
