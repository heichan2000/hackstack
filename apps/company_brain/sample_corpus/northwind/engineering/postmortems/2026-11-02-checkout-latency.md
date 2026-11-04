---
title: "Postmortem: checkout latency spike, 2 November 2026"
owners: [sam.okafor@northwind.test, tomas.silva@northwind.test]
updated_at: 2026-11-02
acl: [group:engineering, group:leadership]
source_type: document
---

# Postmortem: checkout latency spike, 2 November 2026

- **Severity:** SEV-2
- **Duration:** 41 minutes (14:07–14:48 UTC)
- **Incident lead:** Tomás Silva (on-call SRE)
- **Write-up:** Sam Okafor

## What happened

At 14:07 UTC a deploy ran a migration that added an index on `orders.customer_id` **without**
`CONCURRENTLY`. The migration took an exclusive lock on `orders`, and every checkout request
queued behind it. p95 checkout latency went from 180 ms to 9.4 s, and about 6% of checkouts
timed out.

## Timeline (UTC)

- 14:07 Deploy starts; the migration begins.
- 14:11 Latency alert fires; Tomás acknowledges.
- 14:19 Lock identified on the `pg_locks` panel.
- 14:31 Migration cancelled; deploy rolled back.
- 14:48 Latency back to normal.

## Why it happened

The migration was written before the Postgres migrations runbook was updated on 21 October, and
the PR had one approval because it was small. Under EDR-014, migrations always need two
approvals, but the `review-policy` CI check didn't yet recognise migration files.

## Actions

- [x] `review-policy` now treats any file in `migrations/` as needing two approvals (Sam, done 3 Nov).
- [ ] Add a lint that rejects `CREATE INDEX` without `CONCURRENTLY` on hot tables (Ines, due 10 Nov).
- [ ] Add migration duration to the deploy summary in #eng-deploys (Tomás, due 14 Nov).
