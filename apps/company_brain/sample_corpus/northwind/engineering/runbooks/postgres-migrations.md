---
title: "Runbook: Postgres schema migrations"
owners: [sam.okafor@northwind.test]
updated_at: 2026-10-21
acl: [group:engineering, group:leadership]
source_type: page
---

# Runbook: Postgres schema migrations

Owner: Sam Okafor (Staff Engineer, Data). Ask in #eng-database before your first migration.

## Before you write the migration

- Migrations live in `pulse-api/migrations/` and run with `alembic upgrade head` during deploy.
- Every migration must be **backwards compatible** with the code currently in production: add
  columns as nullable, backfill in a separate job, and only then add constraints.
- Never rename or drop a column in the same deploy that stops using it. Do it in a follow-up.

## Locks

The `orders`, `events` and `accounts` tables are hot. On those tables:

- Create indexes with `CREATE INDEX CONCURRENTLY`, in their own migration, outside a transaction.
- Set `lock_timeout = '3s'` at the top of the migration so it fails fast instead of queueing
  behind long queries.
- Don't add a `NOT NULL` constraint or a foreign key directly. Add it as `NOT VALID`, then run
  `VALIDATE CONSTRAINT` separately.

## Running it

1. Run the migration against the staging snapshot (`make migrate-staging`) and record how long it took.
2. Anything over 30 seconds on staging needs a plan in the PR description.
3. Deploy outside peak hours (peak is 13:00–17:00 UTC).
4. Watch the `pg_locks` panel on the Database dashboard for the first ten minutes.

## Reviews

Migrations need two approvals, one from the database owners (Sam Okafor, Ines Duarte).
