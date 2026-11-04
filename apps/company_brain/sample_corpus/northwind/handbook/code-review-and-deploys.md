---
title: Code review and deploys
owners: [priya.raman@northwind.test]
updated_at: 2026-03-04
acl: [group:everyone]
source_type: page
---

# Code review and deploys

This page describes how code gets from a branch into production at Northwind Labs. It applies
to every repository in the `northwind-labs` GitHub organization, including the Pulse API, the
web app and the ingestion workers.

## Reviews

- Every pull request needs **two approving reviews** before it can merge, one of them from a
  code owner of the area it touches.
- Authors should keep PRs small. Anything over 400 changed lines should be split, or come with
  a note explaining why it can't be.
- Database migrations also need a review from the database owners
  (see the Postgres migrations runbook).

## Deploys

- `main` deploys to production automatically after CI passes, through the Argo pipeline.
- **No production deploys on Fridays** or the day before a public holiday. If something must
  ship, the on-call engineer and an engineering manager both have to agree in #eng-deploys.
- Every deploy that changes customer-visible behaviour goes behind a feature flag.

## Rollbacks

Roll back first, debug second. Anyone on the on-call rotation may roll back a deploy without
asking.

## Owner

Priya Raman (Engineering Manager, Platform) owns this page. Proposals to change it go through an
engineering decision record in `engineering/decisions/`.
