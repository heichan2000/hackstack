---
title: "Release notes: Pulse 3.4"
owners: [lena.fischer@northwind.test]
updated_at: 2026-11-02
acl: [group:everyone]
source_type: document
---

# Release notes: Pulse 3.4

Released **2 November 2026**. Owner: Lena Fischer (Product Manager, Alerts).

Pulse is Northwind Labs' product: usage monitoring and alerting for B2B SaaS teams.

## New

- **Usage alerts.** Customers can now get a Slack or email alert when an account's weekly active
  users drop by more than a threshold they choose. On for every plan.
- **Saved views** for the accounts table, shareable by link.

## Changed

- The default data retention on the Starter plan goes from 12 to 6 months for **new** customers.
  Existing customers keep 12 months.
- The `/v1/accounts` API now paginates with cursors. Offset pagination still works until
  31 January 2027.

## Fixed

- CSV exports no longer drop rows with non-ASCII account names.

## Known issues

- Usage alerts can fire twice when an account's timezone changes during the week. A fix is
  planned for 3.4.1.
