---
title: "EDR-014: One approval for small PRs, Friday deploys behind flags"
owners: [priya.raman@northwind.test]
updated_at: 2026-10-30
acl: [group:everyone]
source_type: document
---

# EDR-014: One approval for small PRs, Friday deploys behind flags

- **Status:** Accepted
- **Date:** 30 October 2026
- **Decided in:** #eng, engineering all-hands
- **Owner:** Priya Raman
- **Supersedes:** the "Reviews" and "Deploys" sections of the handbook page *Code review and
  deploys* (March 2026). That page has not been updated yet.

## Context

PRs wait a median of 19 working hours for a second approval. Most of them are under 100 lines.
The Friday deploy freeze means fixes written on Thursday afternoon wait until Monday, and Monday
deploys are now our riskiest because they bundle a week of changes.

## Decision

1. PRs with **fewer than 200 changed lines need one approving review** from a code owner.
   PRs with 200 lines or more still need two.
2. **Database migrations always need two approvals**, one from the database owners, whatever
   their size.
3. **Friday deploys are allowed** when the change is behind a feature flag that is off by default.
   Unflagged changes still don't deploy on Fridays.

## Consequences

- Review latency should drop. We'll check the numbers at the end of November.
- CI enforces the line-count rule with a required check, `review-policy`.
- The on-call engineer can still block any Friday deploy in #eng-deploys.
