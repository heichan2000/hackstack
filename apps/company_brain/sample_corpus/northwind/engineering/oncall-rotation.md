---
title: On-call rotation
owners: [tomas.silva@northwind.test]
updated_at: 2026-11-03
acl: [group:engineering, group:leadership]
source_type: page
---

# On-call rotation

Owner: Tomás Silva (SRE). Questions in #eng-oncall.

## What changed on 3 November 2026

- From **Tuesday 10 November**, the rotation hands off **weekly on Tuesdays at 10:00 UTC**,
  instead of every Monday. Monday handoffs collided with the Monday deploy backlog.
- Each shift now has a **secondary on-call** from a different small team.
- The pager moves from Opsgenie to **incident.io**. Install the app and join the schedule before
  your next shift.

## Who is on call

Six engineers from the Platform and Data teams share the rotation. Check the schedule in
incident.io for the current primary and secondary.

## Expectations

- Acknowledge a page within 5 minutes and a SEV-1 within 2.
- You may roll back any deploy without asking (see *Code review and deploys*).
- Write up every SEV-1 and SEV-2 in `engineering/postmortems/` within three working days.
- On-call pays a flat stipend per week; see the compensation page in HiBob.
