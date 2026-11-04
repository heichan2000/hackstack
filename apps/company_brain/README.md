# company_brain: Northwind Brain

The reference app: a "company brain" agent for **Northwind Labs**, a fictional, remote-first
company of about 60 people that makes **Pulse**, a usage-monitoring and alerting product for B2B
SaaS teams. Select it with `app: company_brain` in `hackstack.yaml` (the default).

| File | What it is |
|---|---|
| `app.yaml` | UI title, prompt, tools, users and connector config |
| `prompts/system.md` | The system prompt |
| `users.yaml` | Four demo personas |
| `sample_corpus/handbook/` | 17 PostHog handbook pages, playing Northwind's general handbook (MIT, see `THIRD_PARTY.md`) |
| `sample_corpus/northwind/` | 13 hand-written Northwind documents with the planted moments below |
| `demo-script.md` | The pitch and demo questions (stub until *Demo hardening*) |
| `tools.py` | The app's own agent tools, `find_experts` and `whats_changed` (below) |
| `people.yaml` | Names and job titles for document owners, used by `find_experts` |

## Tools

Besides the kit's `search_kb` and `get_document`, the agent has two tools of its own. Both read
only what the asker may read:

- **`find_experts(topic)`** searches, then ranks the `owners` of the documents that came back.
  Each document counts by how high it ranked (1/(rank+1) per passage) times how recent it is
  (a 180-day half-life). It returns the ranked people, their titles and the documents they own,
  plus one citable passage per document as evidence. Owners are looked up under the asker's
  principals too, so a restricted document never surfaces its owner's name.
- **`whats_changed(since, topic?)`** lists the documents updated on or after `since`, newest first,
  with each one's opening passage, or runs a date-filtered search when a `topic` is given.
  `since` is a date or a relative phrase as said ("monday", "yesterday", "this week").

Dated conflicts are handled by the prompt: every passage title ends with "(updated
YYYY-MM-DD)", and `prompts/system.md` asks for both dated sources and a recommendation of the
newer one.

## Personas

| `--as` | Name | Role | Groups |
|---|---|---|---|
| `eng@northwind.test` | Alex Chen | Software Engineer, Platform | everyone, engineering |
| `hr@northwind.test` | Jordan Lee | HR Business Partner | everyone, hr |
| `finance@northwind.test` | Riley Park | Finance Analyst | everyone, finance |
| `ceo@northwind.test` | Morgan Blake | CEO | everyone, leadership |

Out of the 30 documents, eng sees 26, hr 25, finance 25 and ceo all 30.

## Planted moments

All dates are absolute. The demo assumes it runs on or just after **Wednesday 4 November 2026**,
so "since Monday" means since **2 November 2026**.

### 1. Dated policy contradiction: PTO carryover

- `northwind/handbook/pto-policy.md` (**12 March 2026**): carry over **up to 5 days**, they
  expire **March 31**.
- `northwind/announcements/2026-10-29-pto-carryover-change.md` (**29 October 2026**): from
  1 November, **up to 10 days**, expiring **June 30**. It says the handbook page is out of date.
- Both are visible to everyone. Expected: the answer cites both with dates and recommends the
  October announcement.
- Ask: *"What is the PTO carryover policy?"* (the README's example, `--as finance@northwind.test`).

### 2. Dated policy contradiction: code review and Friday deploys

- `northwind/handbook/code-review-and-deploys.md` (**4 March 2026**): every PR needs **two
  approvals**, **no Friday deploys**.
- `northwind/engineering/decisions/2026-10-30-review-and-friday-deploys.md` (EDR-014,
  **30 October 2026**): **one approval under 200 lines**, migrations always two, **Friday deploys
  allowed behind a flag**. It says it supersedes the handbook, which hasn't been updated.
- Ask: *"How many approvals does my PR need, and can I deploy on Friday?"*

### 3. What changed since Monday (2 November 2026)

| Date | Document | Who can see it |
|---|---|---|
| 2 Nov | `engineering/postmortems/2026-11-02-checkout-latency.md` (SEV-2, migration lock) | engineering, leadership |
| 2 Nov | `product/release-notes-pulse-3-4.md` (usage alerts, Starter retention cut) | everyone |
| 3 Nov | `engineering/oncall-rotation.md` (Tuesday handoffs, incident.io) | engineering, leadership |
| 3 Nov | `finance/2027-pricing-change.md` (Starter $59, Growth $349) | finance, leadership |
| 3 Nov | `hr/h2-2026-performance-cycle.md` (calibration moved to 24–26 Nov) | hr, leadership |

Older updates for contrast: `finance/expense-policy.md` (27 Oct), `finance/q4-2026-budget.md`
(28 Oct), the PTO announcement (29 Oct), EDR-014 (30 Oct). The handbook slice is dated
2 March 2026. Ask *"What changed since Monday?"* as `hr@` and then as `eng@`: each sees a
different set (eng: postmortem, release notes, on-call; hr: release notes, performance cycle;
ceo: all five).

### 4. Who should I ask?

Every Northwind document has `owners`:

| Person | Owns |
|---|---|
| `sam.okafor@northwind.test` (Staff Engineer, Data) | Postgres migrations runbook, checkout postmortem |
| `tomas.silva@northwind.test` (SRE) | On-call rotation, checkout postmortem |
| `priya.raman@northwind.test` (EM, Platform) | Code review handbook page, EDR-014 |
| `maria.santos@northwind.test` (Head of People) | PTO policy, PTO announcement, comp bands, performance cycle |
| `dan.whitaker@northwind.test` (Head of Finance) | Expense policy, Q4 budget, 2027 pricing |
| `lena.fischer@northwind.test` (PM, Alerts) | Pulse 3.4 release notes, 2027 pricing |

Ask: *"Who should I ask about database migrations?"* → as `eng@`, Sam Okafor (runbook and
postmortem), then Tomás Silva and Priya Raman, whose code-review page and EDR-014 set the
migration approval rules. The runbook and postmortem are engineering-only, so `finance@` never
hears of Sam or Tomás: it gets only Priya, from the everyone-visible code-review documents.

### 5. ACL-restricted HR and finance pages

| Document | Groups |
|---|---|
| `hr/compensation-bands-2026.md` | hr, leadership |
| `hr/h2-2026-performance-cycle.md` | hr, leadership |
| `finance/q4-2026-budget.md` | finance, leadership |
| `finance/2027-pricing-change.md` | finance, leadership |

The role switch: *"What are the salary bands for senior engineers?"* as `eng@` (no access, so
"I couldn't find that") and then as `hr@` (E3: $128,000–$164,000). Likewise *"What is our Q4
cloud budget?"* as `eng@` and then as `finance@` ($412,000).

## Corpus metadata

The `local_files` connector reads each document's metadata from YAML front matter, falling back
to the nearest `_defaults.yaml` in the document's folder or a parent folder (inside the corpus):

```yaml
---
title: Paid time off (PTO) policy      # else the first heading, else the file name
owners: [maria.santos@northwind.test]  # people to ask; drives find_experts
updated_at: 2026-03-12                 # drives whats_changed and conflict dates
acl: [group:everyone]                  # principals: group:<g> or user:<id>
source_type: page                      # optional, default "document"
---
```

- **No ACL means nobody can read the document** (fail closed).
- Files whose names start with `_` (like `_defaults.yaml`) are configuration, not documents.
- Don't rely on file modification times for `updated_at`: a git checkout resets them.

`tests/test_northwind_corpus.py` checks this contract and the planted moments above.
