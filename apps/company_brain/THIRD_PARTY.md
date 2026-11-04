# Third-party content

## PostHog handbook (`sample_corpus/handbook/`)

17 files from the PostHog handbook, copied **unmodified** from
[`PostHog/posthog.com`](https://github.com/PostHog/posthog.com), folder `contents/handbook/`, at
commit `4fc3246084b49b31922fd4fc9482505b12367efe` (2026-11-04). Subfolders are flattened into the
file name (`people/benefits.md` → `people-benefits.md`). In the demo they play Northwind Labs'
general handbook. They are PostHog's text and still mention PostHog.

Files: `values.md`, `help.md`, `finance.md`, `people/{benefits, spending-money, feedback,
offboarding, side-gigs, training}.md`, `company/{communication, culture, small-teams,
management}.md`, `engineering/{development-process, security, bug-prioritization,
feature-ownership}.md`. `people/time-off.md` and `engineering/how-we-review.md` were left out
on purpose: they would compete with the planted Northwind PTO and code-review documents.

The repository's `LICENSE` grants MIT terms **for content in the `/contents/` folder only** (the
rest of the website is not licensed for reuse). That license text follows.

```
Copyright (c) 2020-2025 PostHog Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The same license file says PostHog's logo may be used to reference PostHog, "but not for you to
pretend to be us". Hackstack does neither: it ships no PostHog branding, and this file credits the
source.

## Everything else

Everything under `sample_corpus/northwind/` is hand-written for Hackstack. Northwind Labs, its
people and its numbers are fictional.
