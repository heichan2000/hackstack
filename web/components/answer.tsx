"use client";

import { segment } from "@/lib/citations";
import { inline } from "@/lib/inline";
import { cn } from "@/lib/utils";

// Answer text with each [n] marker rendered as a clickable chip that selects source n.
export function Answer({
  text,
  selected,
  onCite,
}: {
  text: string;
  selected: number | null;
  onCite: (n: number) => void;
}) {
  return (
    <div className="leading-relaxed whitespace-pre-wrap">
      {segment(text).map((s, i) =>
        s.kind === "text" ? (
          <Rich key={i} text={s.text} />
        ) : (
          <span key={i} className="whitespace-nowrap">
            {s.numbers.map((n) => (
              <button
                key={n}
                type="button"
                onClick={() => onCite(n)}
                aria-label={`Show source ${n}`}
                className={cn(
                  "mx-0.5 inline-flex h-5 min-w-5 -translate-y-px items-center justify-center rounded-md border px-1 align-middle font-mono text-[11px] font-semibold transition-colors",
                  selected === n
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-muted text-foreground hover:border-primary/60",
                )}
              >
                {n}
              </button>
            ))}
          </span>
        ),
      )}
    </div>
  );
}

function Rich({ text }: { text: string }) {
  return (
    <>
      {inline(text).map((t, i) =>
        t.kind === "bold" ? (
          <strong key={i} className="font-semibold">
            {t.text}
          </strong>
        ) : t.kind === "code" ? (
          <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-[0.9em]">
            {t.text}
          </code>
        ) : t.kind === "link" ? (
          <a key={i} href={t.href} target="_blank" rel="noreferrer" className="text-primary underline">
            {t.text}
          </a>
        ) : (
          // A bold span cut in two by a [n] marker leaves stray asterisks; drop them.
          <span key={i}>{t.text.replaceAll("**", "")}</span>
        ),
      )}
    </>
  );
}
