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
});
