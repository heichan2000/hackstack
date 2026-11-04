import { describe, expect, it } from "vitest";
import { segment } from "./citations";

describe("segment", () => {
  it("splits [n] markers out of the text", () => {
    expect(segment("Five days [1].")).toEqual([
      { kind: "text", text: "Five days " },
      { kind: "cite", numbers: [1] },
      { kind: "text", text: "." },
    ]);
  });

  it("handles grouped markers like [1, 3] and adjacent markers", () => {
    expect(segment("A [1, 3][2]")).toEqual([
      { kind: "text", text: "A " },
      { kind: "cite", numbers: [1, 3] },
      { kind: "cite", numbers: [2] },
    ]);
  });

  it("leaves non-numeric brackets alone", () => {
    expect(segment("see [docs] and [a1]")).toEqual([{ kind: "text", text: "see [docs] and [a1]" }]);
  });

  it("returns nothing for empty text", () => {
    expect(segment("")).toEqual([]);
  });
});
