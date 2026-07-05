import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import Markdown from "../components/Markdown";

describe("Markdown", () => {
  it("renders bold, bulleted lists, and inline code", () => {
    const { container } = render(
      <Markdown>{"Drive your **knees out**.\n\n- cue one\n- cue two\n\nUse `tempo`."}</Markdown>
    );
    expect(container.querySelector("strong")?.textContent).toBe("knees out");
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelector("code")?.textContent).toBe("tempo");
  });

  it("never emits a raw <script> element (sanitized)", () => {
    const { container } = render(<Markdown>{"Hi <script>alert('xss')</script> there"}</Markdown>);
    expect(container.querySelector("script")).toBeNull();
  });

  it("drops links so the coach can't emit navigable URLs, keeping the visible text", () => {
    // Exercises the rehype-sanitize schema (which strips the <a> tag but keeps its children).
    const { container } = render(<Markdown>{"[click me](https://evil.example)"}</Markdown>);
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("click me");
  });

  it("renders italics, ordered lists, and headings (collapsed to a quiet subhead)", () => {
    const { container } = render(
      <Markdown>{"# H1\n\n## H2\n\n### H3\n\n_stay tall_\n\n1. first\n2. second"}</Markdown>
    );
    expect(container.querySelector("em")?.textContent).toBe("stay tall");
    expect(container.querySelectorAll("ol > li")).toHaveLength(2);
    // Every heading level maps to the same quiet <h3> subhead.
    expect(container.querySelectorAll("h3")).toHaveLength(3);
  });

  it("renders partial/incomplete markdown mid-stream without throwing", () => {
    // A half-streamed answer with an unterminated **bold** must not crash the render.
    expect(() => render(<Markdown>{"Drive your **kne"}</Markdown>)).not.toThrow();
  });

  it("renders GFM pipe tables into a real <table> with header and body cells", () => {
    const md = ["| Fault | Cue |", "| --- | --- |", "| Knees in | Drive knees out |"].join("\n");
    const { container } = render(<Markdown>{md}</Markdown>);
    const table = container.querySelector("table");
    expect(table).not.toBeNull();
    expect(container.querySelectorAll("th")).toHaveLength(2);
    expect(container.querySelector("th")?.textContent).toBe("Fault");
    expect(container.querySelectorAll("tbody td")).toHaveLength(2);
    expect(container.querySelector("tbody td")?.textContent).toBe("Knees in");
  });

  it("still strips links inside a table cell, keeping the visible text", () => {
    // remark-gfm turns a bare URL into an auto-link; the sanitize schema must drop its <a>.
    const md = ["| Ref |", "| --- |", "| See https://evil.example now |"].join("\n");
    const { container } = render(<Markdown>{md}</Markdown>);
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("evil.example");
  });

  it("renders a fenced code block inside <pre> without the inline pill", () => {
    const { container } = render(<Markdown>{"```\nconst x = 1\nreturn x\n```"}</Markdown>);
    const code = container.querySelector("pre code");
    expect(code).not.toBeNull();
    expect(code?.textContent).toContain("const x = 1");
    // Block code must not carry the inline pill background.
    expect(code?.className).not.toContain("bg-content");
    // Inline code still gets the pill.
    const { container: inline } = render(<Markdown>{"use `tempo` here"}</Markdown>);
    expect(inline.querySelector("code")?.className).toContain("bg-content");
    expect(inline.querySelector("pre")).toBeNull();
  });

  it("renders blockquotes and horizontal rules", () => {
    const { container } = render(<Markdown>{"> stay tall\n\n---\n\ndone"}</Markdown>);
    expect(container.querySelector("blockquote")?.textContent).toContain("stay tall");
    expect(container.querySelector("hr")).not.toBeNull();
  });

  it("renders strikethrough and collapses h4–h6 to the quiet subhead", () => {
    const { container } = render(
      <Markdown>{"~~old cue~~\n\n#### H4\n\n##### H5\n\n###### H6"}</Markdown>
    );
    expect(container.querySelector("del")?.textContent).toBe("old cue");
    // Every heading level (incl. 4–6) maps to <h3>; none leak an <h4>/<h5>/<h6>.
    expect(container.querySelectorAll("h3")).toHaveLength(3);
    expect(container.querySelector("h4, h5, h6")).toBeNull();
  });

  it("renders a gfm task list with disabled checkboxes and no disc bullet", () => {
    const { container } = render(<Markdown>{"- [x] done\n- [ ] todo"}</Markdown>);
    const boxes = container.querySelectorAll('input[type="checkbox"]');
    expect(boxes).toHaveLength(2);
    // Checkboxes are inert (the coach's text, not an interactive control).
    expect((boxes[0] as HTMLInputElement).disabled).toBe(true);
    expect((boxes[0] as HTMLInputElement).checked).toBe(true);
    // The task-list <ul> drops the disc bullet.
    expect(container.querySelector("ul")?.className).toContain("list-none");
  });
});
