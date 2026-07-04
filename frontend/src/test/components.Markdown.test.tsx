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
});
