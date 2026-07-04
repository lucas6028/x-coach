import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import ModelIcon from "../components/ModelIcon";

describe("ModelIcon", () => {
  it("renders a brand SVG for each known model id", () => {
    for (const id of [
      "deepseek/deepseek-v4-flash",
      "xiaomi/mimo-v2.5",
      "minimax/minimax-m3",
      "tencent/hy3-preview",
    ]) {
      const { container } = render(<ModelIcon id={id} />);
      expect(container.querySelector("svg")).not.toBeNull();
    }
  });

  it("renders nothing for an unknown id", () => {
    const { container } = render(<ModelIcon id="who/knows" />);
    expect(container.querySelector("svg")).toBeNull();
  });
});
