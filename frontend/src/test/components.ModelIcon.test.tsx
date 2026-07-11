import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import ModelIcon, { modelLabel } from "../components/ModelIcon";

describe("modelLabel", () => {
  it("maps a curated slug to a friendly name and passes an unknown slug through", () => {
    expect(modelLabel("deepseek/deepseek-v4-flash")).toBe("DeepSeek V4 Flash");
    expect(modelLabel("some/self-hosted-model")).toBe("some/self-hosted-model");
  });
});

describe("ModelIcon", () => {
  it("renders a brand SVG for each curated OpenRouter default", () => {
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

  it("resolves a brand by vendor prefix across the OpenRouter and NVIDIA NIM catalogs", () => {
    // Common models on both providers, matched by the slug's vendor prefix (part before "/") — a
    // NIM Nemotron and Nvidia-hosted Gemma/Phi/Llama, plus OpenRouter's OpenAI/Anthropic/Google/etc.
    for (const id of [
      "nvidia/llama-3.1-nemotron-70b-instruct",
      "meta/llama-3.3-70b-instruct",
      "meta-llama/llama-3.1-70b-instruct",
      "mistralai/mixtral-8x22b-instruct-v0.1",
      "deepseek-ai/deepseek-r1",
      "qwen/qwen2.5-coder-32b-instruct",
      "google/gemma-2-27b-it",
      "google/gemini-2.0-flash",
      "minimaxai/minimax-m2.7",
      "microsoft/phi-3-medium-4k-instruct",
      "openai/gpt-oss-120b",
      "anthropic/claude-3.5-sonnet",
      "x-ai/grok-2",
    ]) {
      const { container } = render(<ModelIcon id={id} />);
      expect(container.querySelector("svg")).not.toBeNull();
    }
  });

  it("renders a generic fallback icon for an unknown (self-hosted) id", () => {
    const { container } = render(<ModelIcon id="who/knows" />);
    // Not null: a self-hoster's custom slug still gets a chip icon rather than an empty badge.
    expect(container.querySelector("svg")).not.toBeNull();
  });
});
