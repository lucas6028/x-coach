import type { ComponentType } from "react";
import { Cpu } from "@phosphor-icons/react";
// Deep-import only the plain logo marks (Color / Mono). The package's Avatar/Combine variants pull
// in antd-style; importing the leaf SVG components keeps that whole dependency out of the bundle.
// A few brands ship only a monochrome mark (OpenAI, xAI/Grok, Xiaomi MiMo); the rest use colour.
import DeepSeek from "@lobehub/icons/es/DeepSeek/components/Color";
import Minimax from "@lobehub/icons/es/Minimax/components/Color";
import Hunyuan from "@lobehub/icons/es/Hunyuan/components/Color";
import XiaomiMiMo from "@lobehub/icons/es/XiaomiMiMo/components/Mono";
import Meta from "@lobehub/icons/es/Meta/components/Color";
import Mistral from "@lobehub/icons/es/Mistral/components/Color";
import Qwen from "@lobehub/icons/es/Qwen/components/Color";
import Nvidia from "@lobehub/icons/es/Nvidia/components/Color";
import Microsoft from "@lobehub/icons/es/Microsoft/components/Color";
import Gemini from "@lobehub/icons/es/Gemini/components/Color";
import Gemma from "@lobehub/icons/es/Gemma/components/Color";
import OpenAI from "@lobehub/icons/es/OpenAI/components/Mono";
import Claude from "@lobehub/icons/es/Claude/components/Color";
import Grok from "@lobehub/icons/es/Grok/components/Mono";
import Yi from "@lobehub/icons/es/Yi/components/Color";
import Cohere from "@lobehub/icons/es/Cohere/components/Color";

type BrandIcon = ComponentType<{ size?: number | string }>;

// Brand logo keyed by the vendor prefix of a model slug (the part before "/"). OpenRouter and
// NVIDIA NIM host the same open-weight model makers plus their own, so a single vendor→logo table
// covers the common models of both catalogs; NIM's `nvidia/*` Nemotron and NIM-hosted `google/gemma-*`
// are the additions over the OpenRouter defaults. The server owns the authoritative model list —
// these are purely cosmetic, so an unknown (self-hosted) vendor falls back to a generic chip icon.
const VENDOR_ICONS: Record<string, BrandIcon> = {
  openai: OpenAI as BrandIcon,
  anthropic: Claude as BrandIcon,
  meta: Meta as BrandIcon, // NIM `meta/llama-*`
  "meta-llama": Meta as BrandIcon, // OpenRouter `meta-llama/llama-*`
  mistral: Mistral as BrandIcon,
  mistralai: Mistral as BrandIcon,
  deepseek: DeepSeek as BrandIcon,
  "deepseek-ai": DeepSeek as BrandIcon, // NIM `deepseek-ai/deepseek-r1`
  qwen: Qwen as BrandIcon,
  nvidia: Nvidia as BrandIcon, // NIM Nemotron family
  microsoft: Microsoft as BrandIcon, // NIM Phi family
  "x-ai": Grok as BrandIcon,
  "01-ai": Yi as BrandIcon,
  cohere: Cohere as BrandIcon,
  google: Gemini as BrandIcon, // `google/gemma-*` overridden to Gemma below
  xiaomi: XiaomiMiMo as BrandIcon,
  minimax: Minimax as BrandIcon,
  minimaxai: Minimax as BrandIcon, // NIM `minimaxai/minimax-m*`
  tencent: Hunyuan as BrandIcon, // Tencent's LLM family is Hunyuan (e.g. Hy3)
};

// Friendly display names for the curated OpenRouter defaults; any other slug shows raw.
const LABELS: Record<string, string> = {
  "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
  "xiaomi/mimo-v2.5": "MiMo V2.5",
  "minimax/minimax-m3": "MiniMax M3",
  "tencent/hy3-preview": "Hy3 Preview",
};

// The brand logo component for a model slug, or undefined for an unrecognised (self-hosted) vendor.
function brandFor(id: string): BrandIcon | undefined {
  const vendor = id.split("/")[0];
  // Google ships two families under one prefix: Gemini (OpenRouter) and Gemma (NIM open weights).
  if (vendor === "google" && id.includes("gemma")) return Gemma as BrandIcon;
  return VENDOR_ICONS[vendor];
}

// A friendly display name for a model id, or the raw slug for a self-hosted model.
export function modelLabel(id: string): string {
  return LABELS[id] ?? id;
}

// The brand logo for a coach model; a generic chip icon for an unknown (self-hosted) slug.
export default function ModelIcon({ id, size = 20 }: { id: string; size?: number }) {
  const Icon = brandFor(id);
  return Icon ? <Icon size={size} /> : <Cpu size={size} className="text-muted" />;
}
