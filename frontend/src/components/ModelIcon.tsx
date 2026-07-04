import type { ComponentType } from "react";
import { Cpu } from "@phosphor-icons/react";
// Deep-import only the plain logo marks (Color / Mono). The package's Avatar/Combine variants pull
// in antd-style; importing the leaf SVG components keeps that whole dependency out of the bundle.
// Xiaomi MiMo ships only a monochrome mark; the others use their colour logo.
import DeepSeek from "@lobehub/icons/es/DeepSeek/components/Color";
import Minimax from "@lobehub/icons/es/Minimax/components/Color";
import Hunyuan from "@lobehub/icons/es/Hunyuan/components/Color";
import XiaomiMiMo from "@lobehub/icons/es/XiaomiMiMo/components/Mono";

type BrandIcon = ComponentType<{ size?: number | string }>;

// Curated brand presentation for known model ids (OpenRouter slugs): logo + display name. The
// server owns the authoritative list; these are purely cosmetic, so an unknown (self-hosted) slug
// falls back to a generic chip icon and its raw id as the label. Tencent's Hy3 is the Hunyuan family.
const ICONS: Record<string, BrandIcon> = {
  "deepseek/deepseek-v4-flash": DeepSeek as BrandIcon,
  "xiaomi/mimo-v2.5": XiaomiMiMo as BrandIcon,
  "minimax/minimax-m3": Minimax as BrandIcon,
  "tencent/hy3-preview": Hunyuan as BrandIcon,
};

const LABELS: Record<string, string> = {
  "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
  "xiaomi/mimo-v2.5": "MiMo V2.5",
  "minimax/minimax-m3": "MiniMax M3",
  "tencent/hy3-preview": "Hy3 Preview",
};

// A friendly display name for a model id, or the raw slug for a self-hosted model.
export function modelLabel(id: string): string {
  return LABELS[id] ?? id;
}

// The brand logo for a coach model; a generic chip icon for an unknown (self-hosted) slug.
export default function ModelIcon({ id, size = 20 }: { id: string; size?: number }) {
  const Icon = ICONS[id];
  return Icon ? <Icon size={size} /> : <Cpu size={size} className="text-muted" />;
}
