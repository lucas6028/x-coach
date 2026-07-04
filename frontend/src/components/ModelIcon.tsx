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

// Model id (OpenRouter slug) -> its provider's brand logo. Tencent's Hy3 is the Hunyuan family.
const ICONS: Record<string, BrandIcon> = {
  "deepseek/deepseek-v4-flash": DeepSeek as BrandIcon,
  "xiaomi/mimo-v2.5": XiaomiMiMo as BrandIcon,
  "minimax/minimax-m3": Minimax as BrandIcon,
  "tencent/hy3-preview": Hunyuan as BrandIcon,
};

// The brand logo for a coach model. Since the catalog is server-driven, a self-hoster's custom slug
// may not have a known logo — those fall back to a generic chip icon rather than rendering nothing.
export default function ModelIcon({ id, size = 20 }: { id: string; size?: number }) {
  const Icon = ICONS[id];
  return Icon ? <Icon size={size} /> : <Cpu size={size} className="text-muted" />;
}
