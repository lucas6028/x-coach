import { useEffect, useRef, useState } from "react";
import { CaretDown, Check } from "@phosphor-icons/react";

export interface MenuOption {
  value: string;
  label: string;
}

interface Props {
  id?: string;
  /** The caption above the value, and the first half of the control's accessible name. */
  label: string;
  value: string;
  /** What the trigger reads — usually the current option's label, but the caller owns it so a
   *  value the option list doesn't know can still be displayed. */
  display: string;
  /** Optional leading glyph in a round tint. The studio's two controls carry one; the history
   *  filter row does not — four tinted circles in a row read as status, not as filters. */
  icon?: React.ReactNode;
  /** Tailwind classes for the icon's round tint, so each control keeps its own colour. */
  tint?: string;
  options: MenuOption[];
  onChange: (v: string) => void;
  /** Which edge the menu aligns to. Left for controls at the start of a row, where a
   *  right-aligned menu would hang off toward the page edge. */
  align?: "left" | "right";
}

// The shell's dropdown: a small glass card carrying a caption over the current value, with a caret
// on the right, opening a `menu` of `menuitemradio`s on the shared `.glass-popup` surface.
//
// It USED to be a native <select> stretched invisibly across the card, which bought keyboard
// support and the accessible name for free — but the list a native select opens is drawn by the
// browser, so it ignored the whole palette and dropped a system-white menu on top of the glass.
//
// Shared rather than copied: the studio header (movement, extraction tier) and the history filter
// row (movement, result, period) are the same control, and dismissal — outside click and Escape —
// is the part that goes subtly wrong when it is reimplemented per site.
export default function MenuCard({
  id,
  label,
  value,
  display,
  icon,
  tint,
  options,
  onChange,
  align = "right",
}: Props) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={ref} className="relative">
      <button
        id={id}
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        // The label and the current value together, so the control announces both — the visible
        // caption is a <span>, not a <label>, now that the trigger is a button.
        aria-label={`${label}: ${display}`}
        className="glass-control flex min-w-[150px] items-center gap-3 rounded-2xl px-4 py-2.5 transition-colors hover:border-[#c9bcff]"
      >
        {icon && (
          <span className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full ${tint}`}>
            {icon}
          </span>
        )}
        <span className="min-w-0 flex-1 text-left leading-tight">
          <span className="block text-[10px] font-medium text-[#65719f]">{label}</span>
          <span className="-mt-0.5 block truncate text-[13px] font-semibold text-[#1e2142]">
            {display}
          </span>
        </span>
        <CaretDown
          size={13}
          weight="bold"
          className={`shrink-0 text-[#63709f] transition-transform ${open ? "rotate-180" : ""}`}
        />
      </button>

      {open && (
        <div
          role="menu"
          aria-label={label}
          className={`glass-popup absolute z-50 mt-1.5 min-w-full overflow-hidden rounded-xl p-1 ${
            align === "left" ? "left-0" : "right-0"
          }`}
        >
          {options.map((o) => {
            const active = o.value === value;
            return (
              <button
                key={o.value}
                type="button"
                role="menuitemradio"
                aria-checked={active}
                onClick={() => {
                  onChange(o.value);
                  setOpen(false);
                }}
                className={`flex w-full items-center gap-2.5 whitespace-nowrap rounded-lg px-2.5 py-2 text-[13px] transition-colors ${
                  active
                    ? "bg-primary/10 font-semibold text-primary"
                    : "text-[#59648f] hover:bg-white/60 hover:text-[#1e2142]"
                }`}
              >
                <span className="flex-1 text-left">{o.label}</span>
                {active && <Check size={15} weight="bold" />}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
