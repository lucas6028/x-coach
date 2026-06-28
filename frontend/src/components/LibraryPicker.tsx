import { useEffect, useState } from "react";
import { X } from "@phosphor-icons/react";
import { api, type LibraryItem } from "../api";
import { useI18n, viewLabel } from "../lib/i18n";

interface Props {
  onClose: () => void;
  onPick: (videoId: string) => void;
}

const FAULT_FILTERS = [
  { id: "", labelKey: "library.filter.all" },
  { id: "knees_inward", labelKey: "library.filter.knees_inward" },
  { id: "knees_forward", labelKey: "library.filter.knees_forward" },
  { id: "shallow_depth", labelKey: "library.filter.shallow_depth" },
  { id: "excessive_forward_lean", labelKey: "library.filter.excessive_forward_lean" },
];

export default function LibraryPicker({ onClose, onPick }: Props) {
  const { t } = useI18n();
  const [items, setItems] = useState<LibraryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [fault, setFault] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api
      .listVideos(40, 0, fault || undefined)
      .then((p) => {
        setItems(p.items);
        setTotal(p.total);
      })
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, [fault]);

  return (
    <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4" onClick={onClose}>
      <div
        className="bg-surface-dark border border-border-dark rounded-lg w-full max-w-2xl max-h-[80vh] flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-4 border-b border-border-dark flex items-center justify-between">
          <h2 className="text-sm font-bold text-content">
            {t("library.title")} {total ? `(${total})` : ""}
          </h2>
          <button onClick={onClose} aria-label={t("a11y.close")} className="text-muted hover:text-content">
            <X size={20} />
          </button>
        </div>
        <div className="p-3 flex gap-2 flex-wrap border-b border-border-dark">
          {FAULT_FILTERS.map((f) => (
            <button
              key={f.id}
              onClick={() => setFault(f.id)}
              className={`px-2.5 py-1 rounded-full text-[11px] border ${
                fault === f.id
                  ? "bg-primary/20 text-primary border-primary/40"
                  : "text-muted border-border-dark hover:text-content"
              }`}
            >
              {t(f.labelKey)}
            </button>
          ))}
        </div>
        <div className="overflow-y-auto p-3 grid grid-cols-2 sm:grid-cols-3 gap-2 scrollbar-thin">
          {loading ? (
            <p className="col-span-full text-center text-muted text-sm py-8">{t("library.loading")}</p>
          ) : (
            items.map((it) => (
              <button
                key={it.video_id}
                onClick={() => onPick(it.video_id)}
                className="text-left bg-background rounded border border-border-dark p-3 hover:border-primary transition-colors"
              >
                <div className="font-mono text-xs text-content truncate">{it.video_id}</div>
                <div className="text-[10px] text-muted mb-1">
                  {viewLabel(t, it.view_type)} · {it.split}
                </div>
                <div className="flex flex-wrap gap-1">
                  {it.faults.length === 0 ? (
                    <span className="text-[9px] text-secondary">{t("library.clean")}</span>
                  ) : (
                    it.faults.map((f) => (
                      <span
                        key={f}
                        className="text-[9px] bg-danger/15 text-danger px-1 rounded font-mono"
                      >
                        {f}
                      </span>
                    ))
                  )}
                </div>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
