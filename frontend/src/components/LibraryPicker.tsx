import { useEffect, useState } from "react";
import { api, type LibraryItem } from "../api";
import { titleCase } from "../lib/format";

interface Props {
  onClose: () => void;
  onPick: (videoId: string) => void;
}

const FAULT_FILTERS = [
  { id: "", label: "All" },
  { id: "knees_inward", label: "Knee Valgus" },
  { id: "knees_forward", label: "Knees Forward" },
  { id: "shallow_depth", label: "Shallow" },
  { id: "excessive_forward_lean", label: "Forward Lean" },
];

export default function LibraryPicker({ onClose, onPick }: Props) {
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
          <h2 className="text-sm font-bold text-white">Sample Library {total ? `(${total})` : ""}</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <span className="material-symbols-outlined">close</span>
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
                  : "text-gray-400 border-border-dark hover:text-white"
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
        <div className="overflow-y-auto p-3 grid grid-cols-2 sm:grid-cols-3 gap-2 scrollbar-thin">
          {loading ? (
            <p className="col-span-full text-center text-gray-500 text-sm py-8">Loading…</p>
          ) : (
            items.map((it) => (
              <button
                key={it.video_id}
                onClick={() => onPick(it.video_id)}
                className="text-left bg-[#1c1f24] rounded border border-border-dark p-3 hover:border-primary transition-colors"
              >
                <div className="font-mono text-xs text-white truncate">{it.video_id}</div>
                <div className="text-[10px] text-gray-500 mb-1">
                  {titleCase(it.view_type)} · {it.split}
                </div>
                <div className="flex flex-wrap gap-1">
                  {it.faults.length === 0 ? (
                    <span className="text-[9px] text-secondary">clean</span>
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
