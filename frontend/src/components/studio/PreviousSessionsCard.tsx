import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ClockCounterClockwise } from "@phosphor-icons/react";
import { api, type HistoryItem } from "../../api";
import { useAuth } from "../../lib/auth";
import { movementLabel, useI18n } from "../../lib/i18n";
import StudioCard from "./StudioCard";

const SHOWN = 3;

// The reference's "Your Previous Sessions" panel, on the real history rows.
//
// The reference badges each row with a form score; the history LIST endpoint promotes only
// `fault_count` (not per-fault severities), so a score here would either be a different formula
// from the one on the ring above or a guess. It shows the fault count instead — the number the
// row actually carries.
export default function PreviousSessionsCard({ currentVideoId }: { currentVideoId?: string }) {
  const { t, lang } = useI18n();
  const { user } = useAuth();
  const [items, setItems] = useState<HistoryItem[]>([]);
  const [thumbs, setThumbs] = useState<Record<string, string>>({});
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    if (!user) {
      setState("ready");
      setItems([]);
      return;
    }
    let active = true;
    setState("loading");
    api
      .listAnalyses(SHOWN + 1)
      .then(async (page) => {
        if (!active) return;
        // Drop the analysis currently on screen — "previous" sessions, not this one.
        const rows = page.items.filter((it) => it.video_id !== currentVideoId).slice(0, SHOWN);
        setItems(rows);
        setState("ready");
        try {
          const media = await api.uploadMediaBatch(rows.map((r) => r.video_id));
          if (active)
            setThumbs(
              Object.fromEntries(Object.entries(media).map(([id, m]) => [id, m.thumbnail_url]))
            );
        } catch {
          // Thumbnails are decoration; a storage problem must not blank the list.
          if (active) setThumbs({});
        }
      })
      .catch(() => active && setState("error"));
    return () => {
      active = false;
    };
  }, [user, currentVideoId]);

  const fmtDate = (iso: string) => {
    const d = new Date(iso);
    return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString(lang, { dateStyle: "medium" });
  };

  return (
    <StudioCard
      icon={<ClockCounterClockwise size={14} weight="bold" />}
      title={t("studio.previous")}
      action={
        user ? (
          <Link to="/history" className="text-[11px] font-semibold text-primary hover:underline">
            {t("studio.viewAll")}
          </Link>
        ) : undefined
      }
    >
      {state === "error" ? (
        <p className="text-[11px] leading-relaxed text-[#63709f]">{t("studio.previousError")}</p>
      ) : !user ? (
        <p className="text-[11px] leading-relaxed text-[#63709f]">{t("studio.previousSignIn")}</p>
      ) : state === "loading" ? (
        <div className="space-y-3">
          {Array.from({ length: SHOWN }).map((_, i) => (
            <div key={i} className="h-[54px] animate-pulse rounded-xl bg-[#f5f3ff]" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <p className="text-[11px] leading-relaxed text-[#63709f]">{t("studio.previousEmpty")}</p>
      ) : (
        <ul className="space-y-3">
          {items.map((it) => (
            <li key={it.id}>
              <Link
                to={`/app?analysis=${it.id}`}
                className="glass-control flex items-center gap-3 rounded-xl p-2 transition-colors"
              >
                <span className="relative h-[40px] w-[52px] shrink-0 overflow-hidden rounded-lg bg-[#eef0f5]">
                  {thumbs[it.video_id] && (
                    <img
                      src={thumbs[it.video_id]}
                      alt=""
                      loading="lazy"
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none";
                      }}
                    />
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[11px] font-semibold leading-none text-[#1e2142]">
                    {movementLabel(t, it.movement ?? "Squat")}
                    <span className="font-normal text-[#63709f]"> — {fmtDate(it.created_at)}</span>
                  </span>
                </span>
                <span
                  className={`whitespace-nowrap rounded-full border px-2 py-1 text-[10px] font-bold ${
                    it.fault_count === 0
                      ? "border-[#c6e9d6] bg-[#e6f7ed] text-[#1a9e5a]"
                      : "border-[#ffe0e0] bg-[#fff5f5] text-[#e05252]"
                  }`}
                >
                  {it.fault_count === 0
                    ? t("studio.sessionClean")
                    : t("studio.sessionFaults", { n: it.fault_count })}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </StudioCard>
  );
}
