interface Props {
  open: boolean;
  onToggle: () => void;
  onOpenLibrary: () => void;
}

// Labelled bar when `open`, slim icon rail when collapsed — mirrors demo/index.html.
// The menu toggle lives in the top row so it stays reachable in either state.
export default function Sidebar({ open, onToggle, onOpenLibrary }: Props) {
  return (
    <aside
      className={`${
        open ? "w-60" : "w-16"
      } shrink-0 border-r border-border-dark bg-surface-dark flex flex-col justify-between overflow-hidden transition-[width] duration-200 ease-in-out`}
    >
      <div>
        <div className="h-16 flex items-center gap-2 px-3 border-b border-border-dark">
          <button
            onClick={onToggle}
            aria-label={open ? "Hide navigation" : "Show navigation"}
            title={open ? "Hide navigation" : "Show navigation"}
            className="shrink-0 w-10 h-10 flex items-center justify-center rounded-lg text-gray-400 hover:bg-white/5 hover:text-white transition-colors"
          >
            <span className="material-symbols-outlined">{open ? "menu_open" : "menu"}</span>
          </button>
          {open && (
            <div className="flex items-center min-w-0">
              <div className="w-8 h-8 rounded bg-primary flex items-center justify-center text-white shrink-0">
                <span className="material-symbols-outlined text-lg">biotech</span>
              </div>
              <span className="ml-2 font-bold tracking-wide truncate">X-Coach</span>
            </div>
          )}
        </div>
        <nav className="flex flex-col gap-1 p-2">
          <a
            className={`flex items-center gap-3 px-3 py-3 rounded-lg bg-primary/10 text-primary border border-primary/20 ${
              open ? "" : "justify-center"
            }`}
            href="#"
          >
            <span className="material-symbols-outlined">video_camera_front</span>
            {open && <span className="text-sm font-medium">Analyse</span>}
          </a>
          <button
            onClick={onOpenLibrary}
            className={`flex items-center gap-3 px-3 py-3 rounded-lg text-gray-400 hover:bg-white/5 hover:text-white transition-colors ${
              open ? "" : "justify-center"
            }`}
          >
            <span className="material-symbols-outlined">folder_data</span>
            {open && <span className="text-sm font-medium">Library</span>}
          </button>
        </nav>
      </div>
      {open && (
        <div className="p-3 border-t border-border-dark flex flex-col gap-1">
          <p className="text-[10px] text-gray-600 uppercase tracking-wider">Prototype v0.1</p>
          <p className="text-[10px] text-gray-600">Pose · Rules · GraphRAG</p>
        </div>
      )}
    </aside>
  );
}
