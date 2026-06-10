interface Props {
  onOpenLibrary: () => void;
}

// Slim icon rail on mobile, labelled on desktop — mirrors demo/index.html.
export default function Sidebar({ onOpenLibrary }: Props) {
  return (
    <aside className="w-16 lg:w-60 shrink-0 border-r border-border-dark bg-surface-dark flex flex-col justify-between">
      <div>
        <div className="h-16 flex items-center justify-center lg:justify-start lg:px-5 border-b border-border-dark">
          <div className="w-8 h-8 rounded bg-primary flex items-center justify-center text-white">
            <span className="material-symbols-outlined text-lg">biotech</span>
          </div>
          <span className="ml-3 font-bold tracking-wide hidden lg:block">X-Coach</span>
        </div>
        <nav className="flex flex-col gap-1 p-2">
          <a
            className="flex items-center gap-3 px-3 py-3 rounded-lg bg-primary/10 text-primary border border-primary/20"
            href="#"
          >
            <span className="material-symbols-outlined">video_camera_front</span>
            <span className="hidden lg:block text-sm font-medium">Analyse</span>
          </a>
          <button
            onClick={onOpenLibrary}
            className="flex items-center gap-3 px-3 py-3 rounded-lg text-gray-400 hover:bg-white/5 hover:text-white transition-colors"
          >
            <span className="material-symbols-outlined">folder_data</span>
            <span className="hidden lg:block text-sm font-medium">Library</span>
          </button>
        </nav>
      </div>
      <div className="p-3 border-t border-border-dark hidden lg:flex flex-col gap-1">
        <p className="text-[10px] text-gray-600 uppercase tracking-wider">Prototype v0.1</p>
        <p className="text-[10px] text-gray-600">Pose · Rules · GraphRAG</p>
      </div>
    </aside>
  );
}
