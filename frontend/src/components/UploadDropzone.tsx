import { useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
  statusMsg: string;
}

export default function UploadDropzone({ onFile, loading, statusMsg }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
    <div className="w-full max-w-md">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleFiles(e.dataTransfer.files);
        }}
        onClick={() => !loading && inputRef.current?.click()}
        className={`cursor-pointer rounded-xl border-2 border-dashed p-10 text-center transition-colors ${
          dragOver ? "border-primary bg-primary/5" : "border-border-dark hover:border-primary/60"
        } ${loading ? "opacity-60 pointer-events-none" : ""}`}
      >
        <span className="material-symbols-outlined text-4xl text-primary mb-2 block">
          {loading ? "hourglass_top" : "upload"}
        </span>
        <p className="text-sm text-content font-medium">
          {loading ? "Analysing…" : "Drop a squat video or tap to upload"}
        </p>
        <p className="text-[11px] text-muted mt-1">
          {loading ? statusMsg : "MP4 / MOV · single athlete · side or rear view"}
        </p>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
    </div>
  );
}
