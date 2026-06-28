import { useRef, useState } from "react";
import { CircleNotch, UploadSimple } from "@phosphor-icons/react";
import { useI18n } from "../lib/i18n";

interface Props {
  onFile: (file: File) => void;
  loading: boolean;
  statusMsg: string;
}

export default function UploadDropzone({ onFile, loading, statusMsg }: Props) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files: FileList | null) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
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
      className={`group relative cursor-pointer rounded-2xl border border-dashed p-6 text-center transition-colors sm:p-8 ${
        dragOver
          ? "border-primary bg-primary/[0.06]"
          : "border-border-dark bg-content/[0.02] hover:border-primary/60 hover:bg-content/[0.03]"
      } ${loading ? "pointer-events-none opacity-70" : ""}`}
    >
      <span
        className={`mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary ${
          loading ? "animate-pulse" : "transition-transform group-hover:-translate-y-0.5"
        }`}
      >
        {loading ? (
          <CircleNotch size={30} className="animate-spin" />
        ) : (
          <UploadSimple size={30} />
        )}
      </span>
      <p className="text-content font-medium">
        {loading ? t("upload.analysing") : t("upload.prompt")}
      </p>
      <p className="mt-1.5 font-mono text-[11px] text-muted">{loading ? statusMsg : t("upload.hint")}</p>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
}
