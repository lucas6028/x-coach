import { useRef, useState } from "react";
import { UploadSimple } from "@phosphor-icons/react";
import { useI18n } from "../lib/i18n";

interface Props {
  onFile: (file: File) => void;
}

// The idle upload target. The analysis *waiting* state is owned by Lumen (see DemoIntro, which
// swaps this dropzone for <LumenLoader variant="scan" /> while `loading`), so this component only
// ever renders the resting prompt — no spinner branch of its own.
export default function UploadDropzone({ onFile }: Props) {
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
      onClick={() => inputRef.current?.click()}
      className={`group relative cursor-pointer rounded-2xl border border-dashed p-6 text-center transition-colors sm:p-8 ${
        dragOver
          ? "border-primary bg-primary/[0.06]"
          : "border-border-dark bg-content/[0.02] hover:border-primary/60 hover:bg-content/[0.03]"
      }`}
    >
      <span className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-primary/10 text-primary transition-transform group-hover:-translate-y-0.5">
        <UploadSimple size={30} />
      </span>
      <p className="text-content font-medium">{t("upload.prompt")}</p>
      <p className="mt-1.5 font-mono text-[11px] text-muted">{t("upload.hint")}</p>
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
