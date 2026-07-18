import { useCallback } from "react";

interface Props {
  // Called on every drag step with the horizontal delta (px) since the last step.
  onResize: (deltaX: number) => void;
  onResizeStart?: () => void;
  onResizeEnd?: () => void;
  className?: string;
}

// A thin draggable vertical divider used to resize adjacent panels.
export default function ResizeHandle({ onResize, onResizeStart, onResizeEnd, className = "" }: Props) {
  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      let lastX = e.clientX;
      onResizeStart?.();
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";

      const handleMove = (ev: MouseEvent) => {
        const delta = ev.clientX - lastX;
        lastX = ev.clientX;
        onResize(delta);
      };
      const handleUp = () => {
        window.removeEventListener("mousemove", handleMove);
        window.removeEventListener("mouseup", handleUp);
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        onResizeEnd?.();
      };
      window.addEventListener("mousemove", handleMove);
      window.addEventListener("mouseup", handleUp);
    },
    [onResize, onResizeStart, onResizeEnd]
  );

  return (
    <div
      onMouseDown={handleMouseDown}
      role="separator"
      aria-orientation="vertical"
      className={`shrink-0 w-1 cursor-col-resize bg-transparent hover:bg-primary/60 active:bg-primary transition-colors ${className}`}
    />
  );
}
