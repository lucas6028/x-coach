import type { WebTarget, WebTrace, Point } from "../../lib/webslinger/engine";

export { createPoseLandmarker } from "../poseLandmarker";

export type WebSlingerScene = {
  targets: WebTarget[];
  traces: WebTrace[];
  wrists: (Point | null)[];
};

const mirrorX = (x: number, width: number) => (1 - x) * width;

export function drawWebSlingerScene(
  ctx: CanvasRenderingContext2D,
  scene: WebSlingerScene,
  width: number,
  height: number
): void {
  ctx.clearRect(0, 0, width, height);

  for (const target of scene.targets) {
    const x = mirrorX(target.x, width);
    const y = target.y * height;
    const r = target.radius * height;
    ctx.save();
    ctx.translate(x, y);
    ctx.strokeStyle = "rgba(248,250,252,0.92)";
    ctx.fillStyle = "rgba(190,24,45,0.58)";
    ctx.lineWidth = Math.max(3, height * 0.006);
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = "rgba(248,250,252,0.72)";
    ctx.lineWidth = Math.max(2, height * 0.003);
    ctx.beginPath();
    ctx.moveTo(-r * 0.62, 0);
    ctx.lineTo(r * 0.62, 0);
    ctx.moveTo(0, -r * 0.62);
    ctx.lineTo(0, r * 0.62);
    ctx.stroke();
    ctx.restore();
  }

  for (const trace of scene.traces) {
    const x1 = mirrorX(trace.x, width);
    const y1 = trace.y * height;
    const x2 = mirrorX(trace.x2, width);
    const y2 = trace.y2 * height;
    ctx.save();
    ctx.globalAlpha = Math.max(0, trace.life);
    ctx.strokeStyle = trace.hit ? "#f8fafc" : "rgba(226,232,240,0.72)";
    ctx.lineCap = "round";
    ctx.lineWidth = trace.hit ? Math.max(4, height * 0.008) : Math.max(2, height * 0.004);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.lineWidth = Math.max(1, height * 0.002);
    for (let t = 0.16; t < 0.94; t += 0.16) {
      const x = x1 + (x2 - x1) * t;
      const y = y1 + (y2 - y1) * t;
      ctx.beginPath();
      ctx.moveTo(x - 5, y - 5);
      ctx.lineTo(x + 5, y + 5);
      ctx.moveTo(x + 5, y - 5);
      ctx.lineTo(x - 5, y + 5);
      ctx.stroke();
    }
    ctx.restore();
  }

  for (const wrist of scene.wrists) {
    if (!wrist) continue;
    ctx.fillStyle = "#f8fafc";
    ctx.strokeStyle = "#be123c";
    ctx.lineWidth = Math.max(3, height * 0.005);
    ctx.beginPath();
    ctx.arc(mirrorX(wrist.x, width), wrist.y * height, height * 0.016, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }
}
