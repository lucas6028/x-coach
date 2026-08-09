import type { WebTarget, WebTrace, Point } from "../../lib/webslinger/engine";

export { createPoseLandmarker } from "../poseLandmarker";

export type WebSlingerScene = {
  targets: WebTarget[];
  traces: WebTrace[];
  wrists: (Point | null)[];
};

const mirrorX = (x: number, width: number) => (1 - x) * width;
const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
const easeOutCubic = (value: number) => 1 - Math.pow(1 - clamp01(value), 3);

type CanvasPoint = { x: number; y: number };

function curvePoint(
  start: CanvasPoint,
  control: CanvasPoint,
  end: CanvasPoint,
  t: number
): CanvasPoint {
  const inv = 1 - t;
  return {
    x: inv * inv * start.x + 2 * inv * t * control.x + t * t * end.x,
    y: inv * inv * start.y + 2 * inv * t * control.y + t * t * end.y,
  };
}

function drawSilkBundle(
  ctx: CanvasRenderingContext2D,
  start: CanvasPoint,
  end: CanvasPoint,
  trace: WebTrace,
  height: number
): CanvasPoint {
  const travel = easeOutCubic(trace.progress);
  const tip = {
    x: start.x + (end.x - start.x) * travel,
    y: start.y + (end.y - start.y) * travel,
  };
  const dx = tip.x - start.x;
  const dy = tip.y - start.y;
  const length = Math.max(1, Math.hypot(dx, dy));
  const nx = -dy / length;
  const ny = dx / length;
  const snap = Math.sin(trace.seed * 18 + (1 - trace.life) * 16) * trace.life;
  const bend = Math.min(height * 0.026, length * 0.08) * snap;
  const gravity = Math.min(height * 0.022, length * 0.055) * travel;
  const midpoint = {
    x: (start.x + tip.x) / 2 + nx * bend,
    y: (start.y + tip.y) / 2 + ny * bend + gravity,
  };

  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const fibers = [
    { offset: -2.2, width: 0.8, alpha: 0.42 },
    { offset: -0.7, width: 1.25, alpha: 0.72 },
    { offset: 0.7, width: 1.1, alpha: 0.62 },
    { offset: 2.1, width: 0.65, alpha: 0.34 },
  ];
  for (const fiber of fibers) {
    ctx.strokeStyle = `rgba(248,250,252,${fiber.alpha * trace.life})`;
    ctx.lineWidth = Math.max(0.75, fiber.width * (height / 480));
    ctx.beginPath();
    ctx.moveTo(start.x + nx * fiber.offset, start.y + ny * fiber.offset);
    ctx.quadraticCurveTo(
      midpoint.x + nx * fiber.offset * 1.8,
      midpoint.y + ny * fiber.offset * 1.8,
      tip.x + nx * fiber.offset * 0.35,
      tip.y + ny * fiber.offset * 0.35
    );
    ctx.stroke();
  }

  // Tiny cross-fibres make the strand read as spun webbing instead of a glowing beam.
  ctx.strokeStyle = `rgba(255,255,255,${0.32 * trace.life})`;
  ctx.lineWidth = Math.max(0.6, height * 0.0012);
  for (let t = 0.14; t < travel - 0.04; t += 0.13) {
    const point = curvePoint(start, midpoint, tip, t / Math.max(travel, 0.001));
    const half = 2.5 + 2 * Math.sin(trace.seed * 20 + t * 17);
    ctx.beginPath();
    ctx.moveTo(point.x - nx * half, point.y - ny * half);
    ctx.lineTo(point.x + nx * half, point.y + ny * half);
    ctx.stroke();
  }

  // The leading silk glob stretches forward while the shot is still airborne.
  if (trace.progress < 1) {
    const pulse = 1 + Math.sin(trace.progress * Math.PI) * 0.5;
    ctx.fillStyle = `rgba(255,255,255,${0.88 * trace.life})`;
    ctx.beginPath();
    ctx.ellipse(tip.x, tip.y, height * 0.006 * pulse, height * 0.0035, Math.atan2(dy, dx), 0, Math.PI * 2);
    ctx.fill();
    for (let i = 0; i < 5; i += 1) {
      const angle = trace.seed * 9 + (i / 5) * Math.PI * 2;
      ctx.strokeStyle = `rgba(248,250,252,${0.48 * trace.life})`;
      ctx.beginPath();
      ctx.moveTo(tip.x, tip.y);
      ctx.lineTo(tip.x + Math.cos(angle) * height * 0.012, tip.y + Math.sin(angle) * height * 0.012);
      ctx.stroke();
    }
  }
  return tip;
}

function drawStickyImpact(
  ctx: CanvasRenderingContext2D,
  center: CanvasPoint,
  trace: WebTrace,
  height: number
): void {
  if (!trace.hit || trace.progress < 0.78) return;
  const spread = easeOutCubic((trace.progress - 0.78) / 0.22);
  const radius = Math.max(height * 0.042, trace.impactRadius * height * 1.35) * spread;
  const points: CanvasPoint[] = [];
  const spokes = 10;
  for (let i = 0; i < spokes; i += 1) {
    const angle = (i / spokes) * Math.PI * 2 + trace.seed * 0.7;
    const irregular = 0.76 + 0.22 * Math.sin(i * 4.71 + trace.seed * 19);
    points.push({
      x: center.x + Math.cos(angle) * radius * irregular,
      y: center.y + Math.sin(angle) * radius * irregular,
    });
  }

  ctx.strokeStyle = `rgba(248,250,252,${0.76 * trace.life * spread})`;
  ctx.lineWidth = Math.max(0.8, height * 0.0022);
  for (const point of points) {
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.quadraticCurveTo(
      (center.x + point.x) / 2 + Math.sin(point.y) * 2,
      (center.y + point.y) / 2 + Math.cos(point.x) * 2,
      point.x,
      point.y
    );
    ctx.stroke();
  }

  // Two irregular rings bind the radial strands into a sticky, expanding web patch.
  for (const ring of [0.48, 0.82]) {
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = center.x + (point.x - center.x) * ring;
      const y = center.y + (point.y - center.y) * ring;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }

  ctx.fillStyle = `rgba(255,255,255,${0.9 * trace.life * spread})`;
  ctx.beginPath();
  ctx.arc(center.x, center.y, Math.max(2, height * 0.006), 0, Math.PI * 2);
  ctx.fill();
}

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
    const liveWrist = scene.wrists[trace.hand];
    const start = {
      x: mirrorX(liveWrist?.x ?? trace.x, width),
      y: (liveWrist?.y ?? trace.y) * height,
    };
    const end = { x: mirrorX(trace.x2, width), y: trace.y2 * height };
    ctx.save();
    const tip = drawSilkBundle(ctx, start, end, trace, height);
    drawStickyImpact(ctx, end, trace, height);

    // A brief spray at the palm sells the pressure release from the web shooter.
    if (trace.progress < 0.34) {
      const burst = 1 - trace.progress / 0.34;
      for (let i = 0; i < 6; i += 1) {
        const angle = trace.seed * 13 + i * 1.31;
        const distance = height * 0.018 * burst;
        ctx.fillStyle = `rgba(248,250,252,${0.65 * burst})`;
        ctx.beginPath();
        ctx.arc(
          start.x + Math.cos(angle) * distance,
          start.y + Math.sin(angle) * distance,
          Math.max(0.8, height * 0.0025 * burst),
          0,
          Math.PI * 2
        );
        ctx.fill();
      }
    }

    // Keep the hit point bright while tension travels down the attached strand.
    if (trace.hit && trace.progress >= 1) {
      ctx.fillStyle = `rgba(255,255,255,${0.55 * trace.life})`;
      ctx.beginPath();
      ctx.arc(tip.x, tip.y, Math.max(1.5, height * 0.004), 0, Math.PI * 2);
      ctx.fill();
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
