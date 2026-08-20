import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { WebTarget, WebTrace, Point } from "../../lib/webslinger/engine";

export { createPoseLandmarker } from "../poseLandmarker";

export type WebSlingerScene = {
  targets: WebTarget[];
  traces: WebTrace[];
  wrists: (Point | null)[];
  face: FaceLandmarks | null;
};

export type FaceLandmarks = {
  nose: Point;
  leftEye: Point;
  rightEye: Point;
  leftEar: Point;
  rightEar: Point;
};

const mirrorX = (x: number, width: number) => (1 - x) * width;
const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
const easeOutCubic = (value: number) => 1 - Math.pow(1 - clamp01(value), 3);
const PHOTOGRAPHIC_MASK_SRC = "/assets/web-slinger/spider-mask-photo.jpg";

let photographicMask: HTMLImageElement | null = null;

function getPhotographicMask(): HTMLImageElement | null {
  if (typeof Image === "undefined") return null;
  if (!photographicMask) {
    photographicMask = new Image();
    photographicMask.decoding = "async";
    photographicMask.src = PHOTOGRAPHIC_MASK_SRC;
  }
  return photographicMask.complete && photographicMask.naturalWidth > 0
    ? photographicMask
    : null;
}

type CanvasPoint = { x: number; y: number };

const FACE_INDICES = { nose: 0, leftEye: 2, rightEye: 5, leftEar: 7, rightEar: 8 } as const;

export function extractFaceLandmarks(
  landmarks: NormalizedLandmark[] | null
): FaceLandmarks | null {
  if (!landmarks) return null;
  const points = Object.entries(FACE_INDICES).map(([key, index]) => {
    const landmark = landmarks[index];
    return [
      key,
      landmark && (landmark.visibility ?? 1) >= 0.35
        ? { x: landmark.x, y: landmark.y }
        : null,
    ] as const;
  });
  if (points.some(([, point]) => point === null)) return null;
  return Object.fromEntries(points) as FaceLandmarks;
}

function drawEyeLens(
  ctx: CanvasRenderingContext2D,
  center: CanvasPoint,
  width: number,
  height: number,
  direction: -1 | 1
): void {
  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(direction * -0.08);
  ctx.scale(direction, 1);

  const lens = new Path2D();
  lens.moveTo(-width * 0.45, -height * 0.18);
  lens.bezierCurveTo(-width * 0.18, -height * 0.42, width * 0.19, -height * 0.63, width * 0.48, -height * 0.48);
  lens.bezierCurveTo(width * 0.54, -height * 0.05, width * 0.38, height * 0.42, -width * 0.34, height * 0.56);
  lens.bezierCurveTo(-width * 0.48, height * 0.24, -width * 0.5, height * 0.02, -width * 0.45, -height * 0.18);
  lens.closePath();

  // A deep offset shadow and layered bevel make the frames feel attached to the suit.
  ctx.save();
  ctx.translate(direction * width * 0.025, height * 0.06);
  ctx.fillStyle = "rgba(0,0,0,0.52)";
  ctx.filter = `blur(${Math.max(1, width * 0.025)}px)`;
  ctx.fill(lens);
  ctx.restore();

  ctx.lineJoin = "round";
  ctx.fillStyle = "#080a0d";
  ctx.strokeStyle = "#030405";
  ctx.lineWidth = Math.max(4, width * 0.13);
  ctx.fill(lens);
  ctx.stroke(lens);

  const pearl = ctx.createLinearGradient(-width * 0.4, -height * 0.55, width * 0.38, height * 0.5);
  pearl.addColorStop(0, "#ffffff");
  pearl.addColorStop(0.32, "#dce8eb");
  pearl.addColorStop(0.72, "#aebfc3");
  pearl.addColorStop(1, "#f9ffff");
  ctx.fillStyle = pearl;
  ctx.strokeStyle = "rgba(116,132,138,0.95)";
  ctx.lineWidth = Math.max(1.5, width * 0.035);
  ctx.fill(lens);
  ctx.stroke(lens);

  ctx.save();
  ctx.clip(lens);
  ctx.strokeStyle = "rgba(76,96,102,0.28)";
  ctx.lineWidth = Math.max(0.55, width * 0.009);
  const mesh = Math.max(3, width * 0.075);
  for (let x = -width; x < width; x += mesh) {
    ctx.beginPath();
    ctx.moveTo(x, -height);
    ctx.lineTo(x + height * 0.72, height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, -height);
    ctx.lineTo(x - height * 0.72, height);
    ctx.stroke();
  }

  const reflection = ctx.createRadialGradient(-width * 0.24, -height * 0.38, 0, -width * 0.24, -height * 0.38, width * 0.42);
  reflection.addColorStop(0, "rgba(255,255,255,0.8)");
  reflection.addColorStop(0.35, "rgba(255,255,255,0.22)");
  reflection.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = reflection;
  ctx.fillRect(-width, -height, width * 2, height * 2);
  ctx.restore();

  ctx.strokeStyle = "rgba(112,124,130,0.9)";
  ctx.lineWidth = Math.max(1, width * 0.022);
  ctx.stroke(lens);
  ctx.restore();
}

function drawTrackedMask(
  ctx: CanvasRenderingContext2D,
  face: FaceLandmarks,
  width: number,
  height: number
): void {
  const leftEar = { x: mirrorX(face.leftEar.x, width), y: face.leftEar.y * height };
  const rightEar = { x: mirrorX(face.rightEar.x, width), y: face.rightEar.y * height };
  const leftEye = { x: mirrorX(face.leftEye.x, width), y: face.leftEye.y * height };
  const rightEye = { x: mirrorX(face.rightEye.x, width), y: face.rightEye.y * height };
  const screenLeftEar = leftEar;
  const screenRightEar = rightEar;
  const earDx = screenRightEar.x - screenLeftEar.x;
  const earDy = screenRightEar.y - screenLeftEar.y;
  const earDistance = Math.hypot(earDx, earDy);
  if (earDistance < height * 0.035) return;

  const angle = Math.atan2(earDy, earDx);
  const maskWidth = earDistance * 1.5;
  const maskHeight = maskWidth * 1.44;
  const center = {
    x: (leftEar.x + rightEar.x) / 2,
    y: (leftEar.y + rightEar.y) / 2 + maskHeight * 0.025,
  };
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const toLocal = (point: CanvasPoint): CanvasPoint => {
    const dx = point.x - center.x;
    const dy = point.y - center.y;
    return { x: dx * cos + dy * sin, y: -dx * sin + dy * cos };
  };
  const localLeftEye = toLocal(rightEye);
  const localRightEye = toLocal(leftEye);
  const localNose = toLocal({ x: mirrorX(face.nose.x, width), y: face.nose.y * height });

  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(angle);

  const maskPath = new Path2D();
  maskPath.moveTo(0, -maskHeight * 0.53);
  maskPath.bezierCurveTo(
    maskWidth * 0.2,
    -maskHeight * 0.54,
    maskWidth * 0.42,
    -maskHeight * 0.38,
    maskWidth * 0.46,
    -maskHeight * 0.16
  );
  maskPath.bezierCurveTo(
    maskWidth * 0.5,
    maskHeight * 0.14,
    maskWidth * 0.27,
    maskHeight * 0.45,
    0,
    maskHeight * 0.53
  );
  maskPath.bezierCurveTo(
    -maskWidth * 0.27,
    maskHeight * 0.45,
    -maskWidth * 0.5,
    maskHeight * 0.14,
    -maskWidth * 0.46,
    -maskHeight * 0.16
  );
  maskPath.bezierCurveTo(
    -maskWidth * 0.42,
    -maskHeight * 0.38,
    -maskWidth * 0.2,
    -maskHeight * 0.54,
    0,
    -maskHeight * 0.53
  );
  maskPath.closePath();

  ctx.clip(maskPath);
  const maskPhoto = getPhotographicMask();
  // Avoid ever flashing the old illustrated fallback while the real texture loads.
  if (!maskPhoto) {
    ctx.restore();
    return;
  }
  if (maskPhoto) {
    // This crop is a real photographed fabric mask. A small counter-rotation
    // levels the source eyes before the whole texture follows the player's head.
    ctx.save();
    ctx.rotate(-0.11);
    ctx.drawImage(
      maskPhoto,
      390,
      245,
      175,
      255,
      -maskWidth * 0.52,
      -maskHeight * 0.56,
      maskWidth * 1.04,
      maskHeight * 1.12
    );
    ctx.restore();

    // Preserve the photograph while adding live contouring at the temples and jaw.
    const photoContour = ctx.createLinearGradient(-maskWidth * 0.5, 0, maskWidth * 0.5, 0);
    photoContour.addColorStop(0, "rgba(3,0,2,0.48)");
    photoContour.addColorStop(0.2, "rgba(3,0,2,0.04)");
    photoContour.addColorStop(0.5, "rgba(255,255,255,0.035)");
    photoContour.addColorStop(0.8, "rgba(3,0,2,0.04)");
    photoContour.addColorStop(1, "rgba(3,0,2,0.48)");
    ctx.fillStyle = photoContour;
    ctx.fillRect(-maskWidth, -maskHeight, maskWidth * 2, maskHeight * 2);

    const photoJaw = ctx.createLinearGradient(0, maskHeight * 0.18, 0, maskHeight * 0.54);
    photoJaw.addColorStop(0, "rgba(0,0,0,0)");
    photoJaw.addColorStop(1, "rgba(5,0,2,0.3)");
    ctx.fillStyle = photoJaw;
    ctx.fillRect(-maskWidth, maskHeight * 0.12, maskWidth * 2, maskHeight * 0.46);
    ctx.restore();

    ctx.save();
    ctx.translate(center.x, center.y);
    ctx.rotate(angle);
    ctx.strokeStyle = "rgba(12,4,7,0.82)";
    ctx.lineWidth = Math.max(1.5, maskWidth * 0.008);
    ctx.stroke(maskPath);
    ctx.restore();
    return;
  }

  const red = ctx.createRadialGradient(-maskWidth * 0.12, -maskHeight * 0.22, 0, 0, 0, maskWidth * 0.7);
  red.addColorStop(0, "#f0444f");
  red.addColorStop(0.32, "#c8172d");
  red.addColorStop(0.7, "#8f0c22");
  red.addColorStop(1, "#3e0713");
  ctx.fillStyle = red;
  ctx.fillRect(-maskWidth, -maskHeight, maskWidth * 2, maskHeight * 2);

  // Sculpt cheekbones, brow, nose and jaw with soft suit-conforming shadows.
  const sideShade = ctx.createLinearGradient(-maskWidth * 0.5, 0, maskWidth * 0.5, 0);
  sideShade.addColorStop(0, "rgba(20,0,7,0.62)");
  sideShade.addColorStop(0.22, "rgba(20,0,7,0.08)");
  sideShade.addColorStop(0.5, "rgba(255,116,116,0.08)");
  sideShade.addColorStop(0.78, "rgba(20,0,7,0.08)");
  sideShade.addColorStop(1, "rgba(20,0,7,0.62)");
  ctx.fillStyle = sideShade;
  ctx.fillRect(-maskWidth, -maskHeight, maskWidth * 2, maskHeight * 2);

  const noseShade = ctx.createRadialGradient(localNose.x, localNose.y - maskHeight * 0.04, 0, localNose.x, localNose.y, maskWidth * 0.24);
  noseShade.addColorStop(0, "rgba(255,122,122,0.27)");
  noseShade.addColorStop(0.36, "rgba(111,0,19,0.04)");
  noseShade.addColorStop(1, "rgba(45,0,10,0.32)");
  ctx.fillStyle = noseShade;
  ctx.fillRect(-maskWidth, -maskHeight, maskWidth * 2, maskHeight * 2);

  const jawShade = ctx.createLinearGradient(0, maskHeight * 0.12, 0, maskHeight * 0.54);
  jawShade.addColorStop(0, "rgba(0,0,0,0)");
  jawShade.addColorStop(1, "rgba(20,0,6,0.5)");
  ctx.fillStyle = jawShade;
  ctx.fillRect(-maskWidth, maskHeight * 0.08, maskWidth * 2, maskHeight * 0.5);

  // Interlocking micro-mesh reads as real technical fabric at webcam scale.
  ctx.strokeStyle = "rgba(255,171,171,0.13)";
  ctx.lineWidth = Math.max(0.45, maskWidth * 0.0018);
  const weave = Math.max(3.5, maskWidth * 0.021);
  for (let y = -maskHeight * 0.58; y < maskHeight * 0.58; y += weave) {
    ctx.beginPath();
    ctx.moveTo(-maskWidth * 0.58, y);
    ctx.lineTo(maskWidth * 0.58, y + weave * 0.55);
    ctx.stroke();
  }
  ctx.strokeStyle = "rgba(36,0,8,0.2)";
  for (let x = -maskWidth * 0.58; x < maskWidth * 0.58; x += weave) {
    ctx.beginPath();
    ctx.moveTo(x, -maskHeight * 0.58);
    ctx.lineTo(x + weave * 0.55, maskHeight * 0.58);
    ctx.stroke();
  }

  // Deterministic fiber flecks break up the synthetic smoothness without shimmer.
  for (let i = 0; i < 180; i += 1) {
    const noiseX = ((Math.sin(i * 91.17) + 1) * 0.5 - 0.5) * maskWidth;
    const noiseY = ((Math.sin(i * 47.31 + 1.7) + 1) * 0.5 - 0.5) * maskHeight * 1.08;
    ctx.fillStyle = i % 3 === 0 ? "rgba(255,190,190,0.075)" : "rgba(20,0,6,0.085)";
    ctx.fillRect(noiseX, noiseY, Math.max(0.5, maskWidth * 0.002), Math.max(0.5, maskWidth * 0.002));
  }

  const webOrigin = { x: 0, y: (localLeftEye.y + localRightEye.y) * 0.5 + maskHeight * 0.035 };
  const drawWeb = (offsetY: number, color: string, lineWidth: number): void => {
    ctx.save();
    ctx.translate(0, offsetY);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (let i = 0; i < 12; i += 1) {
      const rayAngle = -Math.PI / 2 + (i / 12) * Math.PI * 2;
      ctx.beginPath();
      ctx.moveTo(webOrigin.x, webOrigin.y);
      ctx.lineTo(
        webOrigin.x + Math.cos(rayAngle) * maskWidth * 0.68,
        webOrigin.y + Math.sin(rayAngle) * maskHeight * 0.76
      );
      ctx.stroke();
    }
    for (const ring of [0.18, 0.32, 0.48, 0.65, 0.84]) {
      ctx.beginPath();
      for (let step = 0; step <= 48; step += 1) {
        const theta = (step / 48) * Math.PI * 2;
        const scallop = 1 - 0.055 * Math.abs(Math.sin(theta * 6));
        const x = webOrigin.x + Math.cos(theta) * maskWidth * ring * 0.68 * scallop;
        const y = webOrigin.y + Math.sin(theta) * maskHeight * ring * 0.76 * scallop;
        if (step === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
    ctx.restore();
  };
  drawWeb(maskWidth * 0.009, "rgba(20,0,5,0.58)", Math.max(2.5, maskWidth * 0.017));
  drawWeb(0, "rgba(12,14,17,0.96)", Math.max(1.5, maskWidth * 0.011));
  drawWeb(-maskWidth * 0.0025, "rgba(125,132,138,0.5)", Math.max(0.6, maskWidth * 0.003));

  ctx.restore();

  // Draw eye lenses after restoring the clip so the black rims stay sharp.
  ctx.save();
  ctx.translate(center.x, center.y);
  ctx.rotate(angle);
  const lensWidth = maskWidth * 0.225;
  const lensHeight = maskHeight * 0.16;
  drawEyeLens(ctx, localLeftEye, lensWidth, lensHeight, -1);
  drawEyeLens(ctx, localRightEye, lensWidth, lensHeight, 1);
  const rim = ctx.createLinearGradient(-maskWidth * 0.5, 0, maskWidth * 0.5, 0);
  rim.addColorStop(0, "rgba(15,0,4,0.9)");
  rim.addColorStop(0.5, "rgba(255,120,120,0.34)");
  rim.addColorStop(1, "rgba(15,0,4,0.9)");
  ctx.strokeStyle = rim;
  ctx.lineWidth = Math.max(1.5, maskWidth * 0.009);
  ctx.stroke(maskPath);
  ctx.restore();
}

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

  if (scene.face) drawTrackedMask(ctx, scene.face, width, height);

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
