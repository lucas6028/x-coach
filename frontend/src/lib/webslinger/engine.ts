export type Point = { x: number; y: number };

export type WebTarget = {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
};

export type WebRay = Point & { x2: number; y2: number };

export type WebTrace = WebRay & { life: number; hit: boolean };

export type WebGameState = {
  targets: WebTarget[];
  traces: WebTrace[];
  score: number;
  combo: number;
  bestCombo: number;
  hits: number;
  nextId: number;
  nextSpawnAt: number;
};

export const ROUND_SECONDS = 45;
const MAX_TARGETS = 5;
const SPAWN_MS = 850;

const distance = (a: Point, b: Point) => Math.hypot(a.x - b.x, a.y - b.y);

export function detectWebFlick(
  elbow: Point,
  wrist: Point,
  previousWrist: Point | null,
  dtMs: number
): WebRay | null {
  if (!previousWrist || dtMs <= 0) return null;
  const armX = wrist.x - elbow.x;
  const armY = wrist.y - elbow.y;
  const armLength = Math.hypot(armX, armY);
  if (armLength < 0.12) return null;

  const moveX = wrist.x - previousWrist.x;
  const moveY = wrist.y - previousWrist.y;
  const speed = Math.hypot(moveX, moveY) / (dtMs / 1000);
  const outwardSpeed = (moveX * armX + moveY * armY) / armLength / (dtMs / 1000);
  if (speed < 0.48 || outwardSpeed < 0.28) return null;

  const dx = armX / armLength;
  const dy = armY / armLength;
  return { x: wrist.x, y: wrist.y, x2: wrist.x + dx * 1.5, y2: wrist.y + dy * 1.5 };
}

export function createTarget(id: number, rng: () => number = Math.random): WebTarget {
  const angle = rng() * Math.PI * 2;
  const speed = 0.055 + rng() * 0.06;
  return {
    id,
    x: 0.16 + rng() * 0.68,
    y: 0.18 + rng() * 0.48,
    vx: Math.cos(angle) * speed,
    vy: Math.sin(angle) * speed,
    radius: 0.045 + rng() * 0.012,
  };
}

export function createWebGameState(now: number, rng: () => number = Math.random): WebGameState {
  return {
    targets: [createTarget(1, rng), createTarget(2, rng), createTarget(3, rng)],
    traces: [],
    score: 0,
    combo: 0,
    bestCombo: 0,
    hits: 0,
    nextId: 4,
    nextSpawnAt: now + SPAWN_MS,
  };
}

export function advanceWorld(
  state: WebGameState,
  dtMs: number,
  now: number,
  rng: () => number = Math.random
): WebGameState {
  const dt = Math.max(0, Math.min(dtMs, 100)) / 1000;
  const targets = state.targets.map((target) => {
    let { x, y, vx, vy } = target;
    x += vx * dt;
    y += vy * dt;
    if (x < 0.1 || x > 0.9) {
      x = Math.max(0.1, Math.min(0.9, x));
      vx *= -1;
    }
    if (y < 0.12 || y > 0.78) {
      y = Math.max(0.12, Math.min(0.78, y));
      vy *= -1;
    }
    return { ...target, x, y, vx, vy };
  });

  let nextId = state.nextId;
  let nextSpawnAt = state.nextSpawnAt;
  if (now >= nextSpawnAt) {
    if (targets.length < MAX_TARGETS) targets.push(createTarget(nextId++, rng));
    nextSpawnAt = now + SPAWN_MS;
  }

  return {
    ...state,
    targets,
    traces: state.traces
      .map((trace) => ({ ...trace, life: trace.life - dt * 3.5 }))
      .filter((trace) => trace.life > 0),
    nextId,
    nextSpawnAt,
  };
}

function distanceToRay(target: Point, ray: WebRay): number {
  const dx = ray.x2 - ray.x;
  const dy = ray.y2 - ray.y;
  const length2 = dx * dx + dy * dy;
  if (length2 === 0) return Number.POSITIVE_INFINITY;
  const t = ((target.x - ray.x) * dx + (target.y - ray.y) * dy) / length2;
  if (t < 0 || t > 1) return Number.POSITIVE_INFINITY;
  return distance(target, { x: ray.x + t * dx, y: ray.y + t * dy });
}

export function fireWeb(state: WebGameState, ray: WebRay): WebGameState {
  let hit: WebTarget | null = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const target of state.targets) {
    const d = distanceToRay(target, ray);
    if (d <= target.radius + 0.025 && d < bestDistance) {
      hit = target;
      bestDistance = d;
    }
  }

  if (!hit) {
    return { ...state, combo: 0, traces: [...state.traces, { ...ray, life: 1, hit: false }] };
  }

  const combo = state.combo + 1;
  const trace = { ...ray, x2: hit.x, y2: hit.y, life: 1, hit: true };
  return {
    ...state,
    targets: state.targets.filter((target) => target.id !== hit?.id),
    traces: [...state.traces, trace],
    score: state.score + 100 + Math.min(combo - 1, 8) * 25,
    combo,
    bestCombo: Math.max(state.bestCombo, combo),
    hits: state.hits + 1,
  };
}
