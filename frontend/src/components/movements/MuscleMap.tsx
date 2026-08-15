import type { ReactElement } from "react";
import type { Muscle } from "../../lib/movementDetail";

// The body map on the movement detail page: a grey mannequin with the trained muscle groups lit
// up — violet for the prime movers, pale violet for the supporting ones.
//
// WHY IT IS DRAWN AND NOT AN IMAGE: the reference mock ships one anterior and one posterior PNG
// with the quadriceps already highlighted, which is right for the squat and wrong for the other
// fifteen movements. A highlight that cannot change is a picture of the squat, not a body map, so
// the regions are paths here and the fill comes from the movement's own muscle list.
//
// It is deliberately a mannequin rather than an anatomical plate: schematic blocks that read at
// 200px tall, not a claim about where the vastus medialis ends. The muscle NAMES beside the map
// carry the precision; the figure carries the "front of the thighs, both sides" part.

interface Props {
  side: "front" | "back";
  primary: Muscle[];
  secondary: Muscle[];
  className?: string;
}

// Which side of the body each group is visible from. A group absent from a side simply is not
// drawn there — the legend still lists it, and the other side shows it.
const FRONT: Muscle[] = [
  "shoulders",
  "chest",
  "biceps",
  "forearms",
  "abs",
  "obliques",
  "hipFlexors",
  "quads",
  "adductors",
];

const BACK: Muscle[] = [
  "shoulders",
  "upperBack",
  "lats",
  "lowerBack",
  "triceps",
  "forearms",
  "glutes",
  "hamstrings",
  "calves",
];

// The mannequin, on a 200x360 grid: head at the top, feet at 340. Both sides share it; only the
// highlighted regions differ.
function Mannequin() {
  return (
    <g className="fill-content/[0.08]">
      <ellipse cx="100" cy="26" rx="16" ry="19" />
      <rect x="92" y="40" width="16" height="20" rx="6" />
      {/* Torso: wide at the shoulders, pulled in at the waist, flaring back out to the hips. */}
      <path d="M62 64q38-13 76 0l-7 40q-9 15-9 25 0 12 4 23H74q4-11 4-23 0-10-9-25z" />
      {/* Pelvis */}
      <path d="M78 150h44l3 28q-25 14-50 0z" />
      {/* Arms. They overlap the torso's shoulder line by ~8 units, so the shoulder reads as a
          joint rather than as a limb parked beside the trunk. */}
      <rect x="53" y="62" width="17" height="60" rx="8" />
      <rect x="130" y="62" width="17" height="60" rx="8" />
      <rect x="50" y="116" width="15" height="52" rx="7" />
      <rect x="135" y="116" width="15" height="52" rx="7" />
      <ellipse cx="57" cy="176" rx="8" ry="11" />
      <ellipse cx="143" cy="176" rx="8" ry="11" />
      {/* Legs */}
      <rect x="78" y="174" width="21" height="82" rx="10" />
      <rect x="101" y="174" width="21" height="82" rx="10" />
      <rect x="81" y="250" width="17" height="76" rx="8" />
      <rect x="102" y="250" width="17" height="76" rx="8" />
      <ellipse cx="88" cy="330" rx="12" ry="8" />
      <ellipse cx="112" cy="330" rx="12" ry="8" />
    </g>
  );
}

// One highlight region per muscle group, positioned onto the mannequin above. Mirrored pairs are
// drawn as two shapes so a single-sided highlight is never implied.
const REGION: Record<Muscle, ReactElement> = {
  shoulders: (
    <>
      <ellipse cx="66" cy="72" rx="12" ry="13" />
      <ellipse cx="134" cy="72" rx="12" ry="13" />
    </>
  ),
  chest: (
    <>
      <path d="M88 68q-13 1-14 12 0 12 14 14 4-13 0-26z" />
      <path d="M112 68q13 1 14 12 0 12-14 14-4-13 0-26z" />
    </>
  ),
  biceps: (
    <>
      <rect x="55" y="72" width="13" height="40" rx="6" />
      <rect x="132" y="72" width="13" height="40" rx="6" />
    </>
  ),
  triceps: (
    <>
      <rect x="55" y="74" width="13" height="42" rx="6" />
      <rect x="132" y="74" width="13" height="42" rx="6" />
    </>
  ),
  forearms: (
    <>
      <rect x="51" y="120" width="13" height="44" rx="6" />
      <rect x="136" y="120" width="13" height="44" rx="6" />
    </>
  ),
  abs: <rect x="89" y="96" width="22" height="52" rx="8" />,
  obliques: (
    <>
      <path d="M86 100q-9 3-10 16 0 16 8 30 5-24 2-46z" />
      <path d="M114 100q9 3 10 16 0 16-8 30-5-24-2-46z" />
    </>
  ),
  upperBack: <path d="M74 62q26-8 52 0l-5 40H79z" />,
  lats: (
    <>
      <path d="M76 100q-4 22 6 44 8-6 10-20-2-16-16-24z" />
      <path d="M124 100q4 22-6 44-8-6-10-20 2-16 16-24z" />
    </>
  ),
  lowerBack: <rect x="84" y="122" width="32" height="28" rx="8" />,
  glutes: (
    <>
      <ellipse cx="89" cy="168" rx="15" ry="16" />
      <ellipse cx="111" cy="168" rx="15" ry="16" />
    </>
  ),
  hipFlexors: (
    <>
      <ellipse cx="90" cy="176" rx="10" ry="14" />
      <ellipse cx="110" cy="176" rx="10" ry="14" />
    </>
  ),
  quads: (
    <>
      <rect x="79" y="182" width="19" height="66" rx="9" />
      <rect x="102" y="182" width="19" height="66" rx="9" />
    </>
  ),
  hamstrings: (
    <>
      <rect x="79" y="186" width="19" height="64" rx="9" />
      <rect x="102" y="186" width="19" height="64" rx="9" />
    </>
  ),
  adductors: (
    <>
      <rect x="91" y="180" width="8" height="56" rx="4" />
      <rect x="101" y="180" width="8" height="56" rx="4" />
    </>
  ),
  calves: (
    <>
      <rect x="82" y="256" width="15" height="46" rx="7" />
      <rect x="103" y="256" width="15" height="46" rx="7" />
    </>
  ),
};

export default function MuscleMap({ side, primary, secondary, className }: Props) {
  const visible = side === "front" ? FRONT : BACK;
  const show = (list: Muscle[]) => list.filter((m) => visible.includes(m));

  return (
    <svg
      viewBox="0 0 200 360"
      className={className}
      role="img"
      aria-hidden="true"
      focusable="false"
    >
      <Mannequin />
      {/* Secondary first, so a group listed in both (which the data does not do today, but
          nothing stops) reads as primary rather than as whichever was drawn last. */}
      <g className="fill-primary/30">
        {show(secondary).map((m) => (
          <g key={m}>{REGION[m]}</g>
        ))}
      </g>
      <g className="fill-primary">
        {show(primary).map((m) => (
          <g key={m}>{REGION[m]}</g>
        ))}
      </g>
    </svg>
  );
}
