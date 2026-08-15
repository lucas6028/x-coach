// The authored half of the movement detail page: what a movement IS, how it is performed, and
// which muscles it trains. Everything on that page that comes from the pipeline instead — the
// fault list, the user's own records, whether the movement is analyzable at all — is fetched, not
// stated here.
//
// WHY THIS IS A SEPARATE MODULE, not more keys in i18n.tsx: the dictionary there is ~1900 lines of
// UI chrome, and this is content — sixteen movements' worth of prose, in two languages, that reads
// as a table only when the two languages sit side by side. Section headings and the enum labels
// (muscle names, "Beginner", "Bodyweight") stay in i18n; this holds the per-movement text.
//
// The keys are the canonical English movement names from lib/movements.ts, matching the backend's
// spelling verbatim — the same key the KG, /api/movements and the ?movement= parameter use.

import type { Lang } from "./i18n";

/** A muscle group the body map can highlight. Every value MUST have a matching region in
 *  components/movements/MuscleMap.tsx and a `muscle.*` label in i18n. */
export type Muscle =
  | "shoulders"
  | "chest"
  | "biceps"
  | "triceps"
  | "forearms"
  | "abs"
  | "obliques"
  | "upperBack"
  | "lats"
  | "lowerBack"
  | "glutes"
  | "quads"
  | "hamstrings"
  | "hipFlexors"
  | "adductors"
  | "calves";

export type Difficulty = "beginner" | "intermediate" | "advanced";
export type Equipment = "bodyweight" | "dumbbell" | "barbell" | "band" | "mat";
export type ExerciseType = "strength" | "mobility" | "stability" | "conditioning";
export type Mechanic = "compound" | "isolation";
export type Force = "push" | "pull" | "static" | "rotation" | "dynamic";

/** One step of the movement. `image` is a path under public/; steps without one fall back to the
 *  movement's own card art (see MovementSteps). */
export interface Step {
  text: Record<Lang, string>;
  image?: string;
}

export interface MovementDetail {
  description: Record<Lang, string>;
  difficulty: Difficulty;
  equipment: Equipment;
  type: ExerciseType;
  mechanic: Mechanic;
  force: Force;
  primary: Muscle[];
  secondary: Muscle[];
  steps: Step[];
  /** An illustrated anterior+posterior muscle plate under public/movements/muscles-worked/, for
   *  the movements one has been drawn for. Where it is absent the page falls back to the drawn
   *  body map (components/movements/MuscleMap.tsx), which highlights the same `primary` and
   *  `secondary` groups listed above — so a movement without a plate is never blank. */
  plate?: string;
  /** A clip under public/demo/, when one exists. Four movements have one; the rest omit the
   *  demo card rather than borrow another movement's footage. */
  demo?: { clip: string };
}

// Only Squat has step illustrations so far (five figures, one per step, trimmed and re-encoded by
// scripts/prep_movement_detail_art.py). The other fifteen fall back to their card art, which is
// one pose rather than five — the numbered cues are still the useful part, and a code change
// cannot honestly invent fifteen more figure sets. Adding a set is a drop-in: put the PNGs in
// frontend/public/movements/steps/, run the script, and list the WebPs here.
const SQUAT_STEP_ART = [
  "/movements/steps/squat-1.webp",
  "/movements/steps/squat-2.webp",
  "/movements/steps/squat-3.webp",
  "/movements/steps/squat-4.webp",
  "/movements/steps/squat-5.webp",
];

const step = (en: string, zh: string, image?: string): Step => ({
  text: { en, "zh-Hant": zh },
  ...(image ? { image } : {}),
});

export const MOVEMENT_DETAIL: Record<string, MovementDetail> = {
  Squat: {
    description: {
      en: "A fundamental lower-body pattern that trains the quadriceps, glutes and hamstrings together, and the movement the analyzer knows best.",
      "zh-Hant": "最基本的下肢動作，同時訓練股四頭肌、臀肌與腿後肌，也是分析器目前掌握得最完整的動作。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "strength",
    mechanic: "compound",
    force: "push",
    primary: ["quads", "glutes"],
    secondary: ["hamstrings", "lowerBack", "abs", "calves"],
    steps: [
      step("Stand with feet shoulder-width apart.", "雙腳與肩同寬站好。", SQUAT_STEP_ART[0]),
      step("Push the hips back and bend the knees to lower down.", "臀部向後推、屈膝往下坐。", SQUAT_STEP_ART[1]),
      step("Keep the chest up and the knees tracking over the toes.", "保持胸口挺起，膝蓋對齊腳尖方向。", SQUAT_STEP_ART[2]),
      step("Lower until the thighs are parallel to the ground.", "下蹲到大腿與地面平行。", SQUAT_STEP_ART[3]),
      step("Push through the heels to return to standing.", "腳跟踩穩地面推起，回到站姿。", SQUAT_STEP_ART[4]),
    ],
    plate: "/movements/muscles-worked/squat.webp",
    demo: { clip: "squat" },
  },

  Lunge: {
    description: {
      en: "A single-leg pattern that builds leg strength one side at a time and exposes balance and hip control a two-legged squat can hide.",
      "zh-Hant": "單腳訓練動作，一次練一側，能夠暴露雙腳深蹲藏得住的平衡與髖部控制問題。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "strength",
    mechanic: "compound",
    force: "push",
    primary: ["quads", "glutes"],
    secondary: ["hamstrings", "abs", "calves"],
    steps: [
      step("Stand tall with the feet hip-width apart.", "雙腳與髖同寬，站直。"),
      step("Take a controlled step forward and plant the whole foot.", "有控制地向前跨一步，整個腳掌踩實。"),
      step("Bend both knees until the back knee is just above the floor.", "雙膝同時彎曲，後腳膝蓋接近地面。"),
      step("Keep the front knee over the mid-foot and the torso upright.", "前腳膝蓋對齊腳掌中央，軀幹保持直立。"),
      step("Drive through the front heel to stand, then switch sides.", "前腳跟推地站起，再換邊。"),
    ],
  },

  Deadlift: {
    description: {
      en: "A hip hinge that loads the whole posterior chain — glutes, hamstrings and back — with the bar travelling in a straight line.",
      "zh-Hant": "以髖關節鉸鏈為主的動作，槓鈴走直線，訓練整條後側鏈：臀肌、腿後肌與背部。",
    },
    difficulty: "intermediate",
    equipment: "barbell",
    type: "strength",
    mechanic: "compound",
    force: "pull",
    primary: ["glutes", "hamstrings"],
    secondary: ["lowerBack", "upperBack", "lats", "forearms", "quads"],
    steps: [
      step("Stand with the mid-foot under the bar, feet about hip-width.", "腳掌中央對準槓下，雙腳與髖同寬。"),
      step("Hinge down and grip the bar just outside the knees.", "髖部後推下蹲，雙手握在膝蓋外側。"),
      step("Flatten the back, pull the slack out of the bar and brace.", "背部打平，先拉緊槓鈴的鬆弛，核心收緊。"),
      step("Push the floor away and stand tall, bar close to the legs.", "推地站直，槓鈴貼著大腿走。"),
      step("Hinge at the hips first, then bend the knees to lower it.", "先髖部後推，再屈膝把槓放回地面。"),
    ],
  },

  "Leg Abduction": {
    description: {
      en: "A hip abduction drill from the rehabilitation protocol: the leg travels sideways while the pelvis stays level, training the lateral hip.",
      "zh-Hant": "來自復健流程的髖外展訓練：腿往側邊抬起、骨盆維持水平，訓練髖部外側肌群。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "mobility",
    mechanic: "isolation",
    force: "push",
    primary: ["glutes"],
    secondary: ["obliques", "lowerBack", "abs"],
    steps: [
      step("Stand tall with the weight on the supporting leg.", "站直，重心放在支撐腳上。"),
      step("Brace the core so the trunk does not lean away.", "收緊核心，軀幹不要往反方向倒。"),
      step("Lift the working leg out to the side, toes forward.", "工作腳向側邊抬起，腳尖朝前。"),
      step("Stop where the pelvis starts to tip and hold briefly.", "抬到骨盆快要傾斜就停住，稍作停留。"),
      step("Lower under control without letting the foot crash down.", "有控制地放下，不要讓腳直接落地。"),
    ],
  },

  "Shoulder Bridge": {
    description: {
      en: "A floor hip extension that teaches the glutes to do the work the lower back usually takes over.",
      "zh-Hant": "在地面進行的髖伸展動作，教臀肌接手平常被下背代償的工作。",
    },
    difficulty: "beginner",
    equipment: "mat",
    type: "stability",
    mechanic: "compound",
    force: "push",
    primary: ["glutes", "hamstrings"],
    secondary: ["lowerBack", "abs"],
    steps: [
      step("Lie on your back with the knees bent and feet flat.", "仰躺，屈膝、雙腳踩地。"),
      step("Tuck the ribs down so the lower back stays long.", "肋骨微收，讓下背保持延展。"),
      step("Press through the heels and lift the hips.", "腳跟推地，把髖部抬起。"),
      step("Stop where hips, knees and shoulders form one line.", "抬到髖、膝、肩成一直線就停住。"),
      step("Lower one vertebra at a time back to the floor.", "一節一節把脊椎放回地面。"),
    ],
  },

  "Push-up": {
    description: {
      en: "A horizontal press that trains the chest, shoulders and triceps while the whole trunk holds a plank.",
      "zh-Hant": "水平推的動作，訓練胸、肩與三頭肌，同時整個軀幹必須維持棒式。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "strength",
    mechanic: "compound",
    force: "push",
    primary: ["chest", "triceps"],
    secondary: ["shoulders", "abs", "obliques"],
    steps: [
      step("Start in a high plank with the hands under the shoulders.", "從高棒式開始，雙手位於肩膀正下方。"),
      step("Make one straight line from the head to the heels.", "頭到腳跟連成一直線。"),
      step("Bend the elbows about 45° from the torso and lower.", "手肘與軀幹約 45 度彎曲，身體下降。"),
      step("Lower the chest to just above the floor.", "胸口下降到接近地面。"),
      step("Push the floor away without letting the hips sag.", "推地回到起始，過程中髖部不要下沉。"),
    ],
    demo: { clip: "pushups" },
  },

  "Overhead Press": {
    description: {
      en: "A vertical press to full lockout overhead, where the ribs and glutes decide whether the spine or the shoulders take the load.",
      "zh-Hant": "垂直推到頭頂完全鎖定的動作；肋骨與臀部的位置，決定負荷落在脊椎還是肩膀。",
    },
    difficulty: "intermediate",
    equipment: "dumbbell",
    type: "strength",
    mechanic: "compound",
    force: "push",
    primary: ["shoulders", "triceps"],
    secondary: ["upperBack", "abs", "lowerBack"],
    steps: [
      step("Hold the weights at shoulder height, wrists stacked.", "重量置於肩膀高度，手腕與前臂對齊。"),
      step("Ribs down, glutes tight, chest tall.", "肋骨下收、臀部夾緊、胸口挺起。"),
      step("Press straight up, moving the head out of the way.", "垂直推起，頭部稍微後讓開路徑。"),
      step("Finish with the arms locked and biceps by the ears.", "手臂鎖定，二頭肌貼近耳朵。"),
      step("Lower under control back to the shoulders.", "有控制地降回肩膀高度。"),
    ],
  },

  Row: {
    description: {
      en: "A horizontal pull for the upper back, where the elbows lead and the torso stays still.",
      "zh-Hant": "水平拉的上背動作，由手肘帶動，軀幹全程不晃。",
    },
    difficulty: "beginner",
    equipment: "dumbbell",
    type: "strength",
    mechanic: "compound",
    force: "pull",
    primary: ["lats", "upperBack"],
    secondary: ["biceps", "forearms", "lowerBack"],
    steps: [
      step("Hinge forward with a flat back and soft knees.", "背部打平、膝蓋微彎，身體前傾。"),
      step("Let the arms hang straight under the shoulders.", "手臂自然垂在肩膀正下方。"),
      step("Pull the elbows back toward the hips.", "手肘往髖部方向後拉。"),
      step("Squeeze the shoulder blades without shrugging.", "肩胛骨後收，但不要聳肩。"),
      step("Lower the weight all the way without rocking the torso.", "完全放下重量，軀幹不要前後擺動。"),
    ],
  },

  "Bicep Curl": {
    description: {
      en: "A single-joint elbow flexion. The point of the exercise is what does NOT move: the shoulders and the trunk.",
      "zh-Hant": "單關節的肘屈曲動作。重點在於「不動的部分」：肩膀與軀幹。",
    },
    difficulty: "beginner",
    equipment: "dumbbell",
    type: "strength",
    mechanic: "isolation",
    force: "pull",
    primary: ["biceps"],
    secondary: ["forearms"],
    steps: [
      step("Stand tall with the weights hanging at the sides.", "站直，重量自然垂在身體兩側。"),
      step("Pin the elbows against the ribs.", "手肘固定貼在肋骨旁。"),
      step("Curl the weights up by bending the elbows only.", "只靠屈肘把重量捲起。"),
      step("Stop before the elbows drift forward.", "在手肘往前跑之前停住。"),
      step("Lower slowly to a full stretch.", "緩慢放下，回到完全伸展。"),
    ],
  },

  "Band Pull Apart": {
    description: {
      en: "A light band drill for the rear shoulder and mid-back — the muscles a day at a desk leaves long and weak.",
      "zh-Hant": "使用彈力帶的輕負荷動作，訓練後三角與中背——久坐一天最容易被拉長、變弱的位置。",
    },
    difficulty: "beginner",
    equipment: "band",
    type: "mobility",
    mechanic: "isolation",
    force: "pull",
    primary: ["upperBack", "shoulders"],
    secondary: ["lats"],
    steps: [
      step("Hold the band at shoulder width, arms out in front.", "雙手與肩同寬握住彈力帶，手臂前伸。"),
      step("Raise the band to chest height with straight arms.", "手臂打直，把彈力帶抬到胸口高度。"),
      step("Pull the band apart, leading with the hands.", "由手部帶動，把彈力帶往兩側拉開。"),
      step("Stop when the band touches the chest, ribs down.", "拉到彈力帶碰到胸口就停，肋骨保持下收。"),
      step("Return under tension rather than letting it snap back.", "維持張力回到起始，不要讓帶子彈回去。"),
    ],
  },

  "Arm Abduction": {
    description: {
      en: "Raising the arms out to the side — the shoulder-range drill from the rehabilitation protocol, judged on how level the movement stays.",
      "zh-Hant": "手臂向側邊抬起，是復健流程中的肩關節活動度訓練，重點在於動作是否維持平穩對稱。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "mobility",
    mechanic: "isolation",
    force: "push",
    primary: ["shoulders"],
    secondary: ["upperBack"],
    steps: [
      step("Stand tall with the arms relaxed at the sides.", "站直，雙手自然放鬆在身側。"),
      step("Turn the thumbs slightly up.", "拇指略微朝上。"),
      step("Raise both arms out to the side at the same speed.", "雙手以相同速度向側邊抬起。"),
      step("Stop at shoulder height without shrugging.", "抬到肩膀高度，不要聳肩。"),
      step("Lower under control, keeping both sides level.", "有控制地放下，兩側保持同高。"),
    ],
  },

  "Arm VW": {
    description: {
      en: "The V-to-W drill: arms overhead in a V, then pulled down into a W. A shoulder-blade exercise disguised as an arm one.",
      "zh-Hant": "V 字到 W 字的訓練：手臂先在頭頂張成 V，再下拉成 W。看起來練手臂，實際上練的是肩胛骨。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "mobility",
    mechanic: "compound",
    force: "pull",
    primary: ["upperBack", "shoulders"],
    secondary: ["lats"],
    steps: [
      step("Stand or lie face down with the arms overhead in a V.", "站姿或俯臥，雙手在頭頂張成 V 字。"),
      step("Keep the ribs down and the neck long.", "肋骨下收，頸部保持延長。"),
      step("Pull the elbows down and back into a W.", "手肘下拉後收，形成 W 字。"),
      step("Squeeze the shoulder blades together at the bottom.", "在最低點把肩胛骨往中間夾。"),
      step("Return to the V slowly, keeping the arms in view.", "緩慢回到 V 字，手臂維持在視線範圍內。"),
    ],
  },

  "Sit-up": {
    description: {
      en: "A full trunk flexion from the floor. Done well it is an abdominal exercise; done fast it becomes a hip-flexor one.",
      "zh-Hant": "從地面完成的完整軀幹屈曲。做得好是腹部訓練，做得急就變成髖屈肌訓練。",
    },
    difficulty: "beginner",
    equipment: "mat",
    type: "strength",
    mechanic: "compound",
    force: "pull",
    primary: ["abs"],
    secondary: ["obliques", "hipFlexors"],
    steps: [
      step("Lie on your back, knees bent, feet flat on the floor.", "仰躺，屈膝、雙腳踩地。"),
      step("Rest the hands on the chest or beside the head.", "雙手放在胸前或頭部兩側。"),
      step("Curl the head and shoulders up first.", "先把頭與肩膀捲起。"),
      step("Continue up until the torso is off the floor.", "繼續往上，直到軀幹離開地面。"),
      step("Lower one segment at a time, without dropping back.", "一段一段放下，不要整個往後倒。"),
    ],
    demo: { clip: "situps" },
  },

  "Torso Twist": {
    description: {
      en: "Rotation through the trunk with the hips facing forward — the drill that shows whether the rotation comes from the spine or the feet.",
      "zh-Hant": "髖部朝前、由軀幹旋轉的動作，可以看出旋轉究竟來自脊椎，還是靠腳在轉。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "mobility",
    mechanic: "compound",
    force: "rotation",
    primary: ["obliques"],
    secondary: ["abs", "lowerBack"],
    steps: [
      step("Stand with the feet planted hip-width apart.", "雙腳與髖同寬踩穩。"),
      step("Hold the arms in front or across the chest.", "雙手前伸或交叉於胸前。"),
      step("Rotate the ribcage to one side, hips facing forward.", "肋廓向一側旋轉，髖部保持朝前。"),
      step("Move only as far as the hips can stay square.", "轉到髖部快要跟著轉之前就停。"),
      step("Return through the middle and repeat the other way.", "回到中間再轉向另一側。"),
    ],
  },

  "Jumping Jacks": {
    description: {
      en: "A whole-body conditioning movement — arms and legs open and close on the same beat, which is exactly what makes it a rhythm exercise.",
      "zh-Hant": "全身性的體能動作：手腳在同一個節拍開合，也因此本質上是一個節奏訓練。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "conditioning",
    mechanic: "compound",
    force: "dynamic",
    primary: ["shoulders", "calves"],
    secondary: ["quads", "glutes", "adductors", "abs"],
    steps: [
      step("Stand with the feet together and the arms at the sides.", "雙腳併攏，雙手放在身側。"),
      step("Jump the feet out wider than the hips.", "跳開雙腳，比髖部更寬。"),
      step("Sweep the arms overhead on the same beat.", "同一拍把雙手往頭頂揮上。"),
      step("Land softly on the balls of the feet.", "以前腳掌輕柔落地。"),
      step("Jump back to the start and keep the tempo even.", "跳回起始位置，維持穩定節奏。"),
    ],
  },

  "High Knee": {
    description: {
      en: "Running on the spot with the knees driven to hip height — a conditioning drill scored on height and cadence, not on load.",
      "zh-Hant": "原地跑並把膝蓋抬到髖部高度的體能動作，看的是抬膝高度與節奏，而不是負荷。",
    },
    difficulty: "beginner",
    equipment: "bodyweight",
    type: "conditioning",
    mechanic: "compound",
    force: "dynamic",
    primary: ["hipFlexors", "quads"],
    secondary: ["calves", "abs", "glutes"],
    steps: [
      step("Stand tall with the feet under the hips.", "站直，雙腳位於髖部正下方。"),
      step("Drive one knee up toward hip height.", "把一邊膝蓋抬向髖部高度。"),
      step("Swap legs with a light bounce off the floor.", "以輕盈的彈跳換腳。"),
      step("Keep the torso upright rather than leaning back.", "軀幹保持直立，不要往後仰。"),
      step("Land on the balls of the feet and keep the rhythm.", "以前腳掌落地，維持節奏。"),
    ],
    demo: { clip: "highknee" },
  },
};

/** The detail record for a movement, or undefined for a name outside the catalog. */
export const movementDetail = (movement: string): MovementDetail | undefined =>
  MOVEMENT_DETAIL[movement];
