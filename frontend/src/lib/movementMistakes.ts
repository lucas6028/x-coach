// The common-mistakes half of the movement detail page: for every movement, the faults its rule
// detector can actually report, written out for a reader rather than for the analyzer.
//
// WHY THE LIST LIVES HERE AND NOT IN THE KNOWLEDGE GRAPH. The tab used to render whatever
// `GET /api/knowledge/faults` returned, which is every Fault node the graph defines for a movement
// — for a flagship that is dozens of nodes, most of which no detector will ever fire. What a user
// opening "common mistakes" wants is the shorter, honest list: the things this app will actually
// tell them about. That list is `src/pose/movements/<movement>.py`'s rules, so that is what this
// file mirrors, in each detector module's own rule order.
//
// The mirror is guarded. `tests/test_movement_mistakes_roster.py` parses this file and asserts its
// (movement -> fault_id, kgQuery) table equals the one it extracts from the detector modules, so a
// rule added, removed, renamed or re-pointed at a different KG node fails a Python test rather
// than silently dropping a card. Same bargain as `catalog.py` + `test_movement_catalog.py`: the
// list is written out because deriving it would need Python in the browser, and a test pays for it.
//
// WHY THE PROSE IS AUTHORED HERE AND NOT FETCHED. The graph stores concept NAMES ("Knee Valgus",
// "Weak Gluteus Medius") — three-word nodes that render as pills, not as the sentences this page
// needs. Each `why` below is a plain-language rendering of the `citation_support` string sitting
// beside that rule's `build_detection(...)` call, so the reason a fault is worth avoiding traces
// back to the same paper the analyzer cites when it reports one. The graph is still reached, one
// hop deeper: expanding a card fetches its causes / risks / cues, which is what `kgQuery` is for.
//
// It is a separate module from `movementDetail.ts` for the same reason that file gives for not
// being in `i18n.tsx`: this is content, in two languages, that reads as a table only when the two
// sit side by side. Section headings and button labels stay in i18n.
//
// The keys are the canonical English movement names, matching the backend, the KG and ?movement=.

import type { Lang } from "./i18n";

/** A wrong / correct illustration pair for one fault, as paths under `public/`. */
export interface MistakeArt {
  wrong: string;
  correct: string;
}

export interface Mistake {
  /** The detector's own `fault_id`. This is the join to the pipeline: a fault detected in a clip
   *  and the same fault read about here carry one identity, not two. */
  id: string;
  /** The detector's own `kg_query` — what `api.graph()` is asked for when the card is expanded.
   *  Verbatim from `build_detection(kg_query=...)`, so the card's causes / risks / cues are the
   *  ones a real detection of this fault would retrieve. */
  kgQuery: string;
  title: Record<Lang, string>;
  subtitle: Record<Lang, string>;
  /** Why the fault matters, in one sentence, derived from the rule's cited `citation_support`. */
  why: Record<Lang, string>;
  fixes: Record<Lang, readonly string[]>;
  /** The wrong / correct pair, for the faults one has been drawn for. Absent means the card
   *  renders without the visual pair rather than with a placeholder body — the same optionality
   *  `movementDetail`'s `plate` and `demo` use. Adding a pair is one `art(...)` call. */
  art?: MistakeArt;
}

/** Build the two illustration paths for a fault from its id. Call it on an entry once the two
 *  files exist under `frontend/public/movements/mistakes/`.
 *
 *  The fault_id's underscores become hyphens, because everything already in `public/movements/`
 *  is hyphenated (`squat-1.webp`, `high-knee.webp`) and one asset directory with two separator
 *  conventions is a trap for whoever adds the next pair. So `knees_inward` files as
 *  `knees-inward-wrong.webp` / `knees-inward-correct.webp`. */
export const art = (id: string): MistakeArt => {
  const slug = id.replace(/_/g, "-");
  return {
    wrong: `/movements/mistakes/${slug}-wrong.webp`,
    correct: `/movements/mistakes/${slug}-correct.webp`,
  };
};

/** One language's copy for a fault. Both languages fill the same shape, so a missing sentence in
 *  one of them is a type error rather than a card that renders half-translated. */
interface Copy {
  title: string;
  subtitle: string;
  why: string;
  fixes: readonly string[];
}

/** `drawn` is the pair, and it is a trailing argument rather than a field inside one of the two
 *  Copy objects because the drawing is not copy: one picture serves both languages. It stays
 *  trailing and optional so the (id, kgQuery) header of every call keeps the one rigid shape
 *  `tests/test_movement_mistakes_roster.py` reads the roster out of. */
const mistake = (id: string, kgQuery: string, en: Copy, zh: Copy, drawn?: MistakeArt): Mistake => ({
  id,
  kgQuery,
  title: { en: en.title, "zh-Hant": zh.title },
  subtitle: { en: en.subtitle, "zh-Hant": zh.subtitle },
  why: { en: en.why, "zh-Hant": zh.why },
  fixes: { en: en.fixes, "zh-Hant": zh.fixes },
  ...(drawn ? { art: drawn } : {}),
});

export const MOVEMENT_MISTAKES: Record<string, readonly Mistake[]> = {
  Squat: [
    mistake(
      "knees_inward",
      "Knee Valgus",
      {
        title: "Knees caving in",
        subtitle: "Knees collapse inward during the squat movement.",
        why: "This puts excessive stress on your knee joints and can lead to injury over time.",
        fixes: [
          "Push your knees out in line with your toes.",
          "Think about spreading the floor apart.",
          "Strengthen your glutes with exercises like glute bridges and band walks.",
        ],
      },
      {
        title: "膝蓋內夾",
        subtitle: "下蹲過程中膝蓋往內塌陷。",
        why: "這會讓膝關節承受過大的壓力，長期下來容易造成傷害。",
        fixes: [
          "把膝蓋往外推，對齊腳尖的方向。",
          "想像用雙腳把地板往兩側撐開。",
          "用臀橋、彈力帶側走等動作強化臀肌。",
        ],
      },
      art("knees_inward")
    ),
    mistake(
      "knees_forward",
      "Anterior Knee Translation",
      {
        title: "Knees travelling too far forward",
        subtitle: "The knees slide well past the toes on the way down.",
        why: "Letting the knee move in front of the toes raised peak patellar tendon stress by 11% and the knee extension moment by 26%.",
        fixes: [
          "Sit the hips back first, then bend the knees.",
          "Keep the shins as upright as the movement allows.",
          "Work on ankle mobility so depth doesn't have to come from the knees.",
        ],
      },
      {
        title: "膝蓋過度前移",
        subtitle: "下蹲時膝蓋明顯滑到腳尖前方。",
        why: "膝蓋移到腳尖前方時，髕骨肌腱的尖峰應力增加約 11%、膝伸展力矩增加約 26%。",
        fixes: [
          "先把髖部往後坐，再彎膝蓋。",
          "在動作允許的範圍內讓小腿盡量保持直立。",
          "加強腳踝活動度，不要靠膝蓋前推換取深度。",
        ],
      },
      art("knees_forward")
    ),
    mistake(
      "shallow_depth",
      "Shallow Depth",
      {
        title: "Not squatting deep enough",
        subtitle: "The squat stops above parallel, so the thighs never reach horizontal.",
        why: "Heavy half and quarter squats favour long-term degenerative change in the knee and spine compared with squatting deep.",
        fixes: [
          "Aim for the thighs at least parallel to the floor.",
          "Use a box or bench as a depth target.",
          "Improve hip and ankle mobility so the depth is available.",
        ],
      },
      {
        title: "蹲得不夠深",
        subtitle: "還沒到達大腿與地面平行就停住往上。",
        why: "相較於深蹲，長期以大重量做半蹲、四分之一蹲，較容易造成膝關節與脊椎的退化性變化。",
        fixes: [
          "至少蹲到大腿與地面平行。",
          "用箱子或板凳當作深度的參考點。",
          "改善髖與腳踝活動度，讓深度做得出來。",
        ],
      },
      art("shallow_depth")
    ),
    mistake(
      "excessive_forward_lean",
      "Excessive Forward Lean",
      {
        title: "Leaning too far forward",
        subtitle: "The chest drops toward the floor and the torso angle collapses.",
        why: "A forward trunk raises spinal flexion torque, so the lower-back extensors have to work much harder to hold the spine.",
        fixes: [
          "Keep the chest up and the eyes forward.",
          "Brace the core before you start the descent.",
          "Widen the stance slightly so the hips have room to travel back.",
        ],
      },
      {
        title: "軀幹過度前傾",
        subtitle: "胸口往地板倒，軀幹角度整個垮下來。",
        why: "軀幹前傾會提高脊椎的屈曲力矩，下背的豎脊肌必須出更多力才撐得住。",
        fixes: [
          "維持挺胸、視線向前。",
          "下蹲前先把核心撐好。",
          "站距稍微加寬，讓髖部有往後移動的空間。",
        ],
      },
      art("excessive_forward_lean")
    ),
    mistake(
      "heel_rise",
      "heel rise squat ankle dorsiflexion",
      {
        title: "Heels lifting off the floor",
        subtitle: "The heels come off the ground as you reach the bottom.",
        why: "Limited ankle dorsiflexion pushes the compensation up the kinetic chain; raising the heel measurably increased ankle excursion and squat depth.",
        fixes: [
          "Drive through the whole foot, heel included.",
          "Stretch the calves and train ankle dorsiflexion.",
          "Use a small heel wedge or lifting shoes while mobility improves.",
        ],
      },
      {
        title: "腳跟離地",
        subtitle: "蹲到底時腳跟被拉離地面。",
        why: "腳踝背屈受限會把代償往上半身推；把腳跟墊高後，腳踝活動範圍與下蹲深度都明顯增加。",
        fixes: [
          "用整個腳掌推地，腳跟也要出力。",
          "伸展小腿並訓練腳踝背屈。",
          "活動度還沒改善前，可以用小的墊片或舉重鞋。",
        ],
      },
      art("heel_rise")
    ),
  ],

  "Overhead Press": [
    mistake(
      "ohp_incomplete_lockout",
      "Incomplete Elbow Lockout",
      {
        title: "Not locking out at the top",
        subtitle: "The elbows stop short of straight at the top of the press.",
        why: "The elbow extensors only become dominant near full extension, so a rep that stops short skips the part that defines a completed press.",
        fixes: [
          "Press until the elbows are fully straight.",
          "Slow the last few centimetres instead of rushing them.",
          "Reduce the load until every rep finishes.",
        ],
      },
      {
        title: "頂端沒有完全鎖死",
        subtitle: "推到最高點時手肘還沒完全伸直。",
        why: "手肘伸肌要接近完全伸直才成為主要出力者，停在半途等於略過了定義一下完整推舉的那一段。",
        fixes: [
          "推到手肘完全打直為止。",
          "最後幾公分放慢，不要草草帶過。",
          "把重量降到每一下都推得完。",
        ],
      },
      art("ohp_incomplete_lockout")
    ),
    mistake(
      "ohp_lumbar_hyperextension",
      "Excessive Lower Back Arching",
      {
        title: "Leaning back and arching the lower back",
        subtitle: "The ribs flare and the lower back arches to get the weight up.",
        why: "The accentuated backbend caused enough lower-back injuries that the strict press was eventually removed from weightlifting competition.",
        fixes: [
          "Squeeze the glutes and keep the ribs pulled down.",
          "Press slightly around the head rather than out in front of it.",
          "Use a half-kneeling or seated press to take the lean away.",
        ],
      },
      {
        title: "後仰、下背過度反弓",
        subtitle: "肋骨外翻、腰往後折，靠身體後仰把重量推上去。",
        why: "過度後仰造成的下背傷害多到讓推舉最後被排除在舉重比賽項目之外。",
        fixes: [
          "夾緊臀部，把肋骨往下收。",
          "槓走頭部後方，不要停在臉前推。",
          "改用單膝跪姿或坐姿推舉，直接消掉後仰空間。",
        ],
      },
      art("ohp_lumbar_hyperextension")
    ),
    mistake(
      "ohp_asymmetric_press",
      "Muscle Imbalance",
      {
        title: "One arm leading the press",
        subtitle: "One side reaches lockout noticeably before the other.",
        why: "A shoulder-girdle difference of more than about 7 degrees counts as scapular dyskinesis, and heavy pressing to failure was shown to produce exactly that asymmetry.",
        fixes: [
          "Press with dumbbells so each side carries its own load.",
          "Film from the front and check both hands finish together.",
          "Set the tempo from the weaker side.",
        ],
      },
      {
        title: "單邊搶快",
        subtitle: "一邊明顯比另一邊先推到頂。",
        why: "肩帶兩側差距超過約 7 度即屬肩胛動作異常，而大重量推到力竭正好會製造出這種不對稱。",
        fixes: [
          "改用啞鈴，讓兩邊各自負擔重量。",
          "從正面錄影，確認兩手同時到頂。",
          "以比較弱的那一邊決定節奏。",
        ],
      },
      art("ohp_asymmetric_press")
    ),
    mistake(
      "ohp_insufficient_elevation",
      "Limited Shoulder Elevation",
      {
        title: "Not pressing fully overhead",
        subtitle: "The weight stalls in front of the head instead of finishing overhead.",
        why: "The full overhead position is what loads upper trapezius, deltoids and triceps together; a press that stalls below it never reaches that end position.",
        fixes: [
          "Finish with the arms beside the ears.",
          "Move the head through once the weight has cleared it.",
          "Check overhead mobility — restriction, not strength, is often the limit.",
        ],
      },
      {
        title: "沒有推到頭頂正上方",
        subtitle: "重量停在頭部前方，沒有推到頭頂上方鎖定。",
        why: "完整的過頭位置才會同時徵召上斜方肌、三角肌與三頭肌；推不到那個位置，等於整組動作都沒有到位。",
        fixes: [
          "結束時手臂貼在耳朵兩側。",
          "重量越過頭之後，頭要往前穿過去。",
          "檢查過頭活動度，常常是活動度而不是肌力在限制。",
        ],
      },
      art("ohp_insufficient_elevation")
    ),
    mistake(
      "ohp_forward_head",
      "Forward Head Posture",
      {
        title: "Head poking forward at lockout",
        subtitle: "The head stays pushed forward when the arms lock out.",
        why: "Overhead pressing measurably drives the head forward, and forward head posture narrows the subacromial space and reduces shoulder range.",
        fixes: [
          "Draw the chin back as the weight passes your face.",
          "Stack the ears over the shoulders at lockout.",
          "Loosen the chest and upper back between sets.",
        ],
      },
      {
        title: "鎖定時頭往前伸",
        subtitle: "手臂鎖定時頭還維持在前伸的位置。",
        why: "過頭推舉本身就會把頭往前帶，而頭部前傾會壓縮肩峰下空間、縮小肩關節活動範圍。",
        fixes: [
          "重量經過臉部時把下巴往後收。",
          "鎖定時讓耳朵對齊肩膀上方。",
          "組間放鬆胸口與上背。",
        ],
      },
      art("ohp_forward_head")
    ),
  ],

  "Push-up": [
    mistake(
      "pushup_hip_sag",
      "Trunk Sagging",
      {
        title: "Hips sagging",
        subtitle: "The hips drop toward the floor and the plank line breaks.",
        why: "Push-up trunk posture governs how much load reaches the lumbar spine, and a sagging trunk is the posture that raises it.",
        fixes: [
          "Squeeze the glutes and brace the abs before you lower.",
          "Hold one straight line from head to heels.",
          "Drop to your knees rather than let the hips sag.",
        ],
      },
      {
        title: "臀部下沉",
        subtitle: "髖部往地板掉，身體失去一直線。",
        why: "伏地挺身的軀幹姿勢直接決定腰椎承受的負荷，而臀部下沉正是把負荷推高的那個姿勢。",
        fixes: [
          "下降前先夾臀、收緊腹部。",
          "從頭到腳跟維持一直線。",
          "撐不住時改用跪姿，不要讓髖部塌下去。",
        ],
      },
      art("pushup_hip_sag")
    ),
    mistake(
      "pushup_shallow_depth",
      "Limited Range Of Motion",
      {
        title: "Not going deep enough",
        subtitle: "The elbows never reach deep flexion before you press back up.",
        why: "Force demand was highest around 90 degrees of elbow flexion and lowest near lockout, so a shallow rep forfeits most of the stimulus.",
        fixes: [
          "Lower until the elbows reach about 90 degrees.",
          "Put a target under the chest — a fist or a rolled towel.",
          "Elevate the hands so you can finish full reps.",
        ],
      },
      {
        title: "下降深度不足",
        subtitle: "手肘還沒彎到位就撐回去。",
        why: "手肘約 90 度彎曲時所需的力量最高、接近伸直時最低，做半程等於放掉大部分的訓練刺激。",
        fixes: [
          "下降到手肘約 90 度。",
          "在胸口下方放一個目標物，例如拳頭或捲起的毛巾。",
          "把手墊高，讓每一下都做得完整。",
        ],
      },
      art("pushup_shallow_depth")
    ),
    mistake(
      "pushup_head_drop",
      "Forward Head Posture",
      {
        title: "Head dropping forward",
        subtitle: "The chin leads the descent and the neck loses its neutral line.",
        why: "Correct form keeps head, spine and pelvis in one straight line; a forward head reduces the subacromial space at the shoulder.",
        fixes: [
          "Look at a spot on the floor about a hand ahead of you.",
          "Keep the chin tucked rather than reaching.",
          "Lower the chest, not the face.",
        ],
      },
      {
        title: "頭部前伸下沉",
        subtitle: "用下巴帶著往下，脖子失去中立位置。",
        why: "正確的姿勢要讓頭、脊椎與骨盆維持一直線；頭往前伸會壓縮肩膀的肩峰下空間。",
        fixes: [
          "視線看向前方大約一個手掌距離的地板。",
          "下巴微收，不要往前伸。",
          "下降的是胸口，不是臉。",
        ],
      },
      art("pushup_head_drop")
    ),
    mistake(
      "pushup_elbow_flare",
      "Elbow Valgus Torque",
      {
        title: "Elbows flaring out",
        subtitle: "The elbows swing out toward 90 degrees from the ribs, or the hands are set too wide.",
        why: "Hand position strongly changes elbow loading: valgus torque carried by the medial ligament rose significantly outside the normal hand position.",
        fixes: [
          "Keep the elbows at roughly 45 degrees to the ribs.",
          "Set the hands about shoulder width apart.",
          "Screw the hands into the floor to hold the position.",
        ],
      },
      {
        title: "手肘外開",
        subtitle: "手肘往外張到接近與身體垂直，或手掌放得太寬。",
        why: "手掌位置會大幅改變手肘負荷：偏離一般手掌位置時，內側韌帶承受的外翻力矩顯著上升。",
        fixes: [
          "手肘與身體維持約 45 度。",
          "雙手大約放在肩寬。",
          "把手掌像旋螺絲一樣抓穩地板，固定位置。",
        ],
      },
      art("pushup_elbow_flare")
    ),
  ],

  Lunge: [
    mistake(
      "lunge_insufficient_depth",
      "Decreased Knee Flexion",
      {
        title: "Not lunging deep enough",
        subtitle: "The front knee stops well above 90 degrees of flexion.",
        why: "Patellofemoral load rises progressively as knee flexion increases, so depth is what produces the training stimulus.",
        fixes: [
          "Lower until the front knee reaches about 90 degrees.",
          "Let the back knee travel down toward the floor.",
          "Take a longer step so the depth is available.",
        ],
      },
      {
        title: "弓步蹲得不夠深",
        subtitle: "前腳膝蓋還遠不到 90 度就停住。",
        why: "膝關節彎曲角度愈大，髕股關節的負荷愈高，深度就是訓練刺激的來源。",
        fixes: [
          "下降到前腳膝蓋約 90 度。",
          "讓後腳膝蓋往地板方向下沉。",
          "步幅跨大一點，讓深度做得出來。",
        ],
      },
      art("lunge_insufficient_depth")
    ),
    mistake(
      "lunge_knee_past_toes",
      "Knee Anterior To Toes",
      {
        title: "Front knee sliding past the toes",
        subtitle: "The lead knee travels well in front of the foot as you descend.",
        why: "The knee-past-toes variation produced 11% greater peak patellar tendon stress and a 26% greater knee extension moment.",
        fixes: [
          "Step further forward before you descend.",
          "Drop the back knee straight down.",
          "Keep the front shin close to vertical.",
        ],
      },
      {
        title: "前腳膝蓋超過腳尖",
        subtitle: "下降時前腳膝蓋明顯滑到腳掌前方。",
        why: "膝蓋超過腳尖的版本，髕骨肌腱尖峰應力高出約 11%、膝伸展力矩高出約 26%。",
        fixes: [
          "下降前先把步幅跨得更遠。",
          "後腳膝蓋垂直往下降。",
          "前腳小腿盡量保持接近垂直。",
        ],
      },
      art("lunge_knee_past_toes")
    ),
    mistake(
      "lunge_knee_valgus",
      "Knee Valgus",
      {
        title: "Front knee caving inward",
        subtitle: "The lead knee collapses toward the midline under load.",
        why: "Knee abduction moment predicted future ACL injury with 73% sensitivity and 78% specificity.",
        fixes: [
          "Track the knee over the second toe.",
          "Grip the floor with the front foot.",
          "Strengthen the hip abductors with band walks and side-lying raises.",
        ],
      },
      {
        title: "前腳膝蓋內夾",
        subtitle: "承重時前腳膝蓋往身體中線塌陷。",
        why: "膝外展力矩可以用 73% 敏感度、78% 特異度預測未來的前十字韌帶傷害。",
        fixes: [
          "讓膝蓋對準第二根腳趾的方向。",
          "前腳腳掌抓穩地板。",
          "以彈力帶側走、側躺抬腿強化髖外展肌群。",
        ],
      },
      art("lunge_knee_valgus")
    ),
    mistake(
      "lunge_pelvic_drop",
      "Trendelenburg Posture",
      {
        title: "Hip dropping on the free side",
        subtitle: "The pelvis tilts down on the unsupported side and the trunk leans to compensate.",
        why: "Failing to produce hip abduction force shows up as a Trendelenburg posture, with the opposite side of the pelvis dropping.",
        fixes: [
          "Keep the hips level and square through the rep.",
          "Brace the side of the trunk before you descend.",
          "Add glute medius work — side planks and band walks.",
        ],
      },
      {
        title: "非支撐側骨盆下掉",
        subtitle: "沒有承重的那一側骨盆往下掉，軀幹跟著側傾代償。",
        why: "髖外展力量不足時，就會出現對側骨盆下掉的特倫德倫堡姿勢。",
        fixes: [
          "整組動作維持骨盆左右等高、朝向正前方。",
          "下降前先撐住軀幹側邊。",
          "加入側棒式、彈力帶側走等臀中肌訓練。",
        ],
      },
      art("lunge_pelvic_drop")
    ),
  ],

  Deadlift: [
    mistake(
      "deadlift_lumbar_flexion",
      "Lumbar Flexion",
      {
        title: "Rounding the lower back",
        subtitle: "The lumbar spine flexes as the bar leaves the floor.",
        why: "The lift-off position already generates the greatest lumbar shear force, and it is the back extensors holding a neutral spine that keeps it down.",
        fixes: [
          "Set the back before the bar moves — chest up, lats tight.",
          "Take the slack out of the bar first.",
          "Reduce the load until you can hold a neutral spine.",
        ],
      },
      {
        title: "下背拱起",
        subtitle: "槓離地的瞬間腰椎就彎了。",
        why: "起始離地本來就是腰椎剪力最大的位置，靠的正是豎脊肌維持中立脊椎把剪力壓下來。",
        fixes: [
          "槓還沒動之前先固定背部：挺胸、夾緊背闊肌。",
          "先把槓的空隙拉緊再發力。",
          "把重量降到能維持中立脊椎為止。",
        ],
      },
      art("deadlift_lumbar_flexion")
    ),
    mistake(
      "deadlift_incomplete_lockout",
      "deadlift incomplete lockout hip and knee extension",
      {
        title: "Not finishing the lockout",
        subtitle: "The hips and knees stop short of full extension at the top.",
        why: "Full extension near 180 degrees is the measured lockout position; the lift counts as complete only standing fully upright.",
        fixes: [
          "Finish by squeezing the glutes and standing tall.",
          "Don't lean back past standing.",
          "Pause a beat at the top of every rep.",
        ],
      },
      {
        title: "頂端沒有站直鎖定",
        subtitle: "最高點時髖與膝還沒完全伸直。",
        why: "接近 180 度的完全伸直是實測的鎖定位置，站到完全直立才算完成一下。",
        fixes: [
          "最後靠夾臀站直收尾。",
          "站直即可，不要再往後仰。",
          "每一下在頂端停一拍。",
        ],
      },
      art("deadlift_incomplete_lockout")
    ),
    mistake(
      "deadlift_hips_shoot_up",
      "deadlift trunk position electromyographic activity lift-off mid-pull lockout",
      {
        title: "Hips shooting up first",
        subtitle: "The hips rise ahead of the shoulders and the pull turns into a stiff-leg lift.",
        why: "Leaning the trunk further forward raises spinal flexion torque, so the erector spinae has to work much harder to resist it.",
        fixes: [
          "Push the floor away with the legs as the bar breaks.",
          "Keep hips and shoulders rising together.",
          "Slow the first few centimetres of the pull.",
        ],
      },
      {
        title: "臀部先竄高",
        subtitle: "髖部比肩膀先抬起來，整下變成直腿硬舉。",
        why: "軀幹愈往前傾，脊椎屈曲力矩愈大，豎脊肌要出更多力才抵抗得住。",
        fixes: [
          "槓離地時用腿把地板推開。",
          "髖部與肩膀同步上升。",
          "把起始的前幾公分放慢。",
        ],
      },
      art("deadlift_hips_shoot_up")
    ),
  ],

  Row: [
    mistake(
      "row_torso_rising",
      "Trunk Extension",
      {
        title: "Torso rising out of the hinge",
        subtitle: "The chest lifts as you pull, so the hinge angle opens up.",
        why: "The free-weight bent-over row imposes a sustained trunk-extensor demand — exactly what a rising torso abandons.",
        fixes: [
          "Hold the hinge: hips back, chest over the weight.",
          "Pull with the arms, not with the back angle.",
          "Lighten the load until the torso stays still.",
        ],
      },
      {
        title: "軀幹在划船過程中抬起來",
        subtitle: "一拉就挺胸站起，髖鉸鏈的角度整個打開。",
        why: "自由重量的俯身划船需要軀幹伸肌持續穩定出力，而軀幹抬起等於直接放掉這個需求。",
        fixes: [
          "維持髖鉸鏈：臀部後推、胸口壓在重量上方。",
          "用手臂拉，不要用軀幹角度換取行程。",
          "把重量降到軀幹不會晃動為止。",
        ],
      },
      art("row_torso_rising")
    ),
    mistake(
      "row_incomplete_rom",
      "Scapular Protraction",
      {
        title: "Not completing the pull",
        subtitle: "The pull stops short of the body, so the shoulder blades never fully retract.",
        why: "Lat excitation was significantly highest in the upper half of the range — the part a short pull leaves out.",
        fixes: [
          "Pull until the handle reaches your abdomen.",
          "Finish by squeezing the shoulder blades together.",
          "Pause a beat at the top of each rep.",
        ],
      },
      {
        title: "沒有把重量拉到底",
        subtitle: "拉到一半就放，肩胛從來沒有完全後收。",
        why: "背闊肌的活化在行程的上半段顯著最高，而那正是拉不到底時被省略掉的一段。",
        fixes: [
          "拉到握把碰到腹部。",
          "結束時把兩側肩胛夾在一起。",
          "每一下在頂端停一拍。",
        ],
      },
      art("row_incomplete_rom")
    ),
    mistake(
      "row_momentum_jerk",
      "Loss Of Neutral Body Position",
      {
        title: "Using momentum",
        subtitle: "The body jerks to start the pull instead of moving the weight under control.",
        why: "Accelerating the load spikes concentric force demand and unloads the eccentric — the opposite of the controlled tension the row is for.",
        fixes: [
          "Take about two seconds up and two seconds down.",
          "Start each rep from a dead stop.",
          "Reduce the load until the tempo holds.",
        ],
      },
      {
        title: "用甩的帶動重量",
        subtitle: "靠身體一抖起拉，而不是控制著把重量拉起來。",
        why: "把重量加速拉起會拉高向心階段的力量需求、卻讓離心階段幾乎沒有負荷，正好和划船想要的持續張力相反。",
        fixes: [
          "上拉約兩秒、下放約兩秒。",
          "每一下都從完全靜止開始。",
          "把重量降到維持得住節奏為止。",
        ],
      },
      art("row_momentum_jerk")
    ),
    mistake(
      "row_asymmetric_pull",
      "Asymmetry",
      {
        title: "One side pulling harder",
        subtitle: "One arm reaches the top of the pull before the other.",
        why: "A one-sided pull loads the external oblique the way a deliberately unilateral row does, and breaks the coordinated scapular motion the exercise trains.",
        fixes: [
          "Row single-arm so each side works on its own.",
          "Film from behind and check both elbows finish level.",
          "Set the tempo from the weaker side.",
        ],
      },
      {
        title: "單邊拉得比較多",
        subtitle: "一隻手臂比另一隻先拉到頂。",
        why: "單邊搶拉會像刻意的單手划船一樣加重腹外斜肌負荷，也破壞了這個動作要訓練的雙側肩胛協調。",
        fixes: [
          "改做單手划船，讓兩邊各自完成。",
          "從後方錄影，確認兩邊手肘同高收尾。",
          "以比較弱的那一邊決定節奏。",
        ],
      },
      art("row_asymmetric_pull")
    ),
  ],

  "Band Pull Apart": [
    mistake(
      "bpa_shrugging",
      "Shoulder Shrugging",
      {
        title: "Shrugging the shoulders",
        subtitle: "The shoulders ride up toward the ears as the band opens.",
        why: "The drill is meant to target the middle and lower trapezius; upper-trap dominance inverts that pattern and can narrow the subacromial space.",
        fixes: [
          "Set the shoulders down before you pull.",
          "Keep the neck long throughout the set.",
          "Use a lighter band until the shrug disappears.",
        ],
      },
      {
        title: "聳肩",
        subtitle: "拉開彈力帶時肩膀往耳朵方向聳起來。",
        why: "這個動作要練的是中、下斜方肌；上斜方肌主導會讓徵召模式整個顛倒，也可能壓縮肩峰下空間。",
        fixes: [
          "拉之前先把肩膀往下沉。",
          "整組維持頸部拉長。",
          "換更輕的彈力帶，直到不再聳肩。",
        ],
      },
      art("bpa_shrugging")
    ),
    mistake(
      "bpa_incomplete_rom",
      "Bent Elbows",
      {
        title: "Not spreading the hands fully",
        subtitle: "The band stops short, so the arms never reach full spread.",
        why: "The larger the excursion worked against the band, the higher the trapezius activation — which a truncated pull gives up.",
        fixes: [
          "Pull until the band touches your chest.",
          "Keep the elbows straight the whole way out.",
          "Choose a band you can open all the way.",
        ],
      },
      {
        title: "沒有把雙手完全拉開",
        subtitle: "彈力帶拉到一半就停，手臂沒有展到底。",
        why: "對抗彈力帶的行程愈大，斜方肌活化愈高，拉不開就等於放掉這部分效果。",
        fixes: [
          "拉到彈力帶碰到胸口。",
          "全程保持手肘打直。",
          "選一條能完全拉開的彈力帶。",
        ],
      },
      art("bpa_incomplete_rom")
    ),
    mistake(
      "bpa_trunk_extension_compensation",
      "No Compensatory Trunk Movement",
      {
        title: "Leaning back to open the band",
        subtitle: "The trunk whips backward instead of the arms opening.",
        why: "The load is meant to come from horizontal abduction at the shoulder girdle; a backward lean diverts it off the target muscles.",
        fixes: [
          "Stand tall with the ribs pulled down.",
          "Brace the core and keep the trunk still.",
          "Drop to a lighter band.",
        ],
      },
      {
        title: "靠身體後仰把彈力帶拉開",
        subtitle: "軀幹往後甩，而不是靠手臂往外展開。",
        why: "這個動作的負荷應該來自肩帶的水平外展，身體後仰會把它導離目標肌群。",
        fixes: [
          "站直，肋骨往下收。",
          "撐住核心，軀幹保持不動。",
          "換更輕的彈力帶。",
        ],
      },
      art("bpa_trunk_extension_compensation")
    ),
  ],

  "Bicep Curl": [
    mistake(
      "curl_elbow_drift_forward",
      "Elbow Drift Forward",
      {
        title: "Elbows drifting forward",
        subtitle: "The elbows leave the ribs and swing forward as you curl.",
        why: "Correct execution keeps the elbows close to the torso throughout; letting them drift moves the work off the biceps.",
        fixes: [
          "Pin the elbows to your sides.",
          "Curl with the upper arm completely still.",
          "Try a preacher or incline curl to remove the drift.",
        ],
      },
      {
        title: "手肘往前跑",
        subtitle: "彎舉時手肘離開身體側邊、往前晃。",
        why: "正確的做法是整個過程手肘都貼在軀幹旁；手肘一跑掉，出力就從二頭肌轉移出去了。",
        fixes: [
          "把手肘固定在身體兩側。",
          "上臂完全不動，只彎前臂。",
          "改做斜板彎舉或牧師椅彎舉，直接消掉手肘位移。",
        ],
      },
      art("curl_elbow_drift_forward")
    ),
    mistake(
      "curl_trunk_swing_momentum",
      "Using Momentum",
      {
        title: "Swinging the body",
        subtitle: "The trunk rocks back and forth to help the weight up.",
        why: "Trunk movement is treated as a cheating deviation in controlled curl protocols and is deliberately monitored out of the lift.",
        fixes: [
          "Stand with your back against a wall.",
          "Lower for a slow count of three.",
          "Use a load you can curl without the swing.",
        ],
      },
      {
        title: "身體前後擺盪",
        subtitle: "靠軀幹前後晃動把重量帶上來。",
        why: "在受控的彎舉流程中，軀幹晃動被視為作弊的偏差動作，是要被排除掉的。",
        fixes: [
          "背靠牆做。",
          "下放時數三秒慢慢放。",
          "換成不用擺盪就舉得起來的重量。",
        ],
      },
      art("curl_trunk_swing_momentum")
    ),
    mistake(
      "curl_incomplete_rom",
      "Incomplete Range Of Motion",
      {
        title: "Half reps",
        subtitle: "The curl neither reaches the top nor returns to a straight arm.",
        why: "Full range from 0 to 140 degrees produced greater strength gains than partial range in trained lifters.",
        fixes: [
          "Straighten the arm fully at the bottom.",
          "Curl all the way to the top.",
          "Lower the weight rather than the range.",
        ],
      },
      {
        title: "只做半程",
        subtitle: "上沒有捲到頂，下也沒有回到手臂打直。",
        why: "在有訓練經驗的族群中，0 到 140 度的完整行程比部分行程帶來更大的肌力進步。",
        fixes: [
          "最低點手臂完全打直。",
          "最高點確實捲到底。",
          "要減的是重量，不是行程。",
        ],
      },
      art("curl_incomplete_rom")
    ),
  ],

  "Arm Abduction": [
    mistake(
      "arm_abd_contralateral_trunk_lean",
      "Trunk Lean Compensation",
      {
        title: "Leaning away from the raising arm",
        subtitle: "The trunk tips sideways to help the arm get up.",
        why: "Compensating instead of controlling the scapula during elevation is part of what drives shoulder impingement.",
        fixes: [
          "Stand tall and keep both hips level.",
          "Raise only as far as you can without leaning.",
          "Perform it with your back against a wall.",
        ],
      },
      {
        title: "軀幹往反方向側傾",
        subtitle: "靠身體往旁邊倒，把手臂帶上去。",
        why: "手臂上抬時用代償取代肩胛控制，正是造成肩夾擠的成因之一。",
        fixes: [
          "站直，骨盆兩側維持等高。",
          "只抬到不需要側傾的高度。",
          "背靠牆做這個動作。",
        ],
      }
    ),
    mistake(
      "arm_abd_lr_asymmetry",
      "Muscle Imbalance",
      {
        title: "One arm lagging behind",
        subtitle: "One arm reaches the top well before the other.",
        why: "Side-to-side asymmetry in the 10–15% range is associated with higher injury risk and reduced performance.",
        fixes: [
          "Raise both arms to the same height every rep.",
          "Work in front of a mirror, or film yourself.",
          "Set the pace from the slower arm.",
        ],
      },
      {
        title: "單邊落後",
        subtitle: "一隻手臂明顯比另一隻先抬到頂。",
        why: "左右差在 10–15% 這個區間，通常伴隨較高的受傷風險與較差的表現。",
        fixes: [
          "每一下都讓兩手抬到相同高度。",
          "對著鏡子做，或錄影下來檢查。",
          "以比較慢的那一手決定節奏。",
        ],
      }
    ),
  ],

  "Arm VW": [
    mistake(
      "vw_incomplete_excursion",
      "Insufficient Scapular Retraction",
      {
        title: "Too little travel between V and W",
        subtitle: "The arms move only a short distance between the V and the W position.",
        why: "Greater scapular excursion is what raises trapezius activation, so a small travel collects little of the effect.",
        fixes: [
          "Reach fully into the V before you pull down.",
          "Bring the elbows all the way down and back into the W.",
          "Slow the movement down so the range is deliberate.",
        ],
      },
      {
        title: "V 到 W 的行程太小",
        subtitle: "手臂在 V 與 W 之間只移動一小段距離。",
        why: "肩胛移動幅度愈大，斜方肌活化愈高，行程太小就收不到多少效果。",
        fixes: [
          "先把手臂完全伸展成 V，再往下收。",
          "手肘要確實往下、往後收成 W。",
          "放慢速度，讓行程是刻意做出來的。",
        ],
      }
    ),
    mistake(
      "vw_loss_of_elevation",
      "Insufficient Scapular Retraction",
      {
        title: "V position too low",
        subtitle: "The arms sit below the overhead band where the lower trapezius works best.",
        why: "Lower-trapezius activation peaks around 120–145 degrees of abduction; a low V never reaches that band.",
        fixes: [
          "Take the arms high overhead into the V, thumbs up.",
          "Keep the ribs down so the range comes from the shoulders.",
          "Check overhead mobility if you can't reach the position.",
        ],
      },
      {
        title: "V 的位置太低",
        subtitle: "手臂停在下斜方肌最有效的過頭區間以下。",
        why: "下斜方肌的活化在外展約 120–145 度時最高，V 太低就進不到這個區間。",
        fixes: [
          "手臂高舉過頭成 V，拇指朝上。",
          "肋骨往下收，讓行程來自肩膀而不是腰。",
          "做不到位置時，先檢查過頭活動度。",
        ],
      }
    ),
    mistake(
      "vw_lr_asymmetry",
      "Muscle Imbalance",
      {
        title: "One arm lagging behind",
        subtitle: "One arm reaches the V or the W ahead of the other.",
        why: "Side-to-side asymmetry in the 10–15% range is associated with higher injury risk and reduced performance.",
        fixes: [
          "Match both arms at the V and at the W.",
          "Film from the front and compare the two sides.",
          "Set the pace from the slower side.",
        ],
      },
      {
        title: "單邊落後",
        subtitle: "一隻手臂比另一隻先到 V 或先到 W。",
        why: "左右差在 10–15% 這個區間，通常伴隨較高的受傷風險與較差的表現。",
        fixes: [
          "V 和 W 兩個位置都讓雙手對齊。",
          "從正面錄影，比較兩側。",
          "以比較慢的那一側決定節奏。",
        ],
      }
    ),
  ],

  "Sit-up": [
    mistake(
      "situp_incomplete_rom",
      "Incomplete Forward Reach",
      {
        title: "Shoulder blades never leaving the mat",
        subtitle: "The curl-up stops before the scapulae clear the floor.",
        why: "The exercise is defined as lifting until the shoulder blades leave the surface, so stopping short never completes a repetition.",
        fixes: [
          "Curl up until the shoulder blades clear the mat.",
          "Roll up one vertebra at a time.",
          "Slow the return so the abs, not momentum, do the work.",
        ],
      },
      {
        title: "肩胛始終沒有離開地面",
        subtitle: "捲腹停在肩胛還沒離地的位置。",
        why: "這個動作的定義就是捲到肩胛離開接觸面，停在之前等於沒有完成一下。",
        fixes: [
          "捲到肩胛確實離開墊子。",
          "一節一節脊椎慢慢捲起來。",
          "放慢下放，讓腹肌而不是慣性在做功。",
        ],
      }
    ),
  ],

  "Shoulder Bridge": [
    mistake(
      "bridge_incomplete_hip_extension",
      "Incomplete Hip Extension",
      {
        title: "Hips not reaching the top",
        subtitle: "The hips stop short of the straight knee–hip–shoulder line.",
        why: "The repetition is defined as lifting until the hips reach a neutral position, in line with the knees and shoulders.",
        fixes: [
          "Lift until knees, hips and shoulders form one line.",
          "Squeeze the glutes at the top rather than arching the back.",
          "Pause a beat at the top of every rep.",
        ],
      },
      {
        title: "臀部沒有推到頂",
        subtitle: "髖部停在膝—髖—肩連成一直線之前。",
        why: "這個動作的定義是把髖部推到中立位置，與膝蓋、肩膀連成一直線。",
        fixes: [
          "推到膝蓋、髖部與肩膀成一直線。",
          "頂端用夾臀收尾，不要用腰去反弓。",
          "每一下在頂端停一拍。",
        ],
      },
      art("bridge_incomplete_hip_extension")
    ),
  ],

  "Leg Abduction": [
    mistake(
      "abd_pelvic_drop_trunk_lean",
      "Trunk Lean",
      {
        title: "Leaning the trunk to lift the leg",
        subtitle: "The trunk sways sideways as the leg goes out.",
        why: "Excessive lateral trunk lean mechanically offloads the stance limb and reduces the abductor demand the exercise exists for.",
        fixes: [
          "Keep the trunk upright and square.",
          "Hold a support lightly rather than leaning on it.",
          "Raise the leg only as high as you can without swaying.",
        ],
      },
      {
        title: "靠軀幹側傾把腿抬起來",
        subtitle: "腿往外抬時身體跟著往旁邊倒。",
        why: "軀幹過度側傾會在力學上卸掉支撐腳的負荷，也就削弱了這個動作原本要練的外展肌需求。",
        fixes: [
          "軀幹保持直立、正對前方。",
          "手只是輕扶支撐物，不要靠在上面。",
          "只抬到不需要側傾的高度。",
        ],
      },
      art("abd_pelvic_drop_trunk_lean")
    ),
  ],

  "Torso Twist": [
    mistake(
      "tt_trunk_not_braced",
      "Poor Abdominal Engagement",
      {
        title: "Losing the braced torso",
        subtitle: "The trunk angle collapses or drifts away from the position the set started in.",
        why: "Stabilising the joints during a twist matters far more to the lumbar spine than producing large amounts of axial torque.",
        fixes: [
          "Hold the trunk at a steady angle throughout.",
          "Brace the abs before the first rotation.",
          "Slow the twist and shorten the range until the brace holds.",
        ],
      },
      {
        title: "軀幹失去支撐",
        subtitle: "軀幹角度垮掉，或偏離這一組開始時的姿勢。",
        why: "對腰椎而言，旋轉過程中把關節穩住，遠比製造大量軸向扭力重要。",
        fixes: [
          "整組維持固定的軀幹角度。",
          "第一次旋轉前先把腹部撐好。",
          "放慢旋轉、縮小幅度，直到撐得住為止。",
        ],
      },
      art("tt_trunk_not_braced")
    ),
  ],
};

/** The authored mistakes for a movement, or an empty list.
 *
 *  Empty is a real answer, not a gap: `Jumping Jacks` and `High Knee` are in the catalog but have
 *  no registered detector — every rule of theirs is permanently silent or withdrawn — so there is
 *  nothing this app can tell a user to watch for, and the page says so rather than borrowing
 *  another movement's faults. */
export const movementMistakes = (movement: string): readonly Mistake[] =>
  MOVEMENT_MISTAKES[movement] ?? [];
