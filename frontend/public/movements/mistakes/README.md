# Common-mistake illustrations

The wrong / correct pair shown on `/movements/<name>?tab=mistakes`, one pair per fault the
movement's rule detector can report. The page holds both slots open already: until a pair lands
they render as a tinted panel with its ✗ / ✓ badge and a "Common mistake" / "Correct form" label,
so dropping the files in changes what is inside the slot and reflows nothing around it.

## Adding a pair

1. Export the two images here under the exact names listed below. WebP; the panel is 152px tall
   and roughly square, and the image is `object-contain`, so anything portrait-ish works — match
   the framing across a movement's set rather than across the whole directory.

   Art that arrives as ONE two-panel sheet (wrong left, correct right) does not get cut by hand:
   drop the PNG here, add its `<sheet>.png -> <fault-slug>` line to `SHEETS` in
   `scripts/prep_mistake_art.py`, and run

   ```
   .venv\Scripts\python.exe scripts/prep_mistake_art.py [--preview]
   ```

   It finds the two drawn figures and splits between them, knocks the background and the ground
   shadow out to transparency (the panel behind is tinted, so an opaque plate renders as a
   rectangle), puts back any piece of the drawing that knock-out cut adrift — the white socks are
   the page's exact tone, so without this the shoes float free of the legs — fits both halves into
   one box, shared rows, so a shared floor line stays level and `object-contain` cannot render the
   same body at two sizes, and refuses to write a pair whose red and green annotations say the
   halves are swapped. The source PNG is gitignored; only the WebPs are committed.

   Thirty-four sheets in, it has needed generalising four times, and every time the sheets
   disagreed about something invisible until it broke: the greens differ by a factor of two in
   saturation (a threshold tuned on squat finds ZERO pixels of the push-up green, which reads
   downstream as "this half has no green, so the pair is swapped"); whether the socks survive the
   knock-out is an accident of whether that drawing encloses them; a bicep curl panel is a figure
   PLUS two magnified insets, so a panel is not one mass; and band pull apart draws the band itself
   in RED, so a correct half carries a thousand red pixels of equipment and the orientation has to
   be decided on green alone. Expect the next one to disagree about something else again, and fix
   it in the script rather than by hand — the whole point of the file is that the remaining pairs
   cost one line each.
2. In `frontend/src/lib/movementMistakes.ts`, pass `art("<fault_id>")` as the last argument of that
   fault's `mistake(...)` call. `art()` hyphenates the id for you, which is why the files below are
   hyphenated and the ids in the TS file are not.
3. `yarn vitest run src/test/lib.movementMistakes.test.ts` — it fails if a declared pair points at
   a file that is not here. Then look at the pair on the page, or with `--preview`: nothing in the
   test suite can tell that a correctly-shaped, correctly-coloured pair landed on the wrong card.

Both halves must be genuinely different drawings. The one thing the page refuses to do is show a
single picture captioned both ways, which is why nothing renders by default.

## The 80 filenames, grouped by movement

Order matches each detector module's rule order, which is the order the cards are numbered in.

```
## Squat
knees-inward-wrong.webp
knees-inward-correct.webp
knees-forward-wrong.webp
knees-forward-correct.webp
shallow-depth-wrong.webp
shallow-depth-correct.webp
excessive-forward-lean-wrong.webp
excessive-forward-lean-correct.webp
heel-rise-wrong.webp
heel-rise-correct.webp

## Overhead Press
ohp-incomplete-lockout-wrong.webp
ohp-incomplete-lockout-correct.webp
ohp-lumbar-hyperextension-wrong.webp
ohp-lumbar-hyperextension-correct.webp
ohp-asymmetric-press-wrong.webp
ohp-asymmetric-press-correct.webp
ohp-insufficient-elevation-wrong.webp
ohp-insufficient-elevation-correct.webp
ohp-forward-head-wrong.webp
ohp-forward-head-correct.webp

## Push-up
pushup-hip-sag-wrong.webp
pushup-hip-sag-correct.webp
pushup-shallow-depth-wrong.webp
pushup-shallow-depth-correct.webp
pushup-head-drop-wrong.webp
pushup-head-drop-correct.webp
pushup-elbow-flare-wrong.webp
pushup-elbow-flare-correct.webp

## Lunge
lunge-insufficient-depth-wrong.webp
lunge-insufficient-depth-correct.webp
lunge-knee-past-toes-wrong.webp
lunge-knee-past-toes-correct.webp
lunge-knee-valgus-wrong.webp
lunge-knee-valgus-correct.webp
lunge-pelvic-drop-wrong.webp
lunge-pelvic-drop-correct.webp

## Deadlift
deadlift-lumbar-flexion-wrong.webp
deadlift-lumbar-flexion-correct.webp
deadlift-incomplete-lockout-wrong.webp
deadlift-incomplete-lockout-correct.webp
deadlift-hips-shoot-up-wrong.webp
deadlift-hips-shoot-up-correct.webp

## Row
row-torso-rising-wrong.webp
row-torso-rising-correct.webp
row-incomplete-rom-wrong.webp
row-incomplete-rom-correct.webp
row-momentum-jerk-wrong.webp
row-momentum-jerk-correct.webp
row-asymmetric-pull-wrong.webp
row-asymmetric-pull-correct.webp

## Band Pull Apart
bpa-shrugging-wrong.webp
bpa-shrugging-correct.webp
bpa-incomplete-rom-wrong.webp
bpa-incomplete-rom-correct.webp
bpa-trunk-extension-compensation-wrong.webp
bpa-trunk-extension-compensation-correct.webp

## Bicep Curl
curl-elbow-drift-forward-wrong.webp
curl-elbow-drift-forward-correct.webp
curl-trunk-swing-momentum-wrong.webp
curl-trunk-swing-momentum-correct.webp
curl-incomplete-rom-wrong.webp
curl-incomplete-rom-correct.webp

## Arm Abduction
arm-abd-contralateral-trunk-lean-wrong.webp
arm-abd-contralateral-trunk-lean-correct.webp
arm-abd-lr-asymmetry-wrong.webp
arm-abd-lr-asymmetry-correct.webp

## Arm VW
vw-incomplete-excursion-wrong.webp
vw-incomplete-excursion-correct.webp
vw-loss-of-elevation-wrong.webp
vw-loss-of-elevation-correct.webp
vw-lr-asymmetry-wrong.webp
vw-lr-asymmetry-correct.webp

## Sit-up
situp-incomplete-rom-wrong.webp
situp-incomplete-rom-correct.webp

## Shoulder Bridge
bridge-incomplete-hip-extension-wrong.webp
bridge-incomplete-hip-extension-correct.webp

## Leg Abduction
abd-pelvic-drop-trunk-lean-wrong.webp
abd-pelvic-drop-trunk-lean-correct.webp

## Torso Twist
tt-trunk-not-braced-wrong.webp
tt-trunk-not-braced-correct.webp

## 80 files total
```
