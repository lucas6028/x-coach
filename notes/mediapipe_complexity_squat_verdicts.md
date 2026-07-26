# MediaPipe complexity vs squat rule-detector verdicts

- clips: 40
- Lite==Heavy verdict agreement: 50.0% (20/40)

| clip | lite | full | heavy | lite==heavy |
|---|---|---|---|---|
| 32903_8.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 32907_3.mp4 | excessive_forward_lean,knees_forward,shallow_depth | knees_forward,shallow_depth | excessive_forward_lean,knees_forward,knees_inward,shallow_depth | NO |
| 32939_2.mp4 | excessive_forward_lean,knees_forward,shallow_depth | excessive_forward_lean,knees_forward | excessive_forward_lean,knees_forward,shallow_depth | yes |
| 32964_1.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward | NO |
| 32971_2.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 32975_6.mp4 | excessive_forward_lean,knees_forward,shallow_depth | excessive_forward_lean,knees_forward,shallow_depth | knees_forward | NO |
| 32977_1.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 32979_1.mp4 | knees_forward | knees_forward,shallow_depth | knees_forward | yes |
| 32987_5.mp4 | knees_forward,knees_inward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | NO |
| 32990_6.mp4 | knees_forward,shallow_depth | knees_forward | knees_forward,shallow_depth | yes |
| 32991_2.mp4 | — | — | — | yes |
| 32995_3.mp4 | shallow_depth | knees_forward,shallow_depth | shallow_depth | yes |
| 32996_2.mp4 | heel_rise,shallow_depth | — | — | NO |
| 33006_2.mp4 | knees_forward,knees_inward | knees_forward,knees_inward,shallow_depth | knees_forward,knees_inward,shallow_depth | NO |
| 33010_3.mp4 | excessive_forward_lean,knees_forward | excessive_forward_lean,knees_forward,shallow_depth | knees_forward | NO |
| 33018_1.mp4 | excessive_forward_lean,knees_forward | excessive_forward_lean,knees_inward,shallow_depth | — | NO |
| 33024_2.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 33025_2.mp4 | knees_inward,shallow_depth | knees_inward | knees_inward,shallow_depth | yes |
| 33028_2.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 33029_1.mp4 | knees_forward,shallow_depth | shallow_depth | knees_forward,shallow_depth | yes |
| 33031_2.mp4 | knees_forward | — | — | NO |
| 33035_4.mp4 | — | — | — | yes |
| 33036_2.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward | NO |
| 33044_11.mp4 | knees_forward,knees_inward,shallow_depth | knees_forward,shallow_depth | — | NO |
| 33045_1.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | — | NO |
| 33048_1.mp4 | knees_forward,knees_inward,shallow_depth | knees_forward,knees_inward,shallow_depth | knees_forward,knees_inward,shallow_depth | yes |
| 33051_1.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 33052_1.mp4 | shallow_depth | — | — | NO |
| 33053_1.mp4 | knees_forward,shallow_depth | excessive_forward_lean,knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 33058_3.mp4 | — | — | — | yes |
| 33063_6.mp4 | excessive_forward_lean,knees_forward,shallow_depth | excessive_forward_lean,shallow_depth | — | NO |
| 33071_1.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 33072_1.mp4 | knees_forward,shallow_depth | knees_forward | excessive_forward_lean,knees_forward,knees_inward,shallow_depth | NO |
| 33073_2.mp4 | excessive_forward_lean,heel_rise,knees_forward,knees_inward,shallow_depth | knees_forward,knees_inward,shallow_depth | knees_forward,knees_inward | NO |
| 33082_1.mp4 | knees_forward | knees_forward,shallow_depth | knees_forward,shallow_depth | NO |
| 33088_1.mp4 | knees_forward,shallow_depth | knees_forward,shallow_depth | knees_forward,shallow_depth | yes |
| 33090_7.mp4 | — | knees_forward | — | yes |
| 33095_3.mp4 | excessive_forward_lean,knees_inward,shallow_depth | excessive_forward_lean,knees_forward,shallow_depth | excessive_forward_lean,knees_forward,shallow_depth | NO |
| 33096_3.mp4 | knees_forward | knees_forward,shallow_depth | excessive_forward_lean,knees_forward,knees_inward | NO |
| 33103_2.mp4 | knees_forward,shallow_depth | — | — | NO |
