# Temporal Convolutional Networks, explained (video)

A 3Blue1Brown-style explainer video on TCNs, built with
[Manim Community](https://www.manim.community/) (the open-source descendant of the
library 3Blue1Brown uses). Narration appears as captions along the bottom of the
frame, and as an ElevenLabs voice-over once the clips exist (see
[Narration](#narration-elevenlabs)). Without the clips, the video renders silent.

Output: `tcn-explained.mp4` (1920×1080, 30 fps, 11:16, about 33 MB). Git doesn't track
the MP4 (see `.gitignore`); run `./render.sh` to build it.

## Sources

- S. Bai, J. Z. Kolter, V. Koltun. *An Empirical Evaluation of Generic Convolutional
  and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271, 2018.
- Official code: [locuslab/TCN](https://github.com/locuslab/TCN). The code on screen
  comes from `TCN/tcn.py` (`Chomp1d`, `TemporalBlock`, `TemporalConvNet`); the
  `TemporalBlock` listing is abridged where marked. Each classification head uses
  `self.linear(y1[:, :, -1])`, from `TCN/mnist_pixel/model.py`. The receptive-field
  table uses the default `--ksize` and `--levels` from `add_test.py`, `pmnist_test.py`
  and `copymem_test.py`.

## Scenes

| Scene | Content |
|---|---|
| `S01_Intro` | Sequences, RNNs and their sequential bottleneck, the paper |
| `S02_Task` | Sequence modelling setup and the causality constraint |
| `S03_Causal` | Centred vs. causal 1-D convolution, left zero-padding, linear receptive-field growth |
| `S04_Dilated` | Dilated convolution `F(s) = Σ f(i)·x_{s−d·i}`, Figure 1(a) (k = 3, d = 1, 2, 4), exponential growth |
| `S05_Chomp` | How the code gets causality: symmetric `Conv1d` padding plus `Chomp1d` |
| `S06_Residual` | Residual block (Figure 1(b)): conv → WeightNorm → ReLU → Dropout ×2, optional 1×1 conv, `TemporalBlock` code |
| `S07_FullTCN` | `TemporalConvNet`, `R = 1 + 2(k−1)(2^L−1)`, repo defaults vs. sequence lengths, last-step linear head |
| `S08_Compare` | Parallelism and gradient-path length vs. RNNs; the paper's advantages and disadvantages |
| `S09_Results` | Table 1 (11 tasks, LSTM/GRU/RNN/TCN), copy-memory guessing baseline, (P-)MNIST bars |
| `S10_Outro` | Summary, the paper's conclusion, references |

## Rendering

Requirements: Manim Community 0.22, a LaTeX distribution with `dvisvgm`
(on Debian/Ubuntu: `texlive-latex-extra`, `texlive-fonts-recommended`,
`texlive-science`, `dvisvgm`), the CMU fonts (`fonts-cmu`), and `ffmpeg`. See
Manim's [installation guide](https://docs.manim.community/en/stable/installation.html).
Install Manim in its own virtual environment; don't add it to the research `.venv`.

```bash
./render.sh            # final cut -> tcn-explained.mp4 (about 6 min on 4 cores)
./render.sh preview    # 480p15 draft -> tcn-explained-preview.mp4
MANIM=/path/to/venv/bin/manim JOBS=8 ./render.sh
```

To iterate on a single scene:

```bash
manim render -ql tcn_video.py S04_Dilated
```

Caption timing lives in `TCNScene.say()`. A line with a voice clip stays up for
the clip's length plus a 0.5 s pause; without a clip, for `words / 2.5` seconds.
Either way it also stays up while the animations under it run, so lines never
overlap.

## Narration (ElevenLabs)

Each `self.say(caption, spoken=...)` call is one narration line. The voice reads
`spoken` when it's given and the caption otherwise. `spoken` exists so that a
caption like `(k − 1)·d` is read aloud as "k minus one, times d".

`tts.py` turns these lines into MP3 clips in `narration/clips/` and lists them in
`narration/manifest.json`. It uses only the Python standard library, so any Python 3.9+
works; on Windows, from the repo root, run
`.venv\Scripts\python.exe docs\explainers\tcn-video\tts.py`. It reads
`ELEVENLABS_API_KEY` from the environment. In a Claude Code cloud environment, a
network secret for `api.elevenlabs.io` (header `xi-api-key`, no prefix) works
instead and keeps the key out of the session. The voice defaults to
`onwK4e9ZLuTAKqWW03F9` ("Daniel - Steady Broadcaster") and the model to
`eleven_multilingual_v2`. On the free plan, the API only accepts ElevenLabs'
premade voices. Voice Library voices need a paid plan.

```bash
python tts.py                        # dry run: lines still to generate, and their cost
python tts.py --generate --limit 1   # one line; listen to it before paying for the rest
python tts.py --generate             # everything still missing
./render.sh                          # picks the clips up automatically
```

- **Cost.** The full script is about 8,500 characters. That is about 8,500 credits
  with `eleven_multilingual_v2`, or about half that with `--model eleven_flash_v2_5`.
  `tts.py` checks your remaining credits first and won't start a run it can't
  finish; `--max-credits N` sets your own ceiling.
- **Caching.** Clips are keyed by voice, model and text. A re-run only pays for lines
  you added or edited. Commit `narration/`: regenerating it costs credits.
- `python tts.py --prune` deletes clips the script no longer uses.
- `TCN_NARRATION=0 ./render.sh` renders without the voice.
- **Attribution.** When clips are present, the references card credits ElevenLabs,
  as their free plan requires. The free plan also excludes commercial use; check
  their current terms before publishing.
