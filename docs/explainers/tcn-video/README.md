# Temporal Convolutional Networks, explained (video)

A 3Blue1Brown-style explainer video on TCNs, built with
[Manim Community](https://www.manim.community/) (the open-source descendant of the
library 3Blue1Brown uses). Narration appears as on-screen captions along the bottom
of the frame, so the video plays without a voice track.

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

Caption timing lives in `TCNScene.say()`. Each caption stays up for at least
`words / 2.5` seconds, or for as long as the animations under it take, whichever
is longer.
