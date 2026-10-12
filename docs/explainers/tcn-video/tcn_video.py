"""Temporal Convolutional Networks, explained: a 3Blue1Brown-style Manim video.

Sources:
  * S. Bai, J. Z. Kolter, V. Koltun. "An Empirical Evaluation of Generic
    Convolutional and Recurrent Networks for Sequence Modeling".
    arXiv:1803.01271, 2018.
  * Official code: https://github.com/locuslab/TCN (TCN/tcn.py and the
    per-task *_test.py / model.py files).

Each scene is self-contained. render.sh renders them in order and
concatenates the clips into one MP4; see README.md.
"""

from __future__ import annotations

import textwrap

import numpy as np
from manim import *

config.background_color = "#0c0c0c"

SERIF = "CMU Serif"
MONO = "DejaVu Sans Mono"

C_IN = BLUE_C
C_HID = TEAL_C
C_OUT = YELLOW_C
C_FUT = RED_C
C_PAD = GREY_C
C_EDGE = GREY_B
C_KER = PURPLE_B
C_RES = GREEN_C

CAPTION_CHARS = 84
CAPTION_MAX_WIDTH = 13.4
CAPTION_LINE_PITCH = 0.38
WORDS_PER_SECOND = 2.5


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def txt(s: str, size: float = 32, color=WHITE, **kw) -> Text:
    return Text(s, font=SERIF, font_size=size, color=color, **kw)


def make_caption(s: str) -> VGroup:
    # One Text per line. Manim's Paragraph maps glyphs back to lines by
    # counting characters, and an apostrophe renders no glyph, which shifts
    # letters onto the wrong line.
    lines = textwrap.wrap(s, CAPTION_CHARS)
    if len(lines) > 2:
        logger.warning("caption wraps to %d lines: %s", len(lines), s)
    cap = VGroup()
    for i, line in enumerate(lines):
        t = Text(line, font=SERIF, font_size=24, color=GREY_A)
        t.shift((-i * CAPTION_LINE_PITCH - t.get_top()[1]) * UP)  # align line tops
        cap.add(t)
    if cap.width > CAPTION_MAX_WIDTH:
        cap.scale_to_fit_width(CAPTION_MAX_WIDTH)
    cap.to_edge(DOWN, buff=0.3)
    return cap


def code_block(src: str, width: float | None = None) -> Code:
    code = Code(
        code_string=src,
        language="python",
        formatter_style="monokai",
        add_line_numbers=False,
        background="rectangle",
        background_config={"fill_color": "#1a1a1a", "stroke_color": GREY_D, "stroke_width": 1},
        paragraph_config={"font": MONO},
    )
    if width is not None:
        code.scale_to_fit_width(width)
    return code


def line_box(code: Code, first: int, last: int, color=YELLOW_C) -> SurroundingRectangle:
    lines = VGroup(*[code.code_lines[i] for i in range(first, last + 1) if len(code.code_lines[i])])
    return SurroundingRectangle(lines, color=color, buff=0.06, corner_radius=0.05, stroke_width=2.5)


def node(x: float, y: float, color=C_IN, radius: float = 0.13, fill: float = 0.0, width: float = 2.5) -> Circle:
    c = Circle(radius=radius, color=color, stroke_width=width).set_fill(color, fill)
    return c.move_to([x, y, 0])


def link(a: Mobject, b: Mobject, color=C_EDGE, width: float = 2.0, opacity: float = 1.0) -> Line:
    return Line(
        a.get_center(), b.get_center(), buff=a.width / 2, color=color,
        stroke_width=width, stroke_opacity=opacity,
    )


def window(mobs, color=C_KER) -> SurroundingRectangle:
    return SurroundingRectangle(VGroup(*mobs), color=color, buff=0.12, corner_radius=0.12, stroke_width=3)


def taps(srcs, dst, color=C_KER, width: float = 3.0) -> VGroup:
    return VGroup(*[Line(s.get_top(), dst.get_bottom(), color=color, stroke_width=width) for s in srcs])


def backprop_arcs(cells, width: float = 5.0) -> VGroup:
    """Red arcs hopping right-to-left over a chain of RNN cells, fading as they go."""
    hops = list(reversed(range(len(cells) - 1)))
    arcs = VGroup()
    for j, i in enumerate(hops):
        fade = 1 - 0.8 * j / max(1, len(hops) - 1)
        arc = CurvedArrow(
            cells[i + 1].get_top() + 0.06 * UP, cells[i].get_top() + 0.06 * UP,
            angle=PI / 2, color=C_FUT, stroke_width=1 + width * fade, tip_length=0.16,
        )
        arcs.add(arc.set_opacity(fade))
    return arcs


def block_box(label: str, color, width: float = 2.5, height: float = 0.36, size: float = 22) -> VGroup:
    rect = Rectangle(width=width, height=height, color=color, stroke_width=2).set_fill(color, 0.35)
    t = txt(label, size)
    if t.width > width - 0.2:
        t.scale_to_fit_width(width - 0.2)
    t.move_to(rect)
    return VGroup(rect, t)


class TCNScene(Scene):
    """Scene with timed narration captions along the bottom edge.

    say() swaps the caption and records how long it needs to stay readable;
    the next say() (or finish()) waits out whatever reading time the
    animations in between did not already use.
    """

    def setup(self):
        self.cap: Mobject | None = None
        self.cap_until = 0.0

    def say(self, text: str, extra: float = 0.0):
        self.hold()
        new = make_caption(text)
        if self.cap is None:
            self.play(FadeIn(new, shift=0.1 * UP), run_time=0.5)
        else:
            self.play(FadeOut(self.cap), FadeIn(new, shift=0.1 * UP), run_time=0.5)
        self.cap = new
        self.cap_until = self.time + max(2.4, len(text.split()) / WORDS_PER_SECOND) + extra

    def hold(self, extra: float = 0.0):
        remaining = self.cap_until - self.time + extra
        if remaining > 1 / config.frame_rate:
            self.wait(remaining)

    def clear_stage(self, run_time: float = 0.8, keep=(), hold: bool = True):
        # By default the visuals stay up until the current caption has been read;
        # pass hold=False when the caption introduces what comes next.
        if hold:
            self.hold()
        mobs = [m for m in self.mobjects if m is not self.cap and m not in keep]
        if mobs:
            self.play(*[FadeOut(m) for m in mobs], run_time=run_time)

    def finish(self):
        self.hold(0.5)
        if self.mobjects:
            self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
        self.wait(0.4)


# --------------------------------------------------------------------------
# 1. intro
# --------------------------------------------------------------------------


class S01_Intro(TCNScene):
    def construct(self):
        title = txt("Temporal Convolutional Networks", 60)
        sub = txt("convolutions that respect the arrow of time", 32, GREY_B)
        VGroup(title, sub).arrange(DOWN, buff=0.45)
        self.play(Write(title), run_time=2)
        self.play(FadeIn(sub, shift=0.2 * UP))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(sub))

        # Sequences are everywhere.
        def sig(x):
            return 0.9 * np.sin(1.1 * x) + 0.4 * np.sin(2.9 * x + 1.0)

        base = -1.5
        curve = FunctionGraph(lambda x: sig(x) + 0.3, x_range=[-6, 6], color=C_IN, stroke_width=3)
        ts = np.linspace(-6, 6, 33)
        dots = VGroup(*[Dot([t, sig(t) + 0.3, 0], radius=0.055, color=C_IN) for t in ts])
        stems = VGroup(
            *[Line([t, base, 0], [t, sig(t) + 0.3, 0], stroke_width=1.5, color=GREY_D) for t in ts]
        )
        time_arrow = Arrow([-6.3, base - 0.25, 0], [6.5, base - 0.25, 0], buff=0, stroke_width=3, color=GREY_B)
        time_label = txt("time", 26, GREY_B).next_to(time_arrow.get_end(), DOWN, buff=0.12).shift(0.3 * LEFT)

        self.say(
            "Sequences are everywhere: audio, text, sensor readings, "
            "the joint angles of someone doing a squat."
        )
        self.play(GrowArrow(time_arrow), FadeIn(time_label), Create(curve), run_time=2)
        self.play(
            LaggedStart(
                *[AnimationGroup(Create(s), FadeIn(d, scale=0.5)) for s, d in zip(stems, dots)],
                lag_ratio=0.04,
            ),
            curve.animate.set_stroke(opacity=0.35),
            run_time=2,
        )
        label = txt("audio", 40, C_IN).move_to(2.6 * UP)
        self.play(FadeIn(label, shift=0.2 * DOWN))
        for kind in ["text", "sensor readings", "joint angles"]:
            new = txt(kind, 40, C_IN).move_to(label)
            self.play(FadeOut(label, shift=0.2 * UP), FadeIn(new, shift=0.2 * UP), run_time=0.6)
            label = new
            self.wait(0.4)

        self.say("For years, the default tool for modelling them was the recurrent neural network, or RNN.")
        self.play(FadeOut(VGroup(curve, dots, stems, label, time_arrow, time_label)))

        n = 8
        xs = np.linspace(-5.25, 5.25, n)
        ins = VGroup(*[node(x, -1.5, C_IN, 0.2, 0.15, 3) for x in xs])
        cells = VGroup(
            *[
                RoundedRectangle(corner_radius=0.12, width=0.95, height=0.7, color=GREY_B, stroke_width=3)
                .move_to([x, 0.2, 0])
                for x in xs
            ]
        )
        hs = VGroup(*[MathTex(f"h_{{{i + 1}}}", font_size=32).move_to(c) for i, c in enumerate(cells)])
        xl = VGroup(
            *[MathTex(f"x_{{{i + 1}}}", font_size=30, color=C_IN).next_to(c, DOWN, buff=0.15) for i, c in enumerate(ins)]
        )
        ups = VGroup(
            *[
                Arrow(a.get_top(), b.get_bottom(), buff=0.06, stroke_width=3, color=GREY_B,
                      max_tip_length_to_length_ratio=0.3)
                for a, b in zip(ins, cells)
            ]
        )
        sides = VGroup(
            *[
                Arrow(cells[i].get_right(), cells[i + 1].get_left(), buff=0.04, stroke_width=3, color=GREY_B,
                      max_tip_length_to_length_ratio=0.35)
                for i in range(n - 1)
            ]
        )
        eq = MathTex(r"h_t = f(h_{t-1},\ x_t)", font_size=48).move_to(2.3 * UP)

        self.play(LaggedStart(*[GrowFromCenter(c) for c in ins], lag_ratio=0.06), FadeIn(xl))
        self.play(
            LaggedStart(
                *[AnimationGroup(GrowArrow(u), Create(c), FadeIn(h)) for u, c, h in zip(ups, cells, hs)],
                lag_ratio=0.08,
            ),
            run_time=1.6,
        )
        self.play(LaggedStart(*[GrowArrow(s) for s in sides], lag_ratio=0.08), Write(eq))

        self.say("It reads the sequence one step at a time, carrying a hidden state h forward from each step to the next.")
        for i in range(n):
            anims = [cells[i].animate.set_fill(C_OUT, 0.3).set_stroke(C_OUT), ups[i].animate.set_color(C_OUT)]
            if i > 0:
                anims.append(sides[i - 1].animate.set_color(C_OUT))
            self.play(*anims, run_time=0.4)

        self.say(
            "Step t cannot start until step t − 1 has finished. Long sequences mean long chains: "
            "slow to train, and gradients that fade or explode on their way back."
        )
        arcs = backprop_arcs(cells)
        grad_lab = txt("gradient", 28, C_FUT).next_to(arcs[0], UP, buff=0.12)
        self.play(FadeIn(grad_lab), LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.6), run_time=4)

        self.clear_stage()
        self.say("In 2018, Shaojie Bai, J. Zico Kolter and Vladlen Koltun ran a careful head-to-head comparison.")
        card = VGroup(
            txt("An Empirical Evaluation of Generic Convolutional", 36),
            txt("and Recurrent Networks for Sequence Modeling", 36),
            txt("Shaojie Bai   ·   J. Zico Kolter   ·   Vladlen Koltun", 26, GREY_B),
            txt("arXiv:1803.01271  (2018)", 24, GREY_C),
        ).arrange(DOWN, buff=0.22)
        card[2].shift(0.15 * DOWN)
        card[3].shift(0.15 * DOWN)
        card.move_to(1.0 * UP)
        frame = SurroundingRectangle(card, buff=0.45, color=GREY_BROWN, corner_radius=0.1)
        self.play(Create(frame), FadeIn(card, shift=0.2 * UP), run_time=1.5)
        self.say("Generic convolutional networks against LSTMs and GRUs, on the recurrent networks' home turf.")

        self.say(
            "Their verdict: a simple convolutional architecture, the Temporal Convolutional Network or TCN, "
            "outperformed the recurrent baselines across a broad range of tasks."
        )
        tcn = txt("TCN", 72, C_OUT).next_to(frame, DOWN, buff=0.45)
        self.play(Write(tcn))
        self.say("Let's build a TCN from the ground up, and then read the official code.")
        self.finish()


# --------------------------------------------------------------------------
# 2. the sequence modelling task and causality
# --------------------------------------------------------------------------


class S02_Task(TCNScene):
    def construct(self):
        n, sp, r = 10, 1.1, 0.18
        xs = [(i - (n - 1) / 2) * sp for i in range(n)]
        ins = VGroup(*[node(x, -1.4, C_IN, r, 0.15, 3) for x in xs])
        outs = VGroup(*[node(x, 1.4, C_OUT, r, 0.0, 3) for x in xs])
        xl = VGroup(*[MathTex(f"x_{{{i}}}", font_size=32, color=C_IN).next_to(c, DOWN, buff=0.16) for i, c in enumerate(ins)])
        yl = VGroup(
            *[MathTex(rf"\hat y_{{{i}}}", font_size=32, color=C_OUT).next_to(c, UP, buff=0.16) for i, c in enumerate(outs)]
        )
        box = Rectangle(width=ins.width + 0.8, height=1.5, color=GREY_B).set_fill(GREY_E, 0.7)
        box_label = txt("sequence model  f", 34)
        eq = MathTex(
            r"\hat y_0,\ \hat y_1,\ \dots,\ \hat y_T", r"\;=\;", r"f(x_0,\ x_1,\ \dots,\ x_T)", font_size=46
        ).to_edge(UP, buff=0.45)
        eq[0].set_color(C_OUT)

        self.say("Here's the problem. We're given an input sequence: x0, x1, and so on up to xT.")
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in ins], lag_ratio=0.08),
            LaggedStart(*[FadeIn(l) for l in xl], lag_ratio=0.08),
        )
        self.say("We want a prediction at every time step, so the output sequence is as long as the input.")
        self.play(FadeIn(box), Write(box_label))
        self.play(
            LaggedStart(*[GrowFromCenter(c) for c in outs], lag_ratio=0.08),
            LaggedStart(*[FadeIn(l) for l in yl], lag_ratio=0.08),
        )
        self.play(Write(eq))

        self.say(
            "There is one rule. The prediction at time t may only use the inputs up to time t. "
            "No peeking into the future."
        )
        rule = MathTex(r"\hat y_t", r"\text{ depends only on }", r"x_0,\ \dots,\ x_t", font_size=46).move_to(eq)
        rule[0].set_color(C_OUT)
        rule[2].set_color(C_IN)
        self.play(FadeOut(box), FadeOut(box_label), ReplacementTransform(eq, rule))

        tr = ValueTracker(5)

        def k_now() -> int:
            return int(round(tr.get_value()))

        def cone():
            k = k_now()
            g = VGroup()
            for i in range(n):
                if i <= k:
                    g.add(Line(ins[i].get_top(), outs[k].get_bottom(), color=C_IN, stroke_width=2.5, stroke_opacity=0.85))
                else:
                    g.add(
                        DashedLine(ins[i].get_top(), outs[k].get_bottom(), color=C_FUT, stroke_width=2,
                                   stroke_opacity=0.7, dash_length=0.08)
                    )
            return g

        lines = always_redraw(cone)
        for i, c in enumerate(ins):
            c.add_updater(
                lambda m, i=i: m.set_stroke(C_IN if i <= k_now() else C_FUT).set_fill(
                    C_IN if i <= k_now() else C_FUT, 0.45 if i <= k_now() else 0.1
                )
            )
            xl[i].add_updater(lambda m, i=i: m.set_color(C_IN if i <= k_now() else C_FUT))
        for i, c in enumerate(outs):
            c.add_updater(lambda m, i=i: m.set_fill(C_OUT, 0.85 if i == k_now() else 0.0))
        self.play(FadeIn(lines))
        self.wait(1)

        self.say(
            "This property is called causality. As t moves forward, the set of allowed inputs grows, "
            "but it never reaches past t."
        )
        self.play(tr.animate.set_value(9), run_time=2.5, rate_func=linear)
        self.play(tr.animate.set_value(1), run_time=3, rate_func=linear)
        self.play(tr.animate.set_value(6), run_time=2, rate_func=linear)

        self.say(
            "An RNN is causal by construction. A convolutional network gets there with two ideas, "
            "and becomes deep with a third."
        )
        for m in [*ins, *outs, *xl]:
            m.clear_updaters()
        self.clear_stage(hold=False)
        ideas = VGroup(
            txt("1.  causal convolutions", 42),
            txt("2.  dilated convolutions", 42),
            txt("3.  residual blocks", 42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        self.play(LaggedStart(*[FadeIn(i, shift=0.3 * RIGHT) for i in ideas], lag_ratio=0.5), run_time=2)
        self.finish()


# --------------------------------------------------------------------------
# 3. causal convolutions
# --------------------------------------------------------------------------


class S03_Causal(TCNScene):
    def construct(self):
        heading = txt("1.  Causal convolutions", 40).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading))

        n = 10
        slots = [-5.5 + j for j in range(n + 2)]  # slots 0, 1 hold the zero padding
        y_in, y_out = -1.5, 0.9
        pads = VGroup(
            *[DashedVMobject(Circle(radius=0.2, color=C_PAD, stroke_width=3), num_dashes=10).move_to([slots[j], y_in, 0]) for j in range(2)]
        )
        pad_l = VGroup(*[MathTex("0", font_size=28, color=C_PAD).move_to(p) for p in pads])
        ins = VGroup(*[node(slots[j + 2], y_in, C_IN, 0.2, 0.2, 3) for j in range(n)])
        xl = VGroup(*[MathTex(f"x_{{{i}}}", font_size=28, color=C_IN).next_to(c, DOWN, buff=0.14) for i, c in enumerate(ins)])
        outs = VGroup(*[node(slots[j + 2], y_out, C_OUT, 0.2, 0.25, 3) for j in range(n)])
        yl = VGroup(*[MathTex(f"y_{{{i}}}", font_size=28, color=C_OUT).next_to(c, UP, buff=0.14) for i, c in enumerate(outs)])

        ker = VGroup(*[Square(0.55, color=C_KER, stroke_width=3).set_fill(C_KER, 0.25) for _ in range(3)]).arrange(RIGHT, buff=0)
        ker_l = VGroup(*[MathTex(f"w_{i}", font_size=30).move_to(s) for i, s in enumerate(ker)])
        kernel = VGroup(ker, ker_l).move_to([-4.6, 2.55, 0])
        ker_cap = MathTex(r"\text{kernel},\ k = 3", font_size=30, color=C_KER).next_to(kernel, RIGHT, buff=0.3)

        self.say("Start with an ordinary one-dimensional convolution: a kernel of k = 3 weights slides along the sequence.")
        self.play(LaggedStart(*[GrowFromCenter(c) for c in ins], lag_ratio=0.05), FadeIn(xl))
        self.play(FadeIn(kernel, shift=0.2 * DOWN), FadeIn(ker_cap))

        win = window([ins[0], ins[2]])
        lines = taps(ins[0:3], outs[1])
        self.play(TransformFromCopy(ker, win), Create(lines), GrowFromCenter(outs[1]), FadeIn(yl[1]))
        for t in range(2, 9):
            self.play(
                Transform(win, window([ins[t - 1], ins[t + 1]])),
                Transform(lines, taps(ins[t - 1:t + 2], outs[t])),
                GrowFromCenter(outs[t]),
                FadeIn(yl[t]),
                run_time=0.45,
            )

        self.say("Each output is a weighted sum of neighbouring inputs. With a centred kernel, that's this formula.")
        eq1 = MathTex(r"y_t", "=", r"w_0\,x_{t-1}", "+", r"w_1\,x_t", "+", r"w_2\,x_{t+1}", font_size=44).move_to([2.6, 2.55, 0])
        self.play(Write(eq1))
        self.play(Transform(win, window([ins[4], ins[6]])), Transform(lines, taps(ins[4:7], outs[5])), run_time=0.8)

        self.say("But look at the last term. x at t + 1 lies in the future: a centred kernel leaks information backwards in time.")
        self.play(
            eq1[6].animate.set_color(C_FUT),
            lines[2].animate.set_color(C_FUT),
            ins[6].animate.set_stroke(C_FUT).set_fill(C_FUT, 0.4),
        )
        self.play(Circumscribe(eq1[6], color=C_FUT), Wiggle(lines[2]), run_time=1.5)

        self.say("The fix: shift the kernel so it only looks backwards. Now y at time t uses x at t − 2, t − 1 and t.")
        eq2 = MathTex(r"y_t", "=", r"w_0\,x_{t-2}", "+", r"w_1\,x_{t-1}", "+", r"w_2\,x_t", font_size=44).move_to(eq1)
        self.play(
            FadeOut(outs[1:9]),
            FadeOut(yl[1:9]),
            ins[6].animate.set_stroke(C_IN).set_fill(C_IN, 0.2),
            ReplacementTransform(eq1, eq2),
            Transform(win, window([ins[3], ins[5]])),
            Transform(lines, taps(ins[3:6], outs[5])),
        )
        self.play(GrowFromCenter(outs[5]), FadeIn(yl[5]))
        self.wait(0.5)

        self.say(
            "At the very start there's nothing to look back at, so we pad the left end with k − 1 = 2 zeros. "
            "Now the output has exactly the same length as the input."
        )
        self.play(FadeOut(outs[5]), FadeOut(yl[5]), FadeIn(pads), FadeIn(pad_l))
        allslots = [*pads, *ins]
        self.play(
            Transform(win, window(allslots[0:3])),
            Transform(lines, taps(allslots[0:3], outs[0])),
            GrowFromCenter(outs[0]),
            FadeIn(yl[0]),
        )
        for t in range(1, n):
            self.play(
                Transform(win, window(allslots[t:t + 3])),
                Transform(lines, taps(allslots[t:t + 3], outs[t])),
                GrowFromCenter(outs[t]),
                FadeIn(yl[t]),
                run_time=0.4,
            )

        self.say("That's a causal convolution: every output depends only on the present and the past.")
        arrow = Arrow([-5.0, -2.45, 0], [5.6, -2.45, 0], buff=0, stroke_width=2.5, color=GREY_C)
        self.play(Indicate(eq2), GrowArrow(arrow))
        self.play(FadeOut(arrow))

        # Stacking causal layers: the receptive field grows only linearly.
        self.clear_stage(keep=(heading,))
        self.say(
            "Stack several causal layers and each output can see further back. "
            "How far back it sees is called its receptive field."
        )
        L, m = 4, 12
        ys = [-1.9, -0.85, 0.2, 1.25, 2.3]
        xs = [-6.3 + 0.75 * j for j in range(m)]
        colors = [C_IN, C_HID, C_HID, C_HID, C_OUT]
        rows = [VGroup(*[node(x, ys[r], colors[r], 0.13, 0.15) for x in xs]) for r in range(L + 1)]
        edges = {}
        for r in range(1, L + 1):
            for t in range(m):
                for src in (t, t - 1):
                    if src >= 0:
                        edges[(r, t, src)] = link(rows[r - 1][src], rows[r][t], opacity=0.35, width=1.5)
        self.play(LaggedStart(*[GrowFromCenter(c) for c in rows[0]], lag_ratio=0.03))
        for r in range(1, L + 1):
            layer_edges = VGroup(*[e for (rr, _, _), e in edges.items() if rr == r])
            self.play(Create(layer_edges), LaggedStart(*[GrowFromCenter(c) for c in rows[r]], lag_ratio=0.03), run_time=0.8)

        cone = {L: {m - 1}}
        for r in range(L, 0, -1):
            cone[r - 1] = {s for t in cone[r] for s in (t, t - 1) if s >= 0}
        self.play(rows[L][m - 1].animate.set_fill(C_OUT, 0.9))
        for r in range(L, 0, -1):
            hl = [edges[(r, t, s)] for t in cone[r] for s in (t, t - 1) if s >= 0]
            self.play(
                *[e.animate.set_stroke(C_OUT, width=3, opacity=1) for e in hl],
                *[rows[r - 1][s].animate.set_fill(colors[r - 1], 0.9) for s in cone[r - 1]],
                run_time=0.6,
            )

        f1 = MathTex(r"R = 1 + L\,(k-1)", font_size=44).move_to([4.9, 0.8, 0])
        f2 = MathTex(r"L = 4,\ k = 2", font_size=36, color=GREY_B).next_to(f1, DOWN, buff=0.35)
        f3 = MathTex(r"\Rightarrow\ R = 5", font_size=40, color=C_OUT).next_to(f2, DOWN, buff=0.3)
        self.say(
            "But it only grows linearly. With L layers of kernel size k, an output sees 1 + L times (k − 1) steps: "
            "here, 5 inputs."
        )
        self.play(Write(f1))
        self.play(FadeIn(f2), FadeIn(f3))
        self.play(Circumscribe(VGroup(*[rows[0][s] for s in cone[0]]), color=C_OUT))

        self.say(
            "To see 1,000 steps back with k = 2 you'd need about a thousand layers. "
            "We need a much faster way to grow the receptive field."
        )
        self.finish()


# --------------------------------------------------------------------------
# 4. dilated convolutions
# --------------------------------------------------------------------------


class S04_Dilated(TCNScene):
    def construct(self):
        heading = txt("2.  Dilated convolutions", 40).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading))
        formula = MathTex(
            r"F(s)", "=", r"(\mathbf{x} *_d f)(s)", "=", r"\sum_{i=0}^{k-1} f(i)\cdot", r"\mathbf{x}_{s-d\cdot i}",
            font_size=46,
        ).next_to(heading, DOWN, buff=0.45)

        n, sp = 16, 0.72
        xs = [(j - 7.5) * sp for j in range(n)]
        ins = VGroup(*[node(x, -1.6, C_IN, 0.14, 0.2) for x in xs])
        s = 13
        out = node(xs[s], 0.5, C_OUT, 0.16, 0.6, 3)
        s_lab = MathTex("s", font_size=32, color=C_OUT).next_to(out, UP, buff=0.12)

        def taps_d(d):
            return VGroup(*[Line(ins[s - d * i].get_top(), out.get_bottom(), color=C_KER, stroke_width=3) for i in range(3)])

        def d_label(d):
            return MathTex(rf"d = {d}", font_size=44, color=C_KER).move_to([-4.6, 0.5, 0])

        self.say("Idea number two is dilation. A dilated convolution spreads its kernel out, skipping over inputs in between.")
        self.play(LaggedStart(*[GrowFromCenter(c) for c in ins], lag_ratio=0.04))
        lines, dl = taps_d(1), d_label(1)
        self.play(GrowFromCenter(out), FadeIn(s_lab), Create(lines), FadeIn(dl))
        hi = [ins[s - i].copy().set_fill(C_KER, 0.8).set_stroke(C_KER) for i in range(3)]
        hi_g = VGroup(*hi)
        self.play(FadeIn(hi_g))

        self.say(
            "The paper defines it like this: f is a filter of size k, and d, the dilation, is the gap between taps. "
            "With d = 1 it's an ordinary convolution."
        )
        self.play(Write(formula), run_time=2)
        self.play(Circumscribe(formula[5], color=C_KER), formula[5].animate.set_color(C_KER))

        for d in (2, 4):
            new_hi = VGroup(*[ins[s - d * i].copy().set_fill(C_KER, 0.8).set_stroke(C_KER) for i in range(3)])
            self.play(Transform(lines, taps_d(d)), Transform(dl, d_label(d)), Transform(hi_g, new_hi), run_time=1.2)
            self.wait(0.8)

        self.say(
            "One such layer reaches (k − 1)·d steps into the past. "
            "So let d grow exponentially with depth: 1, then 2, then 4, and so on."
        )
        reach = MathTex(r"\text{history per layer} = (k-1)\,d", font_size=38).move_to([-3.6, -0.4, 0])
        self.play(FadeIn(reach))
        self.hold()

        # Figure 1(a) of the paper: k = 3, d = 1, 2, 4.
        self.clear_stage(keep=(heading,))
        self.say("This is Figure 1(a) from the paper: kernel size k = 3, with dilation factors 1, 2 and 4.")
        sp = 0.7
        xs = [0.6 + (j - 7.5) * sp for j in range(n)]
        ys = [-1.75, -0.6, 0.55, 1.7]
        dil = [1, 2, 4]
        colors = [C_IN, C_HID, C_HID, C_OUT]
        rows = [VGroup(*[node(x, ys[r], colors[r], 0.12, 0.15) for x in xs]) for r in range(4)]
        names = ["Input", "Hidden", "Hidden", "Output"]
        row_labels = VGroup()
        for r in range(4):
            lab = VGroup(txt(names[r], 24, colors[r]))
            if r > 0:
                lab.add(MathTex(rf"d = {dil[r - 1]}", font_size=28, color=GREY_B))
            lab.arrange(DOWN, buff=0.08).move_to([-5.95, ys[r], 0])
            row_labels.add(lab)
        edges = {}
        for r in range(1, 4):
            d = dil[r - 1]
            for t in range(n):
                for i in range(3):
                    src = t - d * i
                    if src >= 0:
                        edges[(r, t, src)] = link(rows[r - 1][src], rows[r][t], opacity=0.25, width=1.3)

        self.play(FadeIn(row_labels[0]), LaggedStart(*[GrowFromCenter(c) for c in rows[0]], lag_ratio=0.03))
        for r in range(1, 4):
            layer_edges = VGroup(*[e for (rr, _, _), e in edges.items() if rr == r])
            self.play(
                FadeIn(row_labels[r]),
                Create(layer_edges),
                LaggedStart(*[GrowFromCenter(c) for c in rows[r]], lag_ratio=0.03),
                run_time=1.0,
            )

        self.say("Trace the connections back from the newest output…")
        cone = {3: {n - 1}}
        for r in range(3, 0, -1):
            d = dil[r - 1]
            cone[r - 1] = {t - d * i for t in cone[r] for i in range(3) if t - d * i >= 0}
        self.play(rows[3][n - 1].animate.set_fill(C_OUT, 0.95))
        for r in range(3, 0, -1):
            d = dil[r - 1]
            hl = [edges[(r, t, t - d * i)] for t in cone[r] for i in range(3) if t - d * i >= 0]
            self.play(
                *[e.animate.set_stroke(C_OUT, width=3, opacity=1) for e in hl],
                *[rows[r - 1][q].animate.set_fill(colors[r - 1], 0.95) for q in cone[r - 1]],
                run_time=0.9,
            )

        self.say("…and it covers every one of the last 15 inputs, with no gaps.")
        covered = VGroup(*[rows[0][q] for q in sorted(cone[0])])
        brace = Brace(covered, DOWN, buff=0.12, color=GREY_B)
        b_lab = MathTex("15", font_size=32).next_to(brace, DOWN, buff=0.08)
        self.play(GrowFromCenter(brace), FadeIn(b_lab))

        rf = MathTex(
            r"R", "=", r"1 + (k-1)\sum_{\ell} d_\ell", "=", r"1 + 2\,(1 + 2 + 4)", "=", "15", font_size=36
        ).move_to([0.6, 2.55, 0])
        rf[6].set_color(C_OUT)
        self.say(
            "In general, the receptive field is 1 plus (k − 1) times the sum of the dilations. "
            "Here: 1 + 2·(1 + 2 + 4) = 15."
        )
        self.play(Write(rf), run_time=2)

        self.say(
            "Double d at every layer and the receptive field grows exponentially with depth, "
            "while each layer still costs the same."
        )
        self.hold()
        self.clear_stage(keep=(heading,))

        axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 1100, 200], x_length=9, y_length=4.2, tips=False,
            axis_config={"color": GREY_B, "include_numbers": True, "font_size": 24},
        ).move_to([-0.6, 0.65, 0])
        x_lab = txt("number of layers L", 24, GREY_B).next_to(axes.x_axis, DOWN, buff=0.45)
        y_lab = txt("receptive field", 24, GREY_B).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.55)
        lin = axes.plot_line_graph(
            list(range(11)), [1 + q for q in range(11)], line_color=C_IN,
            vertex_dot_radius=0.05, vertex_dot_style={"fill_color": C_IN},
        )
        expo = axes.plot_line_graph(
            list(range(11)), [2**q for q in range(11)], line_color=C_OUT,
            vertex_dot_radius=0.05, vertex_dot_style={"fill_color": C_OUT},
        )
        lin_lab = MathTex(r"1 + L", r"\;=\;11", font_size=34, color=C_IN).next_to(axes.c2p(10, 11), RIGHT, buff=0.2)
        exp_lab = MathTex(r"2^L", r"\;=\;1024", font_size=34, color=C_OUT).next_to(axes.c2p(10, 1024), RIGHT, buff=0.2)
        leg = VGroup(
            txt("plain causal, k = 2", 24, C_IN),
            txt("dilated, k = 2, d = 1, 2, 4, …", 24, C_OUT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15).move_to(axes.c2p(2.8, 850))

        self.say("With kernel size 2, ten plain causal layers see 11 steps. Ten dilated layers see 1,024.")
        self.play(Create(axes), FadeIn(x_lab), FadeIn(y_lab))
        self.play(Create(lin), FadeIn(leg[0]), run_time=1.5)
        self.play(FadeIn(lin_lab))
        self.play(Create(expo), FadeIn(leg[1]), run_time=2.5)
        self.play(FadeIn(exp_lab))

        self.say(
            "Dilated causal convolutions were popularised by WaveNet for raw audio. "
            "The TCN turns them into the backbone of a generic sequence model."
        )
        self.finish()


# --------------------------------------------------------------------------
# 5. how the reference code gets causality: padding + Chomp1d
# --------------------------------------------------------------------------

CONV_SRC = """\
self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                   stride=stride, padding=padding, dilation=dilation))
self.chomp1 = Chomp1d(padding)"""

CHOMP_SRC = """\
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()"""


class S05_Chomp(TCNScene):
    def construct(self):
        conv_code = code_block(CONV_SRC, width=12.5).move_to([0, 2.75, 0])
        pad_eq = MathTex(r"\texttt{padding} = (k-1)\cdot d", font_size=36, color=GREY_A).next_to(conv_code, DOWN, buff=0.25)

        self.say("How does the official PyTorch code make its convolutions causal? Every conv layer comes with a partner.")
        self.play(FadeIn(conv_code, shift=0.2 * DOWN))
        self.play(FadeIn(pad_eq))

        n, p = 8, 2  # sequence length; padding for k = 3, d = 1
        sp = 0.95
        xs = [(j - (n + 2 * p - 1) / 2) * sp for j in range(n + 2 * p)]
        y_in, y_out = -1.55, 0.25
        slot = []
        for j, x in enumerate(xs):
            if p <= j < p + n:
                slot.append(node(x, y_in, C_IN, 0.2, 0.2, 3))
            else:
                slot.append(DashedVMobject(Circle(radius=0.2, color=C_PAD, stroke_width=3), num_dashes=10).move_to([x, y_in, 0]))
        slot = VGroup(*slot)
        xl = VGroup(*[MathTex(f"x_{{{i}}}", font_size=28, color=C_IN).next_to(slot[p + i], DOWN, buff=0.14) for i in range(n)])
        zeros = VGroup(*[MathTex("0", font_size=26, color=C_PAD).move_to(slot[j]) for j in list(range(p)) + list(range(p + n, n + 2 * p))])
        left_pads = VGroup(slot[0], slot[1], zeros[0], zeros[1])
        right_pads = VGroup(slot[-2], slot[-1], zeros[2], zeros[3])

        self.play(LaggedStart(*[GrowFromCenter(slot[p + i]) for i in range(n)], lag_ratio=0.06), FadeIn(xl))
        self.say("nn.Conv1d with padding p = (k − 1)·d adds p zeros to both ends of the sequence. Here k = 3 and d = 1, so p = 2.")
        self.play(FadeIn(left_pads, shift=0.3 * RIGHT), FadeIn(right_pads, shift=0.3 * LEFT))
        p_l = Brace(VGroup(slot[0], slot[1]), DOWN, buff=0.12, color=GREY_B)
        p_r = Brace(VGroup(slot[-2], slot[-1]), DOWN, buff=0.12, color=GREY_B)
        p_lt = MathTex("p", font_size=30).next_to(p_l, DOWN, buff=0.06)
        p_rt = MathTex("p", font_size=30).next_to(p_r, DOWN, buff=0.06)
        self.play(GrowFromCenter(p_l), GrowFromCenter(p_r), FadeIn(p_lt), FadeIn(p_rt))

        self.say("Slide the kernel across and you get L + p outputs, p more than the L inputs we started with.")
        outs = VGroup(*[node(xs[s + 2], y_out, C_OUT, 0.2, 0.25, 3) for s in range(n + p)])
        win = window(slot[0:3])
        lines = taps(slot[0:3], outs[0])
        self.play(Create(win), Create(lines), GrowFromCenter(outs[0]))
        for s in range(1, n + p):
            self.play(
                Transform(win, window(slot[s:s + 3])),
                Transform(lines, taps(slot[s:s + 3], outs[s])),
                GrowFromCenter(outs[s]),
                run_time=0.35,
            )
        count = MathTex(r"L + p = 10 \text{ outputs}", font_size=34, color=C_OUT).move_to([0, 1.15, 0])
        self.play(FadeIn(count))

        self.say(
            "Each output sits above the newest input it touches, so output s sees x from s − p up to s. "
            "The left padding is exactly what causality needs."
        )
        self.play(Transform(win, window(slot[4:7])), Transform(lines, taps(slot[4:7], outs[4])))
        self.play(Indicate(outs[4], color=C_OUT))
        self.wait(0.5)

        self.say(
            "The last p outputs hang past the end of the sequence; they only exist because of the right-hand padding. "
            "Chomp1d chops them off."
        )
        self.play(FadeOut(win), FadeOut(lines), FadeOut(count))
        self.play(*[outs[s].animate.set_stroke(C_FUT).set_fill(C_FUT, 0.5) for s in (n, n + 1)])
        cut_x = (xs[p + n - 1] + xs[p + n]) / 2
        cut = DashedLine([cut_x, y_in - 0.5, 0], [cut_x, y_out + 0.6, 0], color=C_FUT, stroke_width=3)
        chomp_code = code_block(CHOMP_SRC, width=9.0).move_to([0, 2.45, 0])
        self.play(FadeOut(conv_code), FadeOut(pad_eq), FadeIn(chomp_code, shift=0.2 * DOWN))
        self.play(Create(cut))
        self.play(Circumscribe(chomp_code.code_lines[6], color=C_FUT))
        self.play(
            FadeOut(outs[n], shift=0.6 * RIGHT + 0.3 * UP),
            FadeOut(outs[n + 1], shift=0.6 * RIGHT + 0.3 * UP),
            FadeOut(right_pads),
            FadeOut(p_r),
            FadeOut(p_rt),
            FadeOut(cut),
        )

        self.say(
            "Net effect: the same as padding only on the left. "
            "The output is exactly as long as the input, and nothing leaks from the future."
        )
        self.play(Indicate(VGroup(*outs[:n]), color=C_OUT, scale_factor=1.1))
        self.wait(0.5)
        self.finish()


# --------------------------------------------------------------------------
# 6. the residual block
# --------------------------------------------------------------------------

BLOCK_SRC = """\
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size,
                 stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride,
            padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        # ... conv2 / chomp2 / relu2 / dropout2 likewise ...
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) \\
            if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)"""


class S06_Residual(TCNScene):
    def construct(self):
        dx, sx = -4.6, -2.6  # x of the main path and of the skip path
        bw, bh = 2.5, 0.36
        formula = MathTex(
            r"o", "=", r"\mathrm{Activation}", r"\big(", r"\mathbf{x}", "+", r"\mathcal{F}(\mathbf{x})", r"\big)",
            font_size=44,
        ).move_to([2.6, 3.3, 0])
        formula[4].set_color(C_RES)
        formula[6].set_color(C_IN)

        z_in = MathTex(r"\mathbf{z}^{(i-1)}", font_size=34).move_to([dx, -2.4, 0])
        stage_kinds = [
            ("Dilated Causal Conv", BLUE_D),
            ("WeightNorm", GREY_BROWN),
            ("ReLU", GREEN_D),
            ("Dropout", PURPLE_D),
        ]

        def stage(bottom):
            g = VGroup(*[block_box(name, col, bw, bh) for name, col in stage_kinds])
            for i, b in enumerate(g):
                b.move_to([dx, bottom + bh / 2 + i * bh, 0])
            return g

        st1, st2 = stage(-1.95), stage(-0.21)
        plus = VGroup(Circle(radius=0.2, color=WHITE, stroke_width=2.5), MathTex("+", font_size=36)).move_to([dx, 1.73, 0])
        relu = block_box("ReLU", GREEN_D, 1.4, bh).move_to([dx, 2.36, 0])
        z_out = MathTex(r"\mathbf{z}^{(i)}", font_size=34).move_to([dx, 3.05, 0])

        def up_arrow(y0, y1):
            return Arrow([dx, y0, 0], [dx, y1, 0], buff=0, stroke_width=3, color=GREY_B, max_tip_length_to_length_ratio=0.4)

        a0, a1 = up_arrow(-2.2, -1.95), up_arrow(-0.51, -0.21)
        a2, a3, a4 = up_arrow(1.23, 1.53), up_arrow(1.93, 2.18), up_arrow(2.54, 2.82)

        skip_line = VMobject(color=C_RES, stroke_width=3).set_points_as_corners(
            [[dx, -2.2, 0], [sx, -2.2, 0], [sx, 1.73, 0]]
        )
        skip_tip = Arrow([sx, 1.73, 0], [dx + 0.2, 1.73, 0], buff=0, stroke_width=3, color=C_RES, max_tip_length_to_length_ratio=0.2)
        skip = VGroup(skip_line, skip_tip)
        one_by_one = VGroup(
            DashedVMobject(Rectangle(width=1.2, height=0.7, color=C_RES, stroke_width=2.5), num_dashes=24),
            VGroup(txt("1×1 conv", 17), txt("(optional)", 13, GREY_B)).arrange(DOWN, buff=0.05),
        ).move_to([sx, -0.3, 0])
        bg_1x1 = Rectangle(width=1.2, height=0.7, stroke_width=0).set_fill(config.background_color, 1).move_to(one_by_one)

        big_f = VGroup(
            Rectangle(width=bw, height=3.18, color=BLUE_C, stroke_width=2.5).set_fill(BLUE_E, 0.5),
            MathTex(r"\mathcal{F}", font_size=56, color=C_IN),
        ).move_to([dx, (-1.95 + 1.23) / 2, 0])

        self.say("A real TCN is deep, so the paper builds it out of residual blocks.")
        self.play(FadeIn(z_in), GrowArrow(a0), FadeIn(big_f), GrowArrow(a2), FadeIn(plus))
        self.play(Create(skip_line), GrowArrow(skip_tip))
        self.play(GrowArrow(a3), FadeIn(relu), GrowArrow(a4), FadeIn(z_out))

        self.say(
            "Instead of transforming its input from scratch, each block learns a correction F(x) "
            "and adds it back onto x. The paper writes o = Activation(x + F(x))."
        )
        self.play(Write(formula), run_time=2)
        self.play(ShowPassingFlash(skip_line.copy().set_stroke(YELLOW_C, width=7), time_width=0.6), run_time=1.5)
        self.play(Indicate(formula[4], color=C_RES), Indicate(formula[6], color=C_IN))

        self.say(
            "Inside F there are two identical stages: a dilated causal convolution, "
            "weight normalisation, a ReLU, and dropout."
        )
        self.play(FadeOut(big_f))
        self.play(LaggedStart(*[FadeIn(b, shift=0.15 * UP) for b in st1], lag_ratio=0.25), run_time=1.4)
        self.play(GrowArrow(a1))
        self.play(LaggedStart(*[FadeIn(b, shift=0.15 * UP) for b in st2], lag_ratio=0.25), run_time=1.4)

        self.say(
            "Weight normalisation writes each filter as a length times a direction, which helps optimisation; "
            "dropout regularises."
        )
        wn = MathTex(r"\mathbf{w} = g\,\frac{\mathbf{v}}{\lVert \mathbf{v} \rVert}", font_size=48).move_to([2.8, 0.3, 0])
        self.play(Write(wn), Indicate(st1[1]), Indicate(st2[1]))
        self.play(Indicate(st1[3], color=PURPLE_B), Indicate(st2[3], color=PURPLE_B))

        self.say(
            "If a block changes the number of channels, x and F(x) no longer have the same shape. "
            "An optional 1×1 convolution on the skip path fixes that."
        )
        self.play(FadeIn(bg_1x1), FadeIn(one_by_one))
        self.play(Circumscribe(one_by_one, color=C_RES))

        self.say("And here's the same block in the official code, lightly abridged from tcn.py.")
        code = code_block(BLOCK_SRC)
        code.scale_to_fit_height(5.5)
        if code.width > 8.6:
            code.scale_to_fit_width(8.6)
        code.move_to([2.6, 0.07, 0])
        self.play(FadeOut(wn), FadeIn(code, shift=0.2 * LEFT))

        self.say(
            "self.net is the residual function F: conv, chomp, ReLU, dropout, twice over. "
            "downsample is the 1×1 convolution, used only when the channel counts differ."
        )
        box = line_box(code, 11, 13)
        self.play(Create(box), Indicate(VGroup(st1, st2), scale_factor=1.03, color=YELLOW_C))
        box2 = line_box(code, 14, 15, C_RES)
        self.play(ReplacementTransform(box, box2), Indicate(one_by_one, color=C_RES))

        self.say("And forward() is the formula itself: the net's output plus the skip path, through a final ReLU.")
        box3 = line_box(code, 18, 21)
        self.play(ReplacementTransform(box2, box3), Indicate(formula), Indicate(VGroup(plus, relu)))
        self.finish()


# --------------------------------------------------------------------------
# 7. the full network and its receptive field
# --------------------------------------------------------------------------

NET_SRC = """\
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels,
                 kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1,
                dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)"""


class S07_FullTCN(TCNScene):
    def construct(self):
        heading = txt("Putting it together", 40).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading))

        dx = -4.5
        specs = [
            ("Block 1", r"d = 1"),
            ("Block 2", r"d = 2"),
            ("Block 3", r"d = 4"),
            None,
            ("Block L", r"d = 2^{L-1}"),
        ]
        ys = [-1.6, -0.75, 0.1, 0.8, 1.5]
        blocks = VGroup()
        for spec, y in zip(specs, ys):
            if spec is None:
                blocks.add(MathTex(r"\vdots", font_size=40).move_to([dx, y, 0]))
                continue
            name, d = spec
            rect = RoundedRectangle(corner_radius=0.1, width=2.9, height=0.6, color=BLUE_C, stroke_width=2.5).set_fill(BLUE_E, 0.5)
            lab = VGroup(txt(name, 24), MathTex(d, font_size=30, color=C_OUT)).arrange(RIGHT, buff=0.3)
            blocks.add(VGroup(rect, lab.move_to(rect)).move_to([dx, y, 0]))
        x_in = MathTex(r"\mathbf{x}:\ (N,\ C_{in},\ T)", font_size=30, color=C_IN).move_to([dx, -2.4, 0])
        y_out = MathTex(r"(N,\ C_{out},\ T)", font_size=30, color=C_OUT).move_to([dx, 2.35, 0])
        arrows = VGroup(
            Arrow(x_in.get_top(), blocks[0].get_bottom(), buff=0.08, stroke_width=3, color=GREY_B),
            *[
                Arrow(blocks[i].get_top(), blocks[i + 1].get_bottom(), buff=0.06, stroke_width=3, color=GREY_B,
                      max_tip_length_to_length_ratio=0.5)
                for i in (0, 1)
            ],
            Arrow(blocks[4].get_top(), y_out.get_bottom(), buff=0.08, stroke_width=3, color=GREY_B),
        )

        self.say("A full TCN is simply a stack of these residual blocks, with the dilation doubling from one block to the next.")
        self.play(FadeIn(x_in))
        for i, b in enumerate(blocks):
            anims = [FadeIn(b, shift=0.15 * UP)]
            if i == 0:
                anims.append(GrowArrow(arrows[0]))
            elif i in (1, 2):
                anims.append(GrowArrow(arrows[i]))
            self.play(*anims, run_time=0.6)
        self.play(GrowArrow(arrows[3]), FadeIn(y_out))

        self.say(
            "That's the entire TemporalConvNet class. Level i gets dilation 2 to the i, "
            "and padding (k − 1) times that dilation, which Chomp1d then trims."
        )
        code = code_block(NET_SRC, width=8.4).move_to([2.6, -0.05, 0])
        self.play(FadeIn(code, shift=0.2 * LEFT))
        box = line_box(code, 7, 7)
        self.play(Create(box))
        self.play(ReplacementTransform(box, line_box(code, 13, 13)))
        self.wait(1)

        self.say(
            "Each block holds two convolutions with the same dilation. Add up what all L blocks contribute, "
            "and the receptive field is exponential in L."
        )
        self.clear_stage(keep=(heading,), hold=False)
        r1 = MathTex(r"R", "=", r"1 + \sum_{i=0}^{L-1} 2\,(k-1)\,2^{i}", font_size=48).move_to([0, 1.9, 0])
        r2 = MathTex(r"R", "=", r"1 + 2\,(k-1)\,(2^{L} - 1)", font_size=48).move_to([0, 1.9, 0])
        self.play(Write(r1), run_time=2)
        self.wait(0.6)
        self.play(TransformMatchingTex(r1, r2))
        self.play(Circumscribe(r2))

        self.say(
            "Plug in the defaults from the repository's experiment scripts, and compare R "
            "with the length of each task's sequences."
        )
        rows = [
            ("Adding problem", "7", "8", "3061", "200 – 600"),
            ("Sequential MNIST", "7", "8", "3061", "784"),
            ("Copy memory", "8", "8", "3571", "1020"),
        ]
        header = ["task", "k", "L", "R", "sequence length"]
        col_x = [-3.9, -1.3, -0.3, 1.0, 3.6]
        table = VGroup()
        for j, h in enumerate(header):
            table.add(txt(h, 26, GREY_B).move_to([col_x[j], 0.65, 0]))
        rule_line = Line([-5.6, 0.38, 0], [5.4, 0.38, 0], color=GREY_D, stroke_width=1.5)
        cells = VGroup()
        for i, row in enumerate(rows):
            y = 0.0 - 0.55 * i
            for j, v in enumerate(row):
                color = C_OUT if j == 3 else (C_IN if j == 4 else WHITE)
                cells.add(txt(v, 28, color).move_to([col_x[j], y, 0]))
        self.play(FadeIn(table), Create(rule_line))
        self.play(LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=0.05), run_time=2)

        self.say("In every case the receptive field comfortably covers the whole sequence.")
        rcol = VGroup(*[cells[5 * i + 3] for i in range(3)])
        lcol = VGroup(*[cells[5 * i + 4] for i in range(3)])
        self.play(Circumscribe(rcol, color=C_OUT), Circumscribe(lcol, color=C_IN))
        self.hold()

        # Classification head: features at the last time step -> linear layer.
        self.clear_stage(keep=(heading,))
        self.say(
            "To classify a whole sequence, like a digit read one pixel at a time, "
            "the code looks only at the very last time step…"
        )
        rows_n, cols_n, cs = 6, 18, 0.32
        grid = VGroup(
            *[
                Square(cs, stroke_width=1, color=GREY_D).set_fill(BLUE_E, 0.25 + 0.5 * ((r * 7 + c * 3) % 5) / 5)
                .move_to([-5.6 + c * cs, -0.4 + (r - rows_n / 2) * cs, 0])
                for r in range(rows_n)
                for c in range(cols_n)
            ]
        )
        last_col = VGroup(*[grid[r * cols_n + cols_n - 1] for r in range(rows_n)])
        g_lab = MathTex(r"\texttt{y1}:\ (N,\ C,\ T)", font_size=30, color=C_IN).next_to(grid, UP, buff=0.25)
        t_arrow = Arrow(grid.get_corner(DL) + 0.25 * DOWN, grid.get_corner(DR) + 0.25 * DOWN, buff=0, stroke_width=2.5, color=GREY_B)
        t_lab = txt("time", 22, GREY_B).next_to(t_arrow, DOWN, buff=0.08)
        self.play(FadeIn(grid), FadeIn(g_lab), GrowArrow(t_arrow), FadeIn(t_lab))
        sel = SurroundingRectangle(last_col, color=C_OUT, buff=0.04, stroke_width=3)
        self.play(Create(sel), *[c.animate.set_fill(C_OUT, 0.8) for c in last_col])
        self.say("…whose features have seen the entire sequence, and feeds them to a linear layer.")

        lin_box = VGroup(
            RoundedRectangle(corner_radius=0.1, width=1.8, height=1.0, color=C_OUT, stroke_width=2.5).set_fill(GREY_E, 0.6),
            txt("Linear", 28),
        )
        lin_box[1].move_to(lin_box[0])
        lin_box.move_to([1.9, -0.4, 0])
        to_lin = Arrow(sel.get_right(), lin_box.get_left(), buff=0.15, stroke_width=3, color=C_OUT)
        scores = VGroup(*[Rectangle(width=0.22, height=0.15 + 0.6 * v, stroke_width=0).set_fill(C_OUT, 0.8)
                          for v in (0.1, 0.05, 0.2, 0.9, 0.1, 0.15, 0.05, 0.3, 0.1, 0.05)])
        scores.arrange(RIGHT, buff=0.06, aligned_edge=DOWN).next_to(lin_box, RIGHT, buff=0.6).align_to(lin_box, DOWN)
        s_lab = txt("10 class scores", 22, GREY_B).next_to(scores, UP, buff=0.2)
        to_scores = Arrow(lin_box.get_right(), scores.get_left(), buff=0.1, stroke_width=3, color=GREY_B)
        head_code = code_block("o = self.linear(y1[:, :, -1])", width=6.2).move_to([0.6, 1.9, 0])
        self.play(GrowArrow(to_lin), FadeIn(lin_box))
        self.play(GrowArrow(to_scores), LaggedStart(*[GrowFromEdge(b, DOWN) for b in scores], lag_ratio=0.05), FadeIn(s_lab))
        self.play(FadeIn(head_code, shift=0.2 * DOWN))
        self.finish()


# --------------------------------------------------------------------------
# 8. TCN vs RNN
# --------------------------------------------------------------------------


class S08_Compare(TCNScene):
    def construct(self):
        heading = txt("Why convolutions instead of recurrence?", 40).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading))

        # Left: an RNN.
        n = 8
        rx = [-6.0 + 0.72 * i for i in range(n)]
        r_in = VGroup(*[node(x, -1.6, C_IN, 0.13, 0.2) for x in rx])
        r_cell = VGroup(*[RoundedRectangle(corner_radius=0.07, width=0.5, height=0.42, color=GREY_B, stroke_width=2.5).move_to([x, 0.0, 0]) for x in rx])
        r_up = VGroup(*[Line(a.get_top(), b.get_bottom(), color=GREY_B, stroke_width=2) for a, b in zip(r_in, r_cell)])
        r_side = VGroup(*[Line(r_cell[i].get_right(), r_cell[i + 1].get_left(), color=GREY_B, stroke_width=2) for i in range(n - 1)])
        rnn = VGroup(r_in, r_up, r_cell, r_side)
        r_title = txt("RNN", 34).move_to([-3.48, 2.4, 0])

        # Right: a TCN with k = 2 and d = 1, 2, 4.
        tx = [1.0 + 0.72 * i for i in range(n)]
        tys = [-1.6, -0.6, 0.4, 1.4]
        t_rows = [VGroup(*[node(x, tys[r], [C_IN, C_HID, C_HID, C_OUT][r], 0.12, 0.15) for x in tx]) for r in range(4)]
        t_edges = {}
        for r, d in zip((1, 2, 3), (1, 2, 4)):
            for t in range(n):
                for src in (t, t - d):
                    if src >= 0:
                        t_edges[(r, t, src)] = link(t_rows[r - 1][src], t_rows[r][t], width=1.5, opacity=0.4)
        tcn = VGroup(*t_rows, *t_edges.values())
        t_title = txt("TCN", 34).move_to([3.52, 2.4, 0])
        divider = DashedLine([-0.25, -2.2, 0], [-0.25, 2.6, 0], color=GREY_D)

        self.say("The paper lists several reasons a TCN can beat a recurrent network. First: parallelism.")
        self.play(FadeIn(rnn), FadeIn(tcn), FadeIn(r_title), FadeIn(t_title), Create(divider), run_time=1.5)

        self.say(
            "An RNN must finish step t − 1 before it can start step t. A TCN computes an entire layer, "
            "across all time steps, at once. Watch them run on the same clock."
        )
        clock = Integer(0, font_size=36, color=GREY_B)
        clock_lab = txt("tick", 26, GREY_B).next_to(clock, LEFT, buff=0.15)
        clock_g = VGroup(clock_lab, clock).move_to([-0.25, -2.35, 0])
        self.play(FadeIn(clock_g))
        for tick in range(1, n + 1):
            anims = [
                r_cell[tick - 1].animate.set_fill(C_OUT, 0.5).set_stroke(C_OUT),
                clock.animate.set_value(tick),
            ]
            if tick <= 3:
                anims += [c.animate.set_fill([C_HID, C_HID, C_OUT][tick - 1], 0.85) for c in t_rows[tick]]
                anims += [e.animate.set_stroke(opacity=0.9) for (r, _, _), e in t_edges.items() if r == tick]
            self.play(*anims, run_time=0.55)
        r_note = txt("8 sequential steps", 26, C_OUT).next_to(r_title, DOWN, buff=0.2)
        t_note = txt("3 layers, each fully parallel", 26, C_OUT).next_to(t_title, DOWN, buff=0.2)
        self.play(FadeIn(r_note), FadeIn(t_note))
        self.hold()
        self.play(FadeOut(r_note), FadeOut(t_note), FadeOut(clock_g))

        self.say(
            "Second: stable gradients. In an RNN, the gradient from the last output back to the first input "
            "passes through every time step…"
        )
        arcs = backprop_arcs(r_cell, width=4.0)
        r_path = MathTex(r"\text{path length} \propto T", font_size=32, color=C_FUT).next_to(r_title, DOWN, buff=0.2)
        self.play(LaggedStart(*[Create(a) for a in arcs], lag_ratio=0.6), run_time=3.5)
        self.say("…T repeated multiplications, which tend to make it vanish or explode.")
        self.play(FadeIn(r_path))

        self.say(
            "In a TCN, backpropagation runs through the layers, not along time, "
            "so the path is only as long as the network is deep: about log T."
        )
        path = [(3, 7), (2, 3), (1, 1), (0, 0)]
        segs = VGroup(*[Line(t_rows[a][i].get_center(), t_rows[b][j].get_center(), color=C_FUT, stroke_width=7)
                        for (a, i), (b, j) in zip(path[:-1], path[1:])])
        t_path = MathTex(r"\text{path length} \propto \log T", font_size=32, color=C_FUT).next_to(t_title, DOWN, buff=0.2)
        for _ in range(2):
            self.play(LaggedStart(*[ShowPassingFlash(sg.copy(), time_width=0.9) for sg in segs], lag_ratio=0.5), run_time=1.5)
        self.play(Create(segs), FadeIn(t_path))
        self.hold()

        # The advantages / disadvantages list from the paper's discussion of TCNs.
        self.clear_stage(keep=(heading,))
        pros = [
            "Parallelism",
            "Flexible receptive field size",
            "Stable gradients",
            "Low memory requirement for training",
            "Variable-length inputs",
        ]
        cons = [
            "Data storage during evaluation",
            "Potential parameter change for a transfer of domain",
        ]
        pro_g = VGroup(
            *[VGroup(MathTex(r"\checkmark", color=C_RES, font_size=36), txt(p, 28)).arrange(RIGHT, buff=0.25) for p in pros]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        con_g = VGroup(
            *[VGroup(MathTex(r"\times", color=C_FUT, font_size=40), txt(c, 28)).arrange(RIGHT, buff=0.25) for c in cons]
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22)
        lists = VGroup(pro_g, con_g).arrange(DOWN, aligned_edge=LEFT, buff=0.55).move_to([0, 0.15, 0])

        self.say("Here's the paper's full list. You've seen parallelism and stable gradients.")
        self.play(LaggedStart(*[FadeIn(p, shift=0.2 * RIGHT) for p in pro_g], lag_ratio=0.15), run_time=1.5)
        self.play(Indicate(pro_g[0][1], color=C_OUT), Indicate(pro_g[2][1], color=C_OUT))

        self.say(
            "The receptive field is easy to tune through k, d and depth. "
            "Training needs less memory: there are no per-step gate activations to store."
        )
        self.play(Indicate(pro_g[1][1], color=C_OUT))
        self.play(Indicate(pro_g[3][1], color=C_OUT))
        self.say("And inputs can be any length, because the kernels just slide along them.")
        self.play(Indicate(pro_g[4][1], color=C_OUT))

        self.say(
            "The costs: at test time an RNN keeps only its hidden state, while a TCN must keep the raw sequence "
            "as far back as its receptive field."
        )
        self.play(FadeIn(con_g[0], shift=0.2 * RIGHT))
        self.say(
            "And a model built with a small receptive field may do poorly when moved to a domain "
            "that needs much longer memory."
        )
        self.play(FadeIn(con_g[1], shift=0.2 * RIGHT))
        self.finish()


# --------------------------------------------------------------------------
# 9. results (Table 1 of the paper)
# --------------------------------------------------------------------------

RESULTS = [
    # task, LSTM, GRU, RNN, TCN, index of the best column (0 = LSTM ... 3 = TCN)
    ("Seq. MNIST  (accuracy ↑)", "87.2", "96.2", "21.5", "99.0", 3),
    ("Permuted MNIST  (accuracy ↑)", "85.7", "87.3", "25.3", "97.2", 3),
    ("Adding problem T=600  (loss ↓)", "0.164", "5.3e-5", "0.177", "5.8e-5", 1),
    ("Copy memory T=1000  (loss ↓)", "0.0204", "0.0197", "0.0202", "3.5e-5", 3),
    ("Music JSB Chorales  (loss ↓)", "8.45", "8.43", "8.91", "8.10", 3),
    ("Music Nottingham  (loss ↓)", "3.29", "3.46", "4.05", "3.07", 3),
    ("Word-level PTB  (perplexity ↓)", "78.93", "92.48", "114.50", "88.68", 0),
    ("Word-level Wiki-103  (perplexity ↓)", "48.4", "–", "–", "45.19", 3),
    ("Word-level LAMBADA  (perplexity ↓)", "4186", "–", "14725", "1279", 3),
    ("Char-level PTB  (bpc ↓)", "1.36", "1.37", "1.48", "1.31", 3),
    ("Char-level text8  (bpc ↓)", "1.50", "1.53", "1.69", "1.45", 3),
]


class S09_Results(TCNScene):
    def construct(self):
        heading = txt("Results  (Table 1 of the paper)", 40).to_edge(UP, buff=0.35)
        self.play(FadeIn(heading))

        col_x = [-3.2, 1.35, 2.75, 4.15, 5.55]
        top, rh = 2.55, 0.43
        head = VGroup(txt("Sequence modeling task", 24, GREY_B).move_to([col_x[0], top, 0]))
        for j, name in enumerate(["LSTM", "GRU", "RNN", "TCN"]):
            head.add(txt(name, 26, C_OUT if name == "TCN" else GREY_B).move_to([col_x[j + 1], top, 0]))
        head[0].align_to([-6.6, 0, 0], LEFT)
        rule_line = Line([-6.7, top - 0.25, 0], [6.3, top - 0.25, 0], color=GREY_D, stroke_width=1.5)
        rows = VGroup()
        best = []
        for i, (task, *vals, b) in enumerate(RESULTS):
            y = top - 0.5 - rh * i
            row = VGroup(txt(task, 22).move_to([0, y, 0]).align_to([-6.6, 0, 0], LEFT))
            for j, v in enumerate(vals):
                row.add(txt(v, 22, GREY_A).move_to([col_x[j + 1], y, 0]))
            best.append(row[b + 1])
            rows.add(row)

        self.say(
            "The paper pits a generic TCN against LSTM, GRU and vanilla RNN baselines of roughly the same size, "
            "on eleven standard benchmarks."
        )
        self.play(FadeIn(head), Create(rule_line))
        self.play(LaggedStart(*[FadeIn(r, shift=0.1 * UP) for r in rows], lag_ratio=0.12), run_time=3)

        self.say("Yellow marks the best result in each row. The TCN comes out on top in nine of the eleven, with minimal tuning.")
        self.play(*[c.animate.set_color(C_OUT).scale(1.08) for c in best], run_time=1.2)
        tcn_col = VGroup(head[4], *[r[4] for r in rows])
        self.play(Create(SurroundingRectangle(tcn_col, color=C_OUT, buff=0.12, corner_radius=0.08)))

        self.say("It isn't a clean sweep: the GRU edges it out on the adding problem, and the LSTM wins on word-level Penn Treebank.")
        self.play(Circumscribe(rows[2], color=GREY_A), run_time=1.5)
        self.play(Circumscribe(rows[6], color=GREY_A), run_time=1.5)

        self.say(
            "Look at the copy-memory task: recall ten digits after a gap of a thousand steps. "
            "The recurrent nets sit at about 0.02, which is the loss of random guessing."
        )
        guess = MathTex(r"\text{guessing: } \frac{10\,\ln 8}{T + 20} \approx 0.0204", font_size=36).move_to(heading)
        self.play(FadeOut(heading), FadeIn(guess), Circumscribe(rows[3], color=C_OUT, run_time=1.5))
        self.say("The TCN's loss is essentially zero. That's what the authors mean by a longer effective memory.")
        self.play(Indicate(rows[3][4], color=C_OUT, scale_factor=1.4))

        # Bar chart: sequential and permuted MNIST.
        self.hold()
        self.clear_stage()
        self.say(
            "Sequential MNIST makes it vivid: classify a handwritten digit after reading its 784 pixels one at a time."
        )
        groups = [("Sequential MNIST", [87.2, 96.2, 21.5, 99.0]), ("Permuted MNIST", [85.7, 87.3, 25.3, 97.2])]
        models = ["LSTM", "GRU", "RNN", "TCN"]
        model_cols = [BLUE_C, TEAL_C, GREY_BROWN, YELLOW_C]
        axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 100, 20], x_length=11, y_length=4.0, tips=False,
            axis_config={"color": GREY_B}, y_axis_config={"include_numbers": True, "font_size": 24},
            x_axis_config={"include_ticks": False},
        ).move_to([0, 0.75, 0])
        acc_lab = txt("test accuracy (%)", 22, GREY_B).rotate(PI / 2).next_to(axes.y_axis, LEFT, buff=0.45)
        self.play(Create(axes), FadeIn(acc_lab))
        bars_all = []
        for g, (gname, vals) in enumerate(groups):
            x0 = 1.0 + 5.0 * g
            bars = VGroup()
            for m, v in enumerate(vals):
                left = axes.c2p(x0 + m * 0.95, 0)
                right = axes.c2p(x0 + m * 0.95 + 0.8, v)
                bar = Rectangle(width=right[0] - left[0], height=right[1] - left[1], stroke_width=0)
                bar.set_fill(model_cols[m], 0.85).move_to((left + right) / 2)
                val = txt(f"{v:.1f}", 22, model_cols[m]).next_to(bar, UP, buff=0.08)
                name = txt(models[m], 20, GREY_B).next_to(axes.c2p(x0 + m * 0.95 + 0.4, 0), DOWN, buff=0.15)
                bars.add(VGroup(bar, val, name))
            g_lab = txt(gname, 26).next_to(axes.c2p(x0 + 1.85, 0), DOWN, buff=0.55)
            bars_all.append((bars, g_lab))

        bars, g_lab = bars_all[0]
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.2), run_time=1.8)
        self.play(*[FadeIn(b[1]) for b in bars], *[FadeIn(b[2]) for b in bars], FadeIn(g_lab))

        self.say(
            "Permuted MNIST shuffles the pixel order, so long-range dependencies matter even more. "
            "The TCN's lead widens: 97.2% against 87.3% for the best recurrent net."
        )
        bars, g_lab = bars_all[1]
        self.play(LaggedStart(*[GrowFromEdge(b[0], DOWN) for b in bars], lag_ratio=0.2), run_time=1.8)
        self.play(*[FadeIn(b[1]) for b in bars], *[FadeIn(b[2]) for b in bars], FadeIn(g_lab))
        self.play(Indicate(bars[3], color=C_OUT))
        self.finish()


# --------------------------------------------------------------------------
# 10. outro
# --------------------------------------------------------------------------


class S10_Outro(TCNScene):
    def construct(self):
        self.say("So that's a Temporal Convolutional Network: a one-dimensional, fully convolutional network built from three ideas.")

        # Mini icon 1: causal convolution.
        c_in = VGroup(*[node(-0.9 + 0.45 * i, -0.5, C_IN, 0.1, 0.4) for i in range(5)])
        c_out = node(0.9, 0.5, C_OUT, 0.11, 0.8)
        c_lines = VGroup(*[Line(c_in[i].get_top(), c_out.get_bottom(), color=C_KER, stroke_width=2.5) for i in (2, 3, 4)])
        icon1 = VGroup(c_in, c_out, c_lines)

        # Mini icon 2: dilated tree (k = 2).
        rows = [VGroup(*[node(-1.05 + 0.3 * i, -0.6 + 0.4 * r, [C_IN, C_HID, C_HID, C_OUT][r], 0.06, 0.6) for i in range(8)]) for r in range(4)]
        tree = VGroup()
        for r, d in zip((1, 2, 3), (1, 2, 4)):
            for t in range(8):
                for s in (t, t - d):
                    if s >= 0:
                        on_path = (r == 3 and t == 7) or (r == 2 and t in (3, 7)) or (r == 1 and t % 2 == 1)
                        tree.add(link(rows[r - 1][s], rows[r][t], C_OUT if on_path else C_EDGE, 1.5 if on_path else 1.0, 1.0 if on_path else 0.3))
        icon2 = VGroup(*rows, tree)

        # Mini icon 3: residual block.
        fbox = VGroup(Rectangle(width=1.0, height=0.9, color=BLUE_C).set_fill(BLUE_E, 0.5), MathTex(r"\mathcal{F}", font_size=36))
        fbox[1].move_to(fbox[0])
        fbox.move_to([0, 0, 0])
        r_plus = VGroup(Circle(radius=0.15, color=WHITE, stroke_width=2), MathTex("+", font_size=26)).move_to([0, 0.85, 0])
        r_in = Line([0, -0.85, 0], fbox.get_bottom(), color=GREY_B, stroke_width=2.5)
        r_mid = Line(fbox.get_top(), r_plus.get_bottom(), color=GREY_B, stroke_width=2.5)
        r_skip = VMobject(color=C_RES, stroke_width=2.5).set_points_as_corners([[0, -0.7, 0], [0.9, -0.7, 0], [0.9, 0.85, 0], [0.15, 0.85, 0]])
        icon3 = VGroup(r_in, fbox, r_mid, r_plus, r_skip)

        icons = VGroup(icon1, icon2, icon3)
        labels = VGroup(
            txt("causal convolutions", 28),
            txt("dilations", 28),
            txt("residual blocks", 28),
        )
        notes = VGroup(
            txt("keep the future out", 22, GREY_B),
            txt("exponential receptive field", 22, GREY_B),
            txt("make depth trainable", 22, GREY_B),
        )
        for i, x in enumerate((-4.3, 0, 4.3)):
            icons[i].move_to([x, 1.2, 0])
            labels[i].move_to([x, -0.25, 0])
            notes[i].next_to(labels[i], DOWN, buff=0.15)

        for i in range(3):
            self.play(FadeIn(icons[i], shift=0.2 * UP), FadeIn(labels[i]), run_time=0.9)
        self.say(
            "Causal convolutions keep the future out. Dilations make the receptive field grow exponentially. "
            "Residual blocks let the network go deep."
        )
        for i in range(3):
            self.play(FadeIn(notes[i], shift=0.1 * UP), run_time=0.8)
            self.wait(0.6)

        self.clear_stage()
        quote = VGroup(
            txt("“...the common association between sequence modeling and recurrent", 30),
            txt("networks should be reconsidered, and convolutional networks should be", 30),
            txt("regarded as a natural starting point for sequence modeling tasks.”", 30),
        ).arrange(DOWN, buff=0.18).move_to([0, 1.0, 0])
        attribution = txt("— Bai, Kolter & Koltun (2018)", 26, GREY_B).next_to(quote, DOWN, buff=0.35).align_to(quote, RIGHT)
        self.say("The authors' conclusion is worth reading in full.")
        self.play(FadeIn(quote, lag_ratio=0.1), run_time=2.5)
        self.play(FadeIn(attribution))
        self.hold(2.5)

        self.say("The whole model file, tcn.py, is 63 lines of PyTorch. Next time you reach for an LSTM, try a TCN first.")
        self.clear_stage(hold=False)
        refs = VGroup(
            txt("References", 34),
            txt("S. Bai, J. Z. Kolter, V. Koltun. An Empirical Evaluation of Generic Convolutional", 24, GREY_A),
            txt("and Recurrent Networks for Sequence Modeling. arXiv:1803.01271, 2018.", 24, GREY_A),
            txt("Code: github.com/locuslab/TCN  (TCN/tcn.py)", 24, C_IN),
            txt("Dilated causal convolutions: A. van den Oord et al., WaveNet, 2016.", 24, GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).move_to([0, 0.9, 0])
        refs[0].shift(0.1 * UP)
        self.play(FadeIn(refs, lag_ratio=0.15), run_time=2)
        self.finish()

