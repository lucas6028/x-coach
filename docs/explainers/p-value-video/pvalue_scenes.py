"""Manim scenes for a 3Blue1Brown-style explainer: "What is a p-value?"

Each scene voices its segments from audio/timings.json (built by build_audio.py)
and times its animations against them. Render one scene with
    manim -qh pvalue_scenes.py S01Hook
or the whole video with build_video.py.
"""

import json
import math
from pathlib import Path

import numpy as np
from manim import *

HERE = Path(__file__).resolve().parent
AUDIO = HERE / "audio"
TIMINGS = json.loads((AUDIO / "timings.json").read_text(encoding="utf-8"))

HEADS = BLUE_C
TAILS_FILL = GREY_D
BAR = BLUE_D
HI = YELLOW
NULL = BLUE
WARN = RED_B


# --------------------------------------------------------------------------
# Binomial(100, 0.5): the fair-coin world
# --------------------------------------------------------------------------

PMF = {k: math.comb(100, k) / 2**100 for k in range(101)}
PEAK = PMF[50]


def two_sided_p(k):
    d = abs(k - 50)
    return min(1.0, sum(PMF[j] for j in range(101) if abs(j - 50) >= d))


def pattern_with(heads, seed, n=100):
    flips = np.array([1] * heads + [0] * (n - heads))
    np.random.default_rng(seed).shuffle(flips)
    return flips


# --------------------------------------------------------------------------
# Narration plumbing
# --------------------------------------------------------------------------


class Narrated(Scene):
    """A scene whose animations are timed against pre-rendered voice segments."""

    def setup(self):
        self.segments = TIMINGS[type(self).__name__]
        self.seg = None
        self.seg_start = 0.0
        self.seg_end = 0.0

    def now(self):
        return self.renderer.time

    def say(self, i):
        """Start voice segment i now; returns its duration."""
        self.seg = self.segments[i]
        self.add_sound(str(AUDIO / self.seg["file"]))
        for sub in self.seg["subs"]:
            self.add_subcaption(sub["text"], duration=sub["duration"], offset=sub["offset"])
        self.seg_start = self.now()
        self.seg_end = self.seg_start + self.seg["duration"]
        return self.seg["duration"]

    def sentence_at(self, j):
        """Absolute time at which sentence j of the current segment starts."""
        return self.seg_start + self.seg["subs"][j]["offset"]

    def wait_until(self, t):
        dt = t - self.now()
        if dt > 1 / 120:
            self.wait(dt)

    def sync(self, pad=0.35):
        """Hold until the current segment has finished speaking (plus a pause)."""
        self.wait_until(self.seg_end + pad)

    def left(self):
        return max(self.seg_end - self.now(), 0.4)


# --------------------------------------------------------------------------
# Visual building blocks
# --------------------------------------------------------------------------


def make_coin(label="H", radius=0.6, color=YELLOW_D, rim=YELLOW_B):
    body = Circle(radius, fill_color=color, fill_opacity=1, stroke_color=rim, stroke_width=4)
    inner = Circle(radius * 0.78, stroke_color=rim, stroke_width=2, stroke_opacity=0.7)
    text = MathTex(label, color=BLACK).scale(radius * 1.6)
    return VGroup(body, inner, text)


def flip(coin, height=1.8, turns=3, run_time=1.3):
    start = coin.copy()

    def update(m, a):
        m.become(start)
        m.rotate(turns * TAU * a, axis=RIGHT)
        m.shift(UP * height * 4 * a * (1 - a))

    return UpdateFromAlphaFunc(coin, update, run_time=run_time, rate_func=linear)


def dot_grid(radius=0.12, spacing=0.32, rows=10, cols=10):
    grid = VGroup(*[Circle(radius, stroke_color=GREY_B, stroke_width=2, fill_opacity=0) for _ in range(rows * cols)])
    grid.arrange_in_grid(rows, cols, buff=spacing - 2 * radius)
    return grid


def fill_anims(grid, flips):
    return [
        cell.animate.set_fill(HEADS if h else TAILS_FILL, 1).set_stroke(HEADS if h else GREY_C)
        for cell, h in zip(grid, flips)
    ]


def warning_icon(scale=0.32):
    tri = Triangle(color=WARN, stroke_width=5).scale(scale)
    mark = MathTex("!", color=WARN).scale(scale * 2.6).move_to(tri.get_center() + DOWN * scale * 0.12)
    return VGroup(tri, mark)


def trap_header(n):
    return VGroup(warning_icon(), Tex(f"Trap {n}", font_size=50, color=WARN)).arrange(RIGHT, buff=0.25).to_corner(UL, buff=0.5)


class HeadsPlot:
    """A bare x-axis of head counts with bars of the fair-coin distribution."""

    def __init__(self, origin=(-6.0, -2.6), x_length=12.0, height=3.4, lo=25, hi=75, font_size=30):
        self.lo, self.hi, self.H = lo, hi, height
        self.ax = Axes(
            x_range=[lo, hi, 5],
            y_range=[0, 1, 1],
            x_length=x_length,
            y_length=height,
            tips=False,
            x_axis_config={"numbers_to_include": list(range(30, hi, 10)), "font_size": font_size},
        )
        self.ax.shift(np.array([origin[0], origin[1], 0]) - self.ax.c2p(lo, 0))
        self.axis = self.ax.x_axis
        self.bw = self.axis.get_unit_size() * 0.8

    def x(self, k):
        return self.ax.c2p(k, 0)[0]

    def base(self, k):
        return self.ax.c2p(k, 0)

    def bar(self, k, height, color=BAR, opacity=0.9):
        rect = Rectangle(
            width=self.bw,
            height=max(height, 0.001),
            fill_color=color,
            fill_opacity=opacity,
            stroke_color=color,
            stroke_width=1,
            stroke_opacity=opacity,
        )
        return rect.move_to(self.base(k), aligned_edge=DOWN)

    def null_height(self, k):
        return PMF[k] / PEAK * self.H

    def null_bars(self, style=None):
        style = style or (lambda k: (BAR, 0.9))
        return VGroup(*[self.bar(k, self.null_height(k), *style(k)) for k in range(self.lo, self.hi + 1)])

    def index(self, k):
        return k - self.lo


def curve_axes(lo=0.35, hi=0.65, origin=(-5.5, -2.4), x_length=11.0, height=3.0):
    ax = Axes(
        x_range=[lo, hi, 0.05],
        y_range=[0, 1, 1],
        x_length=x_length,
        y_length=height,
        tips=False,
        x_axis_config={
            "numbers_to_include": [round(v, 2) for v in np.arange(lo + 0.05, hi - 0.01, 0.05)],
            "decimal_number_config": {"num_decimal_places": 2},
            "font_size": 28,
        },
    )
    ax.shift(np.array([origin[0], origin[1], 0]) - ax.c2p(lo, 0))
    return ax


def bell(x, center, sd):
    return np.exp(-0.5 * ((x - center) / sd) ** 2)


def bell_curve(ax, center, sd, lo, hi, color=NULL, width=4):
    xs = np.linspace(lo, hi, 700)
    curve = VMobject(stroke_color=color, stroke_width=width)
    curve.set_points_as_corners([ax.c2p(x, bell(x, center, sd)) for x in xs])
    return curve


def bell_area(ax, center, sd, a, b, color, opacity):
    xs = np.linspace(a, b, 300)
    pts = [ax.c2p(x, bell(x, center, sd)) for x in xs] + [ax.c2p(b, 0), ax.c2p(a, 0)]
    return Polygon(*pts, stroke_width=0, fill_color=color, fill_opacity=opacity)


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------


class S01Hook(Narrated):
    def construct(self):
        coin = make_coin("H", 0.9).shift(UP * 0.5)
        quote = Tex(r"``It's fair, I promise.''", font_size=42, color=GREY_A).next_to(coin, DOWN, buff=0.6)

        self.say(0)
        self.play(GrowFromCenter(coin), run_time=0.9)
        self.play(Write(quote), run_time=1.4)
        self.sync()

        grid = dot_grid().move_to(RIGHT * 2.6 + UP * 0.45)
        self.say(1)
        self.play(flip(coin), FadeOut(quote), run_time=1.3)
        self.play(
            coin.animate.scale(0.65).move_to(LEFT * 4.0 + UP * 0.45),
            LaggedStart(*[Create(c) for c in grid], lag_ratio=0.01),
            run_time=1.4,
        )
        self.sync(0.1)

        flips = pattern_with(62, seed=7)
        heads = Integer(0, font_size=52, color=HEADS)
        tails = Integer(0, font_size=52, color=GREY_B)
        heads.set_value(62)
        tails.set_value(38)
        tally = VGroup(
            VGroup(heads, Tex("heads", font_size=38, color=HEADS)).arrange(RIGHT, buff=0.2),
            VGroup(tails, Tex("tails", font_size=38, color=GREY_B)).arrange(RIGHT, buff=0.2),
        ).arrange(RIGHT, buff=0.9).next_to(grid, DOWN, buff=0.45)
        for num, word in tally:
            num.set_value(0)
            word.add_updater(lambda m, num=num: m.next_to(num, RIGHT, buff=0.2))
        self.say(2)
        self.add(tally)
        self.play(
            LaggedStart(*fill_anims(grid, flips), lag_ratio=0.03),
            ChangeDecimalToValue(heads, 62),
            ChangeDecimalToValue(tails, 38),
            run_time=2.6,
        )
        self.play(Indicate(tally[0], color=HI, scale_factor=1.2), run_time=0.8)
        self.sync()

        q1 = Tex("Suspicious?", font_size=56, color=HI)
        q2 = Tex("...or just luck?", font_size=56, color=GREY_A)
        qs = VGroup(q1, q2).arrange(RIGHT, buff=0.5).to_edge(UP, buff=0.55)
        self.say(3)
        self.play(FadeIn(q1, shift=DOWN * 0.3), run_time=0.7)
        self.wait_until(self.sentence_at(1))
        self.play(FadeIn(q2, shift=DOWN * 0.3), run_time=0.7)
        self.sync()

        title = Tex("What is a ", "$p$", "-value?", font_size=88)
        title[1].set_color(HI)
        title.shift(UP * 0.4)
        underline = Line(LEFT, RIGHT, color=GREY_B).set_width(title.width + 0.4).next_to(title, DOWN, buff=0.25)
        sub = Tex(
            r"one of the most used --- and most misunderstood --- numbers in science",
            font_size=34,
            color=GREY_B,
        ).next_to(underline, DOWN, buff=0.5)
        self.say(4)
        self.play(FadeOut(VGroup(coin, grid, tally, qs)), run_time=0.8)
        self.play(Write(title), GrowFromCenter(underline), run_time=1.6)
        self.wait_until(self.sentence_at(1))
        self.play(FadeIn(sub, shift=UP * 0.2), run_time=1.0)
        self.sync(1.0)
        self.play(FadeOut(VGroup(title, underline, sub)), run_time=0.8)


class S02Null(Narrated):
    def construct(self):
        prove = Tex("Prove the coin is biased?", font_size=56).shift(UP * 1.0)
        cross = Cross(prove, stroke_color=WARN, stroke_width=7)
        devil = Tex(r"Play devil's advocate instead.", font_size=56, color=HI).shift(DOWN * 0.6)
        self.say(0)
        self.play(FadeIn(prove, shift=UP * 0.3), run_time=0.8)
        self.wait_until(self.sentence_at(1) + 1.2)
        self.play(Create(cross), run_time=0.6)
        self.play(FadeIn(devil, shift=UP * 0.3), run_time=0.8)
        self.sync()

        coin = make_coin("H", 0.85).shift(LEFT * 4.0)
        boring = Tex("The boring explanation:", font_size=42, color=GREY_B)
        fair = Tex("the coin is fair", font_size=56)
        eq = MathTex(r"P(\text{heads})", "=", "0.5", font_size=64)
        text = VGroup(boring, fair, eq).arrange(DOWN, buff=0.45).shift(RIGHT * 1.6)
        self.say(1)
        self.play(FadeOut(VGroup(prove, cross, devil)), run_time=0.6)
        self.play(GrowFromCenter(coin), FadeIn(boring, shift=DOWN * 0.2), run_time=0.9)
        self.play(Write(fair), run_time=1.0)
        self.wait_until(self.seg_end - 2.2)
        self.play(Write(eq), run_time=1.2)
        self.sync()

        named = Tex("The null hypothesis:", font_size=42, color=NULL).move_to(boring)
        eq2 = MathTex("H_0", ":", r"P(\text{heads})", "=", "0.5", font_size=64).move_to(eq)
        eq2[0].set_color(NULL)
        self.say(2)
        self.wait(1.2)
        self.play(ReplacementTransform(boring, named), run_time=1.0)
        self.wait_until(self.seg_end - 1.2)
        self.play(TransformMatchingTex(eq, eq2), run_time=1.0)
        self.play(Indicate(eq2[0], color=NULL, scale_factor=1.3), run_time=0.8)
        self.sync()

        question = Tex(
            r"If ", r"$H_0$", r" were true, \\ how surprising would \\ ", r"62 heads", r" be?",
            font_size=50,
        )
        question[1].set_color(NULL)
        question[3].set_color(HI)
        box = SurroundingRectangle(question, buff=0.45, corner_radius=0.2, color=GREY_B)
        VGroup(question, box).move_to(DOWN * 0.3)
        self.say(3)
        self.play(
            FadeOut(coin),
            FadeOut(fair),
            VGroup(named, eq2).animate.arrange(RIGHT, buff=0.4).to_edge(UP, buff=0.6),
            run_time=1.0,
        )
        self.wait_until(self.sentence_at(1))
        self.play(Write(question), Create(box), run_time=2.0)
        self.sync(1.0)
        self.play(FadeOut(VGroup(named, eq2, question, box)), run_time=0.8)


class S03Simulate(Narrated):
    def construct(self):
        plot = HeadsPlot()
        xlabel = Tex("number of heads in 100 flips", font_size=30, color=GREY_B).next_to(plot.axis, DOWN, buff=0.45)
        banner = VGroup(
            make_coin("H", 0.24),
            Tex(r"A world where ", r"$H_0$", r" is true: the coin is fair", font_size=38),
        ).arrange(RIGHT, buff=0.3).to_edge(UP, buff=0.35)
        banner[1][1].set_color(NULL)

        self.say(0)
        self.play(FadeIn(banner, shift=DOWN * 0.3), run_time=1.0)
        self.wait(0.6)
        self.play(Create(plot.axis), FadeIn(xlabel), run_time=1.6)
        self.sync(0.2)

        grid = dot_grid(radius=0.065, spacing=0.17).move_to([-4.6, 1.5, 0])
        count = Integer(0, font_size=60).move_to([-2.75, 1.65, 0])
        count_lbl = Tex("heads", font_size=32, color=GREY_B).next_to(count, DOWN, buff=0.15)
        self.say(1)
        self.play(LaggedStart(*[Create(c) for c in grid], lag_ratio=0.004), run_time=0.9)

        unit = 0.22
        stack = {}
        blocks = VGroup()
        first_runs = [53, 48, 51]
        for i, k in enumerate(first_runs):
            if i > 0:
                self.play(
                    *[c.animate.set_fill(opacity=0).set_stroke(GREY_B) for c in grid],
                    FadeOut(count),
                    run_time=0.35,
                )
            self.play(LaggedStart(*fill_anims(grid, pattern_with(k, seed=100 + i)), lag_ratio=0.01), run_time=1.1)
            if i == 0:
                self.sync(0.1)
            self.say(2 + i)
            count.set_value(k)
            self.play(FadeIn(count, scale=1.4), *([FadeIn(count_lbl)] if i == 0 else []), run_time=0.4)
            level = stack.get(k, 0)
            block = plot.bar(k, unit).shift(UP * unit * level)
            stack[k] = level + 1
            start = block.copy().move_to(count.get_center())
            self.play(ReplacementTransform(start, block), run_time=0.8)
            blocks.add(block)
            self.sync(0.15)

        rng = np.random.default_rng(2024)
        results = rng.binomial(100, 0.5, size=10_000)
        results[:3] = first_runs
        cum = np.zeros((results.size + 1, 101), dtype=np.int32)
        cum[1:] = np.cumsum(np.eye(101, dtype=np.int32)[results], axis=0)

        logn = ValueTracker(np.log(3))

        def runs_so_far():
            return int(round(np.exp(logn.get_value())))

        def histogram():
            counts = cum[runs_so_far()]
            scale = min(unit, plot.H / max(int(counts.max()), 1))
            return VGroup(*[plot.bar(k, counts[k] * scale) for k in range(plot.lo, plot.hi + 1)])

        runs = Integer(3, font_size=44)
        runs.add_updater(lambda m: m.set_value(runs_so_far()))
        runs_group = VGroup(Tex("runs:", font_size=36, color=GREY_B), runs).arrange(RIGHT, buff=0.2)
        runs_group.move_to([4.6, 1.8, 0])
        runs_group[0].add_updater(lambda m: m.next_to(runs, LEFT, buff=0.2))

        self.say(5)
        self.play(FadeOut(VGroup(grid, count, count_lbl)), FadeIn(runs_group), run_time=0.6)
        bars = always_redraw(histogram)
        self.remove(blocks)
        self.add(bars)
        self.play(logn.animate.set_value(np.log(10_000)), run_time=6.5, rate_func=rate_functions.ease_in_out_sine)
        bars.clear_updaters()
        runs.clear_updaters()
        self.play(Transform(bars, plot.null_bars()), run_time=0.8)
        self.sync(0.1)

        peak_top = bars[plot.index(50)].get_top()
        common = Tex("common", font_size=36).move_to([2.2, 1.4, 0])
        common_arrow = Arrow(common.get_left(), peak_top + RIGHT * 0.12, buff=0.15, color=WHITE, stroke_width=3)
        rare_r = Tex("rare", font_size=36, color=GREY_A).move_to([4.6, -1.3, 0])
        rare_l = Tex("rare", font_size=36, color=GREY_A).move_to([-4.6, -1.3, 0])
        rare_r_arrow = Arrow(rare_r.get_bottom(), plot.base(66) + UP * 0.12, buff=0.12, color=GREY_A, stroke_width=3)
        rare_l_arrow = Arrow(rare_l.get_bottom(), plot.base(34) + UP * 0.12, buff=0.12, color=GREY_A, stroke_width=3)
        self.say(6)
        self.play(FadeIn(common), GrowArrow(common_arrow), run_time=0.8)
        self.wait_until(self.sentence_at(1))
        self.play(FadeIn(rare_l), FadeIn(rare_r), GrowArrow(rare_l_arrow), GrowArrow(rare_r_arrow), run_time=0.9)
        self.sync()

        name = Tex("the null distribution", font_size=50, color=NULL).move_to([0, 1.75, 0])
        what = Tex("what pure chance looks like", font_size=34, color=GREY_B).next_to(name, DOWN, buff=0.15)
        self.say(7)
        self.play(FadeOut(VGroup(common, common_arrow, rare_l, rare_r, rare_l_arrow, rare_r_arrow)), run_time=0.5)
        self.play(Write(name), run_time=1.2)
        self.wait_until(self.sentence_at(1))
        self.play(FadeIn(what, shift=UP * 0.15), run_time=0.8)
        self.sync(0.8)
        self.play(FadeOut(VGroup(name, what, banner, runs_group)), run_time=0.8)


class S04Tail(Narrated):
    def construct(self):
        plot = HeadsPlot()
        xlabel = Tex("number of heads in 100 flips", font_size=30, color=GREY_B).next_to(plot.axis, DOWN, buff=0.45)
        bars = plot.null_bars()
        self.add(plot.axis, xlabel, bars)

        def marker(k):
            top = bars[plot.index(k)].get_top()
            arrow = Arrow([plot.x(k), -0.85, 0], top + UP * 0.08, buff=0, color=HI, stroke_width=5)
            label = MathTex(str(k), color=HI, font_size=46).next_to(arrow, UP, buff=0.1)
            return arrow, label

        arrow62, lab62 = marker(62)
        ours = Tex("our result", font_size=30, color=HI).next_to(lab62, UP, buff=0.08)
        self.say(0)
        self.play(GrowArrow(arrow62), FadeIn(lab62, shift=DOWN * 0.2), run_time=0.9)
        self.play(FadeIn(ours), run_time=0.5)
        self.sync(0.1)

        ring = Ellipse(width=2.8, height=0.9, color=HI, stroke_width=3).move_to(plot.base(66.5) + UP * 0.18)
        self.say(1)
        self.play(Create(ring), run_time=0.9)
        self.sync(0.1)

        right_tail = [bars[plot.index(k)] for k in range(62, plot.hi + 1)]
        rest = [bars[plot.index(k)] for k in range(plot.lo, 62)]
        extreme = Tex("at least as extreme", font_size=34, color=HI).move_to([5.0, -1.3, 0])
        extreme_arrow = Arrow(extreme.get_bottom(), plot.base(65) + UP * 0.1, buff=0.1, color=HI, stroke_width=3)
        self.say(2)
        self.play(FadeOut(ring), run_time=0.4)
        self.wait(1.5)
        self.play(
            LaggedStart(*[b.animate.set_fill(HI, 1).set_stroke(HI, opacity=1) for b in right_tail], lag_ratio=0.1),
            *[b.animate.set_fill(opacity=0.35).set_stroke(opacity=0.35) for b in rest],
            run_time=1.6,
        )
        self.wait_until(self.seg_end - 2.0)
        self.play(FadeIn(extreme), GrowArrow(extreme_arrow), run_time=0.8)
        self.sync()

        eq1 = MathTex(r"P(\text{heads} \ge 62)", r"\approx", r"0.0105", r"\approx", r"1\%", font_size=50).move_to([0, 2.6, 0])
        eq1[0].set_color(HI)
        self.say(3)
        self.play(Write(eq1[0]), run_time=1.0)
        self.play(Write(eq1[1:3]), run_time=0.8)
        self.wait_until(self.seg_end - 1.0)
        self.play(Write(eq1[3:]), run_time=0.7)
        self.sync()

        arrow38, lab38 = marker(38)
        arc = CurvedArrow(ours.get_top() + UP * 0.1, lab38.get_top() + UP * 0.45, angle=TAU / 4.5, color=GREY_B, stroke_width=3)
        mirror = Tex("mirror image", font_size=32, color=GREY_B).next_to(arc, UP, buff=0.1)
        left_tail = [bars[plot.index(k)] for k in range(plot.lo, 39)]
        self.say(4)
        self.wait_until(self.seg_end - 3.4)
        self.play(Create(arc), FadeIn(mirror), run_time=1.0)
        self.play(GrowArrow(arrow38), FadeIn(lab38, shift=DOWN * 0.2), run_time=0.8)
        self.play(
            LaggedStart(*[b.animate.set_fill(HI, 1).set_stroke(HI, opacity=1) for b in reversed(left_tail)], lag_ratio=0.1),
            run_time=1.2,
        )
        self.sync()

        eq2 = MathTex("p", "=", r"P(\text{heads} \le 38)", "+", r"P(\text{heads} \ge 62)", r"\approx", "0.021", font_size=46).move_to([0, 2.6, 0])
        eq2[2].set_color(HI)
        eq2[4].set_color(HI)
        final = MathTex("p", r"\approx", "0.02", font_size=80).move_to([0, 2.45, 0])
        box = SurroundingRectangle(final, color=HI, buff=0.25, corner_radius=0.1)
        self.say(5)
        self.play(FadeOut(VGroup(arc, mirror)), ReplacementTransform(eq1, eq2), run_time=1.2)
        self.wait_until(self.sentence_at(1))
        self.play(ReplacementTransform(eq2, final), run_time=1.0)
        self.play(Create(box), run_time=0.6)
        self.sync(1.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class S05Definition(Narrated):
    def construct(self):
        title = Tex("The ", "$p$", "-value", font_size=64).to_edge(UP, buff=0.6)
        title[1].set_color(HI)
        rule = Line(LEFT, RIGHT, color=GREY_B).set_width(title.width + 0.5).next_to(title, DOWN, buff=0.2)
        self.say(0)
        self.play(Write(title), GrowFromCenter(rule), run_time=1.2)
        self.sync(0.1)

        words = Tex(
            r"The probability of seeing data ", r"\emph{at least as extreme}",
            r" \\ as what you observed, \\ assuming the ", r"null hypothesis", r" is true.",
            font_size=44,
        ).shift(UP * 0.8)
        words[1].set_color(HI)
        words[3].set_color(NULL)
        eq = MathTex("p", "=", "P(", r"\text{data at least this extreme}", r"\,\mid\,", "H_0", ")", font_size=60).shift(DOWN * 1.1)
        eq[3].set_color(HI)
        eq[5].set_color(NULL)
        self.say(1)
        self.play(Write(words), run_time=self.left() * 0.75)
        self.play(Write(eq), run_time=1.6)
        self.sync()

        bar = eq[4]
        given = SurroundingRectangle(VGroup(eq[4], eq[5]), color=NULL, buff=0.15, corner_radius=0.08)
        given_lbl = Tex(r"``given that ", r"$H_0$", r" is true''", font_size=38, color=NULL).next_to(given, DOWN, buff=0.35)
        confusion = Tex("most confusion starts here", font_size=32, color=WARN).next_to(given_lbl, DOWN, buff=0.3)
        self.say(2)
        self.play(bar.animate.set_color(HI).scale(1.6), run_time=0.6)
        self.play(bar.animate.set_color(WHITE).scale(1 / 1.6), run_time=0.5)
        self.wait_until(self.sentence_at(1))
        self.play(Create(given), FadeIn(given_lbl, shift=UP * 0.2), run_time=1.0)
        self.wait_until(self.sentence_at(2) + 0.6)
        self.play(FadeIn(confusion, shift=UP * 0.2), run_time=0.8)
        self.sync(1.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class S06Threshold(Narrated):
    def construct(self):
        plot = HeadsPlot(origin=(-5.5, 0.4), x_length=11, height=2.5, font_size=26)
        k = ValueTracker(62)

        def kk():
            return int(round(k.get_value()))

        def bars_fn():
            d = abs(kk() - 50)
            return plot.null_bars(lambda j: (HI, 1.0) if abs(j - 50) >= d else (BAR, 0.45))

        bars = always_redraw(bars_fn)
        marker = always_redraw(
            lambda: Triangle(fill_color=HI, fill_opacity=1, stroke_width=0).scale(0.13).move_to(plot.base(kk()) + DOWN * 0.62)
        )
        knum = Integer(62, font_size=36, color=HI)
        klbl = Tex("heads", font_size=32, color=HI)
        kgroup = VGroup(knum, klbl)

        def follow(m):
            knum.set_value(kk())
            m.arrange(RIGHT, buff=0.15, aligned_edge=DOWN)
            m.next_to(marker, DOWN, buff=0.12)

        kgroup.add_updater(follow)

        meter = NumberLine(
            x_range=[0, 0.3, 0.05],
            length=10,
            include_tip=False,
            numbers_to_include=[0, 0.05, 0.1, 0.2, 0.3],
            decimal_number_config={"num_decimal_places": 2},
            font_size=26,
        ).move_to([0.3, -1.95, 0])
        meter_lbl = Tex("$p$-value", font_size=32).next_to(meter, LEFT, buff=0.35)
        a_pt = meter.n2p(0.05)
        alpha_line = Line(a_pt + DOWN * 0.16, a_pt + UP * 0.3, color=HI, stroke_width=5)
        alpha_lbl = MathTex(r"\alpha = 0.05", font_size=36, color=HI).next_to(alpha_line, UP, buff=0.1)

        region = Rectangle(width=a_pt[0] - meter.n2p(0)[0], height=0.36, fill_color=HI, fill_opacity=0.25, stroke_width=0)
        region.move_to(meter.n2p(0), aligned_edge=LEFT)
        region_lbl = Tex(r"reject $H_0$", font_size=30, color=HI).move_to(meter.n2p(0.025) + DOWN * 0.8)

        p_dot = always_redraw(lambda: Dot(meter.n2p(min(two_sided_p(kk()), 0.3)), radius=0.11, color=WHITE))
        p_num = DecimalNumber(two_sided_p(62), num_decimal_places=3, font_size=46)
        p_num.add_updater(lambda m: m.set_value(two_sided_p(kk())))
        readout = VGroup(MathTex("p", "=", font_size=46), p_num).arrange(RIGHT, buff=0.2).move_to([-1.2, -3.35, 0])

        def verdict_fn():
            if two_sided_p(kk()) < 0.05:
                t = Tex("significant", font_size=44, color=HI)
            else:
                t = Tex("not significant", font_size=44, color=GREY_B)
            return t.next_to(readout, RIGHT, buff=0.7)

        verdict = always_redraw(verdict_fn)

        self.say(0)
        self.play(FadeIn(plot.axis), FadeIn(bars), FadeIn(marker), FadeIn(kgroup), run_time=1.0)
        self.play(Create(meter), FadeIn(meter_lbl), run_time=1.2)
        self.wait_until(self.seg_end - 2.0)
        self.play(GrowFromCenter(alpha_line), Write(alpha_lbl), run_time=1.0)
        self.sync()

        self.say(1)
        self.play(FadeIn(region), FadeIn(region_lbl), run_time=1.0)
        self.wait(1.4)
        self.play(FadeIn(p_dot, scale=2), FadeIn(readout), FadeIn(verdict), run_time=1.0)
        self.sync()

        self.say(2)
        self.wait(0.6)
        self.play(k.animate.set_value(53), run_time=3.0, rate_func=smooth)
        self.play(k.animate.set_value(58), run_time=1.8, rate_func=smooth)
        self.sync(0.1)

        self.say(3)
        self.play(k.animate.set_value(60), run_time=0.9)
        self.play(Indicate(readout, color=WHITE, scale_factor=1.15), run_time=0.9)
        self.sync()

        self.say(4)
        self.play(k.animate.set_value(61), run_time=0.6)
        self.play(Indicate(readout, color=HI, scale_factor=1.15), run_time=0.9)
        self.sync()

        self.say(5)
        for target in [60, 61, 60, 61]:
            self.play(k.animate.set_value(target), run_time=0.45)
            self.wait(0.35)
        self.sync(1.0)
        for m in (kgroup, p_num):
            m.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class S07TrapConditional(Narrated):
    def construct(self):
        intro = VGroup(warning_icon(0.5), Tex("Common traps", font_size=72, color=WARN)).arrange(RIGHT, buff=0.4)
        self.say(0)
        self.play(FadeIn(intro, scale=0.8), run_time=0.9)
        self.sync(0.1)

        header = trap_header(1)
        claim_l = MathTex("p = 0.02", font_size=64)
        claim_ne = MathTex(r"\ne", font_size=72, color=WARN)
        claim_r = Tex(r"a 2\% chance the coin is fair", font_size=50)
        claim = VGroup(claim_l, claim_ne, claim_r).arrange(RIGHT, buff=0.45)
        self.say(1)
        self.play(ReplacementTransform(intro, header), run_time=0.9)
        self.wait_until(self.sentence_at(1))
        self.play(Write(claim_l), run_time=0.8)
        self.wait_until(self.seg_end - 2.2)
        self.play(FadeIn(claim_r, shift=LEFT * 0.2), run_time=0.8)
        self.play(GrowFromCenter(claim_ne), run_time=0.5)
        self.sync()

        good = MathTex("p", "=", "P(", r"\text{data}", r"\,\mid\,", "H_0", ")", font_size=68).move_to([0.6, 1.3, 0])
        good[3].set_color(HI)
        good[5].set_color(NULL)
        bad = MathTex(r"\ne", "P(", "H_0", r"\,\mid\,", r"\text{data}", ")", font_size=68)
        bad.move_to([0.6, -1.0, 0]).align_to(good[1], LEFT)
        bad[0].set_color(WARN)
        bad[2].set_color(NULL)
        bad[4].set_color(HI)
        good_lbl = Tex("the probability of the data, given the hypothesis", font_size=32, color=GREY_B).next_to(good, DOWN, buff=0.3)
        bad_lbl = Tex("the probability of the hypothesis, given the data", font_size=32, color=GREY_B).next_to(bad, DOWN, buff=0.3)
        self.say(2)
        self.play(FadeOut(claim), run_time=0.5)
        self.play(Write(good), run_time=1.2)
        self.play(FadeIn(good_lbl), run_time=0.6)
        self.wait_until(self.sentence_at(1))
        self.play(
            TransformFromCopy(good[2], bad[1]),
            TransformFromCopy(good[3], bad[4], path_arc=-PI / 2),
            TransformFromCopy(good[4], bad[3]),
            TransformFromCopy(good[5], bad[2], path_arc=-PI / 2),
            TransformFromCopy(good[6], bad[5]),
            run_time=1.6,
        )
        self.play(FadeIn(bad[0], scale=1.5), FadeIn(bad_lbl), run_time=0.7)
        self.sync()

        circle = Circle(2.3, fill_color=NULL, fill_opacity=0.22, stroke_color=NULL, stroke_width=3).move_to([-3.7, -0.5, 0])
        circle_lbl = Tex("people who speak English", font_size=34, color=NULL).next_to(circle, UP, buff=0.2)
        dot = Dot([-3.1, -1.0, 0], radius=0.05, color=HI)
        dot_lbl = Tex("U.S. presidents", font_size=32, color=HI).move_to([-0.6, -2.9, 0])
        dot_arrow = Arrow(dot_lbl.get_left(), dot.get_center(), buff=0.1, color=HI, stroke_width=3)
        self.say(3)
        self.play(FadeOut(VGroup(good, bad, good_lbl, bad_lbl)), run_time=0.6)
        self.play(DrawBorderThenFill(circle), FadeIn(circle_lbl), run_time=1.2)
        self.play(FadeIn(dot, scale=4), GrowArrow(dot_arrow), FadeIn(dot_lbl), run_time=0.8)
        self.sync(0.1)

        eq1 = MathTex(r"P(\text{English} \mid \text{president})", r"\approx 1", font_size=46).move_to([3.4, 1.0, 0])
        eq2 = MathTex(r"P(\text{president} \mid \text{English})", r"\approx 0", font_size=46).move_to([3.4, -0.6, 0])
        eq2[1].set_color(WARN)
        self.say(4)
        self.play(Write(eq1[0]), run_time=1.4)
        self.play(Flash(dot, color=HI, line_length=0.25, flash_radius=0.2), run_time=0.8)
        self.wait_until(self.seg_end - 1.0)
        self.play(Write(eq1[1]), run_time=0.6)
        self.sync()

        self.say(5)
        self.play(Write(eq2[0]), run_time=1.4)
        self.play(Indicate(circle, color=NULL, scale_factor=1.05), run_time=1.0)
        self.wait_until(self.seg_end - 1.0)
        self.play(Write(eq2[1]), run_time=0.6)
        self.sync(1.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class S08TrapEffectSize(Narrated):
    def construct(self):
        header = trap_header(2)
        claim = Tex(r"A small $p$-value $\ne$ a big effect", font_size=44).next_to(header, DOWN, buff=0.3, aligned_edge=LEFT)
        self.say(0)
        self.play(FadeIn(header), run_time=0.7)
        self.wait_until(self.sentence_at(1))
        self.play(Write(claim), run_time=1.4)
        self.sync()

        lo, hi = 0.35, 0.65
        ax = curve_axes(lo, hi)
        xlabel = Tex("fraction of flips that land heads", font_size=30, color=GREY_B).next_to(ax.x_axis, DOWN, buff=0.5)
        logn = ValueTracker(np.log(100))

        def n():
            return float(np.exp(logn.get_value()))

        def sd():
            return 0.5 / math.sqrt(n())

        def p_of_n():
            z = (0.01 * n() - 0.5) / (0.5 * math.sqrt(n()))
            return math.erfc(z / math.sqrt(2))

        body = always_redraw(lambda: bell_area(ax, 0.5, sd(), lo, hi, NULL, 0.2))
        curve = always_redraw(lambda: bell_curve(ax, 0.5, sd(), lo, hi))
        tails = always_redraw(
            lambda: VGroup(bell_area(ax, 0.5, sd(), lo, 0.49, HI, 0.6), bell_area(ax, 0.5, sd(), 0.51, hi, HI, 0.6))
        )
        mark = DashedLine(ax.c2p(0.51, 0), ax.c2p(0.51, 1.12), color=HI, stroke_width=3)
        mirror = DashedLine(ax.c2p(0.49, 0), ax.c2p(0.49, 1.12), color=HI, stroke_width=2, stroke_opacity=0.5)
        mark_lbl = MathTex(r"51\%", font_size=36, color=HI).next_to(mark, UP, buff=0.1)

        flips = Integer(100, font_size=40)
        heads = Integer(51, font_size=40)
        p_num = DecimalNumber(p_of_n(), num_decimal_places=3, font_size=40)
        flips.add_updater(lambda m: m.set_value(int(round(n()))))
        heads.add_updater(lambda m: m.set_value(int(round(0.51 * n()))))
        p_num.add_updater(lambda m: m.set_value(p_of_n()).set_color(HI if p_of_n() < 0.05 else WHITE))
        labels = VGroup(Tex("flips:", font_size=36), Tex("heads:", font_size=36), MathTex("p =", font_size=40))
        values = VGroup(flips, heads, p_num)
        panel = VGroup(*[VGroup(lbl, val) for lbl, val in zip(labels, values)])
        for i, row in enumerate(panel):
            row[0].move_to([3.3, 2.95 - 0.6 * i, 0], aligned_edge=RIGHT)
            row[1].next_to(row[0], RIGHT, buff=0.25)
            row[1].add_updater(lambda m, row=row: m.next_to(row[0], RIGHT, buff=0.25))

        self.say(1)
        self.play(Create(ax.x_axis), FadeIn(xlabel), run_time=1.0)
        self.play(Create(curve), FadeIn(body), run_time=1.0)
        self.play(Create(mark), Create(mirror), FadeIn(mark_lbl), run_time=0.8)
        self.wait_until(self.sentence_at(1))
        self.play(FadeIn(panel[0]), FadeIn(panel[1]), run_time=0.8)
        self.play(FadeIn(tails), run_time=0.8)
        self.play(FadeIn(panel[2]), run_time=0.6)
        self.sync()

        ghost = bell_curve(ax, 0.5, 0.05, lo, hi, color=GREY_B, width=2)
        ghost = DashedVMobject(ghost, num_dashes=90)
        self.say(2)
        self.add(ghost)
        self.play(logn.animate.set_value(np.log(10_000)), run_time=max(self.left() - 0.4, 3.5), rate_func=smooth)
        self.sync(0.2)

        def half_width(sd_value, height):
            return sd_value * math.sqrt(-2 * math.log(height))

        y_wide, y_narrow = 0.25, 0.5
        w_wide = half_width(0.05, y_wide)
        w_narrow = half_width(0.005, y_narrow)
        wide = DoubleArrow(ax.c2p(0.5 - w_wide, y_wide), ax.c2p(0.5 + w_wide, y_wide), buff=0, color=GREY_B, stroke_width=3, tip_length=0.18)
        narrow = DoubleArrow(ax.c2p(0.5 - w_narrow, y_narrow), ax.c2p(0.5 + w_narrow, y_narrow), buff=0, color=WHITE, stroke_width=3, tip_length=0.12)
        wide_lbl = Tex("100 flips", font_size=30, color=GREY_B).next_to(wide, LEFT, buff=0.15)
        narrow_lbl = Tex("10,000 flips", font_size=30).next_to(narrow, RIGHT, buff=0.6).shift(UP * 0.3)
        same = Tex(r"same 1\% bias", font_size=34, color=HI).next_to(mark_lbl, RIGHT, buff=0.3)
        self.say(3)
        self.play(FadeIn(same, shift=LEFT * 0.2), Indicate(mark_lbl, color=HI), run_time=1.0)
        self.wait_until(self.sentence_at(1) + 0.8)
        self.play(GrowFromCenter(wide), FadeIn(wide_lbl), run_time=0.8)
        self.play(GrowFromCenter(narrow), FadeIn(narrow_lbl), run_time=0.8)
        self.sync(1.0)
        for m in [flips, heads, p_num, *[row[1] for row in panel]]:
            m.clear_updaters()
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class S09TrapAbsence(Narrated):
    def construct(self):
        header = trap_header(3)
        claim = MathTex(r"\text{large } p", r"\;\;\nRightarrow\;\;", r"H_0 \text{ is true}", font_size=56).move_to([0.8, 2.3, 0])
        claim[1].set_color(WARN)
        claim[2].set_color(NULL)
        self.say(0)
        self.play(FadeIn(header), run_time=0.7)
        self.wait_until(self.sentence_at(1))
        self.play(Write(claim), run_time=1.6)
        self.sync()

        lo, hi = 0.3, 0.7
        ax = curve_axes(lo, hi, origin=(-5.5, -2.4), height=3.0)
        xlabel = Tex("fraction of 100 flips that land heads", font_size=30, color=GREY_B).next_to(ax.x_axis, DOWN, buff=0.5)
        fair_fill = bell_area(ax, 0.5, 0.05, lo, hi, NULL, 0.25)
        fair = bell_curve(ax, 0.5, 0.05, lo, hi)
        mark = DashedLine(ax.c2p(0.51, 0), ax.c2p(0.51, 1.04), color=WHITE, stroke_width=3)
        mark_lbl = Tex("51 heads", font_size=32).next_to(mark, UP, buff=0.1)
        meh = Tex(r"not surprising under $H_0$", font_size=32, color=GREY_B).next_to(mark_lbl, RIGHT, buff=0.4)
        self.say(1)
        self.play(Create(ax.x_axis), FadeIn(xlabel), Create(fair), FadeIn(fair_fill), run_time=1.2)
        self.play(Create(mark), FadeIn(mark_lbl), run_time=0.6)
        self.play(FadeIn(meh), run_time=0.6)
        self.sync(0.1)

        fair_lbl = Tex("fair coin", font_size=34, color=NULL).move_to(ax.c2p(0.385, 0.55))
        biased = bell_curve(ax, 0.51, 0.05, lo, hi, color=HI)
        biased_lbl = Tex(r"51\% coin", font_size=34, color=HI).move_to(ax.c2p(0.63, 0.55))
        cant = Tex("100 flips can't tell them apart", font_size=36).move_to([0.8, 1.6, 0])
        self.say(2)
        self.play(FadeOut(meh), FadeIn(fair_lbl), run_time=0.7)
        self.wait_until(self.sentence_at(1))
        self.play(Create(biased), FadeIn(biased_lbl), run_time=1.2)
        self.play(FadeIn(cant, shift=DOWN * 0.2), run_time=0.8)
        self.sync()

        verdict = Tex("undetermined", font_size=72, color=HI)
        verdict_box = SurroundingRectangle(verdict, color=HI, buff=0.3, corner_radius=0.12)
        VGroup(verdict, verdict_box).move_to(claim)
        self.say(3)
        self.play(FadeOut(cant), FadeOut(claim), run_time=0.5)
        self.wait_until(self.seg_end - 1.3)
        self.play(Write(verdict), Create(verdict_box), run_time=1.1)
        self.sync(1.2)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class S10TrapManyTests(Narrated):
    def construct(self):
        header = trap_header(4)
        heads = [47, 52, 55, 49, 44, 50, 53, 58, 46, 51, 48, 56, 43, 61, 54, 50, 45, 52, 57, 49]
        lucky = heads.index(61)
        xs = [-5.6, -4.1, -2.6, -1.1, 0.4]
        ys = [1.9, 0.45, -1.0, -2.45]
        coins, p_labels = VGroup(), VGroup()
        for i, k in enumerate(heads):
            pos = np.array([xs[i % 5], ys[i // 5], 0])
            c = make_coin(str(i + 1), 0.34, color=GREY_C, rim=GREY_A)
            c[2].scale(0.75)
            coins.add(c.move_to(pos))
            p = two_sided_p(k)
            p_text = f"{p:.3f}" if p < 0.05 else f"{p:.2f}"
            p_labels.add(MathTex(f"p = {p_text}", font_size=26, color=GREY_A).next_to(c, DOWN, buff=0.12))
        caption = Tex(r"20 fair coins, \\ 100 flips each", font_size=36, color=GREY_B).move_to([4.4, 2.2, 0])

        self.say(0)
        self.play(FadeIn(header), run_time=0.7)
        self.wait_until(self.sentence_at(1))
        self.play(LaggedStart(*[GrowFromCenter(c) for c in coins], lag_ratio=0.06), FadeIn(caption), run_time=2.0)
        self.play(LaggedStart(*[FadeIn(lbl, shift=UP * 0.1) for lbl in p_labels], lag_ratio=0.08), run_time=2.4)
        self.sync()

        winner = coins[lucky]
        ring = SurroundingRectangle(VGroup(winner, p_labels[lucky]), color=HI, buff=0.12, corner_radius=0.1)
        formula = VGroup(
            MathTex(r"P(\text{at least one } p < 0.05)", font_size=36),
            MathTex(r"= 1 - 0.95^{20}", font_size=40),
            MathTex(r"\approx 0.64", font_size=48, color=HI),
        ).arrange(DOWN, buff=0.25, aligned_edge=LEFT).move_to([4.4, 0.4, 0])
        self.say(1)
        self.play(
            winner[0].animate.set_fill(HI).set_stroke(YELLOW_B),
            p_labels[lucky].animate.set_color(HI),
            Create(ring),
            run_time=0.9,
        )
        self.play(Write(formula[0]), run_time=1.2)
        self.play(Write(formula[1]), run_time=0.9)
        self.wait_until(self.seg_end - 1.0)
        self.play(Write(formula[2]), run_time=0.7)
        self.sync()

        paper = VGroup(
            Tex(r"\textbf{BREAKING}", font_size=30, color=WARN),
            Tex(r"Coin \#14 is biased!", font_size=44, color=BLACK),
            MathTex(r"(p = 0.035)", font_size=32, color=GREY_D),
        ).arrange(DOWN, buff=0.15)
        card = RoundedRectangle(width=paper.width + 0.6, height=paper.height + 0.5, corner_radius=0.1, fill_color=GREY_A, fill_opacity=1, stroke_width=0)
        headline = VGroup(card, paper).rotate(4 * DEGREES).move_to([4.4, -2.2, 0])
        self.say(2)
        self.play(FadeIn(headline, scale=1.4), run_time=0.8)
        self.sync(1.4)
        self.play(FadeOut(Group(*self.mobjects)), run_time=0.8)


class S11Summary(Narrated):
    def construct(self):
        title = Tex("So, what is a ", "$p$", "-value?", font_size=60).to_edge(UP, buff=0.5)
        title[1].set_color(HI)
        self.say(0)
        self.play(Write(title), run_time=1.2)
        self.sync(0.1)

        plot = HeadsPlot(origin=(-4.0, -1.0), x_length=8, height=2.2, font_size=24)
        bars = plot.null_bars(lambda k: (HI, 1.0) if abs(k - 50) >= 12 else (BAR, 0.45))
        surprise = Tex("a measure of ", r"\emph{surprise}", font_size=50).move_to([0, 2.05, 0])
        surprise[1].set_color(HI)
        eq = MathTex("p", "=", "P(", r"\text{data at least this extreme}", r"\,\mid\,", "H_0", ")", font_size=48).move_to([0, -2.4, 0])
        eq[3].set_color(HI)
        eq[5].set_color(NULL)
        self.say(1)
        self.play(FadeIn(plot.axis), FadeIn(bars), run_time=0.9)
        self.play(Write(surprise), run_time=1.0)
        self.wait_until(self.sentence_at(1))
        self.play(Write(eq), run_time=1.8)
        self.sync()

        good = Tex(r"small $p$", r"\quad$\Rightarrow$\quad", r"worth a closer look", font_size=50)
        bad = Tex(r"small $p$", r"\quad$\nRightarrow$\quad", r"a final verdict", font_size=50)
        good[1].set_color(HI)
        bad[1].set_color(WARN)
        lines = VGroup(good, bad).arrange(DOWN, buff=0.6, aligned_edge=LEFT).move_to([0, 0.1, 0])
        self.say(2)
        self.play(FadeOut(VGroup(plot.axis, bars, surprise, eq)), run_time=0.7)
        self.play(FadeIn(good, shift=UP * 0.2), run_time=0.9)
        self.wait_until(self.sentence_at(1))
        self.play(FadeIn(bad, shift=UP * 0.2), run_time=0.9)
        self.sync(2.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.2)
        self.wait(1.0)


SCENES = [
    S01Hook,
    S02Null,
    S03Simulate,
    S04Tail,
    S05Definition,
    S06Threshold,
    S07TrapConditional,
    S08TrapEffectSize,
    S09TrapAbsence,
    S10TrapManyTests,
    S11Summary,
]
