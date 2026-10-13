"""Narration script for the p-value explainer video.

NARRATION maps each Manim scene to a list of segments. A segment is a list of
sentences that are voiced back to back; the scene code starts each segment with
``self.say(i)`` so the animation can be timed against it. A sentence is either a
plain string (shown and spoken as-is) or a ``(caption, spoken)`` pair when the
TTS needs a spelled-out form of numbers or symbols.
"""

NARRATION = {
    "S01Hook": [
        ["A friend hands you a coin, and swears that it's fair."],
        [("To check, you flip it 100 times.", "To check, you flip it one hundred times.")],
        [("It lands heads 62 times.", "It lands heads sixty-two times.")],
        ["Is that suspicious?", "Or is it just the kind of luck you'd expect?"],
        [
            "The tool statisticians reach for here is called a p-value.",
            "It's one of the most used, and most misunderstood, numbers in all of science.",
        ],
    ],
    "S02Null": [
        ["Here's the trick.", "Instead of trying to prove the coin is biased, we play devil's advocate."],
        [(
            "We assume the boring explanation: the coin really is fair, with exactly a 50% chance of heads.",
            "We assume the boring explanation: the coin really is fair, with exactly a fifty percent chance of heads.",
        )],
        [(
            "This boring assumption has a name: the null hypothesis, written H₀.",
            "This boring assumption has a name: the null hypothesis, written H naught.",
        )],
        ["Then we ask one question.", "If the null hypothesis were true, how surprising would our data be?"],
    ],
    "S03Simulate": [
        ["To find out, imagine running the whole experiment again and again, in a world where the coin is perfectly fair."],
        [("Each run, we flip 100 times and count the heads.", "Each run, we flip one hundred times, and count the heads.")],
        [("53.", "Fifty-three.")],
        [("48.", "Forty-eight.")],
        [("51.", "Fifty-one.")],
        ["Stack up thousands of these runs, and a shape emerges."],
        [
            ("Most runs land near 50 heads.", "Most runs land near fifty heads."),
            ("Results far from 50 get rarer and rarer.", "Results far from fifty get rarer, and rarer."),
        ],
        ["This is the null distribution.", "It's a picture of what pure chance looks like."],
    ],
    "S04Tail": [
        [("Now, where does our 62 fall?", "Now, where does our sixty-two fall?")],
        ["Way out here, in the tail."],
        ["The p-value is the fraction of this fair-coin world that produces a result at least as extreme as ours."],
        [("62 or more heads happens only about 1% of the time.", "Sixty-two or more heads happens only about one percent of the time.")],
        [(
            "But a coin biased toward tails would be just as suspicious, so we also count the mirror-image tail: 38 heads or fewer.",
            "But a coin biased toward tails would be just as suspicious, so we also count the mirror image tail: thirty-eight heads, or fewer.",
        )],
        [
            ("Together, that's about 2%.", "Together, that's about two percent."),
            ("Our p-value is 0.02.", "Our p-value is zero point zero two."),
        ],
    ],
    "S05Definition": [
        ["So here's the definition, in full."],
        ["The p-value is the probability of seeing data at least as extreme as what you observed, assuming the null hypothesis is true."],
        [
            "Notice that little vertical bar.",
            ("It means everything is computed inside the world where H₀ holds.", "It means everything is computed inside the world where H naught holds."),
            "Keep an eye on it, because it's the source of most of the confusion.",
        ],
    ],
    "S06Threshold": [
        [(
            "In practice, people choose a cutoff in advance, called alpha, usually 0.05.",
            "In practice, people choose a cutoff in advance, called alpha. Usually, point oh five.",
        )],
        ["If the p-value falls below alpha, the result is called statistically significant, and we reject the null hypothesis."],
        ["Watch what happens as we slide the number of heads."],
        [
            ("At 60 heads, the p-value is about 0.057.", "At sixty heads, the p-value is about point oh five seven."),
            "Not significant.",
        ],
        [
            ("At 61, it drops to 0.035.", "At sixty-one, it drops to point oh three five."),
            "Significant.",
        ],
        [
            "One extra head flips the verdict.",
            ("The line at 0.05 is a convention, not a law of nature.", "The line at point oh five is a convention, not a law of nature."),
        ],
    ],
    "S07TrapConditional": [
        ["Now for the traps."],
        [
            "Trap number one.",
            ("A p-value of 0.02 does not mean there's a 2% chance the coin is fair.", "A p-value of zero point zero two does not mean there's a two percent chance the coin is fair."),
        ],
        [
            "The p-value is the probability of the data, given the hypothesis.",
            "Not the probability of the hypothesis, given the data.",
        ],
        ["And flipping a conditional can change it enormously."],
        [("The chance that someone speaks English, given they're a U.S. president, is nearly 1.", "The chance that someone speaks English, given they're an American president, is nearly one.")],
        [("But the chance that someone is a U.S. president, given they speak English, is almost zero.", "But the chance that someone is an American president, given they speak English, is almost zero.")],
    ],
    "S08TrapEffectSize": [
        ["Trap number two.", "A small p-value does not mean a big effect."],
        [
            ("Picture a coin that lands heads 51% of the time.", "Picture a coin that lands heads fifty-one percent of the time."),
            ("Flip it 100 times, get 51 heads, and the p-value is about 0.92.", "Flip it a hundred times, get fifty-one heads, and the p-value is about point nine two."),
            "Nothing to see.",
        ],
        [(
            "But flip it 10,000 times, get 5,100 heads, and the p-value drops below 0.05.",
            "But flip it ten thousand times, get fifty-one hundred heads, and the p-value drops below point oh five.",
        )],
        [
            "Same tiny bias.",
            "With more data, the null distribution gets narrower, so even a small deviation ends up in the tail.",
        ],
    ],
    "S09TrapAbsence": [
        ["Trap number three.", "A large p-value does not prove the null hypothesis."],
        ["It only means the data aren't surprising under it."],
        ["Maybe the coin is fair.", "Or maybe it's slightly biased, and we just didn't flip it enough to tell."],
        ["The honest conclusion is: undetermined."],
    ],
    "S10TrapManyTests": [
        [
            "Trap number four.",
            ("Even when every null hypothesis is true, p-values below 0.05 still show up, about one time in twenty.", "Even when every null hypothesis is true, p-values below point oh five still show up, about one time in twenty."),
        ],
        [(
            "Test 20 perfectly fair coins, and there's a 64% chance at least one of them looks significant.",
            "Test twenty perfectly fair coins, and there's a sixty-four percent chance at least one of them looks significant.",
        )],
        ["Run enough tests, and chance alone will hand you a discovery."],
    ],
    "S11Summary": [
        ["So, what is a p-value?"],
        ["It's a measure of surprise.", "It tells you how strange your data would look if the boring explanation were true."],
        ["A small p-value means chance alone would rarely do this.", "That's a reason to look closer, not a final verdict."],
    ],
}

SCENE_ORDER = list(NARRATION)
