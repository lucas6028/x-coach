"""Fitness-AQA experiments: does depth matter for in-the-wild form classification?

Companion to the Fit3D / REHAB24-6 depth-bottleneck studies (``src/fit3d``,
``src/rehab24``). Those datasets carry mocap ground truth, so depth error can be
*measured*. Fitness-AQA has no 3D truth, so the evidence here is downstream only:
identical cue features, identical splits, identical classifier -- the single thing
that changes between arms is whether the depth channel is present.
"""
