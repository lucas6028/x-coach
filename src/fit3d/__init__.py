"""Fit3D (AIFit, CVPR'21) loaders and experiments.

Fit3D ships mocap-grade 3D ground truth (``joints3d_25``), 4 synchronised camera
views with calibration, SMPLX meshes, and per-action repetition annotations. This
package exposes that data plus two experiments that exploit the 3D ground truth:

* ``view_dependence`` (experiment 2): how far the 2D squat-rule readings drift
  across camera views relative to the view-invariant 3D truth.
* ``depth_eval`` (experiment 1): how much true depth a monocular 3D method
  (NLF / lifting / pseudo-3D) recovers, measured against the mocap ground truth.
"""
