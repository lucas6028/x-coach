"""pyrender turntable -- ``Renderer`` for ``loop.run_one``.

The Phase 0 render (``data/models/nlf/phase0_6_turntable.py``): the SMPL mesh in the body frame
(pelvis at the origin, y up, from ``geometry.to_body_frame``), an orthographic camera centred on
the body, a directional light at the camera, and 24 views spun about the vertical axis. Angle 0
is the camera's own view; angle 90 (tile 6) is side-on. Each tile carries two faint horizontal
lines at the hip and knee JOINT-CENTRE heights so depth reads at a glance side-on. They are
deliberately unlabelled: joint centres level is not the coaching "parallel" cutoff (hip crease
vs top of the knee), and a label would invite a depth verdict the rules don't make.

The SMPL triangle list is read from the local SMPLify pickle and used only here, on the PC; only
rendered pixels leave it (plan decision 5). pyrender draws through a hidden pyglet window, so the
worker must run in a logged-in desktop session.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh

from src.nlf_worker import geometry
from src.nlf_worker.smpl_faces import load_smpl_faces

# BGR, as drawn by OpenCV onto the rendered tile.
KNEE_LINE_BGR = (60, 160, 60)
HIP_LINE_BGR = (60, 60, 200)
_BASE_COLOR = (0.55, 0.68, 0.82, 1.0)


class TurntableRenderer:
    def __init__(
        self,
        smpl_pkl: str | Path,
        *,
        tile_px: int = 384,
        angles: int = 24,
        webp_quality: int = 80,
    ) -> None:
        self.faces = load_smpl_faces(smpl_pkl)
        self.tile_px = tile_px
        self.angles = angles
        self.webp_quality = webp_quality
        self._offscreen: pyrender.OffscreenRenderer | None = None
        self._material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=_BASE_COLOR, metallicFactor=0.0, roughnessFactor=0.8
        )

    def _renderer(self) -> pyrender.OffscreenRenderer:
        # Created on first use (after any CUDA work, as in Phase 0) and reused for every strip.
        if self._offscreen is None:
            self._offscreen = pyrender.OffscreenRenderer(self.tile_px, self.tile_px)
        return self._offscreen

    def close(self) -> None:
        if self._offscreen is not None:
            self._offscreen.delete()
            self._offscreen = None

    def render_tiles(
        self,
        vertices_mm: np.ndarray | None,
        joints_mm: np.ndarray,
        basis: np.ndarray,
        framing: dict[str, float] | None = None,
    ) -> list[np.ndarray]:
        """``angles`` BGR uint8 tiles, angle 0 first. Inputs are raw camera-coordinate mm;
        ``framing`` is the job-wide ``{cy, mag}`` (``None``: frame this mesh alone)."""
        if vertices_mm is None:
            raise ValueError("render: no mesh for this key frame (nobody detected)")
        vertices_mm = np.asarray(vertices_mm, dtype=float)
        joints_mm = np.asarray(joints_mm, dtype=float)
        if vertices_mm.shape[0] <= int(self.faces.max()) or not np.isfinite(vertices_mm).all():
            raise ValueError(
                f"render: expected a finite SMPL mesh with >{int(self.faces.max())} vertices, "
                f"got shape {vertices_mm.shape}"
            )
        pelvis = joints_mm[geometry.PELVIS]
        verts = geometry.to_body_frame(vertices_mm, pelvis, basis)
        joints = geometry.to_body_frame(joints_mm, pelvis, basis)
        view = framing if framing is not None else geometry.turntable_framing([verts])
        cy, mag = float(view["cy"]), float(view["mag"])
        lines = geometry.guide_line_heights(joints)
        size = self.tile_px
        knee_px = geometry.ortho_pixel_y(lines["knee_y"], cy, mag, size)
        hip_px = geometry.ortho_pixel_y(lines["hip_y"], cy, mag, size)

        mesh = pyrender.Mesh.from_trimesh(
            trimesh.Trimesh(verts, self.faces, process=False), material=self._material, smooth=True
        )
        scene = pyrender.Scene(bg_color=(1.0, 1.0, 1.0, 1.0), ambient_light=(0.35, 0.35, 0.35))
        node = scene.add(mesh)
        cam_pose = np.eye(4)
        cam_pose[:3, 3] = [0.0, cy, 3.0]
        scene.add(pyrender.OrthographicCamera(xmag=mag, ymag=mag), pose=cam_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=cam_pose)

        renderer = self._renderer()
        tiles = []
        for a in range(self.angles):
            th = 2.0 * np.pi * a / self.angles
            pose = np.eye(4)
            pose[:3, :3] = [[np.cos(th), 0.0, np.sin(th)], [0.0, 1.0, 0.0], [-np.sin(th), 0.0, np.cos(th)]]
            scene.set_pose(node, pose)
            color, _ = renderer.render(scene)
            tile = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.line(tile, (0, knee_px), (size - 1, knee_px), KNEE_LINE_BGR, 1)
            cv2.line(tile, (0, hip_px), (size - 1, hip_px), HIP_LINE_BGR, 1)
            tiles.append(tile)
        return tiles

    def render_strip(
        self,
        vertices_mm: np.ndarray | None,
        joints_mm: np.ndarray,
        basis: np.ndarray,
        framing: dict[str, float] | None = None,
    ) -> bytes:
        """WebP bytes of all tiles side by side (``angles * tile_px`` wide, ``tile_px`` tall)."""
        strip = np.concatenate(self.render_tiles(vertices_mm, joints_mm, basis, framing), axis=1)
        ok, buf = cv2.imencode(".webp", strip, [cv2.IMWRITE_WEBP_QUALITY, self.webp_quality])
        if not ok:
            raise RuntimeError("render: WebP encoding failed")
        return buf.tobytes()
