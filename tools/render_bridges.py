#!/usr/bin/env python
"""Offscreen renders of the generated bridge models, for visual QA.

    python tools/render_bridges.py --models Dataset/BridgeModels --out renders
    python tools/render_bridges.py --views side iso --contact-sheet
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import rendering

# One colour per component.
COLORS = {
    "deck":       (0.82, 0.82, 0.86),
    "railings":   (0.95, 0.65, 0.25),
    "piers":      (0.85, 0.35, 0.35),
    "abutments":  (0.40, 0.70, 0.45),
    "ground":     (0.66, 0.60, 0.48),
}

# azimuth 0 looks along -X (down the road); azimuth 90 looks broadside along -Y.
VIEWS = {
    "iso":   (38.0, 20.0),
    "side":  (90.0, 4.0),     # spans, piers, clearance, soffit line
    "front": (0.0, -6.0),     # width, girder/column count
    "under": (62.0, -22.0),   # soffit and pier bases
}


def load_meshes(bridge_dir: Path, with_ground: bool):
    """Component meshes in the same sorted order the scene XML uses."""
    manifest = bridge_dir / "components.json"
    if manifest.is_file():
        names = json.loads(manifest.read_text())["order"]
    else:
        names = [f.stem for f in sorted(bridge_dir.glob("*.obj"))]

    meshes = []
    for name in names:
        if name == "ground" and not with_ground:
            continue
        f = bridge_dir / f"{name}.obj"
        if not f.is_file():
            continue
        m = o3d.io.read_triangle_mesh(str(f))
        if len(m.vertices) == 0:
            continue
        m.compute_vertex_normals()
        m.paint_uniform_color(COLORS.get(name, (0.6, 0.6, 0.6)))
        meshes.append((name, m))
    return meshes


def render(meshes, out_path: Path, azim: float, elev: float, size=(1000, 640),
           bg=(1.0, 1.0, 1.0, 1.0)):
    """Filament offscreen render. Works headless via EGL."""
    w, h = size
    r = rendering.OffscreenRenderer(w, h)
    r.scene.set_background(list(bg))
    r.scene.scene.set_sun_light([-0.4, -0.5, -0.75], [1.0, 1.0, 1.0], 92000)
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.set_indirect_light_intensity(38000)

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_roughness = 0.85

    # frame on the bridge, not the ground
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    for name, m in meshes:
        r.scene.add_geometry(name, m, mat)
        if name in ("ground",):
            continue
        b = m.get_axis_aligned_bounding_box()
        lo = np.minimum(lo, b.get_min_bound())
        hi = np.maximum(hi, b.get_max_bound())

    if not np.isfinite(lo).all():
        r.scene.clear_geometry()
        return False

    centre = (lo + hi) / 2.0
    extent = float(np.linalg.norm(hi - lo))
    dist = extent * 1.05

    a, e = math.radians(azim), math.radians(elev)
    eye = centre + dist * np.array([
        math.cos(e) * math.cos(a),
        math.cos(e) * math.sin(a),
        math.sin(e),
    ])
    r.setup_camera(35.0, centre, eye, [0, 0, 1])

    img = r.render_to_image()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_image(str(out_path), img, 9)
    r.scene.clear_geometry()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Dataset/BridgeModels")
    ap.add_argument("--out", default="Dataset/Renders")
    ap.add_argument("--views", nargs="+", default=["side"], choices=list(VIEWS))
    ap.add_argument("--no-ground", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--width", type=int, default=1000)
    ap.add_argument("--height", type=int, default=640)
    args = ap.parse_args()

    models = Path(args.models)
    dirs = sorted([d for d in models.iterdir() if d.is_dir()],
                  key=lambda d: int(d.name.split("_")[-1]) if d.name.split("_")[-1].isdigit() else 0)
    if args.limit:
        dirs = dirs[:args.limit]

    done = 0
    for d in dirs:
        meshes = load_meshes(d, with_ground=not args.no_ground)
        if not meshes:
            print(f"  skip {d.name}: no meshes")
            continue
        for view in args.views:
            azim, elev = VIEWS[view]
            out = Path(args.out) / view / f"{d.name}.png"
            if render(meshes, out, azim, elev, (args.width, args.height)):
                done += 1
        print(f"  {d.name}: {len(meshes)} components -> {len(args.views)} view(s)")
    print(f"\nrendered {done} images into {args.out}")


if __name__ == "__main__":
    main()
