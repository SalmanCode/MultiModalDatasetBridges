#!/usr/bin/env python
"""Real vs synthetic comparison figure.

Figure 1: real bridges with their closest synthetic match.
Figure 2: v1.0, v2.0 and real distributions.

    python tools/compare_real_synthetic.py --out Dataset/Figures
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

REAL_DIR = Path(os.environ.get("REAL_BRIDGE_DIR", "real_bridges/measurements/out"))
FAMILY_COLOR = {
    "bamberg": "#4C72B0", "cambridge": "#DD8452",
    "japan": "#55A868", "saeed": "#C44E52",
}


# --------------------------------------------------------------- utilities
def normalise(p: np.ndarray) -> np.ndarray:
    """Centre and scale to the unit sphere -- the same treatment the real
    clouds got, so shape is compared and absolute size is not."""
    p = p - p.mean(0)
    r = np.linalg.norm(p, axis=1).max()
    return p / r if r > 0 else p


def load_real():
    out = []
    for f in sorted(REAL_DIR.glob("*/primary_parameter.json")):
        d = json.loads(f.read_text())
        npy = next(f.parent.glob("*_8192.npy"), None)
        if npy is None:
            continue
        pts = np.load(npy)[:, :3].astype(np.float64)
        L, W = d["total_length_m"], d["width_m"]
        H = abs(d.get("bridge_clearance_height_m") or 4.0) + 1.2 + 1.0
        out.append(dict(
            bid=d["bridge_id"], fam=d["bridge_id"].rsplit("_", 1)[0],
            L=L, W=W, lw=L / W, lr=len_over_r(L, W, H),
            spans=d.get("num_spans", 1), pts=normalise(pts),
        ))
    return out


def len_over_r(L, W, H):
    """Analytic len/R for an L x W x H box, R being half the body diagonal."""
    return 2.0 / math.sqrt(1.0 + (W / L) ** 2 + (H / L) ** 2)


def sample_synthetic(bridge_dir: Path, n=8192, with_ground=False):
    """Poisson-disk sample the structural components to n points."""
    manifest = bridge_dir / "components.json"
    names = (json.loads(manifest.read_text())["order"] if manifest.is_file()
             else [f.stem for f in sorted(bridge_dir.glob("*.obj"))])
    merged = o3d.geometry.TriangleMesh()
    for name in names:
        if name == "ground" and not with_ground:
            continue
        f = bridge_dir / f"{name}.obj"
        if f.is_file():
            merged += o3d.io.read_triangle_mesh(str(f))
    if len(merged.vertices) == 0:
        return None
    pcd = merged.sample_points_poisson_disk(n, init_factor=3)
    return normalise(np.asarray(pcd.points))


def scatter3d(ax, p, color, size=1.1):
    """Broadside-ish view: x along the road, y across, z up."""
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=size, c=color, lw=0, alpha=0.75,
               rasterized=True)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=16, azim=-68)
    ax.set_axis_off()


# ------------------------------------------------------------- figure one
def figure_pairs(real, synth, out: Path, n_pairs=8):
    """Each real bridge beside its nearest synthetic match in shape space."""
    # Match on (L/W, len/R, span count), all standardised so no axis dominates.
    def vec(d):
        return np.array([math.log(d["lw"]), d["lr"] * 4.0, min(d["spans"], 5) * 0.35])

    R = np.array([vec(d) for d in real])
    S = np.array([vec(d) for d in synth])
    mu, sd = R.mean(0), R.std(0) + 1e-9
    Rn, Sn = (R - mu) / sd, (S - mu) / sd

    # Spread the shown pairs over the four families.
    chosen, used = [], set()
    order = sorted(range(len(real)), key=lambda i: real[i]["fam"])
    per_fam = {}
    for i in order:
        fam = real[i]["fam"]
        if per_fam.get(fam, 0) >= math.ceil(n_pairs / 4):
            continue
        d = np.linalg.norm(Sn - Rn[i], axis=1)
        for j in np.argsort(d):
            if j not in used:
                used.add(int(j)); chosen.append((i, int(j))); break
        per_fam[fam] = per_fam.get(fam, 0) + 1
        if len(chosen) >= n_pairs:
            break

    ncol = 4
    nrow = math.ceil(len(chosen) / ncol) * 2
    fig = plt.figure(figsize=(3.1 * ncol, 3.25 * nrow))
    fig.suptitle("Real bridges (top of each pair) and their nearest v2.0 synthetic match (below)\n"
                 "8192 points each, centred and scaled to the unit sphere",
                 fontsize=13, y=0.995)

    for k, (i, j) in enumerate(chosen):
        block, col = divmod(k, ncol)
        r, s = real[i], synth[j]
        c = FAMILY_COLOR.get(r["fam"], "#666666")

        ax = fig.add_subplot(nrow, ncol, block * 2 * ncol + col + 1, projection="3d")
        scatter3d(ax, r["pts"], c)
        ax.set_title(f"REAL  {r['bid']}\nL/W {r['lw']:.2f}   len/R {r['lr']:.2f}   "
                     f"{r['spans']} span{'s' if r['spans'] > 1 else ''}",
                     fontsize=8.5, color=c, pad=1)

        ax = fig.add_subplot(nrow, ncol, (block * 2 + 1) * ncol + col + 1, projection="3d")
        scatter3d(ax, s["pts"], "#7D7D7D")
        ax.set_title(f"SYNTH  {s['bid']}  ({s['fam']})\nL/W {s['lw']:.2f}   "
                     f"len/R {s['lr']:.2f}   {s['spans']} span{'s' if s['spans'] > 1 else ''}",
                     fontsize=8.5, color="#333333", pad=1)

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=155)
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------- figure two
def figure_distributions(real, v3, v1, out: Path):
    panels = [
        ("len/R   (shape proxy)", "lr",   np.linspace(0.75, 2.05, 34)),
        ("L / W", "lw",                   np.linspace(0.0, 8.0, 34)),
        ("total length  (m)", "L",        np.linspace(0, 90, 34)),
        ("deck width  (m)", "W",          np.linspace(0, 60, 34)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    fig.suptitle("v1.0 vs v2.0 vs the 32-bridge real bank", fontsize=15, y=0.98)

    for ax, (title, key, bins) in zip(axes.flat, panels):
        for data, label, color, ls in (
            (v1, "v1.0 (old)", "#C44E52", "--"),
            (v3, "v2.0 (new)", "#4C72B0", "-"),
            (real, "real (n=32)", "#333333", "-"),
        ):
            vals = [d[key] for d in data]
            ax.hist(vals, bins=bins, density=True, histtype="step",
                    lw=2.2 if label != "real" else 2.6, color=color, ls=ls, label=label)
        ax.set_title(title, fontsize=11.5)
        ax.set_ylabel("density", fontsize=9)
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    # L/W vs len/R scatter -- the joint shape space
    ax = axes.flat[4]
    ax.scatter([d["lw"] for d in v1], [d["lr"] for d in v1], s=5, alpha=0.20,
               c="#C44E52", lw=0, label="v1.0")
    ax.scatter([d["lw"] for d in v3], [d["lr"] for d in v3], s=7, alpha=0.30,
               c="#4C72B0", lw=0, label="v2.0")
    for d in real:
        ax.scatter(d["lw"], d["lr"], s=52, marker="*", lw=0.5,
                   c=FAMILY_COLOR.get(d["fam"], "#333"), edgecolors="k", zorder=5)
    ax.set_xlabel("L / W", fontsize=10); ax.set_ylabel("len/R", fontsize=10)
    ax.set_title("joint shape space  (stars = real, by family)", fontsize=11.5)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # summary table
    ax = axes.flat[5]; ax.axis("off")
    def cv(v): return float(np.std(v) / np.mean(v))
    rows = [
        ("len/R CV",      cv([d["lr"] for d in v1]),   cv([d["lr"] for d in v3]),   cv([d["lr"] for d in real])),
        ("length median", float(np.median([d["L"] for d in v1])),  float(np.median([d["L"] for d in v3])),  float(np.median([d["L"] for d in real]))),
        ("width max",     max(d["W"] for d in v1),     max(d["W"] for d in v3),     max(d["W"] for d in real)),
        ("L/W median",    float(np.median([d["lw"] for d in v1])), float(np.median([d["lw"] for d in v3])), float(np.median([d["lw"] for d in real]))),
        ("frac L/W < 1",  np.mean([d["lw"] < 1 for d in v1]),      np.mean([d["lw"] < 1 for d in v3]),      np.mean([d["lw"] < 1 for d in real])),
        ("frac 1-span",   np.mean([d["spans"] == 1 for d in v1]),  np.mean([d["spans"] == 1 for d in v3]),  np.mean([d["spans"] == 1 for d in real])),
    ]
    tbl = ax.table(
        cellText=[[n, f"{a:.3f}", f"{b:.3f}", f"{c:.3f}"] for n, a, b, c in rows],
        colLabels=["metric", "v1.0", "v2.0", "real"], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 1.75)
    for j in range(4):
        tbl[0, j].set_facecolor("#E8E8E8"); tbl[0, j].set_text_props(weight="bold")
    for i in range(1, len(rows) + 1):
        tbl[i, 2].set_facecolor("#DCE6F2")     # v2.0 column
        tbl[i, 3].set_facecolor("#EFEFEF")     # real column
    ax.set_title("summary", fontsize=11.5)

    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out, dpi=155)
    plt.close(fig)
    print(f"  wrote {out}")


def cfg_records(cfgs):
    return [dict(bid=c.bridge_id, fam=getattr(c, "family", ""),
                 L=c.total_length_m, W=c.width_m, lw=c.total_length_m / c.width_m,
                 spans=c.num_spans,
                 lr=len_over_r(c.total_length_m, c.width_m,
                               c.bridge_clearance_height + c.depth_of_girder + 1.0))
            for c in cfgs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Dataset/BridgeModels")
    ap.add_argument("--out", default="Dataset/Figures")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--n-dist", type=int, default=3000)
    ap.add_argument("--v1", default="", help="path to the v1.0 package, for the overlay")
    args = ap.parse_args()

    out = Path(args.out)
    real = load_real()
    print(f"loaded {len(real)} real bridges")

    # --- synthetic point clouds, from the generated models -----------------
    sys.path.insert(0, ".")
    summary = {d["bridge_id"]: d for d in
               json.loads(Path("Dataset/bridge_summary.json").read_text())}
    synth = []
    for bd in sorted(Path(args.models).iterdir()):
        if not bd.is_dir() or bd.name not in summary:
            continue
        pts = sample_synthetic(bd)
        if pts is None:
            continue
        c = summary[bd.name]
        L, W = c["total_length_m"], c["width_m"]
        H = c["bridge_clearance_height"] + c["depth_of_girder"] + 1.0
        synth.append(dict(bid=bd.name, fam=c.get("family", ""), L=L, W=W,
                          lw=L / W, lr=len_over_r(L, W, H),
                          spans=c["num_spans"], pts=pts))
    print(f"sampled {len(synth)} synthetic bridges")
    figure_pairs(real, synth, out / "real_vs_synthetic_pairs.png", args.pairs)

    # --- distributions ------------------------------------------------------
    from BridgeModelGeneration.param_gen import generate_bridge_configs
    v3 = cfg_records(generate_bridge_configs(args.n_dist, None, seed=42))

    v1 = []
    if args.v1:
        sys.path.insert(0, args.v1)
        import importlib
        for m in list(sys.modules):
            if m.startswith("BridgeModelGeneration"):
                del sys.modules[m]
        old = importlib.import_module("BridgeModelGeneration.param_gen")
        v1 = [dict(bid=c.bridge_id, fam="", L=c.total_length_m, W=c.width_m,
                   lw=c.total_length_m / c.width_m, spans=c.num_spans,
                   lr=len_over_r(c.total_length_m, c.width_m,
                                 c.bridge_clearance_height + c.depth_of_girder + 1.0))
              for c in old.generate_bridge_configs(args.n_dist, None, seed=42)]
    if v1:
        figure_distributions(real, v3, v1, out / "distribution_comparison.png")


if __name__ == "__main__":
    main()
