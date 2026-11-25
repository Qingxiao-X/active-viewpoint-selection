"""Build results/evidence_v1/scene_overview.png as a schematic diagram instead of an
in-engine render: a real USD-material render puts the camera/light markers at their
true small scale and distance, where they visually blend into the head/skin color
under scene lighting (tried first; not legible). A labeled 3D schematic -- head photo
inset + camera orbit arc + light direction arrow, using the manifest's real pose
numbers -- communicates the setup far more clearly for a presentation slide.

Run with a modern Python (system python3 is 3.6.8): `module load Anaconda3`.
    python3 tools/assets/scene_overview_diagram.py --manifest results/evidence_v1/manifest.json \
        --photo results/evidence_v1/clean_front.png --out results/evidence_v1/scene_overview.png
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.proj3d import proj_transform


def sph_to_vec(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest = json.load(open(args.manifest))
    poses = manifest["poses"]
    radius = poses["clean_front"]["radius_m"]
    light = manifest["lighting_presets"]["glare"]

    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 0.8))

    # Head as a simple ellipsoid centered at the origin (head_center), matching the
    # real scene's coordinate convention (face along +x).
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    hx = 0.11 * np.cos(u) * np.sin(v)
    hy = 0.085 * np.sin(u) * np.sin(v)
    hz = 0.13 * np.cos(v) + 0.02
    ax.plot_surface(hx, hy, hz, color="#e8c9a8", alpha=0.85, linewidth=0, shade=True)

    # Camera positions actually used (real poses from manifest.json, not illustrative).
    cams = {
        "clean_front / glare_front": ("front", poses["clean_front"], "#2e6ff2"),
        "glare_left (LEFT, +15 deg)": ("left", poses["glare_left"], "#2bb32b"),
        "glare_right (RIGHT, -15 deg)": ("right", poses["glare_right"], "#c92bc9"),
    }
    for label, (_, p, color) in cams.items():
        pos = radius * sph_to_vec(p["azimuth_deg"], p["elevation_deg"])
        ax.scatter(*pos, color=color, s=140, edgecolor="black", linewidth=0.6, depthshade=False)
        ax.text(pos[0], pos[1], pos[2] + 0.05, label, color=color, fontsize=8, ha="center")
        arrow = Arrow3D([pos[0], 0], [pos[1], 0], [pos[2], 0.02],
                         mutation_scale=8, lw=1.0, arrowstyle="-", color=color, alpha=0.5)
        ax.add_artist(arrow)

    # Orbit arc at elevation 0 showing the +-15 deg LEFT/RIGHT range actually used.
    arc_az = np.linspace(-20, 20, 60)
    arc = np.array([radius * sph_to_vec(a, 0) for a in arc_az])
    ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="gray", lw=1.5, linestyle="--", alpha=0.6)

    # Key light direction (DistantLight has no position -- this is an arrow showing the
    # direction light arrives FROM, at an arbitrary visualization distance).
    light_dir = sph_to_vec(light["light_az"], light["light_el"]) * 0.45
    larrow = Arrow3D([light_dir[0], 0], [light_dir[1], 0], [light_dir[2], 0.02],
                      mutation_scale=18, lw=2.2, arrowstyle="-|>", color="#e0b000")
    ax.add_artist(larrow)
    ax.text(*(light_dir * 1.12), f"key light\naz={light['light_az']:.0f} el={light['light_el']:.0f}",
            color="#8a6d00", fontsize=8, ha="center")

    ax.scatter([0], [0], [0.02], color="black", s=20)
    ax.text(0, 0, -0.08, "head center\n(origin)", fontsize=7, ha="center", color="black")

    ax.set_xlabel("x (face points +x)")
    ax.set_ylabel("y (+ = head's left)")
    ax.set_zlabel("z (up)")
    lim = radius * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.view_init(elev=22, azim=-55)
    ax.set_title(f"Camera setup: radius={radius} m, 640x480, orbit +-15 deg steps shown\n"
                 f"(real poses from {args.manifest})", fontsize=10)

    # Inset: the actual captured clean_front.png, for a literal "what the head looks like".
    photo = plt.imread(args.photo)
    inset = fig.add_axes([0.02, 0.60, 0.22, 0.32])
    inset.imshow(photo)
    inset.axis("off")
    inset.set_title("clean_front.png", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
