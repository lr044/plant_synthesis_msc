"""
Three dune species, geometry per species rather than per trait.

The trait-combination is no longer used for these test. Each species has one fixed morphology
and variance comes from counts and jitter: how many shoots, how many leaves per
shoot, how many flower heads, plus angles, sizes, camera and lighting. 

  eryngium     Eryngium maritimum, sea holly. Glaucous blue-grey, rounded
               palmately-lobed leaves with hard marginal spines and pale veins,
               globular blue flower heads ringed by spiny bracts, low sprawling
               clump.
  achillea     Achillea maritima, cottonweed. Densely white-woolly throughout,
               small crowded oblong leaves, tight terminal clusters of small
               yellow button heads, bushy with many upright shoots.
  carpobrotus  Carpobrotus acinaciformis. Succulent triangular sabre leaves in
               opposite pairs along trailing stems, large magenta many-rayed
               flowers, prostrate mat.

Backgrounds are sand: a warm gradient with grain, optionally a real dune crop
via --bg-image.

  python build_species.py --species eryngium -n 50 -o out/eryngium
  python build_species.py --species all -n 50 -o out
  python gen_flat.py out/ -n 1 --strength 0.72
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LIGHT = np.array([-.55, -.62, .56])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def to_ground(P3):
    """Project 3D points onto z=0 along the light ray. A point at z=0 maps to
    itself, so a runner lying on the sand is provided with a shadow flush against it."""
    L = LIGHT / np.linalg.norm(LIGHT)
    d = -L                                  # direction light travels
    P3 = np.asarray(P3, float).reshape(-1, 3)
    t = -P3[:, 2] / min(d[2], -1e-6)
    return P3 + t[:, None] * d


class Camera:
    """World: X right, Y into the screen, Z up. Camera looks along +Y."""

    def __init__(self, W, H, dist=2.5, focal=2.0, elev_deg=9.0, tz=0.46):
        self.W, self.H, self.dist, self.f = W, H, dist, focal
        self.e, self.tz = np.deg2rad(elev_deg), tz

    def project(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        ce, se = np.cos(self.e), np.sin(self.e)
        y = np.maximum(P[:, 1] * ce + (P[:, 2] - self.tz) * se + self.dist, .2)
        z = -P[:, 1] * se + (P[:, 2] - self.tz) * ce
        return np.stack([self.W * .5 + self.f * self.W * P[:, 0] / y,
                         self.H * .5 - self.f * self.W * z / y], 1), y


def bezier(p0, p1, p2, n=26):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(a, float) for a in (p0, p1, p2))
    return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2


def _hull2d(pts):
    """Monotone-chain convex hull, used as the fill silhouette for flowers."""
    p = np.unique(np.round(np.asarray(pts, float), 2), axis=0)
    if len(p) < 3:
        return np.asarray(pts, float)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def half(ps):
        h = []
        for q in ps:
            while len(h) >= 2 and np.cross(h[-1] - h[-2], q - h[-2]) <= 0:
                h.pop()
            h.append(q)
        return h
    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


class Organ:
    """Carries its own colour: these species are not all green, and a single
    palette per role cannot express glaucous, woolly white and succulent."""
    __slots__ = ("outline", "inner", "depth", "role", "rgb", "vein_rgb",
                 "margin_rgb", "stroke", "shadow")

    def __init__(self, outline, inner, depth, role, rgb, vein_rgb=None,
                 margin_rgb=None, stroke=True, shadow=None):
        self.outline, self.inner = outline, inner
        self.depth, self.role, self.rgb = depth, role, rgb
        self.vein_rgb, self.margin_rgb = vein_rgb, margin_rgb
        # Silhouette cast onto the ground plane along the light direction.
        self.shadow = shadow
        # stroke=False: still fills for occlusion and colour, but its
        # silhouette is not drawn. The head's convex hull was being outlined as
        # a hard circle, so a bristly ball read as a rimmed disc.
        self.stroke = stroke


# ------------------------------------------------------------------ palettes

SPECIES = {
    "eryngium": dict(
        # Not blue. The reference photographs are silvery grey-green with a
        # white vein network, and the heads are pale khaki-green
        leaf=(186, 196, 182), vein=(240, 243, 236), stem=(202, 208, 196),
        flower=(196, 200, 158), bract=(198, 206, 192),
        name="Eryngium maritimum",
        prompt=("a non-flowering sea holly plant, Eryngium maritimum, with "
                "stiff silvery grey-green holly-like leaves with deep spiny "
                "lobes and a white vein network, growing in white sand dunes"),
        neg=("flowers, flower heads, blooms, buds, blue flowers, "
             "purple flowers, thistle head, soft leaves, smooth leaves, "
             "dense foliage, leafy cabbage, grass, dense meadow")),
    "achillea": dict(
        leaf=(200, 200, 188), vein=(214, 214, 204), stem=(198, 198, 186),
        flower=(234, 204, 72), bract=(206, 206, 194),
        name="Achillea maritima",
        prompt=("a cottonweed plant, Achillea maritima, densely covered in "
                "white woolly felt, with small crowded silver-grey leaves and "
                "tight clusters of small yellow button flower heads, growing "
                "in sand dunes"),
        neg=("green leaves, glossy leaves, large flowers, petals, daisy "
             "petals, grass, dense meadow")),
    "carpobrotus": dict(
        leaf=(104, 132, 82), vein=(168, 92, 62), stem=(140, 122, 84),
        flower=(206, 58, 150), bract=(104, 132, 82),
        name="Carpobrotus acinaciformis",
        prompt=("a hottentot fig plant, Carpobrotus acinaciformis, with thick "
                "smooth triangular succulent leaves in opposite pairs along "
                "trailing stems, and a large magenta flower with many narrow "
                "petals, creeping over sand"),
        neg=("thin leaves, hairy leaves, woolly, grass, dense meadow, "
             "yellow flowers")),
}

STYLE = ("realistic photograph, natural daylight, sharp focus, fine detail, "
         "shot on a 100mm macro lens at f/8")
BASE_NEG = ("drawing, sketch, line art, illustration, cartoon, painting, "
            "flat colour, oversaturated, neon, lowres, worst quality, "
            "watermark, text, person, hands, floating plant, cut out, "
            "heavy bokeh, soft focus")

SAND = dict(sky=((150, 176, 198), (216, 210, 194)),
            ground=((198, 178, 142), (222, 208, 182)))


# -------------------------------------------------------------- primitives

def _tube(centre3, radius, cam, rgb, taper=.5, role="stem"):
    p2, d = cam.project(centre3)
    seg = np.gradient(p2, axis=0)
    n = np.stack([-seg[:, 1], seg[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    t = np.linspace(0, 1, len(p2))
    w = ((radius * (1 + (taper - 1) * t)) * cam.f * cam.W / d)[:, None]
    return Organ(np.concatenate([p2 + n * w, (p2 - n * w)[::-1]]), [],
                 float(d.mean()), role, rgb)


def spiny_lobed_outline(n_lobes=4, n_spines=17, spine=0.16, n=460):
    """
    Eryngium leaf: a broad rounded blade with shallow lobes and many short
    marginal teeth.

    The reference photographs show a leaf roughly as wide as long, with 3-5
    shallow lobes and short spines all round the edge. The previous version
    used deep lobes and long spines, which produced a star or a spearhead --
    recognisably wrong, and it dominated every generated image.
    """
    th = np.linspace(-np.pi * .97, np.pi * .97, n)
    # deep sinuses between lobes, as a holly leaf has. 0.80 + 0.20 gave a disc
    lobe = 0.50 + 0.50 * np.abs(np.cos(n_lobes * th / 2.0)) ** 1.5
    ph = (th - th[0]) / (th[-1] - th[0]) * n_spines
    teeth = 1.0 + spine * (1.0 - np.abs(2 * (ph % 1.0) - 1) ** 0.30)
    # a long hard spine at each lobe apex, on top of the marginal teeth
    apex = 1.0 + 0.34 * np.clip(np.abs(np.cos(n_lobes * th / 2.0)) ** 14, 0, 1)
    r = lobe * teeth * apex
    return np.stack([r * np.cos(th), r * np.sin(th)], 1)


def oblong_outline(n=90):
    """Achillea leaf: small, blunt, spathulate."""
    t = np.linspace(0, 1, n)
    w = 0.17 * np.sin(np.pi * t) ** 0.6 * (0.45 + 0.55 * t)
    top = np.stack([t, w], 1)
    bot = top.copy()
    bot[:, 1] *= -1
    return np.concatenate([top, bot[::-1]])


def sabre_outline(n=110):
    """Carpobrotus leaf: a thick fleshy blade, not a spike. Width tapers only
    in the last third; tapering from the base made these read as yucca."""
    t = np.linspace(0, 1, n)
    # width/length near 0.20 and a blunt tip. At 0.10 with a needle taper the
    # leaves read as grass blades rather .
    # Base ramps up from zero so the leaf attaches at a point. The previous
    # profile started at half-width 0.11. A flat edge 0.23 wide and then
    # bulged to 0.20 by t=0.05 before dropping back to 0.11. The flat base
    # plus the shoulder above it is what read as a triangle stuck on the stem.
    # length:width near 6:1. At 0.185 half-width the leaves were 2.9:1. The base
    # ramp is also spread over 22% of the length instead of 11%, so there is
    # no hard shoulder where the blade meets the stem.
    base = np.clip(t / 0.22, 0, 1) ** 0.8
    w = 0.125 * base * (1.0 - 0.78 * t ** 1.6)
    # `bow` displaces the blade sideways within its own plane. Keep the value small for slight asymmetry; the
    # scimitar curve belongs on the leaf's z axis and is applied by the caller
    # through `droop`, because that is the axis that maps to back-and-up along
    # the blade once the leaf is elevated.
    bow = 0.035 * t ** 1.8
    return np.concatenate([np.stack([t, w + bow], 1),
                           np.stack([t, -w + bow], 1)[::-1]])


def leaf_organ(outline2, size, node, az, elev, roll, droop, curl, cam, rgb,
               vein_rgb=None, veins=3, margin_rgb=None, palmate=False):
    M = rot_z(az) @ rot_y(-elev) @ rot_x(roll)

    def place(p2):
        u, v = p2[:, 0], p2[:, 1]
        z = droop * np.clip(u, 0, None) ** 1.7 + curl * (v ** 2) * 5.0
        return (np.stack([u, v, z], 1) * size) @ M.T + node

    o3 = place(outline2)
    o2, d = cam.project(o3)
    shadow = cam.project(to_ground(o3))[0]
    inner = []
    if vein_rgb is not None and veins:
        if palmate:

            for a in np.linspace(-2.3, 2.3, veins):
                tip = np.array([[0., 0.], [np.cos(a) * .72, np.sin(a) * .72]])
                inner.append(cam.project(place(tip))[0])
        else:
            lo, hi = outline2[:, 0].min(), outline2[:, 0].max()
            for f in np.linspace(0.12, 0.88, veins):
                x = lo + (hi - lo) * f
                band = outline2[np.abs(outline2[:, 0] - x) < (hi - lo) * 0.05]
                if len(band) < 2:
                    continue
                inner.append(cam.project(place(np.array(
                    [[x, band[:, 1].min() * .55],
                     [x, band[:, 1].max() * .55]])))[0])
    return Organ(o2, inner, float(d.mean()), "leaf", rgb, vein_rgb,
                 margin_rgb, True, shadow)


def spiky_ball(size, pos, az, elev, cam, rgb, n_spines=9, rng=None):
    """
    Eryngium head as a star polygon {n/k}, drawn in one continuous stroke.

    The point angle of {n/k} is pi*(n-2k)/n, so the stride has to be near n/2
    for narrow spikes: {9/4} gives 20 degrees, {9/3} gives 60. Using stride
    n//3 as a second pass, fills the interior with blunt chords and
    the whole head becomes a mesh. One pass, few points, maximum stride.

    Tips sit on a tilted plane rather than a sphere. Stride ordering only
    produces a clean star when the points are in angular order around a
    circle; scattering them over a sphere destroys that and gives a tangle.
    """
    rng = rng or np.random.default_rng(0)
    n = int(np.clip(n_spines, 7, 11)) | 1
    k = (n - 1) // 2                            # sharpest star for this n
    if np.gcd(k, n) != 1:
        k -= 1

    a0 = rng.random() * 2 * np.pi
    th = a0 + np.arange(n) * 2 * np.pi / n
    r = 1.0 + 0.14 * (rng.random(n) - .5)       # slightly irregular outline
    tips = np.stack([r * np.cos(th), r * np.sin(th),
                     0.10 * (rng.random(n) - .5)], 1)

    # tilt the plane so the head is not a flat decal facing the camera
    M = (rot_z(rng.random() * 2 * np.pi)
         @ rot_y(np.deg2rad(28 * (rng.random() - .5)))
         @ rot_x(np.deg2rad(58 + 24 * (rng.random() - .5))))
    tips = tips @ M.T + [0, 0, 0.85]

    idx = [(j * k) % n for j in range(n + 1)]
    path = tips[idx]

    R = rot_z(az) @ rot_y(-elev)
    h2, dd = cam.project(tips * size @ R.T + pos)
    return Organ(_hull2d(h2), [cam.project(path * size @ R.T + pos)[0]],
                 float(dd.mean()), "flower", rgb, rgb, None, False)


def dome_head(size, pos, az, elev, cam, rgb, bract_rgb, n_bracts=9):
    """Eryngium inflorescence: a dome of florets ringed by narrow spiny bracts."""
    t = np.linspace(0, 1, 22)
    # ovoid, widest in the middle. sin(pi*t/2) is a monotonically widening
    # cup, which is why the heads came out as funnels.
    prof = np.stack([0.40 * np.sin(np.pi * t) ** 0.55 + 0.06, 0.92 * t], 1)
    th = np.linspace(0, 2 * np.pi, 26)
    lines = [np.stack([prof[-1, 0] * np.cos(th), prof[-1, 0] * np.sin(th),
                       np.full_like(th, prof[-1, 1])], 1)]
    hull = [lines[0]]
    for k in range(5):
        a = k * 2 * np.pi / 5
        m = np.stack([prof[:, 0] * np.cos(a), prof[:, 0] * np.sin(a),
                      prof[:, 1]], 1)
        hull.append(m)
        lines.append(m[8:])
    M = rot_z(az) @ rot_y(-elev)
    h2, d = cam.project(np.concatenate(hull) * size @ M.T + pos)
    head = Organ(_hull2d(h2), [cam.project(l * size @ M.T + pos)[0]
                               for l in lines], float(d.mean()), "flower", rgb)

    bracts = []
    for k in range(n_bracts):
        a = k * 2 * np.pi / n_bracts
        u = np.linspace(0, 1, 16)
        wid = 0.15 * np.sin(np.pi * u) ** 0.6
        r = 0.34 + 1.05 * u
        ca, sa = np.cos(a), np.sin(a)
        lf = np.stack([r * ca - wid * sa, r * sa + wid * ca, 0.06 - 0.10 * u], 1)
        rt = np.stack([r * ca + wid * sa, r * sa - wid * ca, 0.06 - 0.10 * u], 1)
        loop = np.concatenate([lf, rt[::-1], lf[:1]])
        b2, bd = cam.project(loop * size @ M.T + pos)
        bracts.append(Organ(b2, [], float(bd.mean()), "leaf", bract_rgb))
    return head, bracts


def button_head(size, pos, cam, rgb, squash=0.86):
    """
    Achillea capitulum: a small button head.

    Was a flat horizontal disc at a fixed height. At this species' camera elevation (11 degrees)
     a horizontal circle projects at aspect 0.19, so
    every head read as a thin sliver rather than a button, and the crowded
    terminal cluster came out as a stack of ellipses.

    A shell of revolution silhouettes as a circle from any angle, the head remains round under camera jitter without needing the camera raised. `squash`
    flattens it toward the disc it used to be: 1.0 is a ball, 0.7 is visibly
    oblate. Real heads are slightly wider than tall, hence the default.
    """
    u = np.linspace(0, np.pi, 13)
    v = np.linspace(0, 2 * np.pi, 22)
    U, V = np.meshgrid(u, v)
    shell = np.stack([np.sin(U) * np.cos(V),
                      np.sin(U) * np.sin(V),
                      squash * (1 - np.cos(U))], -1).reshape(-1, 3)
    d3 = shell * size + pos
    p2, d = cam.project(d3)

    # two latitude rings, so the head reads as a packed surface of florets
    # rather than a blank blob once ControlNet sees it
    th = np.linspace(0, 2 * np.pi, 22)
    inner = []
    for f in (0.42, 0.72):
        ph = np.pi * f
        ring = np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th),
                         np.full_like(th, squash * (1 - np.cos(ph)))], 1)
        inner.append(cam.project(ring * size + pos)[0])

    return Organ(_hull2d(p2), inner, float(d.mean()), "flower", rgb, None,
                 None, True, _hull2d(cam.project(to_ground(d3))[0]))


def rayed_flower(size, pos, az, elev, cam, rgb, n_pet=34):
    lines, hull = [], []
    for i in range(n_pet):
        a = i * 2 * np.pi / n_pet + (0.06 if i % 2 else 0.0)
        u = np.linspace(0, 1, 12)
        r = 0.24 + 0.78 * u
        w = 0.045 * np.sin(np.pi * u) ** 0.6
        ca, sa = np.cos(a), np.sin(a)
        lf = np.stack([r * ca - w * sa, r * sa + w * ca, 0.05 * (1 - u)], 1)
        rt = np.stack([r * ca + w * sa, r * sa - w * ca, 0.05 * (1 - u)], 1)
        loop = np.concatenate([lf, rt[::-1], lf[:1]])
        lines.append(loop)
        hull.append(loop)
    th = np.linspace(0, 2 * np.pi, 20)
    lines.append(np.stack([0.22 * np.cos(th), 0.22 * np.sin(th),
                           np.full_like(th, 0.07)], 1))
    M = rot_z(az) @ rot_y(-elev)
    h3 = np.concatenate(hull) * size @ M.T + pos
    h2, d = cam.project(h3)
    shadow = _hull2d(cam.project(to_ground(h3))[0])
    # stroke=False: the petals are the drawing. Outlining the convex hull puts
    # a circle round the flower, the same fault the Eryngium head had.
    return Organ(_hull2d(h2), [cam.project(l * size @ M.T + pos)[0]
                               for l in lines], float(d.mean()), "flower",
                 rgb, rgb, None, False, shadow)


# ----------------------------------------------------------------- species

def build_eryngium(cam, rng, J, P):
    """
    Branched, not a leafy mound.

    The photographs show pale branching stems with leaves spaced at the nodes
    and a spiky head terminating each branch, with sand visible between the
    leaves. The previous version packed 12-20 leaves onto one short stem, which
    produced a dense cabbage.
    """
    organs = []
    base = np.array([J(-.12, .12), J(-.10, .10), 0.])

    # a few broad basal leaves, flat to the sand
    for k in range(int(rng.integers(2, 4))):
        organs.append(leaf_organ(
            spiny_lobed_outline(int(rng.integers(3, 6)),
                                int(rng.integers(9, 14)), .24 + J(0, .10)),
            .30 * (1 + J(-.18, .18)), base,
            rng.random() * 2 * np.pi, np.deg2rad(4 + J(-6, 12)),
            np.deg2rad(J(-22, 22)), -.10 + J(-.12, .10), .05,
            cam, P["leaf"], P["vein"], 5, P["vein"], True))

    def branch(start, direction, length, depth, size):
        nonlocal organs          
        tip = start + direction * length
        stem = bezier(start, (start + tip) / 2 + [0, 0, length * .18], tip)
        organs.append(_tube(stem, .015 * size, cam, P["stem"], .7))
        # one or two clasping leaves at the node, not a whorl
        for _ in range(int(rng.integers(1, 3)) if depth else 2):
            organs.append(leaf_organ(
                spiny_lobed_outline(int(rng.integers(3, 6)),
                                    int(rng.integers(9, 14)), .24 + J(0, .10)),
                .15 * size * (1 + J(-.20, .20)), tip,
                rng.random() * 2 * np.pi, np.deg2rad(24 + J(-24, 26)),
                np.deg2rad(J(-24, 24)), -.14 + J(-.14, .12), .05,
                cam, P["leaf"], P["vein"], 5, P["vein"], True))
        if depth > 0 and rng.random() < .85:
            for _ in range(int(rng.integers(2, 4))):
                a = rng.random() * 2 * np.pi
                el = np.deg2rad(58 + J(-20, 24))
                d2 = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el),
                               np.sin(el)])
                branch(tip, d2, length * (.78 + J(0, .20)), depth - 1,
                       size * .82)
        else:
            # the whorl under a head is ordinary spiny leaves, not thin bracts,
            # so build it from the same primitive and let it splay outward
            for b in range(int(rng.integers(4, 7))):
                organs.append(leaf_organ(
                    spiny_lobed_outline(int(rng.integers(3, 6)),
                                        int(rng.integers(8, 13)),
                                        .26 + J(0, .10)),
                    .115 * size * (1 + J(-.22, .22)), tip,
                    b * 2 * np.pi / 5 + J(-.5, .5),
                    np.deg2rad(6 + J(-14, 18)), np.deg2rad(J(-20, 20)),
                    -.10 + J(-.10, .10), .04,
                    cam, P["leaf"], P["vein"], 4, P["vein"], True))
            # No flower heads. They flower rarely in the wild, and every
            # attempt at the head geometry proved unsuccessful. A
            # branch simply terminates in its leaf whorl.

    for _ in range(int(rng.integers(2, 4))):
        a = rng.random() * 2 * np.pi
        el = np.deg2rad(74 + J(-14, 12))
        d = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el),
                      np.sin(el)])
        branch(base, d, .46 + J(-.08, .16), 1, 1.0)
    return organs


def build_achillea(cam, rng, J, P):
    organs = []
    for sh in range(int(rng.integers(4, 8))):
        base = np.array([J(-.34, .34), J(-.26, .26), 0.])
        h = 0.40 + J(-.10, .14)
        lean = np.array([J(-.20, .20), J(-.16, .16), 0.])
        apex = base + lean + [0, 0, h]
        stem = bezier(base + [0, 0, -.2], (base + apex) / 2 + lean * .4, apex)
        organs.append(_tube(stem, .012, cam, P["stem"], .5))
        # side branches: unbranched wands read as grass, and the congested
        # cushion habit is most of what identifies this species
        tips = [apex]
        for _b in range(int(rng.integers(1, 4))):
            fb = .35 + .40 * rng.random()
            nb = stem[int(fb * (len(stem) - 1))]
            a = rng.random() * 2 * np.pi
            bt = nb + [np.cos(a) * (.10 + J(0, .10)),
                       np.sin(a) * (.10 + J(0, .10)), (h * (1 - fb)) * .85]
            organs.append(_tube(bezier(nb, (nb + bt) / 2, bt), .009, cam,
                                P["stem"], .55))
            tips.append(bt)
        az0 = rng.random() * 2 * np.pi
        # leaves are small and crowded: the felted, congested look is the
        # species, so count matters more than individual leaf shape
        n_leaf = int(rng.integers(28, 44))
        for k in range(n_leaf):
            f = .06 + .88 * (k / (n_leaf - 1.0)) + J(-.03, .03)
            node = stem[int(np.clip(f, 0, 1) * (len(stem) - 1))]
            organs.append(leaf_organ(
                oblong_outline(), .078 * (1 + J(-.30, .30)), node,
                az0 + k * 2.399 + J(-.5, .5), np.deg2rad(58 + J(-32, 24)),
                np.deg2rad(J(-30, 30)), -.10 + J(-.16, .12), .10 + J(-.06, .10),
                cam, P["leaf"], None, 0))
        for tip in tips:
            for b in range(int(rng.integers(3, 8))):
                a = b * 2 * np.pi / 6 + J(-.4, .4)
                r = .050 * (1 + J(-.4, .4))
                organs.append(button_head(
                    .021 * (1 + J(-.22, .22)),
                    tip + [np.cos(a) * r, np.sin(a) * r, J(-.02, .03)],
                    cam, P["flower"]))
    return organs


def build_carpobrotus(cam, rng, J, P):
    """
    Structure, read off the reference photographs (1146, 1865, 3223):

      * one runner lying ON the sand, not above it
      * ONE opposite pair of leaves per node. never a cluster of pairs
      * clear internodes between nodes, roughly one leaf-width of bare runner
      * the two leaves of a pair share a single attachment point and open into
        a V, so they diverge rather than overlap
      * successive pairs are decussate, a quarter turn apart, so from the side
        you see alternating V's and foreshortened pairs
      * leaves point up and outward, curving back toward vertical at the tip

    Errors: two to four pairs were stacked at each node with their
    bases a few millimetres apart, which made overlapping tangles instead of
    clean V's, and the pair azimuth was random so a pair could lie flat across
    the frame pointing sideways instead of upward.
    """
    organs = []

    for sh in range(int(rng.integers(2, 5))):
        a0 = rng.random() * 2 * np.pi
        length = 0.78 + J(-.12, .26)
        r_run = .0105
        # centre the runner one radius above z=0 so it RESTS on the sand plane
        # rather than hovering a couple of centimetres over it
        basez = r_run
        base = np.array([J(-.30, .30), J(-.26, .26), basez])
        tip = base + [np.cos(a0) * length, np.sin(a0) * length, 0.]
        bend = np.array([-np.sin(a0), np.cos(a0), 0.]) * J(-.13, .13)
        runner = bezier(base, (base + tip) / 2 + bend, tip)
        organs.append(_tube(runner, r_run, cam, P["stem"], .85))

        n_node = int(rng.integers(5, 9))
        for i in range(n_node):
            f = .07 + .88 * (i / max(n_node - 1, 1)) + J(-.02, .02)
            node = runner[int(np.clip(f, 0, 1) * (len(runner) - 1))]
            # decussate about the runner, starting perpendicular to it so the
            # first pair opens as a V across the line of sight
            az = a0 + np.pi / 2 + i * np.pi / 2 + J(-.18, .18)
            elev = np.deg2rad(66 + J(-9, 11))
            size = .30 * (1 + J(-.10, .10))
            for sgn in (0, np.pi):
                organs.append(leaf_organ(
                    sabre_outline(), size, node, az + sgn,
                    elev, np.deg2rad(J(-10, 10)),
                    .26 + J(-.07, .09), .02,
                    cam, P["leaf"], P["vein"], 1))
    return organs


BUILDERS = {"eryngium": build_eryngium, "achillea": build_achillea,
            "carpobrotus": build_carpobrotus}


def build(species, W, H, seed, jitter=1.0):
    rng = np.random.default_rng(seed)
    J = lambda lo, hi: float(rng.uniform(lo, hi)) * jitter
    P = SPECIES[species]
    cam = Camera(W, H, dist=2.4 + J(-.4, .6), focal=2.0,
                 elev_deg=(58 + J(-6, 8) if species == "carpobrotus"
                            else 11 + J(-10, 14)))
    organs = BUILDERS[species](cam, rng, J, P)

    # frame it: uniform scale and centre, base just past the bottom edge
    pts = [o.outline for o in organs] + [q for o in organs for q in o.inner
                                         if len(q)]
    allp = np.concatenate([np.asarray(q, float).reshape(-1, 2) for q in pts])
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1e-6)
    k = min(W * .86 / span[0], H * .86 / span[1])
    # Anchor the base off the bottom edge, but not at the cost of leaving the
    # frame mostly sky: a prostrate species is wide and shallow, so bottom-
    # anchoring alone strands it in a strip. Use whichever places more of the plant in
    # frame.
    cy = H * .58 - (lo[1] + hi[1]) / 2 * k
    by = H * 1.02 - hi[1] * k
    off = np.array([W / 2 - (lo[0] + hi[0]) / 2 * k, min(cy, by)])
    for o in organs:
        o.outline = np.asarray(o.outline, float) * k + off
        o.inner = [np.asarray(q, float) * k + off for q in o.inner]
        if o.shadow is not None:
            o.shadow = np.asarray(o.shadow, float) * k + off
    return organs, cam


# --------------------------------------------------------------- rendering

def _xy(p):
    return [(float(x), float(y)) for x, y in np.asarray(p)]


def render_lines(organs, W, H, width=3):
    """Hidden-line removal by painter's algorithm: fill each silhouette black
    to erase what sits behind it, then stroke it white."""
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    for o in sorted(organs, key=lambda x: -x.depth):
        xy = _xy(o.outline)
        if len(xy) > 3:
            d.polygon(xy, fill=0)
            if o.stroke:
                d.line(xy + xy[:1], fill=255, width=width, joint="curve")
        for q in o.inner:
            if len(q) > 1:
                d.line(_xy(q), fill=255, width=max(1, width - 1), joint="curve")
    return Image.merge("RGB", (img, img, img))


def render_mask(organs, W, H):
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for o in organs:
        xy = _xy(o.outline)
        if len(xy) > 2:
            d.polygon(xy, fill=255)
            d.line(xy + xy[:1], fill=255, width=3)
    return m


def _shade(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    p = np.stack([xs, ys], 1).astype(np.float64)
    c = p.mean(0)
    _, _, V = np.linalg.svd(p - c, full_matrices=False)
    e2 = V[1]
    pr = (p - c) @ e2
    v = pr / max(np.abs(pr).max(), 1e-9)
    hgt = np.sqrt(np.clip(1 - v ** 2, 0, 1))
    L = LIGHT / np.linalg.norm(LIGHT)
    lam = np.clip(np.stack([v * e2[0], v * e2[1], hgt], 1) @ L, 0, 1)
    out = np.zeros(mask.shape, np.float32)
    out[ys, xs] = .62 + .55 * lam
    return out


def render_flat(organs, W, H, seed, blur=2, bg_image=None):
    """Shaded plant on sand. Colour is painted at the right coordinates, so
    binding is positional and cannot bleed between organs."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)

    if bg_image is not None:
        bgi = cv2.cvtColor(cv2.imread(str(bg_image)), cv2.COLOR_BGR2RGB)
        img[:] = cv2.resize(bgi, (W, H), interpolation=cv2.INTER_AREA)
    else:
        hy = H * (0.12 + rng.random() * 0.16)
        rows = np.arange(H, dtype=np.float32)
        s_lo, s_hi = [np.array(c, np.float32) for c in SAND["sky"]]
        g_far, g_near = [np.array(c, np.float32) for c in SAND["ground"]]
        st = np.clip(rows / max(hy, 1), 0, 1)[:, None]
        sky = s_lo[None] * (1 - st) + s_hi[None] * st
        gt = np.clip((rows - hy) / max(H - hy, 1), 0, 1)[:, None] ** .6
        gnd = g_far[None] * (1 - gt) + g_near[None] * gt
        img[:] = np.where((rows >= hy)[:, None, None], gnd[:, None, :],
                          sky[:, None, :])
        # sand grain, so img2img has texture to refine rather than a flat wash
        n = rng.normal(0, 1, (H // 5, W // 5)).astype(np.float32)
        n = cv2.GaussianBlur(cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC),
                             (0, 0), 3)
        img *= (1 + .07 * n / max(np.abs(n).max(), 1e-6))[:, :, None]

    # Real cast shadows: every organ's silhouette projected onto z=0 along the
    # light ray at build time. A runner lying on the sand therefore gets a
    # shadow flush against it, while a leaf standing 20cm up throws one out
    # across the sand. 
    sm = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(sm)
    for o in organs:
        if o.shadow is None:
            continue
        xy = _xy(o.shadow)
        if len(xy) > 2:
            sd.polygon(xy, fill=255)
    sh = np.array(sm).astype(np.float32) / 255.0
    if sh.any():
        sh = np.clip(cv2.GaussianBlur(sh, (0, 0), max(W * .010, 2.0)), 0, 1)
        img = img * (1 - .21 * sh[:, :, None])

    um = np.array(render_mask(organs, W, H)) > 0
    ground_y = int(np.percentile(np.nonzero(um)[0], 99)) if um.any() else None

    noise = rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32)
    noise = cv2.GaussianBlur(cv2.resize(noise, (W, H),
                                        interpolation=cv2.INTER_CUBIC), (0, 0), 8)
    noise /= max(np.abs(noise).max(), 1e-6)

    for o in sorted(organs, key=lambda x: -x.depth):
        xy = _xy(o.outline)
        if len(xy) < 3:
            continue
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(xy, fill=255)
        mask = np.array(m) > 0
        if not mask.any():
            continue
        sh = _shade(mask)
        sh = sh[mask][:, None] if sh is not None else 1.0
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        edge = np.clip(dt / max(.014 * min(W, H), 1.), 0, 1)[mask][:, None]
        px = np.array(o.rgb, np.float32)[None] * sh * (.82 + .18 * edge)
        px = px * (1 + .06 * noise[mask][:, None])
        img[mask] = np.clip(px, 0, 255)
        # cartilaginous margin. Eryngium's bright white leaf edge is its most distinctive feature and nothing in the render carried it before
        if o.margin_rgb is not None:
            em = Image.new("L", (W, H), 0)
            xyc = xy + xy[:1]
            ImageDraw.Draw(em).line(xyc, fill=255,
                                    width=max(2, int(min(W, H) * .004)))
            sel = (np.array(em) > 0) & mask
            img[sel] = (img[sel] * .25
                        + np.array(o.margin_rgb, np.float32) * .75)

        vc = o.vein_rgb if o.vein_rgb is not None else o.rgb
        for q in o.inner:
            if len(q) < 2:
                continue
            vm = Image.new("L", (W, H), 0)
            ImageDraw.Draw(vm).line(_xy(q), fill=255, width=2)
            sel = (np.array(vm) > 0) & mask
            # blend toward the VEIN colour. This blended toward the organ's own
            # colour, so every vein was invisible.
            img[sel] = img[sel] * .45 + np.array(vc, np.float32) * .55

    # Sand drawn back over the very bottom of the plant, so the runner is
    # half-buried instead of resting on the surface. A prostrate stem in a dune
    # is always partly under the sand; drawn on top of it, it floats.
    if ground_y is not None:
        band = max(int(H * .030), 4)
        top = ground_y - band
        ramp = np.clip((np.arange(H) - top) / max(band, 1), 0, 1)[:, None] ** .8
        sand = img[min(ground_y + 2, H - 1)].mean(0)
        berm = (ramp * (np.arange(H)[:, None] >= top))[:, :, None]
        img = img * (1 - berm) + sand[None, None, :] * berm

    out = Image.fromarray(img.astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(blur)) if blur else out


# -------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="all",
                    choices=list(SPECIES) + ["all"])
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=1.0)
    ap.add_argument("--blur", type=int, default=2)
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--bg-image", default=None,
                    help="a real sand crop to use instead of the painted "
                         "gradient")
    ap.add_argument("-o", "--out", default="out")
    args = ap.parse_args()

    names = list(SPECIES) if args.species == "all" else [args.species]
    out = Path(args.out)
    for sp in names:
        d = out / sp if args.species == "all" else out
        d.mkdir(parents=True, exist_ok=True)
        P = SPECIES[sp]
        for i in range(args.n):
            seed = args.seed + i
            organs, _ = build(sp, args.size, args.size, seed, args.jitter)
            stem = d / f"{sp}_{seed:03d}"
            render_lines(organs, args.size, args.size,
                         args.line_width).save(stem.with_suffix(".png"))
            render_flat(organs, args.size, args.size, seed, args.blur,
                        args.bg_image).save(Path(str(stem) + "_flat.png"))
            render_mask(organs, args.size, args.size).save(
                Path(str(stem) + "_mask.png"))
            stem.with_suffix(".txt").write_text(
                f"{P['prompt']}, {STYLE}")
            stem.with_suffix(".neg.txt").write_text(
                f"{P['neg']}, {BASE_NEG}")
            stem.with_suffix(".json").write_text(json.dumps(
                dict(species=sp, latin=P["name"], seed=seed,
                     jitter=args.jitter), indent=2))
        print(f"{sp}: {args.n} -> {d}")


if __name__ == "__main__":
    main()
