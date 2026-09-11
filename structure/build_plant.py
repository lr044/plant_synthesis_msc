"""
Trait tags -> 3D plant geometry -> line drawing + shaded colour map.

Rebuilt around a minimal 3D core. The previous version laid leaves flat in the
picture plane, both radiating sideways from a single node: a botanical-diagram
pose, not a plant. Every leaf faced the camera square-on and leaving no functional occlusion

This version improves through:
  * leaves are 3D surfaces (arched midrib, folded blade) placed around the stem
    by phyllotaxis, so azimuth alone produces natural foreshortening.  A leaf
    pointing at the camera is short and broad, one  pointing sideways is long
  * several nodes up the stem, leaf siz decreasing with height
  * a perspective camera, so near organs are larger
  * painter's algorithm with fill-black-then-stroke-white, which gives
    hidden-line removal as nearer leaves cover further ones

Jitter is per-organ and far wider than before, but bounded so no seed can push
a leaf out of its margin or shape class. Tooth count and depth are never
jittered; only tooth phase is.

  python build_plant.py --leaf-margin serrated --variants 4 --jitter 1.0 -o out/s
  python build_plant.py --sweep leaf_margin --jitter 0.7 -o out/margin
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

LEAF_SHAPES = ["lanceolate", "linear", "ovate", "cordate", "sagittate",
               "lobed", "palmately lobed"]
LEAF_MARGINS = ["entire", "crenate", "serrated"]
FLOWER_STRUCTURES = ["campanulate", "tubular", "rotate", "ligulate",
                     "papilionaceous", "spurred"]
INFLORESCENCES = ["solitary", "raceme", "spike", "panicle", "corymb", "cyme",
                  "capitulum"]
FLOWER_COLOURS = ["blue", "green", "pink", "purple", "white", "yellow"]
LEAF_SURFACES = ["glabrous", "hairy"]
PHYLLOTAXIS = ["opposite", "alternate", "whorled"]

SURFACE_PHRASE = {"glabrous": "smooth hairless leaves",
                  "hairy": "leaves densely covered in fine hairs"}
LEAF_RGB, STEM_RGB = (74, 118, 60), (104, 138, 72)
# deliberately NOT green: a green field reads as foliage and SDXL fills it in
BG_RGB = (176, 172, 166)
# Sky and ground palettes. LIGHT is shared with _cyl_shade, so the organs, the ground gradient and the cast shadow all agree on where the sun is -- the
# single biggest reason a subject reads as pasted on rather than photographed.
LIGHT = np.array([-.55, -.62, .56])

BACKGROUNDS = {
    "plain": dict(sky=((176, 172, 166), (176, 172, 166)),
                  ground=((176, 172, 166), (176, 172, 166)),
                  haze=(176, 172, 166), shadow=0.0,
                  phrase="on a plain background",
                  extra="studio background, seamless backdrop"),
    # Phrases stay short and carry no blur word. Every "soft", "hazy" or
    # "blurred" here was being obeyed, and the prompt contradicting     # itself by also requiring for sharp focus.
    "beach": dict(sky=((126, 168, 206), (206, 216, 220)),
                  ground=((196, 176, 142), (214, 202, 178)),
                  haze=(212, 214, 210), shadow=0.34,
                  phrase="on a sandy beach with dune grass",
                  extra="marram grass, sand"),
    "field": dict(sky=((150, 178, 202), (208, 214, 208)),
                  ground=((104, 124, 66), (150, 162, 112)),
                  haze=(196, 204, 196), shadow=0.30,
                  phrase="in a green meadow",
                  extra="meadow grass"),
}

FLOWER_RGB = {"blue": (72, 96, 196), "green": (126, 172, 92),
              "pink": (232, 118, 168), "purple": (146, 78, 178),
              "white": (246, 244, 238), "yellow": (240, 202, 66)}


# ------------------------------------------------------------------- 3D core

def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Camera:
    """World: X right, Y into screen, Z up. Camera looks along +Y."""

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


# ------------------------------------------------------------------- outlines

SHAPE_PARAMS = {"linear": (0.30, 0.30, 0.045), "lanceolate": (0.55, .95, .110),
                "ovate": (0.45, 1.25, 0.240), "cordate": (0.30, 1.15, 0.280),
                "sagittate": (0.35, 1.05, 0.150), "lobed": (0.45, 1.15, .260)}


def leaf_half(shape, n=240):
    t = np.linspace(0, 1, n)
    if shape == "palmately lobed":
        th = np.linspace(-np.deg2rad(148), np.deg2rad(148), 2 * n)
        r = 0.55 + 0.45 * np.abs(np.cos(5 * th / 2.0)) ** 1.6
        return np.stack([r * np.cos(th) * .55 + .45,
                         r * np.sin(th) * .55], 1), True
    a, b, amp = SHAPE_PARAMS[shape]
    w = (t ** a) * ((1 - t) ** b)
    w = amp * w / max(w.max(), 1e-9)
    if shape == "lobed":
        # keep the frequency low, or this reads as a serrated margin. 
        w = w * (0.34 + 0.66 * np.abs(np.cos(np.pi * 3 * t)) ** 0.7)
    pts = np.stack([t, w], 1)
    if shape in ("cordate", "sagittate"):
        d = 0.12 if shape == "cordate" else 0.30
        sp = 0.20 if shape == "cordate" else 0.13
        sh = 2.2 if shape == "cordate" else 0.7
        u = np.linspace(0, 1, 56)
        pts = np.concatenate([np.stack([-d * (1 - u) ** sh,
                                        sp * np.sin(np.pi * u * .5) ** .7],
                                       1), pts])
    return pts, False


def apply_margin(pts, margin, phase=0.0):
    """Depth is set so a tooth survives VAE encoding; resolvability matters
    more here than botanically exact tooth counts."""
    if margin == "entire" or len(pts) < 5:
        return pts
    d = np.gradient(pts, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    s = np.cumsum(np.linalg.norm(np.diff(pts, axis=0, prepend=pts[:1]), axis=1))
    s = (s / max(s[-1], 1e-9) + phase) % 1.0
    if margin == "crenate":
        disp = 0.050 * (0.5 - 0.5 * np.cos(2 * np.pi * 8 * s))
    else:
        # asymmetric sawtooth: serrate teeth lean toward the apex, which is
        # what separates them from symmetric rounded crenations
        disp = 0.055 * ((s * 13) % 1.0) ** 0.85
    return pts + n * (disp * np.sin(np.pi * np.clip(s, 0, 1)) ** .5)[:, None]


# --------------------------------------------------------------------- organs

class Organ:
    __slots__ = ("outline", "inner", "depth", "role")

    def __init__(self, outline, inner, depth, role):
        self.outline, self.inner = outline, inner
        self.depth, self.role = depth, role


def _hull2d(pts):
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


def leaf_organ(shape, margin, size, node, az, elev, roll, droop, curl,
               phase, cam, petiole=0.18):
    """`petiole` offsets the blade outward along its own axis, in blade
    lengths. Without it the blade grows straight out of the stem, which looks
    wrong on broad shapes like ovate and cordate."""
    half, closed = leaf_half(shape)
    if closed:
        out2 = apply_margin(half, margin, phase)
        veins2 = [np.array([[.45, 0.], [np.cos(np.deg2rad(-118 + i * 59)) * .52
                                        + .45,
                                        np.sin(np.deg2rad(-118 + i * 59)) * .52]])
                  for i in range(5)]
    else:
        top = apply_margin(half, margin, phase)
        bot = top.copy()
        bot[:, 1] *= -1
        out2 = np.concatenate([top, bot[::-1]])
        pos = half[half[:, 0] >= 0]
        veins2 = [np.array([[0., 0.], [.97, 0.]])]
        for f in (.22, .40, .58, .76):
            w = np.interp(f, pos[:, 0], pos[:, 1])
            for sgn in (1, -1):
                # stop short of the margin: veins reaching the edge enclose
                # regions, and enclosed regions get filled as separate colours
                veins2.append(np.array([[f, 0.], [f + .11, sgn * w * .55]]))

    M = rot_z(az) @ rot_y(-elev) @ rot_x(roll)

    def place(p2):
        u, v = p2[:, 0], p2[:, 1]
        z = droop * np.clip(u, 0, None) ** 1.8 + curl * (v ** 2) * 6.0
        return (np.stack([u + petiole, v, z], 1) * size) @ M.T + node

    o2, d = cam.project(place(out2))
    return Organ(o2, [cam.project(place(v))[0] for v in veins2],
                 float(d.mean()), "leaf")


def petiole_organ(size, node, az, elev, roll, petiole, cam):
    M = rot_z(az) @ rot_y(-elev) @ rot_x(roll)
    u = np.linspace(0, petiole, 10)
    pts = (np.stack([u, np.zeros_like(u), np.zeros_like(u)], 1)
           * size) @ M.T + node
    return tube_organ(pts, 0.008 * size / 0.4, cam, taper=0.7)


def _revolve(profile, n_theta=26, n_merid=3, merid_from=0.18, lobes=5,
             lobe_depth=0.10):
    """
    Surface of revolution with a lobed rim.

    Two fixes over the first version. Meridians were cut to the top 45%, which
    left short strokes clustered around the rim; in projection those radiate
    outward and ControlNet renders them as fringe or filaments, which is why
    petal tips came out feathery. They now run nearly the full length and read
    as petal folds. And the rim itself is scalloped rather than a plain ellipse,
    so petal separation is in the silhouette.
    """
    r, z = profile[:, 0], profile[:, 1]
    th = np.linspace(0, 2 * np.pi, n_theta * 3)
    scallop = 1.0 - lobe_depth * (0.5 - 0.5 * np.cos(lobes * th))
    rim = np.stack([r[-1] * scallop * np.cos(th),
                    r[-1] * scallop * np.sin(th),
                    z[-1] - r[-1] * lobe_depth * 0.8
                    * (0.5 - 0.5 * np.cos(lobes * th))], 1)
    hull, lines = [rim], [rim]
    cut = int(len(r) * merid_from)
    for k in range(n_merid):
        a = k * 2 * np.pi / n_merid
        full = np.stack([r * np.cos(a), r * np.sin(a), z], 1)
        hull.append(full)
        lines.append(full[cut:])
    return np.concatenate(hull), lines


def flower_geometry(kind):
    t = np.linspace(0, 1, 24)
    if kind == "campanulate":
        # width ~0.6 of height. The old profile was as wide as it was
        # tall, which reads as a funnel or a megaphone.
        return _revolve(np.stack([.10 + .21 * t ** 2.1, t], 1))
    if kind == "tubular":
        return _revolve(np.stack([.09 + .14 * t ** 4, t], 1), n_merid=4)
    if kind == "spurred":
        hull, lines = _revolve(np.stack([.10 + .20 * t ** 2.0,
                                         .12 + t * .88], 1))
        s = np.linspace(0, 1, 20)
        spur = np.stack([-.34 * s, np.zeros_like(s), .12 - .46 * s ** 1.2], 1)
        return np.concatenate([hull, spur]), lines + [spur]
    if kind in ("rotate", "ligulate", "capitulum"):
        npet, r0, r1, hw = ((5, .16, .62, .26) if kind == "rotate"
                            else (13, .24, .70, .075))
        lines = []
        for i in range(npet):
            a = i * 2 * np.pi / npet
            u = np.linspace(0, 1, 16)
            r = r0 + (r1 - r0) * u
            w = hw * np.sin(np.pi * u) ** .8
            ca, sa = np.cos(a), np.sin(a)
            lf = np.stack([r * ca - w * sa, r * sa + w * ca, .10 * (1 - u)], 1)
            rt = np.stack([r * ca + w * sa, r * sa - w * ca, .10 * (1 - u)], 1)
            lines.append(np.concatenate([lf, rt[::-1], lf[:1]]))
        th = np.linspace(0, 2 * np.pi, 24)
        dr = .15 if kind == "rotate" else .23
        lines.append(np.stack([dr * np.cos(th), dr * np.sin(th),
                               np.full_like(th, .12)], 1))
        return np.concatenate(lines), lines
    if kind == "papilionaceous":
        th = np.linspace(0, np.pi, 28)
        std = np.stack([.44 * np.cos(th), np.zeros_like(th),
                        .62 + .34 * np.sin(th)], 1)
        lines = [np.concatenate([std, std[:1]])]
        for sgn in (-1, 1):
            a = np.linspace(0, 2 * np.pi, 20)
            lines.append(np.stack([.28 * sgn + .24 * np.cos(a),
                                   .10 * sgn * np.ones_like(a),
                                   .38 + .15 * np.sin(a)], 1))
        u = np.linspace(0, 1, 18)
        keel = np.stack([-.32 * u, np.zeros_like(u),
                         .30 - .22 * np.sin(np.pi * u * .6)], 1)
        lines.append(np.concatenate([keel, (keel + [0, 0, .16])[::-1],
                                     keel[:1]]))
        return np.concatenate(lines), lines
    raise ValueError(kind)


def flower_organ(kind, size, pos, az, elev, cam):
    hull3, lines3 = flower_geometry(kind)
    M = rot_z(az) @ rot_y(-elev)
    h2, d = cam.project(hull3 * size @ M.T + pos)
    return Organ(_hull2d(h2),
                 [cam.project(l * size @ M.T + pos)[0] for l in lines3],
                 float(d.mean()), "flower")


def tube_organ(centre3, radius, cam, role="stem", taper=0.45):
    """
    A stem as a filled ribbon, so it occludes and can be occluded.

    `taper` is the tip radius as a fraction of the base. A constant-width stem
    reads as a hollow tube or a drinking straw; real stems narrow upward, and
    that taper is most of what makes a stalk look like a stalk.
    """
    p2, d = cam.project(centre3)
    seg = np.gradient(p2, axis=0)
    n = np.stack([-seg[:, 1], seg[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    t = np.linspace(0, 1, len(p2))
    r = radius * (1.0 + (taper - 1.0) * t)
    w = (r * cam.f * cam.W / d)[:, None]
    return Organ(np.concatenate([p2 + n * w, (p2 - n * w)[::-1]]), [],
                 float(d.mean()), role)


# ------------------------------------------------------------------- assembly

def bezier(p0, p1, p2, n=26):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = map(lambda a: np.asarray(a, float), (p0, p1, p2))
    return ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2


INFL_GEOM = {"solitary": (1, .210, .60), "raceme": (5, .100, 1.0),
             "spike": (6, .085, 1.0), "panicle": (9, .070, 1.0),
             "corymb": (5, .100, .86), "cyme": (5, .110, .90),
             "capitulum": (1, .160, .74)}


def build_one(tags, cam, origin, scale, rng, jit, phyllotaxis, nodes,
              leaf_scale):
    """One individual. `origin` is its base in world coords, `scale` its size."""
    J = lambda lo, hi: float(rng.uniform(lo, hi)) * jit
    organs = []
    origin = np.asarray(origin, float)

    h = (1.0 + J(-.10, .10)) * scale
    apex = origin + np.array([J(-.16, .16) * scale, J(-.12, .12) * scale, h])
    ctrl = (origin + apex) / 2 + np.array([J(-.10, .10) * scale,
                                           J(-.08, .08) * scale, 0.])
    stem = bezier(origin + [J(-.03, .03), 0., -.30 * scale], ctrl, apex)
    organs.append(tube_organ(stem, .014 * scale, cam, taper=.38))

    # Azimuth 90 and 270 point straight at / away from the camera, where a leaf collapses to an unreadable sliver. az - k*sin(2az) has fixed points every
    # 90 degrees and repels from exactly those two, so leaves keep varied
    # foreshortening without any of them degenerating.
    def readable(a):
        return a - 0.35 * np.sin(2 * a)

    
    top_f = .72 if tags["inflorescence"] in ("solitary", "capitulum") else .52

    az0 = rng.random() * 2 * np.pi
    for k in range(nodes):
        f = .18 + (top_f - .18) * (k / max(nodes - 1, 1))
        node = stem[int(f * (len(stem) - 1))]
        size = leaf_scale * (.52 - .24 * f) * scale
        if phyllotaxis == "alternate":
            azs = [az0 + k * np.deg2rad(137.5)]
        elif phyllotaxis == "whorled":
            azs = [az0 + k * np.deg2rad(40) + i * 2 * np.pi / 3
                   for i in range(3)]
        else:
            azs = [az0 + k * np.pi / 2, az0 + k * np.pi / 2 + np.pi]
        for az in azs:
            a = readable(az + J(-.5, .5))
            el = np.deg2rad(24 + J(-28, 28))
            rl = np.deg2rad(J(-20, 20))
            sz = size * (1 + J(-.18, .18))
            pet = .18 + J(-.06, .10)
            organs.append(petiole_organ(sz, node, a, el, rl, pet, cam))
            organs.append(leaf_organ(
                tags["leaf_shape"], tags["leaf_margin"], sz, node, a, el, rl,
                -.30 + J(-.35, .22),          
                .05 + J(-.05, .10),           # blade fold
                float(rng.random()), cam, petiole=pet))

    kind = tags["inflorescence"]
    n_f, fs, ax = INFL_GEOM[kind]
    axis_len = (h - (stem[int(top_f * (len(stem) - 1))][2] - origin[2])) * ax \
        + .06 * scale
    top = apex
    fsize = fs * (1 + J(-.14, .14)) * scale

    def flower(pos, az, elev):
        organs.append(flower_organ(tags["flower_structure"], fsize,
                                   np.asarray(pos, float), az, elev, cam))

    if kind == "capitulum":
        organs.append(flower_organ("capitulum", fsize, np.asarray(top, float),
                                   az0 + J(-1, 1), np.deg2rad(J(-30, 30)), cam))
    elif kind == "solitary":
        flower(top, az0 + J(-1, 1), np.deg2rad(J(-28, 28)))
    elif kind in ("raceme", "spike"):
        ped = .10 * axis_len if kind == "raceme" else 0.
        for i in range(n_f):
            g = .25 + .75 * i / (n_f - 1)
            a = az0 + i * np.deg2rad(137.5) + J(-.4, .4)
            q = top - np.array([0, 0, axis_len * (1 - g)])
            tip = q + np.array([np.cos(a), np.sin(a), .15]) * ped
            if ped:
                organs.append(tube_organ(bezier(q, (q + tip) / 2, tip),
                                         .006 * scale, cam, taper=.7))
            flower(tip, a, np.deg2rad(58 + J(-22, 22)))
    elif kind == "panicle":
        for i in range(3):
            g = .30 + .66 * i / 2
            q = top - np.array([0, 0, axis_len * (1 - g)])
            for j in range(3):
                a = az0 + (i * 3 + j) * np.deg2rad(137.5) + J(-.4, .4)
                bl = .30 * axis_len * (1 - .4 * g)
                tip = q + np.array([np.cos(a), np.sin(a), .5]) * bl
                organs.append(tube_organ(bezier(q, (q + tip) / 2, tip),
                                         .005 * scale, cam, taper=.7))
                flower(tip, a, np.deg2rad(55 + J(-22, 22)))
    elif kind == "corymb":
        q = top - np.array([0, 0, axis_len * .55])
        for i in range(n_f):
            a = az0 + i * 2 * np.pi / n_f + J(-.3, .3)
            tip = np.array([q[0] + np.cos(a) * .30 * axis_len,
                            q[1] + np.sin(a) * .30 * axis_len, top[2]])
            organs.append(tube_organ(bezier(q, (q + tip) / 2, tip),
                                     .006 * scale, cam, taper=.7))
            flower(tip, a, np.deg2rad(J(-22, 22)))
    elif kind == "cyme":
        node = top - np.array([0, 0, axis_len * .45])
        flower(top, az0, np.deg2rad(J(-16, 16)))
        organs.append(tube_organ(bezier(node, (node + top) / 2, top),
                                 .007 * scale, cam, taper=.7))
        for i in range(4):
            a = az0 + i * np.pi / 2 + J(-.3, .3)
            tip = node + np.array([np.cos(a) * .34 * axis_len,
                                   np.sin(a) * .34 * axis_len,
                                   .22 * axis_len])
            organs.append(tube_organ(bezier(node, (node + tip) / 2, tip),
                                     .006 * scale, cam, taper=.7))
            flower(tip, a, np.deg2rad(30 + J(-22, 22)))
    return organs


def build(tags, W=1024, H=1024, jit=.7, rng=None, phyllotaxis="opposite",
          nodes=3, leaf_scale=1.0, count=1):
    """
    A clump of `count` individuals sharing one camera.

    Individuals are scattered on a disc, so some sit further back and are
    smaller and partly hidden by perspective and depth sorting alone. The count
    itself is jittered: a fixed number across a dataset is a giveaway, and the
    point of the clump is variation.
    """
    rng = rng or np.random.default_rng(0)
    J = lambda lo, hi: float(rng.uniform(lo, hi)) * jit

    cam = Camera(W, H, dist=2.5 + J(-.35, .55), elev_deg=9 + J(-8, 14))
    n = count
    if jit > 0 and count > 1:
        n = int(rng.integers(max(1, count - 2), count + 1))

    spread = .10 + .13 * n
    organs = []
    for i in range(n):
        a = rng.random() * 2 * np.pi
        r = np.sqrt(rng.random()) * spread
        origin = np.array([r * np.cos(a), r * np.sin(a), 0.0]) if n > 1 \
            else np.zeros(3)
        scale = 1.0 if n == 1 else float(rng.uniform(.72, 1.12))
        organs += build_one(tags, cam, origin, scale, rng, jit, phyllotaxis,
                            nodes, leaf_scale)
    organs, k, off = fit_to_frame(organs, W, H)
    return organs, dict(cam=cam, k=k, off=off)


def fit_to_frame(organs, W, H, margin=0.10, anchor_bottom=True):
    """
    Uniformly scale and centre the projected plant to fill the frame.

    Tuning camera distance by hand cannot work: jitter changes the plant's
    extent every seed, so a distance that frames one variant crops the next.
    Fitting after projection is scale-invariant and keeps perspective intact.
    """
    pts = [o.outline for o in organs] + [q for o in organs for q in o.inner
                                         if len(q)]
    if not pts:
        return organs
    allp = np.concatenate([np.asarray(q, float).reshape(-1, 2) for q in pts])
    lo, hi = allp.min(0), allp.max(0)
    span = np.maximum(hi - lo, 1e-6)
    k = min(W * (1 - 2 * margin) / span[0], H * (1 - 2 * margin) / span[1])
    off = np.array([W, H]) / 2 - (lo + hi) / 2 * k
    if anchor_bottom:
        # Run the stem base off the bottom edge. Centring the bounding box
        # leaves a visible cut end floating mid-frame, which is the single
        # clearest tell that the plant was pasted rather than photographed.
        off[1] = H * 1.03 - hi[1] * k
    for o in organs:
        o.outline = np.asarray(o.outline, float) * k + off
        o.inner = [np.asarray(q, float) * k + off for q in o.inner]
    return organs, k, off


# ------------------------------------------------------------------ rendering

def _xy(p):
    return [(float(x), float(y)) for x, y in np.asarray(p)]


def render(organs, W, H, width=3, scene=None, background="plain"):
    """
    White line art on black, with hidden-line removal.

    Painter's algorithm far-to-near: fill each silhouette black to erase what sits behind it, then stroke it white. Without this every leaf shows through every other one and the drawing reads with tangled organs.
    """
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)

    for o in sorted(organs, key=lambda x: -x.depth):
        xy = _xy(o.outline)
        if len(xy) > 3:
            d.polygon(xy, fill=0)
            d.line(xy + xy[:1], fill=255, width=width, joint="curve")
        for q in o.inner:
            if len(q) > 1:
                d.line(_xy(q), fill=255, width=max(1, width - 1), joint="curve")
    return Image.merge("RGB", (img, img, img))


def _cyl_shade(mask):
    """cylindrical shading."""
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


def horizon_row(scene, W, H):
    """
    Screen row of the horizon, clamped into frame.
    """
    cam, k, off = scene["cam"], scene["k"], scene["off"]
    hy = (cam.H * .5 + cam.f * cam.W * np.tan(cam.e)) * k + off[1]
    return float(np.clip(hy, 0.30 * H, 0.58 * H))


def render_background(img, scene, bg, W, H, seed=0):
    """Sky above the horizon, ground receding below it, and a lateral
    brightness ramp on the side the light comes from."""
    import cv2
    rng = np.random.default_rng(seed + 991)
    hy = horizon_row(scene, W, H)
    sky_lo, sky_hi = [np.array(c, np.float32) for c in bg["sky"]]
    g_near, g_far = [np.array(c, np.float32) for c in bg["ground"]]
    haze = np.array(bg["haze"], np.float32)

    rows = np.arange(H, dtype=np.float32)
    sky_t = np.clip(rows / max(hy, 1), 0, 1)[:, None]
    sky = sky_lo[None] * (1 - sky_t) + sky_hi[None] * sky_t

    # ^0.55 compresses the far ground toward the horizon, emulating 
    # perspective 
    gt = np.clip((rows - hy) / max(H - hy, 1), 0, 1)[:, None] ** 0.55
    ground = g_far[None] * (1 - gt) + g_near[None] * gt
    ground = ground * (0.55 + 0.45 * gt) + haze[None] * (1 - gt) * 0.45

    img[:] = np.where((rows >= hy)[:, None, None], ground[:, None, :],
                      sky[:, None, :])

    # texture, so the ground is not a flat wash
    n = rng.normal(0, 1, (H // 6, W // 6)).astype(np.float32)
    n = cv2.GaussianBlur(cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC),
                         (0, 0), 5)
    n /= max(np.abs(n).max(), 1e-6)
    below = (np.arange(H)[:, None] >= hy).astype(np.float32)
    img *= (1.0 + 0.10 * n * below)[:, :, None]

    ramp = np.linspace(1.0, -1.0, W, dtype=np.float32) * (-np.sign(LIGHT[0]))
    img *= (1.0 + 0.06 * ramp)[None, :, None]
    return np.clip(img, 0, 255)


def cast_shadow(img, organ_mask, scene, W, H, strength):
    """
    Uses plant silhouette projected onto the ground and offset it away from the
    light. Not geometrically exact
    """
    import cv2
    if strength <= 0 or not organ_mask.any():
        return img
    ys, xs = np.nonzero(organ_mask)
    base = max(ys.max() * 0.97, horizon_row(scene, W, H) + 0.12 * H)
    dx = -LIGHT[0] / max(abs(LIGHT[2]), 1e-6) * 0.22 * H
    M = np.float32([[1, 0, dx], [0, 0.16, base * (1 - 0.16)]])
    sh = cv2.warpAffine(organ_mask.astype(np.float32), M, (W, H))
    sh = cv2.GaussianBlur(sh, (0, 0), max(W // 90, 3))
    sh = np.clip(sh, 0, 1)[:, :, None]
    return img * (1.0 - strength * sh)


def render_flat(organs, tags, W, H, blur=2, seed=0, scene=None,
                background="plain"):
    """Shaded colour map for img2img. Colour is painted at the right
    coordinates, so binding is positional and cannot bleed between organs."""
    import cv2
    rng = np.random.default_rng(seed)
    bg = BACKGROUNDS[background]
    img = np.zeros((H, W, 3), np.float32)
    if scene is not None and background != "plain":
        img = render_background(img, scene, bg, W, H, seed)
    else:
        img[:] = np.array(BG_RGB, np.float32)

    if scene is not None and bg["shadow"] > 0:
        um = Image.new("L", (W, H), 0)
        ud = ImageDraw.Draw(um)
        for o in organs:
            xy = _xy(o.outline)
            if len(xy) > 2:
                ud.polygon(xy, fill=255)
        img = cast_shadow(img, np.array(um) > 0, scene, W, H, bg["shadow"])

    noise = rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32)
    noise = cv2.GaussianBlur(cv2.resize(noise, (W, H), cv2.INTER_CUBIC),
                             (0, 0), 9)
    noise /= max(np.abs(noise).max(), 1e-6)
    fcol = np.array(FLOWER_RGB[tags["flower_colour"]], np.float32)

    for o in sorted(organs, key=lambda x: -x.depth):
        xy = _xy(o.outline)
        if len(xy) < 3:
            continue
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(xy, fill=255)
        mask = np.array(m) > 0
        if not mask.any():
            continue
        base = fcol if o.role == "flower" else np.array(
            LEAF_RGB if o.role == "leaf" else STEM_RGB, np.float32)
        sh = _cyl_shade(mask)
        sh = sh[mask][:, None] if sh is not None else 1.0
        dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
        edge = np.clip(dt / max(.018 * min(W, H), 1.), 0, 1)[mask][:, None]
        px = base[None] * sh * (.80 + .20 * edge)
        if o.role in ("leaf", "stem"):
            px = px * (1. + .07 * noise[mask][:, None])
        img[mask] = np.clip(px, 0, 255)

        for q in o.inner:                     # veins as shading, not strokes
            if o.role != "leaf" or len(q) < 2:
                continue
            vm = Image.new("L", (W, H), 0)
            ImageDraw.Draw(vm).line(_xy(q), fill=255, width=2)
            img[(np.array(vm) > 0) & mask] *= .88

    out = Image.fromarray(img.astype(np.uint8))
    return out.filter(ImageFilter.GaussianBlur(blur)) if blur else out


def render_mask(organs, W, H, dilate=0):
    """Binary silhouette of the whole plant: white plant, black elsewhere.

    This is what lets the background remain as a real photograph for real background tests. Inpainting is instructed
    to regenerate only inside this mask, so the habitat tile behind it is never
    changed by the model.
    """
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for o in organs:
        xy = _xy(o.outline)
        if len(xy) > 2:
            d.polygon(xy, fill=255)
            d.line(xy + xy[:1], fill=255, width=3)
    a = np.array(m)
    if dilate:
        import cv2
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (dilate * 2 + 1, dilate * 2 + 1))
        a = cv2.dilate(a, k)
    return Image.fromarray(a)


# -------------------------------------------------------------------- prompts

def make_prompt(tags, background="plain", count=1, aperture="f/8"):
    """
    Short, front-loaded, and free of contradictions.

    CLIP weights early tokens more and dilutes with length, so the location
    rides with the subject rather than trailing at the end. An aperture is a
    concrete instruction about depth of field and used over less direct instructions.
    """
    bg = BACKGROUNDS[background]
    subject = ("a small clump of plants" if count > 1 else "a single plant")
    return ", ".join([
        f"{subject} with {tags['flower_colour']} flowers {bg['phrase']}",
        "green leaves",                 # or the flower colour bleeds onto them
        SURFACE_PHRASE[tags["leaf_surface"]],
        "realistic photograph", "natural daylight", "sharp focus",
        "fine leaf and petal detail",
        f"shot on a 100mm macro lens at {aperture}"])


def negative_for(tags, background="plain"):
    others = [c for c in FLOWER_COLOURS if c != tags["flower_colour"]]
    neg = (", ".join(f"{c} flowers" for c in others) + ", "
           "pink leaves, purple leaves, coloured leaves, "
           "seed head, catkin, feathery plume, "
           "drawing, sketch, line art, illustration, cartoon, painting, "
           "oversaturated, neon, blurry subject, lowres, worst quality, "
           "watermark, text, person, human, face, hands, "
           "floating plant, cut out, isolated on white, studio lighting, "
           "heavy bokeh, extreme background blur, dreamy haze, soft focus, "
           "horizon line, hard horizontal edge")
    if background == "plain":
        
        neg += ", extra leaves, additional foliage, multiple plants, grass"
    return neg


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-shape", default="ovate", choices=LEAF_SHAPES)
    ap.add_argument("--leaf-margin", default="entire", choices=LEAF_MARGINS)
    ap.add_argument("--flower-structure", default="campanulate",
                    choices=FLOWER_STRUCTURES)
    ap.add_argument("--inflorescence", default="solitary",
                    choices=INFLORESCENCES)
    ap.add_argument("--flower-colour", default="pink", choices=FLOWER_COLOURS)
    ap.add_argument("--leaf-surface", default="hairy", choices=LEAF_SURFACES)
    ap.add_argument("--phyllotaxis", default="opposite", choices=PHYLLOTAXIS)
    ap.add_argument("--nodes", type=int, default=3)
    ap.add_argument("--leaf-scale", type=float, default=1.0)
    ap.add_argument("--count", type=int, default=1,
                    help="individuals in the clump; jittered "
                         "down by up to 2 when --jitter > 0")
    ap.add_argument("--background", default="field",
                    choices=list(BACKGROUNDS))
    ap.add_argument("--all-backgrounds", action="store_true",
                    help="emit one set per background")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--line-width", type=int, default=3)
    ap.add_argument("--blur", type=int, default=2)
    ap.add_argument("--aperture", default="f/8",
                    help="goes in the prompt; f/11 for a sharper "
                         "background, f/2.8 for more separation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.7)
    ap.add_argument("--variants", type=int, default=1)
    ap.add_argument("--sweep", default=None,
                    choices=["leaf_shape", "leaf_margin", "flower_structure",
                             "inflorescence"])
    ap.add_argument("-o", "--out", default="out/plant")
    args = ap.parse_args()

    base = dict(leaf_shape=args.leaf_shape, leaf_margin=args.leaf_margin,
                flower_structure=args.flower_structure,
                inflorescence=args.inflorescence,
                flower_colour=args.flower_colour,
                leaf_surface=args.leaf_surface)
    values = {"leaf_shape": LEAF_SHAPES, "leaf_margin": LEAF_MARGINS,
              "flower_structure": FLOWER_STRUCTURES,
              "inflorescence": INFLORESCENCES}
    specs = ([{**base, args.sweep: v} for v in values[args.sweep]]
             if args.sweep else [base])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for tags in specs:
        suf = ("_" + tags[args.sweep].replace(" ", "-")) if args.sweep else ""
        for v in range(args.variants):
            seed = args.seed + v
            stem = Path(str(out) + suf + (f"_v{v}" if args.variants > 1 else ""))
            organs, scene = build(tags, args.size, args.size, args.jitter,
                                  np.random.default_rng(seed),
                                  args.phyllotaxis, args.nodes,
                                  args.leaf_scale, args.count)
            bgs = (list(BACKGROUNDS) if args.all_backgrounds
                   else [args.background])
            for bg in bgs:
                st = Path(str(stem) + (f"_{bg}" if len(bgs) > 1 else ""))
                render(organs, args.size, args.size, args.line_width,
                       scene, bg).save(st.with_suffix(".png"))
                render_flat(organs, tags, args.size, args.size, args.blur,
                            seed, scene, bg).save(Path(str(st) + "_flat.png"))
                render_mask(organs, args.size, args.size).save(
                    Path(str(st) + "_mask.png"))
                st.with_suffix(".txt").write_text(
                    make_prompt(tags, bg, args.count, args.aperture))
                st.with_suffix(".neg.txt").write_text(negative_for(tags, bg))
                st.with_suffix(".json").write_text(json.dumps(
                    {**tags, "_seed": seed, "_jitter": args.jitter,
                     "_phyllotaxis": args.phyllotaxis, "_nodes": args.nodes,
                     "_count": args.count, "_background": bg}, indent=2))
                print(f"{st.with_suffix('.png')}  seed={seed}  bg={bg}")


if __name__ == "__main__":
    main()
