"""Generate the corpus-assay logo set as self-contained SVGs (text outlined to paths).

Usage: python docs/assets/make_logos.py [OUT_DIR]

Needs ``fonttools`` and Poppins Bold/Medium (SIL OFL 1.1, Google Fonts) as
``Poppins-Bold.ttf`` / ``Poppins-Medium.ttf`` in ``$POPPINS_DIR`` (default: this
directory). The fonts are only needed to regenerate; the SVGs embed outlines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).parent
FONTS = Path(os.environ.get("POPPINS_DIR", HERE))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#0F172A"
BLUE = "#3B82F6"
LIGHT = "#93C5FD"
ACCENT = "#F59E0B"
SLATE = "#64748B"  # tagline on light backgrounds
WHITE = "#FFFFFF"
SLATE_DARK_BG = "#94A3B8"  # tagline on dark backgrounds

BOLD = TTFont(FONTS / "Poppins-Bold.ttf")
MEDIUM = TTFont(FONTS / "Poppins-Medium.ttf")


def text_path(
    font: TTFont, text: str, size: float, tracking: float = 0.0
) -> tuple[str, float]:
    """Outline ``text`` at ``size`` px (baseline at y=0); ``tracking`` is in em."""
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    x = 0.0
    parts = []
    for i, ch in enumerate(text):
        name = cmap[ord(ch)]
        pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x, 0)))
        parts.append(pen.getCommands())
        x += font["hmtx"][name][0] * scale
        if i < len(text) - 1:
            x += tracking * size
    return " ".join(p for p in parts if p), x


# ---------------------------------------------------------------------------
# Icon geometry, in a 246 x 150 box: two brackets (the right one carries ruler
# ticks), a light "benchmark" token, a blue "corpus" token, and their overlap.
# ---------------------------------------------------------------------------
ICON_W, ICON_H = 246.0, 150.0
STROKE = 18.0
ARM = 46.0
TICK_W = 3.5


def bracket_paths() -> list[str]:
    r = 4.0  # outer corner radius
    left = (
        f"M{ARM} 0 H{r} Q0 0 0 {r} V{ICON_H - r} Q0 {ICON_H} {r} {ICON_H} H{ARM} "
        f"V{ICON_H - STROKE} H{STROKE} V{STROKE} H{ARM} Z"
    )
    x1 = ICON_W
    right = (
        f"M{x1 - ARM} 0 H{x1 - r} Q{x1} 0 {x1} {r} V{ICON_H - r} Q{x1} {ICON_H} {x1 - r} {ICON_H} "
        f"H{x1 - ARM} V{ICON_H - STROKE} H{x1 - STROKE} V{STROKE} H{x1 - ARM} Z"
    )
    return [left, right]


def ticks() -> list[tuple[float, float, float]]:
    """(x, y, length) of ruler ticks along the right bracket's inner edge."""
    inner = ICON_W - STROKE
    out = []
    for i in range(7):
        y = 40 + i * 13.5
        length = 13.0 if i % 2 == 0 else 7.5
        out.append((inner - length, y - TICK_W / 2, length))
    return out


LEFT_TOKEN = (38.0, 33.0, 104.0, 72.0)  # x, y, w, h
RIGHT_TOKEN = (101.0, 54.0, 101.0, 71.0)
TOKEN_R = 12.0
OVERLAP_R = 7.0


def overlap_rect() -> tuple[float, float, float, float]:
    lx, ly, lw, lh = LEFT_TOKEN
    rx, ry, rw, rh = RIGHT_TOKEN
    x0, y0 = max(lx, rx), max(ly, ry)
    x1, y1 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    return x0, y0, x1 - x0, y1 - y0


def rect(
    r: tuple[float, float, float, float], radius: float, fill: str, extra: str = ""
) -> str:
    x, y, w, h = r
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{radius:g}" '
        f'fill="{fill}"{extra}/>'
    )


def icon_group(
    *,
    ink: str = NAVY,
    light: str = LIGHT,
    blue: str = BLUE,
    accent: str = ACCENT,
    brackets: bool = True,
    token_opacity: tuple[str, str, str] | None = None,
) -> str:
    els = []
    if brackets:
        els += [f'<path d="{d}" fill="{ink}"/>' for d in bracket_paths()]
        els += [
            f'<rect x="{x:g}" y="{y:g}" width="{length:g}" height="{TICK_W:g}" rx="1" fill="{ink}"/>'
            for x, y, length in ticks()
        ]
    if token_opacity:
        lo, ro, oo = token_opacity
        els.append(rect(LEFT_TOKEN, TOKEN_R, ink, f' fill-opacity="{lo}"'))
        els.append(rect(RIGHT_TOKEN, TOKEN_R, ink, f' fill-opacity="{ro}"'))
        els.append(rect(overlap_rect(), OVERLAP_R, ink, f' fill-opacity="{oo}"'))
    else:
        els.append(rect(LEFT_TOKEN, TOKEN_R, light))
        els.append(rect(RIGHT_TOKEN, TOKEN_R, blue))
        els.append(rect(overlap_rect(), OVERLAP_R, accent))
    return "\n  ".join(els)


def svg(
    width: float, height: float, body: str, title: str, view: str | None = None
) -> str:
    view = view or f"0 0 {width:g} {height:g}"
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" '
        f'viewBox="{view}" role="img" aria-label="{title}">\n'
        f"  <title>{title}</title>\n  {body}\n</svg>\n"
    )


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content, encoding="utf-8")
    print("wrote", OUT / name)


# ---------------------------------------------------------------------------
# Standalone icons (padded square-ish canvas)
# ---------------------------------------------------------------------------
PAD = 12.0
iw, ih = ICON_W + 2 * PAD, ICON_H + 2 * PAD
tx = f'transform="translate({PAD:g} {PAD:g})"'

write("icon.svg", svg(iw, ih, f"<g {tx}>\n  {icon_group()}\n  </g>", "corpus-assay"))
write(
    "icon-mono.svg",
    svg(
        iw,
        ih,
        f"<g {tx}>\n  {icon_group(token_opacity=('0.25', '0.45', '0.7'))}\n  </g>",
        "corpus-assay",
    ),
)

# On a dark tile: white brackets, same tokens.
tile_pad = 34.0
tw, th = ICON_W + 2 * tile_pad, ICON_H + 2 * tile_pad
dark_tile = (
    f'<rect width="{tw:g}" height="{th:g}" rx="28" fill="{NAVY}"/>\n  '
    f'<g transform="translate({tile_pad:g} {tile_pad:g})">\n  {icon_group(ink=WHITE)}\n  </g>'
)
write("icon-dark.svg", svg(tw, th, dark_tile, "corpus-assay"))

# Favicon / app icon: tokens only, centered in a square.
lx, ly = LEFT_TOKEN[0], LEFT_TOKEN[1]
rx_end = RIGHT_TOKEN[0] + RIGHT_TOKEN[2]
ry_end = RIGHT_TOKEN[1] + RIGHT_TOKEN[3]
tok_w, tok_h = rx_end - lx, ry_end - ly
side = max(tok_w, tok_h) + 24
fx, fy = (side - tok_w) / 2 - lx, (side - tok_h) / 2 - ly
favicon = (
    f'<g transform="translate({fx:g} {fy:g})">\n  {icon_group(brackets=False)}\n  </g>'
)
write("favicon.svg", svg(side, side, favicon, "corpus-assay"))

# ---------------------------------------------------------------------------
# Horizontal logo: icon + wordmark + tagline
# ---------------------------------------------------------------------------
WORD_SIZE = 92.0
word_d, word_w = text_path(BOLD, "corpus-assay", WORD_SIZE, tracking=-0.02)
TAG_TEXT = "AUDIT BENCHMARK OVERLAP"
# Size the tagline so its tracked width matches the wordmark's.
TAG_TRACK = 0.24
probe_d, probe_w = text_path(MEDIUM, TAG_TEXT, 100.0, tracking=TAG_TRACK)
TAG_SIZE = 100.0 * (word_w * 0.965) / probe_w
tag_d, tag_w = text_path(MEDIUM, TAG_TEXT, TAG_SIZE, tracking=TAG_TRACK)

GAP = 30.0
text_x = ICON_W + GAP
word_baseline = 94.0  # within the 150 px icon height
tag_baseline = ICON_H - 2.0
logo_w = text_x + max(word_w, tag_w) + 4
logo_h = ICON_H


def horizontal(ink: str, tagline: str, *, bracket_ink: str | None = None) -> str:
    body = (
        f"<g>\n  {icon_group(ink=bracket_ink or ink)}\n  </g>\n  "
        f'<path transform="translate({text_x:g} {word_baseline:g})" d="{word_d}" fill="{ink}"/>\n  '
        f'<path transform="translate({text_x + 3:g} {tag_baseline:g})" d="{tag_d}" fill="{tagline}"/>'
    )
    pad = 8.0
    return svg(
        logo_w + 2 * pad,
        logo_h + 2 * pad,
        body,
        "corpus-assay: audit benchmark overlap",
        view=f"{-pad:g} {-pad:g} {logo_w + 2 * pad:g} {logo_h + 2 * pad:g}",
    )


write("logo.svg", horizontal(NAVY, SLATE))
write("logo-dark.svg", horizontal(WHITE, SLATE_DARK_BG))
print(
    f"wordmark {word_w:.1f}px, tagline {tag_w:.1f}px at {TAG_SIZE:.1f}px, logo {logo_w:.0f}x{logo_h:.0f}"
)
