"""Pixel-faithful browser renderer for the native SmartMelt furnace canvas.

This module deliberately has no Streamlit dependency so geometry and SVG output
can be regression-tested independently of the web runtime.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import html
import random

SLAG_COL = "#7d6b48"
SLAG_TOP = "#a08a5a"
SCRAP_COL = "#8792a0"
FLUX_COL = "#ece6d4"
LINING = "#4a3527"
LINING_HL = "#5e4433"
COIL = "#c8802f"
CAVITY = "#1a1410"

def metal_colour(T_bath_C: float, tap_aim_C: float) -> str:
    """Exact browser equivalent of gui.theme.metal_colour()."""
    frac = max(0.0, min(1.0, (T_bath_C - 1150.0) / (tap_aim_C + 40.0 - 1150.0)))
    r = int(196 + 59 * frac)
    g = int(46 + 130 * frac)
    b = int(12 + 26 * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


def furnace_geometry(
    melted_pct: float,
    T_bath_C: float,
    slag_kg: float,
    undissolved_kg: float,
    heat_size_t: float = 12.0,
    tap_aim_C: float = 1620.0,
    width: int = 340,
    height: int = 240,
) -> Dict[str, float | str]:
    """Return the exact geometry used by the native Tk FurnaceCanvas.

    Keeping this as a separate pure function lets the regression tests verify
    that liquid level, slag level/thickness and temperature colour are identical
    in the desktop GUI and browser GUI for every process state.
    """
    melted = max(0.0, min(1.0, melted_pct / 100.0))
    W, H = float(width), float(height)
    cx = W * 0.42
    cw = W * 0.42
    x0, x1 = cx - cw / 2.0, cx + cw / 2.0
    top, bot = H * 0.14, H * 0.85
    cav_h = bot - top
    usable = cav_h * 0.90
    liq_h = usable * melted
    liq_top = bot - liq_h
    slag_h = min(16.0, 6.0 + float(slag_kg) / 10.0) if (slag_kg > 0 and liq_h > 3) else 0.0
    slag_top = liq_top - slag_h
    solid_bot = slag_top if slag_h > 0 else liq_top
    solid_h = usable * (1.0 - melted)
    solid_top = max(top + 3.0, solid_bot - solid_h)
    return {
        "melted": melted, "W": W, "H": H, "cx": cx, "cw": cw,
        "x0": x0, "x1": x1, "top": top, "bot": bot, "cav_h": cav_h,
        "usable": usable, "liq_h": liq_h, "liq_top": liq_top,
        "slag_h": slag_h, "slag_top": slag_top, "solid_bot": solid_bot,
        "solid_h": solid_h, "solid_top": solid_top,
        "metal_colour": metal_colour(T_bath_C, tap_aim_C),
        "T_bath_C": float(T_bath_C), "slag_kg": float(slag_kg),
        "undissolved_kg": float(undissolved_kg), "heat_size_t": float(heat_size_t),
        "tap_aim_C": float(tap_aim_C),
    }


def furnace_svg(
    melted_pct: float,
    T_bath_C: float,
    slag_kg: float,
    undissolved_kg: float,
    heat_size_t: float = 12.0,
    tap_aim_C: float = 1620.0,
    height: int = 240,
    previous: Optional[Tuple[float, float, float, float]] = None,
    animate_ms: int = 220,
) -> str:
    """Render a pixel-faithful SVG translation of ``FurnaceCanvas.draw``.

    The desktop code uses a 340 × 240 Tk canvas in the Operator Console. This
    renderer uses that same coordinate system, formulae, random seeds, colours,
    labels and layer order. ``previous`` optionally adds a short interpolation
    from the preceding live state so Streamlit updates appear continuous rather
    than jumping between reruns.
    """
    W, H = 340, 240
    g = furnace_geometry(melted_pct, T_bath_C, slag_kg, undissolved_kg,
                         heat_size_t, tap_aim_C, W, H)
    pg = None
    if previous is not None:
        try:
            pm, pt, ps, pu = previous
            pg = furnace_geometry(pm, pt, ps, pu, heat_size_t, tap_aim_C, W, H)
        except Exception:
            pg = None

    x0, x1 = g["x0"], g["x1"]
    top, bot = g["top"], g["bot"]
    cav_h = g["cav_h"]
    melted = g["melted"]
    liq_h, liq_top = g["liq_h"], g["liq_top"]
    slag_h, slag_top = g["slag_h"], g["slag_top"]
    solid_bot, solid_top = g["solid_bot"], g["solid_top"]
    xr = x1 + 28.0

    def rect(x, y, w, h, fill, stroke="none", sw=0, extra=""):
        return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')

    def anim(attr: str, old: float | str, new: float | str) -> str:
        if pg is None or old == new:
            return ""
        return (f'<animate attributeName="{attr}" from="{old}" to="{new}" '
                f'dur="{max(60, int(animate_ms))}ms" fill="freeze" calcMode="spline" '
                f'keySplines="0.2 0.0 0.2 1.0"/>')

    coils = []
    yy = top + 6.0
    while yy < bot - 6.0:
        coils.append(rect(x0 - 22, yy, 12, 11, COIL))
        coils.append(rect(x1 + 10, yy, 12, 11, COIL))
        yy += 17.0

    liquid = ""
    if liq_h > 1:
        old_top = pg["liq_top"] if pg else liq_top
        old_h = pg["liq_h"] if pg else liq_h
        old_col = pg["metal_colour"] if pg else g["metal_colour"]
        liquid = (
            f'<rect data-layer="liquid" x="{x0:.2f}" y="{old_top:.2f}" '
            f'width="{x1-x0:.2f}" height="{old_h:.2f}" fill="{old_col}">'
            f'{anim("y", old_top, liq_top)}{anim("height", old_h, liq_h)}'
            f'{anim("fill", old_col, g["metal_colour"])}</rect>'
            f'<rect data-layer="liquid-surface" x="{x0:.2f}" y="{old_top:.2f}" '
            f'width="{x1-x0:.2f}" height="4" fill="#ffd166">'
            f'{anim("y", old_top, liq_top)}</rect>'
        )

    slag = ""
    if slag_h > 0:
        if pg and pg["slag_h"] > 0:
            old_top, old_h = pg["slag_top"], pg["slag_h"]
        else:
            old_top, old_h = slag_top, slag_h
        slag = (
            f'<rect data-layer="slag" x="{x0:.2f}" y="{old_top:.2f}" '
            f'width="{x1-x0:.2f}" height="{old_h:.2f}" fill="{SLAG_COL}">'
            f'{anim("y", old_top, slag_top)}{anim("height", old_h, slag_h)}</rect>'
            f'<rect data-layer="slag-surface" x="{x0:.2f}" y="{old_top:.2f}" '
            f'width="{x1-x0:.2f}" height="3" fill="{SLAG_TOP}">'
            f'{anim("y", old_top, slag_top)}</rect>'
        )

    scrap = []
    if melted < 0.985 and (solid_bot - solid_top) > 6:
        rng = random.Random(7)
        n = int(8 + 16 * (1 - melted))
        for _ in range(n):
            px = rng.uniform(x0 + 5, x1 - 14)
            py = rng.uniform(solid_top + 3, solid_bot - 8)
            w = rng.uniform(7, 14)
            h = rng.uniform(5, 9)
            scrap.append(rect(px, py, w, h, SCRAP_COL, "#69727e", 1))

    lumps = []
    if undissolved_kg > 1 and liq_h > 8:
        rng = random.Random(int(undissolved_kg))
        n = int(min(9, 1 + undissolved_kg / 10.0))
        for _ in range(n):
            px = rng.uniform(x0 + 8, x1 - 8)
            py = rng.uniform(liq_top + 3, min(bot - 6, liq_top + 24))
            rr = rng.uniform(3, 5.5)
            lumps.append(f'<ellipse cx="{px:.2f}" cy="{py:.2f}" rx="{rr:.2f}" ry="{rr:.2f}" '
                         f'fill="{FLUX_COL}" stroke="#b7ae95" stroke-width="1"/>')

    def label(y, text, colour):
        yy = max(top + 6, min(bot - 4, y))
        return (f'<line x1="{x1:.2f}" y1="{yy:.2f}" x2="{xr-4:.2f}" y2="{yy:.2f}" '
                f'stroke="{colour}" stroke-width="1"/>'
                f'<text x="{xr:.2f}" y="{yy+3:.2f}" fill="{colour}" font-size="9" '
                f'font-family="Segoe UI, DejaVu Sans, Arial, sans-serif">{html.escape(text)}</text>')

    labels = []
    if liq_h > 3:
        labels.append(label((liq_top + bot) / 2, f"metal {melted*100:.0f}%", "#ffb066"))
    if slag_h > 0:
        labels.append(label(slag_top + slag_h / 2, "slag", SLAG_TOP))
    if melted < 0.985 and (solid_bot - solid_top) > 8:
        labels.append(label((solid_top + solid_bot) / 2, "scrap", "#aab3bf"))
    if undissolved_kg > 1 and liq_h > 8:
        labels.append(label(liq_top + 8, f"flux {undissolved_kg:.0f}kg", FLUX_COL))

    badge_col = "#ffd166" if T_bath_C > (1150 + tap_aim_C) / 2 else "#e07a4a"
    svg = f"""<div class="furnace-frame" data-melted="{melted_pct:.4f}" data-temp="{T_bath_C:.3f}"
         data-liquid-top="{liq_top:.4f}" data-liquid-height="{liq_h:.4f}"
         data-slag-top="{slag_top:.4f}" data-slag-height="{slag_h:.4f}">
<svg viewBox="0 0 {W} {H}" width="340" height="{height}" preserveAspectRatio="xMinYMin meet"
     xmlns="http://www.w3.org/2000/svg" role="img"
     aria-label="Furnace state: {melted_pct:.0f}% melted, bath {T_bath_C:.0f} degrees C"
     style="display:block;background:#0a0d10;shape-rendering:geometricPrecision;text-rendering:geometricPrecision">
  {''.join(coils)}
  {rect(x0-13, top-15, x1-x0+26, cav_h+31, LINING, '#2b1d13', 2)}
  {rect(x0-13, top-15, 4, cav_h+31, LINING_HL)}
  {rect(x0, top, x1-x0, cav_h, CAVITY)}
  {liquid}
  {slag}
  {''.join(scrap)}
  {''.join(lumps)}
  {rect(x0-15, top-18, x1-x0+30, 7, '#5e4433')}
  {rect(x0-8, 8, 100, 24, '#182027', '#232c33', 1)}
  <text x="{x0-2:.2f}" y="25" fill="{badge_col}" font-size="13" font-weight="700"
        font-family="Segoe UI, DejaVu Sans, Arial, sans-serif">{T_bath_C:.0f} °C</text>
  <text x="{x0+62:.2f}" y="24" fill="#aeb7c1" font-size="9"
        font-family="Segoe UI, DejaVu Sans, Arial, sans-serif">bath</text>
  {''.join(labels)}
  <text x="{g['cx']:.2f}" y="{bot+30:.2f}" fill="#aeb7c1" font-size="9" text-anchor="middle"
        font-family="Segoe UI, DejaVu Sans, Arial, sans-serif">coreless induction · {heat_size_t:.0f} t · aim {tap_aim_C:.0f} °C</text>
</svg></div>"""
    return svg
