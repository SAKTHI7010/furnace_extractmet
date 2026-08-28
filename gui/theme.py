"""
SmartMelt Studio — desktop GUI theme + the coloured furnace canvas.

The furnace is drawn on a native Tk Canvas (no external rendering), showing the
live level of molten metal (colour shifts with bath temperature), the slag cap,
remaining solid scrap, floating undissolved flux lumps, the refractory lining
and the copper coil — the same picture the shop-floor operator reads at a glance.
"""
from __future__ import annotations

import tkinter as tk

# ── palette (matches the working HTML console) ──────────────────────────────
BG_DEEP = "#0a0d10"
BG_PANEL = "#12171b"
BG_RAISED = "#182027"
BG_INPUT = "#0e1317"
LINE = "#232c33"
TEXT = "#e9edf0"
TEXT_MUT = "#9aa4af"
TEXT_DIM = "#6b757f"
MOLTEN = "#ff6a34"
MOLTEN_HI = "#ffd166"
STEEL = "#4fa8d8"
GREEN = "#33d17a"
AMBER = "#f0a83c"
RED = "#e5484d"

# furnace element colours
SLAG_COL = "#7d6b48"
SLAG_TOP = "#a08a5a"
SCRAP_COL = "#8792a0"
SCRAP_EDGE = "#69727e"
FLUX_COL = "#ece6d4"
LINING = "#4a3527"
LINING_HL = "#5e4433"
COIL = "#c8802f"
CAVITY = "#1a1410"

FONT = "Segoe UI"
FONT_MONO = "Consolas"


def metal_colour(T_bath_C: float, tap_aim_C: float) -> str:
    """Molten-metal colour: deep red when cool, bright orange-gold when hot."""
    frac = max(0.0, min(1.0, (T_bath_C - 1150.0) / (tap_aim_C + 40.0 - 1150.0)))
    r = int(196 + 59 * frac)
    g = int(46 + 130 * frac)
    b = int(12 + 26 * frac)
    return f"#{r:02x}{g:02x}{b:02x}"


class FurnaceCanvas(tk.Canvas):
    """A coreless-induction cross-section that redraws to reflect live state."""

    def __init__(self, master, width=300, height=380, **kw):
        super().__init__(master, width=width, height=height,
                         bg=BG_DEEP, highlightthickness=0, **kw)
        self._cw = width
        self._ch = height
        self.bind("<Configure>", self._on_resize)
        self.draw(0.0, 30.0, 0.0, 0.0)

    def _on_resize(self, event):
        self._cw, self._ch = event.width, event.height
        if hasattr(self, "_last"):
            self.draw(*self._last)

    def draw(self, melted_pct: float, T_bath_C: float, slag_kg: float,
             undissolved_kg: float, heat_size_t: float = 12.0,
             tap_aim_C: float = 1620.0):
        """Redraw the furnace for the given melt state."""
        self._last = (melted_pct, T_bath_C, slag_kg, undissolved_kg,
                      heat_size_t, tap_aim_C)
        self.delete("all")
        W, H = self._cw, self._ch
        melted = max(0.0, min(1.0, melted_pct / 100.0))

        # geometry
        cx = W * 0.42
        cw = W * 0.42
        x0, x1 = cx - cw / 2, cx + cw / 2
        top, bot = H * 0.14, H * 0.85
        cav_h = bot - top

        # coil turns down both sides
        yy = top + 6
        while yy < bot - 6:
            self.create_rectangle(x0 - 22, yy, x0 - 10, yy + 11, fill=COIL, width=0)
            self.create_rectangle(x1 + 10, yy, x1 + 22, yy + 11, fill=COIL, width=0)
            yy += 17

        # refractory shell
        self.create_rectangle(x0 - 13, top - 15, x1 + 13, bot + 16,
                              fill=LINING, outline="#2b1d13", width=2)
        self.create_rectangle(x0 - 13, top - 15, x0 - 9, bot + 16,
                              fill=LINING_HL, width=0)
        # cavity
        self.create_rectangle(x0, top, x1, bot, fill=CAVITY, width=0)

        usable = cav_h * 0.90
        liq_h = usable * melted
        liq_top = bot - liq_h
        slag_h = min(16, 6 + slag_kg / 10.0) if (slag_kg > 0 and liq_h > 3) else 0
        slag_top = liq_top - slag_h
        solid_bot = slag_top if slag_h > 0 else liq_top
        solid_h = usable * (1 - melted)
        solid_top = max(top + 3, solid_bot - solid_h)

        # molten metal
        if liq_h > 1:
            mc = metal_colour(T_bath_C, tap_aim_C)
            self.create_rectangle(x0, liq_top, x1, bot, fill=mc, width=0)
            # bright surface band → the level is unmistakable
            self.create_rectangle(x0, liq_top, x1, liq_top + 4,
                                  fill=MOLTEN_HI, width=0)

        # slag cap
        if slag_h > 0:
            self.create_rectangle(x0, slag_top, x1, slag_top + slag_h,
                                  fill=SLAG_COL, width=0)
            self.create_rectangle(x0, slag_top, x1, slag_top + 3,
                                  fill=SLAG_TOP, width=0)

        # solid scrap chunks
        if melted < 0.985 and (solid_bot - solid_top) > 6:
            import random
            rng = random.Random(7)
            n = int(8 + 16 * (1 - melted))
            for _ in range(n):
                px = rng.uniform(x0 + 5, x1 - 14)
                py = rng.uniform(solid_top + 3, solid_bot - 8)
                w = rng.uniform(7, 14)
                h = rng.uniform(5, 9)
                self.create_rectangle(px, py, px + w, py + h,
                                      fill=SCRAP_COL, outline=SCRAP_EDGE, width=1)

        # undissolved flux lumps at the interface
        if undissolved_kg > 1 and liq_h > 8:
            import random
            rng = random.Random(int(undissolved_kg))
            n = int(min(9, 1 + undissolved_kg / 10.0))
            for _ in range(n):
                px = rng.uniform(x0 + 8, x1 - 8)
                py = rng.uniform(liq_top + 3, min(bot - 6, liq_top + 24))
                rr = rng.uniform(3, 5.5)
                self.create_oval(px - rr, py - rr, px + rr, py + rr,
                                fill=FLUX_COL, outline="#b7ae95", width=1)

        # lip
        self.create_rectangle(x0 - 15, top - 18, x1 + 15, top - 11,
                              fill="#5e4433", width=0)

        # temperature badge (top-left)
        self.create_rectangle(x0 - 8, 8, x0 + 92, 32, fill=BG_RAISED,
                              outline=LINE, width=1)
        bc = MOLTEN_HI if (T_bath_C > (1150 + tap_aim_C) / 2) else "#e07a4a"
        self.create_text(x0 - 2, 20, anchor="w", fill=bc,
                        font=(FONT, 13, "bold"), text=f"{T_bath_C:.0f} °C")
        self.create_text(x0 + 62, 21, anchor="w", fill=TEXT_MUT,
                        font=(FONT, 8), text="bath")

        # level callout labels (right side)
        xr = x1 + 28
        def label(y, txt, col):
            y = max(top + 6, min(bot - 4, y))
            self.create_line(x1, y, xr - 4, y, fill=col, width=1)
            self.create_text(xr, y, anchor="w", fill=col,
                            font=(FONT, 8), text=txt)

        if liq_h > 3:
            label((liq_top + bot) / 2, f"metal {melted*100:.0f}%", "#ffb066")
        if slag_h > 0:
            label(slag_top + slag_h / 2, "slag", SLAG_TOP)
        if melted < 0.985 and (solid_bot - solid_top) > 8:
            label((solid_top + solid_bot) / 2, "scrap", "#aab3bf")
        if undissolved_kg > 1 and liq_h > 8:
            label(liq_top + 8, f"flux {undissolved_kg:.0f}kg", FLUX_COL)

        # caption
        self.create_text(cx, bot + 30, fill=TEXT_MUT, font=(FONT, 8),
                        text=f"coreless induction · {heat_size_t:.0f} t · aim {tap_aim_C:.0f} °C")
