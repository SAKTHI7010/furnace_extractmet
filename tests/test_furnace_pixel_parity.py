from __future__ import annotations

import math
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from app.furnace_renderer import furnace_geometry, furnace_svg, metal_colour


def _native_formula(melted_pct, slag_kg, width=340, height=240):
    melted=max(0.0,min(1.0,melted_pct/100.0))
    cx=width*0.42; cw=width*0.42
    x0,x1=cx-cw/2,cx+cw/2
    top,bot=height*0.14,height*0.85
    cav_h=bot-top; usable=cav_h*0.90
    liq_h=usable*melted; liq_top=bot-liq_h
    slag_h=min(16,6+slag_kg/10.0) if (slag_kg>0 and liq_h>3) else 0
    slag_top=liq_top-slag_h
    solid_bot=slag_top if slag_h>0 else liq_top
    solid_h=usable*(1-melted)
    solid_top=max(top+3,solid_bot-solid_h)
    return locals()


def test_geometry_is_literal_native_canvas_formula():
    for melted,temp,slag,flux in [(0,30,0,0),(10,1489,122,0),(35,1519,116,0),
                                  (65,1563,117,20),(90,1631,118,50),(100,1660,140,0)]:
        got=furnace_geometry(melted,temp,slag,flux)
        exp=_native_formula(melted,slag)
        for name in ("x0","x1","top","bot","cav_h","usable","liq_h","liq_top",
                     "slag_h","slag_top","solid_bot","solid_h","solid_top"):
            assert math.isclose(float(got[name]),float(exp[name]),rel_tol=0,abs_tol=1e-10), name


def test_liquid_level_rises_and_temperature_colour_brightens():
    levels=[furnace_geometry(p,1200+4*p,110,0)["liq_top"] for p in (0,10,35,65,90,100)]
    assert all(a>b for a,b in zip(levels,levels[1:]))
    cool=metal_colour(1200,1620); hot=metal_colour(1650,1620)
    cool_rgb=tuple(int(cool[i:i+2],16) for i in (1,3,5))
    hot_rgb=tuple(int(hot[i:i+2],16) for i in (1,3,5))
    assert hot_rgb[0]>=cool_rgb[0] and hot_rgb[1]>cool_rgb[1] and hot_rgb[2]>=cool_rgb[2]


def test_slag_level_tracks_liquid_and_slag_inventory():
    low=furnace_geometry(25,1500,20,0)
    high=furnace_geometry(75,1600,20,0)
    thick=furnace_geometry(75,1600,80,0)
    assert high["slag_top"] < low["slag_top"]
    assert thick["slag_h"] > high["slag_h"]
    assert math.isclose(float(thick["slag_top"]),float(thick["liq_top"]-thick["slag_h"]),abs_tol=1e-12)


def test_svg_has_exact_state_layers_and_smooth_interpolation():
    svg=furnace_svg(65,1563,117,30,previous=(35,1519,116,0))
    assert 'viewBox="0 0 340 240"' in svg
    assert 'data-layer="liquid"' in svg
    assert 'data-layer="slag"' in svg
    assert 'fill="#ffd166"' in svg
    assert '<animate attributeName="y"' in svg
    assert '<animate attributeName="height"' in svg
    assert 'metal 65%' in svg and '>slag<' in svg and 'flux 30kg' in svg
    assert re.search(r'data-liquid-top="\d+\.\d+"',svg)
