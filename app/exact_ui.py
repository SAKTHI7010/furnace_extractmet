"""Shared UI helpers for SmartMelt Studio: theme, KPI cards, Plotly defaults,
and the coloured furnace cross-section (liquid metal, slag, solid scrap,
undissolved additions) that visualises the live state of the bath."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# palette (matches the operator console / deck)
INK = "#0d1013"
PANEL = "#151b21"
MOLT = "#e8622e"
MOLT_HI = "#ff8a4c"
STEEL = "#4fa8d8"
GREEN = "#33c17a"
AMBER = "#f5b301"
RED = "#e5484d"
MUT = "#8b93a1"
LINE = "#2a3138"

METAL_HOT = "#ff7a1a"   # molten iron (bright orange when hot)
METAL_MID = "#e24d0f"   # cooler melt (red-orange)
SLAG_COL = "#7d6b48"    # slag (distinct khaki, clearly above metal)
SLAG_TOP = "#a08a5a"    # slag highlight
SCRAP_COL = "#8792a0"   # cold solid charge (light steel-grey, reads on dark)
FLUX_COL = "#ece6d4"    # undissolved flux lumps (near-white)
LINING = "#4a3527"      # refractory (warm brown)
LINING_HL = "#5e4433"   # refractory highlight
COIL = "#c8802f"        # copper coil
CAVITY = "#1a1410"      # empty cavity (dark warm)


def inject_css():
    st.markdown(
        """
        <style>
        :root {
          --bg-deep:#0a0d10; --bg-panel:#12171b; --bg-raised:#182027;
          --line:#232c33; --text:#e9edf0; --text-mut:#9aa4af; --text-dim:#6b757f;
          --molten:#ff6a34; --molten-hi:#ffd166; --steel:#4fa8d8;
          --green:#33d17a; --amber:#f0a83c; --red:#e5484d;
        }
        /* base surfaces — cover EVERY Streamlit container incl. the header */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"],
        .main, .block-container { background: #0a0d10 !important; color: #e9edf0 !important; }
        /* the header/toolbar that was rendering WHITE at the top */
        [data-testid="stHeader"], header[data-testid="stHeader"],
        [data-testid="stToolbar"], [data-testid="stDecoration"] {
          background: #0a0d10 !important; color: #e9edf0 !important;
        }
        [data-testid="stDecoration"] { display: none !important; }  /* rainbow bar */
        [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
          background: #12171b !important;
        }
        [data-testid="stSidebar"] * { color: #d4dae1 !important; }
        /* body text — bright, legible on the near-black background */
        .stApp, .stMarkdown, p, li, label, span,
        [data-testid="stMarkdownContainer"] { color: #e9edf0; }
        h1,h2,h3,h4 { color:#f4f7fa !important; letter-spacing:.2px; }
        [data-testid="stCaptionContainer"], .stCaption, small { color:#9aa4af !important; }
        code { color:#8fd0f0 !important; background:#12171b !important;
               padding:1px 5px; border-radius:4px; }
        /* KPI cards */
        .kpi { background:#182027; border:1px solid #232c33; border-radius:10px;
               padding:12px 14px; height:100%; }
        .kpi .lab { color:#9aa4af; font-size:12px; text-transform:uppercase;
                    letter-spacing:1px; margin-bottom:4px; }
        .kpi .val { color:#ff8a4c; font-size:26px; font-weight:700; line-height:1.15;
                    font-family:'DejaVu Sans',sans-serif; }
        .kpi .sub { color:#aab3bf; font-size:11px; margin-top:2px; }
        /* status pills — solid, high-contrast */
        .pill { display:inline-block; padding:4px 12px; border-radius:20px;
                font-size:12px; font-weight:700; border:1px solid transparent; }
        .ok  { background:rgba(51,209,122,.18); color:#5fe3a3; border-color:rgba(51,209,122,.45);}
        .warn{ background:rgba(240,168,60,.18); color:#ffc766; border-color:rgba(240,168,60,.45);}
        .bad { background:rgba(229,72,77,.20);  color:#ff8286; border-color:rgba(229,72,77,.5);}
        .muted{ color:#aab3bf; font-size:13px; }
        div[data-testid="stMetricValue"]{ color:#ff8a4c; }
        div[data-testid="stMetricLabel"]{ color:#9aa4af; }
        /* buttons */
        .stButton button { background:#182027; color:#e9edf0; border:1px solid #2c3742; }
        .stButton button:hover { border-color:#ff6a34; color:#ffd166; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="kpi"><div class="lab">{label}</div>'
        f'<div class="val">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "ok") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def style_fig(fig: go.Figure, height: int = 320, title: str = "",
              ylab: str = "", y2lab: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0f1418",
        height=height,
        margin=dict(l=55, r=55 if y2lab else 15, t=34 if title else 12, b=38),
        title=dict(text=title, font=dict(size=14, color="#ecf0f3")) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        font=dict(family="DejaVu Sans, sans-serif", size=12, color="#c6ccd4"),
    )
    fig.update_xaxes(gridcolor="#20262c", zeroline=False)
    fig.update_yaxes(gridcolor="#20262c", zeroline=False, title_text=ylab)
    if y2lab:
        fig.update_layout(yaxis2=dict(title=y2lab, overlaying="y", side="right",
                                      gridcolor="rgba(0,0,0,0)"))
    return fig


# ────────────────────────────────────────────────────────────────────────────
# Coloured furnace cross-section — literal translation of gui/theme.py
# ────────────────────────────────────────────────────────────────────────────
from app.furnace_renderer import furnace_svg, furnace_geometry, metal_colour

def legend_row() -> str:
    items = [("Molten metal", METAL_HOT), ("Slag", SLAG_COL),
             ("Solid scrap", SCRAP_COL), ("Undissolved flux", FLUX_COL),
             ("Refractory", LINING), ("Coil", COIL)]
    chips = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;'
        f'font-size:12px;color:#c6ccd4;">'
        f'<span style="width:12px;height:12px;border-radius:3px;background:{c};'
        f'display:inline-block;margin-right:5px;"></span>{name}</span>'
        for name, c in items
    )
    return f'<div style="margin-top:6px;">{chips}</div>'

# Extra browser layout rules for the faithful single-window replica.
def inject_exact_css():
    inject_css()
    st.markdown(
        """
        <style>
        /* Stable desktop workspace. Stale elements stay visible during fragment
           reruns, so the screen never flashes white or appears to disappear. */
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {display:none !important;}
        [data-testid="stHeader"] {height:0 !important;min-height:0 !important;}
        [data-stale="true"] {opacity:1 !important;filter:none !important;}
        [data-testid="stAppViewContainer"], [data-testid="stMain"], .stApp,
        .block-container {transition:none !important;animation:none !important;}
        .block-container {max-width:1440px !important;width:98vw !important;
          padding:0.35rem 0.55rem 1rem !important;}
        .stApp {font-family:"Segoe UI",Arial,sans-serif;font-size:13px;}

        /* Native notebook row using tracked/lazy st.tabs. Full labels remain
           readable and the row scrolls horizontally on narrow screens. */
        .stTabs [data-baseweb="tab-list"] {gap:0;border-bottom:1px solid #66717a;
          overflow-x:auto !important;overflow-y:hidden;scrollbar-width:thin;
          flex-wrap:nowrap !important;padding-bottom:1px;}
        .stTabs [data-baseweb="tab"] {height:38px;min-width:max-content !important;
          flex:0 0 auto !important;padding:0 14px;background:#12171b;color:#aeb7c1;
          border:1px solid #66717a;border-bottom:0;border-radius:0;font-size:12px;
          white-space:nowrap;overflow:visible;text-overflow:clip;}
        .stTabs [aria-selected="true"] {background:#182027 !important;color:#ff7a3d !important;
          font-weight:700;}
        .stTabs [data-baseweb="tab-highlight"] {background:#ff6a34 !important;height:2px;}
        .stTabs [data-baseweb="tab-border"] {display:none;}
        .stTabs [data-baseweb="tab-panel"] {padding-top:0.35rem !important;}

        /* Controls: no cropped labels, no negative margins, comfortable hit area. */
        .stSlider {padding-top:0 !important;padding-bottom:0.1rem !important;}
        .stSlider [data-testid="stWidgetLabel"] {margin-bottom:1px !important;}
        .stSlider [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] p {font-size:12px !important;line-height:1.25 !important;
          color:#d7dde4 !important;}
        .stButton>button, .stDownloadButton>button {min-height:34px;height:auto !important;
          padding:5px 9px;border-radius:0;font-size:12px;line-height:1.15;
          text-overflow:ellipsis !important;}
        .stSelectbox [data-baseweb="select"]>div, .stNumberInput input, .stTextInput input {
          min-height:34px !important;height:34px !important;border-radius:0 !important;
          background:#0e1317 !important;color:#f0f3f6 !important;font-size:13px !important;}
        .stSelectbox [data-baseweb="select"] span {color:#f0f3f6 !important;font-size:12px !important;}
        .stNumberInput button {min-height:34px !important;}
        [data-testid="stDataFrame"] {border:1px solid #46515a;}

        .section-title {font-size:16px;font-weight:700;color:#f3f6f8;margin:5px 0 4px;
          line-height:1.3;overflow:visible;}
        .section-title-sm {font-size:13px;font-weight:700;color:#f3f6f8;margin:3px 0 3px;
          line-height:1.3;overflow:visible;}
        .kpi {border-radius:0 !important;padding:9px 11px !important;min-height:84px;}
        .kpi .lab {font-size:10.5px !important;line-height:1.2;}
        .kpi .val {font-size:22px !important;line-height:1.12;white-space:normal;}
        .kpi .sub {font-size:10.5px !important;line-height:1.2;}
        .adv-card {background:#182027;border:1px solid #232c33;padding:8px 9px;}
        .adv-card>div {min-width:0;}
        .adv-title {font-size:11.5px;font-weight:700;color:#f0f3f6;line-height:1.25;
          white-space:normal;overflow-wrap:anywhere;}
        .adv-msg {font-size:10.5px;line-height:1.35;color:#c2cad3;white-space:normal;
          overflow-wrap:anywhere;}
        .logbox {background:#0e1317;border:1px solid #66717a;padding:7px;min-height:68px;
          max-height:94px;overflow:auto;font:11px/1.35 Consolas,monospace;color:#c0c8d1;
          white-space:pre-wrap;overflow-wrap:anywhere;}
        .status-ok{color:#33d17a}.status-warn{color:#f0a83c}.status-bad{color:#e5484d}
        .thin-note{font-size:11px;color:#aeb7c1;line-height:1.35;white-space:normal;
          overflow-wrap:anywhere;}
        .pill {font-size:12px;line-height:1.25;white-space:normal;max-width:100%;}

        /* The calculation banner replaces blocking spinners. */
        .calc-banner {display:grid;grid-template-columns:12px 1fr;gap:2px 8px;
          align-items:center;background:#181b16;border:1px solid #8d6a21;padding:7px 8px;
          margin:3px 0 5px;color:#ffd071;font-size:11px;line-height:1.25;}
        .calc-banner span:last-child {grid-column:2;color:#c4cbd2;font-weight:400;}
        .calc-dot {width:9px;height:9px;border-radius:50%;background:#f0a83c;
          box-shadow:0 0 0 0 rgba(240,168,60,.6);animation:calcPulse 1.2s infinite !important;}
        @keyframes calcPulse {0%{box-shadow:0 0 0 0 rgba(240,168,60,.6)}
          70%{box-shadow:0 0 0 7px rgba(240,168,60,0)}100%{box-shadow:0 0 0 0 rgba(240,168,60,0)}}

        .furnace-frame {width:340px;height:240px;max-width:100%;overflow:visible;
          background:#0a0d10;contain:layout paint;}
        .furnace-frame svg {width:340px;height:240px;max-width:100%;display:block;
          overflow:visible;}
        .live-trend {contain:layout paint;min-height:340px;}
        .live-trend svg {display:block;width:100%;overflow:visible;}
        hr {margin:0.3rem 0 !important;border-color:#232c33 !important;}

        /* Header text must not disappear when the status sentence is long. */
        .smartmelt-head-title {font-size:20px;font-weight:700;padding-top:4px;white-space:nowrap;}
        .smartmelt-head-meta {font-size:12px;color:#aeb7c1;padding-top:10px;white-space:nowrap;}
        .smartmelt-head-status {font-size:12px;padding-top:8px;text-align:right;line-height:1.25;
          white-space:normal;overflow-wrap:anywhere;min-height:34px;}

        @media (max-width:1100px) {
          .block-container {width:99vw !important;padding-left:.3rem !important;padding-right:.3rem !important;}
          .kpi .val {font-size:19px !important;}
          .kpi .lab,.kpi .sub {font-size:9.5px !important;}
          .adv-msg {font-size:9.5px;}
          .stButton>button {font-size:10.5px;padding:4px 5px;}
        }
        </style>
        """, unsafe_allow_html=True)


def section_title(text: str, small: bool = False):
    cls = "section-title-sm" if small else "section-title"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def advisory_card(level: str, title: str, message: str):
    colour = {"ok":"#33d17a", "warn":"#f0a83c", "bad":"#e5484d"}.get(level,"#9aa4af")
    badge = {"ok":"OK", "warn":"!", "bad":"!!!"}.get(level,"—")
    st.markdown(
        f'<div class="adv-card" style="border-color:{colour if level != "ok" else "#232c33"}">'
        f'<div style="display:flex;gap:8px;align-items:flex-start">'
        f'<div style="font-weight:800;color:{colour};font-size:14px;min-width:25px">{badge}</div>'
        f'<div><div class="adv-title">{title}</div><div class="adv-msg">{message}</div></div>'
        f'</div></div>', unsafe_allow_html=True)
# End of exact UI helpers.
