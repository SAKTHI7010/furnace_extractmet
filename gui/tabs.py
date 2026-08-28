"""All tabs for the SmartMelt desktop GUI. Each tab is a ttk.Frame that binds
to the shared engine bridge and renders plots via embedded matplotlib."""
from __future__ import annotations

import numpy as np
import tkinter as tk
from tkinter import ttk

import engine as E
from gui import theme as T
from gui.theme import FurnaceCanvas
from gui.app import KPI, Pill, mpl, section


# helper: a horizontal row of KPI cards
def kpi_row(master, specs):
    row = tk.Frame(master, bg=T.BG_PANEL)
    cards = {}
    for key, label in specs:
        c = KPI(row, label)
        c.pack(side="left", fill="both", expand=True, padx=3)
        cards[key] = c
    return row, cards


def slider(master, label, lo, hi, init, step=None, fmt="{:.0f}"):
    """A labelled slider that shows its current value."""
    fr = tk.Frame(master, bg=T.BG_PANEL)
    top = tk.Frame(fr, bg=T.BG_PANEL); top.pack(fill="x")
    tk.Label(top, text=label, bg=T.BG_PANEL, fg=T.TEXT_MUT,
             font=(T.FONT, 9), anchor="w").pack(side="left")
    valv = tk.StringVar(value=fmt.format(init))
    tk.Label(top, textvariable=valv, bg=T.BG_PANEL, fg=T.MOLTEN_HI,
             font=(T.FONT_MONO, 9), anchor="e").pack(side="right")
    var = tk.DoubleVar(value=init)
    def _upd(v):
        valv.set(fmt.format(float(v)))
    sc = ttk.Scale(fr, from_=lo, to=hi, variable=var, orient="horizontal",
                   command=_upd)
    sc.pack(fill="x")
    fr.var = var
    return fr


def style_axes(ax, title=None, xlabel=None, ylabel=None, legend=False):
    ax.set_facecolor("#0f1418")
    ax.grid(True, color="#20262c", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(T.LINE)
    if title:
        ax.set_title(title, fontsize=9, color=T.TEXT, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=T.TEXT_MUT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=T.TEXT_MUT)
    if legend:
        ax.legend(fontsize=7, labelcolor=T.TEXT, facecolor=T.BG_RAISED,
                  edgecolor=T.LINE, framealpha=0.9)
    ax.tick_params(labelsize=7)


# ════════════════════════════════════════════════════════════════════════════
# 1 · Operator Console
# ════════════════════════════════════════════════════════════════════════════
class OperatorConsoleTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self._frame_idx = 0
        self._playing = False

        # ---- left: controls + furnace ----
        left = tk.Frame(self, bg=T.BG_PANEL, width=340)
        left.pack(side="left", fill="y", padx=(4, 8), pady=4)
        left.pack_propagate(False)

        ctl = section(left, "Heat setup")
        ctl.pack(fill="x")
        self.s_charge = slider(ctl, "Charge (t)", 4, 14, 12, fmt="{:.1f}")
        self.s_charge.pack(fill="x", pady=2)
        self.s_power = slider(ctl, "Power (kW)", 1000, 8000, 5200)
        self.s_power.pack(fill="x", pady=2)
        self.s_carbon = slider(ctl, "Charge C (%)", 0.1, 1.5, 0.6, fmt="{:.2f}")
        self.s_carbon.pack(fill="x", pady=2)

        adds = tk.Frame(ctl, bg=T.BG_PANEL); adds.pack(fill="x", pady=(6, 2))
        self.v_lime = tk.BooleanVar(value=True)
        self.v_fesi = tk.BooleanVar(value=True)
        self.v_scale = tk.BooleanVar(value=True)
        for txt, var in [("Lime 48 kg @10 min", self.v_lime),
                         ("FeSi75 15 kg @45 min", self.v_fesi),
                         ("Mill scale 150 kg @60 min", self.v_scale)]:
            tk.Checkbutton(adds, text=txt, variable=var, bg=T.BG_PANEL,
                           fg=T.TEXT, selectcolor=T.BG_INPUT, activebackground=T.BG_PANEL,
                           activeforeground=T.MOLTEN_HI, font=(T.FONT, 9),
                           anchor="w").pack(fill="x")

        btns = tk.Frame(left, bg=T.BG_PANEL); btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="⏵ Run heat", command=self.run_heat).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="▶ Play", command=self.play).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="⏮ Reset", command=self.reset).pack(side="left", expand=True, fill="x", padx=2)

        tk.Label(left, text="Furnace state", bg=T.BG_PANEL, fg=T.TEXT,
                 font=(T.FONT, 11, "bold"), anchor="w").pack(fill="x", pady=(8, 2))
        self.furnace = FurnaceCanvas(left, width=320, height=380)
        self.furnace.pack(fill="both", expand=True)

        # scrub
        self.scrub_var = tk.DoubleVar(value=0)
        self.scrub = ttk.Scale(left, from_=0, to=100, variable=self.scrub_var,
                               orient="horizontal", command=self._on_scrub)
        self.scrub.pack(fill="x", pady=4)

        # ---- right: KPIs + live trend + advice ----
        right = tk.Frame(self, bg=T.BG_PANEL)
        right.pack(side="left", fill="both", expand=True, pady=4)

        self.k_row1, self.k1 = kpi_row(right, [
            ("T", "Bath °C"), ("C", "Carbon %"), ("melt", "Melted %"), ("sec", "SEC kWh/t")])
        self.k_row1.pack(fill="x", pady=2)
        self.k_row2, self.k2 = kpi_row(right, [
            ("feo", "Slag FeO %"), ("b2", "B2"), ("p", "P %"), ("time", "Time min")])
        self.k_row2.pack(fill="x", pady=2)

        stat = tk.Frame(right, bg=T.BG_PANEL); stat.pack(fill="x", pady=6)
        self.pill = Pill(stat, "press Run heat", "warn"); self.pill.pack(side="left")

        self.fig, self.canvas = mpl(right, figsize=(7, 4.4))
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # endpoint/advice row
        adv = tk.Frame(right, bg=T.BG_PANEL); adv.pack(fill="x", pady=(6, 0))
        self.lbl_end = tk.Label(adv, text="", bg=T.BG_PANEL, fg=T.TEXT_MUT,
                                font=(T.FONT, 9), anchor="w", justify="left")
        self.lbl_end.pack(fill="x")

        self.run_heat()

    def _specs(self):
        specs = []
        if self.v_lime.get():
            specs.append(E.AdditionSpec("Lime (92% CaO)", 10, 48))
        if self.v_fesi.get():
            specs.append(E.AdditionSpec("FeSi75", 45, 15))
        if self.v_scale.get():
            specs.append(E.AdditionSpec("Mill scale (FeO)", 60, 150))
        return specs

    def run_heat(self):
        self.app.set_status("simulating heat…", T.AMBER)
        charge = self.s_charge.var.get()
        power = self.s_power.var.get()
        cpct = self.s_carbon.var.get()
        specs = self._specs()
        cfg = self.app.cfg

        def work():
            comp = dict(E.DEFAULT_CHARGE_COMP); comp["C"] = cpct / 100.0
            return E.run_heat(cfg, charge * 1000.0, comp, power,
                              additions=E.build_additions(specs), dt=2.0)

        def done(res):
            self.res = res
            df = res.df.copy()
            scols = [c for c in df.columns if c.startswith("slag_") and c.endswith("_kg")]
            df["slag_total_kg"] = df[scols].sum(axis=1) if scols else 20.0
            self.df = df
            self._frame_idx = len(df) - 1
            self.scrub.config(to=float(df["t_min"].iloc[-1]))
            self.scrub_var.set(float(df["t_min"].iloc[-1]))
            self._render(self._frame_idx)
            self._render_endpoint()
            self.app.set_status("heat ready", T.GREEN)

        self.app.run_async(work, done)

    def _render(self, i):
        df = self.df
        row = df.iloc[i]
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        heat_t = getattr(self.app.cfg.plant, "heat_size_kg", 12000) / 1000

        self.furnace.draw(row["melted_pct"], row["T_bath_C"],
                          row.get("slag_total_kg", 20), row["undissolved_kg"],
                          heat_size_t=heat_t, tap_aim_C=aim)

        self.k1["T"].set(f'{row["T_bath_C"]:.0f}', f'aim {aim:.0f}')
        self.k1["C"].set(f'{row["pct_C"]:.3f}', f'Si {row["pct_Si"]:.3f}')
        self.k1["melt"].set(f'{row["melted_pct"]:.0f}', f'undis {row["undissolved_kg"]:.0f}kg')
        self.k1["sec"].set(f'{row["SEC_kWh_t"]:.0f}', f'{row["E_kWh"]:.0f} kWh')
        self.k2["feo"].set(f'{row["slag_FeO_pct"]:.1f}', "")
        self.k2["b2"].set(f'{row["B2"]:.2f}', "basicity")
        self.k2["p"].set(f'{row["pct_P"]:.4f}', f'S {row["pct_S"]:.4f}')
        self.k2["time"].set(f'{row["t_min"]:.0f}', f'tap {df["t_min"].iloc[-1]:.0f}')

        ready = row["melted_pct"] > 99 and row["T_bath_C"] >= aim - 5
        if ready:
            self.pill.set("READY TO TAP — on temperature & fully melted", "ok")
        elif row["melted_pct"] < 2:
            self.pill.set("charging / frozen heel — heating solid", "warn")
        else:
            self.pill.set(f"melting — {aim - row['T_bath_C']:.0f} °C below aim", "warn")

        # trend up to now
        self.fig.clear()
        ax = self.fig.add_subplot(111); style_axes(ax)
        d = df.iloc[: i + 1]
        ax.plot(d["t_min"], d["T_bath_C"], color=T.MOLTEN, lw=2, label="bath °C")
        ax.plot(d["t_min"], d["T_solid_C"], color=T.SCRAP_COL, lw=1.2, label="solid °C")
        ax.axhline(aim, color=T.GREEN, ls="--", lw=1)
        ax.set_ylabel("°C"); ax.set_xlabel("min")
        ax2 = ax.twinx()
        ax2.plot(d["t_min"], d["melted_pct"], color=T.STEEL, lw=1.4, label="melted %")
        ax2.set_ylim(0, 105); ax2.set_ylabel("melted %", color=T.STEEL)
        ax2.tick_params(axis="y", colors=T.STEEL)
        ax.legend(loc="upper left", fontsize=7, facecolor=T.BG_RAISED, edgecolor=T.LINE, labelcolor=T.TEXT)
        self.fig.tight_layout()
        self.canvas.draw()

    def _render_endpoint(self):
        res = self.res
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        hit = abs(res.endpoint["T_C"] - aim) <= 15
        note = "on aim — safe to tap" if hit else "trim power to hit tap temperature"
        if res.undissolved_kg > 5:
            note = "late additions not fully dissolved — hold before tap"
        self.lbl_end.config(text=(
            f"Endpoint:  tap {res.endpoint['T_C']:.0f} °C  ·  C {res.endpoint['pct_C']:.3f} %  "
            f"·  tap time {res.tap_min:.0f} min\n"
            f"Conservation:  element ledger {res.ledger_max_pct:.2f}%  ·  "
            f"first-law closure {res.energy['residual_pct']:+.1f}%  ·  "
            f"undissolved {res.undissolved_kg:.0f} kg\n"
            f"Advisory:  {note}"))

    def _on_scrub(self, v):
        if not hasattr(self, "df"):
            return
        idx = int(np.searchsorted(self.df["t_min"].to_numpy(), float(v)))
        idx = max(0, min(idx, len(self.df) - 1))
        self._frame_idx = idx
        self._render(idx)

    def play(self):
        if not hasattr(self, "df") or self._playing:
            return
        self._playing = True
        self._frame_idx = 0
        step = max(1, len(self.df) // 60)
        self._animate(step)

    def _animate(self, step):
        if not self._playing:
            return
        i = self._frame_idx
        if i >= len(self.df):
            self._playing = False
            self._render(len(self.df) - 1)
            return
        self._render(i)
        self.scrub_var.set(float(self.df["t_min"].iloc[min(i, len(self.df) - 1)]))
        self._frame_idx += step
        self.after(60, lambda: self._animate(step))

    def reset(self):
        self._playing = False
        self._frame_idx = 0
        if hasattr(self, "df"):
            self._render(0)
            self.scrub_var.set(0)


# ════════════════════════════════════════════════════════════════════════════
# 2 · Process Trajectory (six-panel)
# ════════════════════════════════════════════════════════════════════════════
class TrajectoryTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ctl = tk.Frame(self, bg=T.BG_PANEL); ctl.pack(fill="x", padx=6, pady=4)
        self.banner = tk.Label(ctl, text="", bg=T.BG_PANEL, fg=T.STEEL,
                               font=(T.FONT, 9), anchor="w")
        self.banner.pack(side="left", fill="x", expand=True)
        ttk.Button(ctl, text="↻ Use operator's heat", command=self.run).pack(side="left", padx=6)

        self.k_row, self.k = kpi_row(self, [
            ("tap", "Tap °C"), ("c", "Carbon %"), ("time", "Tap min"),
            ("sec", "SEC kWh/t"), ("ledg", "Ledger %")])
        self.k_row.pack(fill="x", padx=6, pady=2)

        self.fig, self.canvas = mpl(self, figsize=(11, 6))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        # re-run automatically whenever the operator taps a new heat
        self.app.register_spec_listener(self.run)
        # animate live while a console heat is running
        self.app.register_live_listener(self._on_live)
        self.after(400, self.run)

    def _on_live(self, hist, running):
        """Called each console frame — draw the 6 trajectory panels live from the
        growing snapshot history (the SAME state the operator sees). When the heat
        is tapped (running=False) this holds the final state instead of reverting."""
        if not hist or len(hist) < 2:
            return
        self._live_frozen = not running        # remember we're showing a tapped heat
        if running:
            self.banner.config(text=f"● LIVE from the running heat — {len(hist)} samples, "
                                    f"{hist[-1]['t_min']:.0f} min")
        else:
            self.banner.config(text=f"■ TAPPED heat — final state at {hist[-1]['t_min']:.0f} min "
                                    f"({len(hist)} samples). This is the actual heat that ran.")
        self._plot_from_snaps(hist)

    def _plot_from_snaps(self, d):
        """Draw the 6 trajectory panels from a list of snapshot dicts (live path)."""
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        floor = E.theoretical_floor_kWh_t(self.app.cfg)
        t = [s["t_min"] for s in d]
        self.k["tap"].set(f'{d[-1]["T_bath_C"]:.0f}', f'aim {aim}')
        self.k["c"].set(f'{d[-1]["pct_C"]:.3f}', "")
        self.k["time"].set(f'{d[-1]["t_min"]:.0f}', "live")
        self.k["sec"].set(f'{d[-1]["SEC_kWh_t"]:.0f}', f'floor {floor:.0f}')
        self.k["ledg"].set("live", "running")

        self.fig.clear()
        (a1, a2, a3), (a4, a5, a6) = self.fig.subplots(2, 3)
        for a in (a1, a2, a3, a4, a5, a6):
            style_axes(a); a.set_xlabel("Time (min)", fontsize=8, color=T.TEXT_MUT)

        a1.set_title("Temperatures", fontsize=9, color=T.TEXT, fontweight="bold")
        a1.plot(t, [s["T_bath_C"] for s in d], color=T.MOLTEN, lw=2, label="bath")
        a1.plot(t, [s["T_solid_C"] for s in d], color=T.SCRAP_COL, lw=1.2, label="solid")
        a1.axhline(aim, color=T.GREEN, ls="--", lw=1, label=f"tap aim {aim}")
        a1.set_ylabel("Temperature (°C)", fontsize=8, color=T.TEXT_MUT)
        a1.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a2.set_title("Inventories & dissolution", fontsize=9, color=T.TEXT, fontweight="bold")
        a2.plot(t, [s["M_solid_t"] for s in d], color=T.SCRAP_COL, label="solid")
        a2.plot(t, [s["M_liquid_t"] for s in d], color=T.MOLTEN, label="liquid")
        a2b = a2.twinx()
        a2b.plot(t, [s["undissolved_kg"] for s in d], color=T.STEEL, lw=1)
        a2b.set_ylabel("Undissolved (kg)", color=T.STEEL, fontsize=8)
        a2b.tick_params(axis="y", colors=T.STEEL)
        a2.set_ylabel("Metal mass (t)", fontsize=8, color=T.TEXT_MUT)
        a2.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a3.set_title("Bath composition", fontsize=9, color=T.TEXT, fontweight="bold")
        for el, c in [("pct_C", T.MOLTEN), ("pct_Si", T.STEEL), ("pct_Mn", T.GREEN), ("pct_S", T.SLAG_TOP)]:
            a3.plot(t, [s[el] for s in d], color=c, label=el.replace("pct_", ""))
        a3.set_ylabel("Element content (wt %)", fontsize=8, color=T.TEXT_MUT)
        a3.legend(fontsize=6, ncol=2, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a4.set_title("Slag chemistry & basicity", fontsize=9, color=T.TEXT, fontweight="bold")
        a4.plot(t, [s.get("slag_FeO_kg", 0) for s in d], color=T.RED, lw=2, label="FeO kg")
        a4.plot(t, [s.get("slag_CaO_kg", 0) for s in d], color=T.STEEL, label="CaO kg")
        a4.plot(t, [s.get("slag_SiO2_kg", 0) for s in d], color=T.SCRAP_COL, label="SiO₂ kg")
        a4b = a4.twinx(); a4b.plot(t, [s["B2"] for s in d], color=T.GREEN, ls="--", lw=1.5)
        a4b.set_ylabel("Basicity B2", color=T.GREEN, fontsize=8)
        a4b.tick_params(axis="y", colors=T.GREEN)
        a4.set_ylabel("Slag mass (kg)", fontsize=8, color=T.TEXT_MUT)
        a4.legend(fontsize=6, ncol=3, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a5.set_title("Power & heat flows", fontsize=9, color=T.TEXT, fontweight="bold")
        for key, c, nm in [("Q_useful_kW", T.MOLTEN, "useful"), ("Q_wall_kW", T.SLAG_TOP, "wall loss"),
                           ("Q_rad_kW", T.RED, "radiation"), ("Q_chem_kW", T.GREEN, "chemical")]:
            vals = [s.get(key, float("nan")) for s in d]
            if any(v == v for v in vals):
                a5.plot(t, vals, color=c, label=nm)
        a5.set_ylabel("Heat flow (kW)", fontsize=8, color=T.TEXT_MUT)
        a5.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a6.set_title("Energy & specific consumption", fontsize=9, color=T.TEXT, fontweight="bold")
        a6.plot(t, [s["E_kWh"] for s in d], color=T.SCRAP_COL, label="energy")
        a6b = a6.twinx(); a6b.plot(t, [s["SEC_kWh_t"] for s in d], color=T.MOLTEN)
        a6b.axhline(floor, color=T.GREEN, ls="--", lw=1)
        a6b.set_ylabel("SEC (kWh/t)", color=T.MOLTEN, fontsize=8)
        a6b.tick_params(axis="y", colors=T.MOLTEN)
        a6.set_ylabel("Cumulative energy (kWh)", fontsize=8, color=T.TEXT_MUT)
        for add in self.app.heat_spec["schedule"]:
            for a in (a1, a2, a3):
                a.axvline(add["time_min"], color=T.AMBER, ls=":", lw=0.7)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _spec_banner(self):
        s = self.app.heat_spec
        adds = ", ".join(f"{a['mass']:.0f}kg {a['material'].split(' (')[0]}@{a['time_min']:.0f}m"
                         for a in s["schedule"]) or "no additions"
        self.banner.config(text=f"Operator's heat: {s['charge_t']:.1f} t · {s['power_kW']:.0f} kW · "
                                f"C {s['charge_C_pct']:.2f}% · additions: {adds}")

    def run(self):
        # Do not overwrite a live or just-tapped heat with a batch re-simulation.
        # The trajectory should show the ACTUAL heat that ran until the operator
        # explicitly starts a new one.
        if getattr(self, "_live_frozen", False) or self.app.live_running:
            return
        self.app.set_status("running operator's heat…", T.AMBER)
        self._spec_banner()
        run_spec = self.app.run_spec_heat

        def work():
            return run_spec()

        def done(res):
            self.res = res
            self._plot(res)
            self.app.set_status("trajectory ready", T.GREEN)
        self.app.run_async(work, done)

    def _plot(self, res):
        d = res.df; t = d["t_min"]
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        floor = E.theoretical_floor_kWh_t(self.app.cfg)
        self.k["tap"].set(f'{res.endpoint["T_C"]:.0f}', f'aim {aim}')
        self.k["c"].set(f'{res.endpoint["pct_C"]:.3f}', "")
        self.k["time"].set(f'{res.tap_min:.0f}', "")
        self.k["sec"].set(f'{d["SEC_kWh_t"].iloc[-1]:.0f}', f'floor {floor:.0f}')
        self.k["ledg"].set(f'{res.ledger_max_pct:.2f}', "closure")

        self.fig.clear()
        axes = self.fig.subplots(2, 3)
        (a1, a2, a3), (a4, a5, a6) = axes
        for a in axes.flat:
            style_axes(a); a.set_xlabel("Time (min)", fontsize=8, color=T.TEXT_MUT)

        a1.set_title("Temperatures", fontsize=9, color=T.TEXT, fontweight="bold")
        a1.plot(t, d["T_bath_C"], color=T.MOLTEN, label="bath")
        a1.plot(t, d["T_solid_C"], color=T.SCRAP_COL, label="solid charge")
        if "T_hotface_C" in d:
            a1.plot(t, d["T_hotface_C"], color=T.SLAG_TOP, ls=":", label="lining hot face")
        a1.axhline(aim, color=T.GREEN, ls="--", lw=1, label=f"tap aim {aim:.0f}")
        a1.set_ylabel("Temperature (°C)", fontsize=8, color=T.TEXT_MUT)
        a1.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a2.set_title("Inventories & dissolution", fontsize=9, color=T.TEXT, fontweight="bold")
        a2.plot(t, d["M_solid_t"], color=T.SCRAP_COL, label="solid")
        a2.plot(t, d["M_liquid_t"], color=T.MOLTEN, label="liquid")
        a2b = a2.twinx()
        a2b.plot(t, d["undissolved_kg"], color=T.STEEL, lw=1)
        a2b.set_ylabel("Undissolved additions (kg)", color=T.STEEL, fontsize=8)
        a2b.tick_params(axis="y", colors=T.STEEL)
        a2.set_ylabel("Metal mass (t)", fontsize=8, color=T.TEXT_MUT)
        a2.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a3.set_title("Bath composition", fontsize=9, color=T.TEXT, fontweight="bold")
        for el, c in [("C", T.MOLTEN), ("Si", T.STEEL), ("Mn", T.GREEN), ("S", T.SLAG_TOP)]:
            if f"pct_{el}" in d:
                a3.plot(t, d[f"pct_{el}"], color=c, label=el)
        a3.set_ylabel("Element content (wt %)", fontsize=8, color=T.TEXT_MUT)
        a3.legend(fontsize=6, ncol=2, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a4.set_title("Slag chemistry & basicity", fontsize=9, color=T.TEXT, fontweight="bold")
        a4.plot(t, d["slag_FeO_pct"], color=T.MOLTEN, label="FeO")
        a4b = a4.twinx(); a4b.plot(t, d["B2"], color=T.STEEL)
        a4b.set_ylabel("Basicity B2 (CaO/SiO₂)", color=T.STEEL, fontsize=8)
        a4b.tick_params(axis="y", colors=T.STEEL)
        a4.set_ylabel("Slag FeO (wt %)", fontsize=8, color=T.TEXT_MUT)
        a4.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a5.set_title("Heat-flow breakdown", fontsize=9, color=T.TEXT, fontweight="bold")
        for key, c, nm in [("Q_wall_kW", T.SLAG_TOP, "lining loss"), ("Q_rad_kW", T.RED, "radiation"),
                           ("Q_bath_to_scrap_kW", T.SCRAP_COL, "bath→scrap"),
                           ("Q_chem_kW", T.GREEN, "chemical")]:
            if key in d:
                a5.plot(t, d[key], color=c, label=nm)
        a5.set_ylabel("Heat flow (kW)", fontsize=8, color=T.TEXT_MUT)
        a5.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a6.set_title("Energy & specific consumption", fontsize=9, color=T.TEXT, fontweight="bold")
        a6.plot(t, d["E_kWh"], color=T.SCRAP_COL, label="cumulative kWh")
        a6b = a6.twinx(); a6b.plot(t, d["SEC_kWh_t"], color=T.MOLTEN)
        a6b.axhline(floor, color=T.GREEN, ls="--", lw=1)
        a6b.set_ylabel("Specific energy (kWh/t)", color=T.MOLTEN, fontsize=8)
        a6b.tick_params(axis="y", colors=T.MOLTEN)
        a6.set_ylabel("Cumulative energy (kWh)", fontsize=8, color=T.TEXT_MUT)

        # mark the operator's actual additions on the time axes
        for add in self.app.heat_spec["schedule"]:
            for a in (a1, a2, a3):
                a.axvline(add["time_min"], color=T.AMBER, ls=":", lw=0.7)
        self.fig.tight_layout()
        self.canvas.draw()


# ════════════════════════════════════════════════════════════════════════════
# 3 · Physics & Energy
# ════════════════════════════════════════════════════════════════════════════
class PhysicsEnergyTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ctl = tk.Frame(self, bg=T.BG_PANEL); ctl.pack(fill="x", padx=6, pady=4)
        self.banner = tk.Label(ctl, text="", bg=T.BG_PANEL, fg=T.STEEL,
                               font=(T.FONT, 9), anchor="w")
        self.banner.pack(side="left", fill="x", expand=True)
        ttk.Button(ctl, text="↻ Use operator's heat", command=self.run).pack(side="left", padx=6)

        self.k_row, self.k = kpi_row(self, [
            ("ledg", "Element ledger %"), ("clo", "First-law closure %"),
            ("sec", "Final SEC"), ("uf", "Useful fraction %")])
        self.k_row.pack(fill="x", padx=6, pady=2)

        self.fig, self.canvas = mpl(self, figsize=(11, 5))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        self.app.register_spec_listener(self.run)
        self.after(1200, self.run)

    def run(self):
        self.app.set_status("running operator's heat…", T.AMBER)
        s = self.app.heat_spec
        adds = ", ".join(f"{a['mass']:.0f}kg {a['material'].split(' (')[0]}@{a['time_min']:.0f}m"
                         for a in s["schedule"]) or "no additions"
        self.banner.config(text=f"Operator's heat: {s['charge_t']:.1f} t · {s['power_kW']:.0f} kW · "
                                f"additions: {adds}")
        run_spec = self.app.run_spec_heat

        def work():
            return run_spec()

        def done(res):
            self._plot(res); self.app.set_status("energy audit ready", T.GREEN)
        self.app.run_async(work, done)

    def _plot(self, res):
        d = res.df; en = res.energy; floor = E.theoretical_floor_kWh_t(self.app.cfg)
        self.k["ledg"].set(f'{res.ledger_max_pct:.2f}', "worst species")
        self.k["clo"].set(f'{en["residual_pct"]:+.1f}', "in − out")
        self.k["sec"].set(f'{d["SEC_kWh_t"].iloc[-1]:.0f}', f'floor {floor:.0f}')
        self.k["uf"].set(f'{100*en.get("useful_fraction",0):.0f}', "of grid input")

        self.fig.clear()
        (a1, a2), (a3, a4) = self.fig.subplots(2, 2)
        for a in (a1, a2, a3, a4):
            style_axes(a)

        a1.set_title("Heat-flow breakdown through the heat", fontsize=9, color=T.TEXT, fontweight="bold")
        for key, c, nm in [("Q_useful_kW", T.MOLTEN, "useful (to metal)"), ("Q_wall_kW", T.SLAG_TOP, "lining loss"),
                           ("Q_rad_kW", T.RED, "radiation"), ("Q_chem_kW", T.GREEN, "chemical"),
                           ("Q_offgas_kW", T.STEEL, "off-gas")]:
            if key in d:
                a1.plot(d["t_min"], d[key], color=c, lw=1.4, label=nm)
        a1.set_xlabel("Time (min)", fontsize=8, color=T.TEXT_MUT)
        a1.set_ylabel("Heat flow (kW)", fontsize=8, color=T.TEXT_MUT)
        a1.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a2.set_title("Energy split — grid input to tapped steel", fontsize=9, color=T.TEXT, fontweight="bold")
        total = en.get("grid_kWh", d["E_kWh"].iloc[-1])
        parts = [("converter", en.get("converter_loss_kWh", 0)),
                 ("coil water", en.get("coil_water_loss_kWh", 0)),
                 ("lining", en.get("lining_loss_kWh", 0)),
                 ("radiation", en.get("radiation_loss_kWh", 0)),
                 ("off-gas", en.get("offgas_loss_kWh", 0))]
        useful = en.get("useful_melt_kWh", 0)
        labels = ["grid in"] + [p[0] for p in parts] + ["to steel"]
        vals = [total] + [-p[1] for p in parts] + [useful]
        cum = 0; bottoms = []; heights = []; colours = []
        for i, v in enumerate(vals):
            if i == 0:
                bottoms.append(0); heights.append(v); colours.append(T.MOLTEN); cum = v
            elif i == len(vals) - 1:
                bottoms.append(0); heights.append(v); colours.append(T.STEEL)
            else:
                bottoms.append(cum + v); heights.append(-v); colours.append(T.RED); cum += v
        a2.bar(range(len(vals)), heights, bottom=bottoms, color=colours, width=0.6)
        a2.set_xticks(range(len(labels)))
        a2.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        a2.set_ylabel("Energy (kWh)", fontsize=8, color=T.TEXT_MUT)

        a3.set_title("Element reaction rates", fontsize=9, color=T.TEXT, fontweight="bold")
        plotted = False
        for key, c, nm in [("rate_C", T.MOLTEN, "C"), ("rate_Si", T.STEEL, "Si"),
                           ("rate_Mn", T.GREEN, "Mn"), ("rate_P", T.SLAG_TOP, "P")]:
            if key in d:
                a3.plot(d["t_min"], d[key], color=c, lw=1.3, label=nm); plotted = True
        if not plotted:
            # derive rates from composition if explicit rate cols absent
            for el, c in [("C", T.MOLTEN), ("Si", T.STEEL), ("Mn", T.GREEN)]:
                if f"pct_{el}" in d:
                    a3.plot(d["t_min"], np.gradient(d[f"pct_{el}"], d["t_min"]),
                            color=c, lw=1.3, label=f"d{el}/dt")
        a3.axhline(0, color=T.TEXT_DIM, lw=0.6)
        a3.set_xlabel("Time (min)", fontsize=8, color=T.TEXT_MUT)
        a3.set_ylabel("Rate (wt %/min)", fontsize=8, color=T.TEXT_MUT)
        a3.legend(fontsize=6, ncol=2, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a4.set_title("Cumulative energy: input vs useful", fontsize=9, color=T.TEXT, fontweight="bold")
        a4.plot(d["t_min"], d["E_kWh"], color=T.MOLTEN, lw=1.6, label="grid input")
        if "Q_useful_kW" in d:
            dt_h = np.gradient(d["t_min"]) / 60.0
            cum_useful = np.cumsum(np.clip(d["Q_useful_kW"].to_numpy(), 0, None) * dt_h)
            a4.plot(d["t_min"], cum_useful, color=T.GREEN, lw=1.6, label="useful (to metal)")
            a4.fill_between(d["t_min"], cum_useful, d["E_kWh"], color=T.RED, alpha=0.12, label="losses")
        a4.set_xlabel("Time (min)", fontsize=8, color=T.TEXT_MUT)
        a4.set_ylabel("Cumulative energy (kWh)", fontsize=8, color=T.TEXT_MUT)
        a4.legend(fontsize=6, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        self.fig.tight_layout()
        self.canvas.draw()


# ════════════════════════════════════════════════════════════════════════════
# 4 · Virtual Sensor (EKF)
# ════════════════════════════════════════════════════════════════════════════
class VirtualSensorTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ctl = tk.Frame(self, bg=T.BG_PANEL); ctl.pack(fill="x", padx=6, pady=4)
        self.s_eta = slider(ctl, "True η electrical", 0.80, 1.0, 0.90, fmt="{:.2f}")
        self.s_eta.pack(side="left", fill="x", expand=True, padx=4)
        self.s_ua = slider(ctl, "True wall-loss scale", 0.8, 1.8, 1.35, fmt="{:.2f}")
        self.s_ua.pack(side="left", fill="x", expand=True, padx=4)
        self.s_dips = slider(ctl, "Immersion dips", 1, 6, 3)
        self.s_dips.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(ctl, text="Run live (~1 min)", command=self.run).pack(side="left", padx=6)

        tk.Label(self, text="Default result is pre-computed and loads instantly. A live "
                 "run recomputes the Kalman filter (finite-difference Jacobians over 34 "
                 "states ≈ 1 min); the window stays responsive meanwhile.",
                 bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 8), anchor="w",
                 wraplength=1100, justify="left").pack(fill="x", padx=6)

        self.k_row, self.k = kpi_row(self, [
            ("err", "Final error °C"), ("eta", "η̂ electrical"),
            ("sig", "σ_T end °C"), ("dips", "Dips used")])
        self.k_row.pack(fill="x", padx=6, pady=2)

        self.fig, self.canvas = mpl(self, figsize=(11, 5))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        # load the pre-computed default immediately (instant)
        cached = E.load_default_ekf()
        if cached is not None:
            self._plot(cached)

    def run(self):
        self.app.set_status("running EKF live (~1 min)…", T.AMBER)
        eta = self.s_eta.var.get(); ua = self.s_ua.var.get()
        ndips = int(self.s_dips.var.get()); cfg = self.app.cfg

        def work():
            dips = tuple(np.linspace(30, 78, ndips))
            return E.run_ekf_demo(cfg, true_eta=eta, true_UA_scale=ua,
                                  dip_times_min=dips, seed=1)

        def done(ek):
            self._plot(ek); self.app.set_status("EKF ready", T.GREEN)
        self.app.run_async(work, done)

    def _plot(self, ek):
        df = ek.df
        eta_final = ek.theta_path["eta_electrical"].iloc[-1]
        self.k["err"].set(f'{ek.final_error_C:+.1f}', "est − truth")
        self.k["eta"].set(f'{eta_final:.3f}', "converged")
        self.k["sig"].set(f'{df["sigma_T"].iloc[-1]:.1f}', "uncertainty")
        self.k["dips"].set(f'{len(ek.dip_df)}', "measurements")

        self.fig.clear()
        a1, a2 = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1]})
        style_axes(a1); style_axes(a2)

        a1.set_title("Bath temperature — truth vs EKF estimate", fontsize=9, color=T.TEXT, fontweight="bold")
        a1.fill_between(df["t_min"], df["T_est_C"] - 2 * df["sigma_T"],
                        df["T_est_C"] + 2 * df["sigma_T"],
                        color=T.MOLTEN, alpha=0.18, label="±2σ confidence")
        a1.plot(df["t_min"], df["T_true_C"], color="#cfd6dd", lw=2, label="true (hidden)")
        a1.plot(df["t_min"], df["T_est_C"], color=T.MOLTEN, lw=2, label="EKF estimate")
        if len(ek.dip_df):
            a1.scatter(ek.dip_df["t_min"], ek.dip_df["T_meas_C"], color=T.STEEL,
                       s=55, marker="D", zorder=5, label="immersion dip")
        a1.set_xlabel("Time (min)", fontsize=8, color=T.TEXT_MUT)
        a1.set_ylabel("Temperature (°C)", fontsize=8, color=T.TEXT_MUT)
        a1.legend(fontsize=7, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a2.set_title("Tracked parameters converging to truth", fontsize=9, color=T.TEXT, fontweight="bold")
        a2.plot(ek.theta_path["t_min"], ek.theta_path["eta_electrical"],
                color=T.MOLTEN, label="η electrical")
        if "UA_lining_scale" in ek.theta_path:
            a2.plot(ek.theta_path["t_min"], ek.theta_path["UA_lining_scale"],
                    color=T.STEEL, label="UA wall-loss scale")
        a2.set_xlabel("Time (min)", fontsize=8, color=T.TEXT_MUT)
        a2.set_ylabel("Parameter value", fontsize=8, color=T.TEXT_MUT)
        a2.legend(fontsize=7, labelcolor=T.TEXT, facecolor=T.BG_RAISED)
        self.fig.tight_layout()
        self.canvas.draw()


# ════════════════════════════════════════════════════════════════════════════
# 5 · Machine Learning
# ════════════════════════════════════════════════════════════════════════════
class MachineLearningTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ctl = tk.Frame(self, bg=T.BG_PANEL); ctl.pack(fill="x", padx=6, pady=4)
        self.s_split = slider(ctl, "Train fraction", 0.5, 0.85, 0.7, fmt="{:.2f}")
        self.s_split.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(ctl, text="Train on cached data", command=self.train_cached).pack(side="left", padx=4)
        self.s_n = slider(ctl, "Live heats", 20, 80, 40)
        self.s_n.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(ctl, text="Generate live (slow)", command=self.run).pack(side="left", padx=4)

        tk.Label(self, text="The hybrid model = the SAME physics engine used on the Operator "
                 "Console, plus a Gaussian-process residual head fitted on many heats. Physics "
                 "predicts, the ML head corrects it, and it gates itself off until it can prove "
                 "an out-of-time improvement. A 60-heat dataset is pre-computed and trains "
                 "instantly; live generation (~3–4 s/heat) is an explicit action.",
                 bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 8), anchor="w",
                 wraplength=1100, justify="left").pack(fill="x", padx=6)

        self.pill = Pill(self, "loading cached data…", "warn"); self.pill.pack(anchor="w", padx=6, pady=2)
        self.k_row, self.k = kpi_row(self, [
            ("t15", "T hit ±15°C"), ("tmae", "T MAE °C"),
            ("c02", "C hit ±0.02%"), ("cmae", "C MAE %")])
        self.k_row.pack(fill="x", padx=6, pady=2)

        self.fig, self.canvas = mpl(self, figsize=(11, 4.6))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        # train on cached data immediately (fast)
        self.after(200, self.train_cached)

    def train_cached(self):
        df = E.load_cached_dataset()
        if df is None:
            self.pill.set("no cached dataset found — use Generate live", "warn")
            return
        self.app.set_status("training on cached dataset…", T.AMBER)
        split = self.s_split.var.get(); cfg = self.app.cfg

        def work():
            return E.train_hybrid(cfg, df, split_frac=split)

        def done(ml):
            self._plot(ml); self.app.set_status("ML ready", T.GREEN)
        self.app.run_async(work, done)

    def run(self):
        self.app.set_status("simulating heats & fitting ML… (may take ~1 min)", T.AMBER)
        self.pill.set("working — generating heats…", "warn")
        n = int(self.s_n.var.get()); split = self.s_split.var.get(); cfg = self.app.cfg

        def work():
            df = E.generate_dataset(cfg, n_heats=n, seed=0)
            ml = E.train_hybrid(cfg, df, split_frac=split)
            return ml

        def done(ml):
            self._plot(ml); self.app.set_status("ML ready", T.GREEN)
        self.app.run_async(work, done)

    def _plot(self, ml):
        m = ml.metrics; p = ml.pred_df
        self.pill.set(
            f"maturity: {m['maturity']}  ·  T-ML {'active' if m['ml_T_active'] else 'gated off'}"
            f"  ·  C-ML {'active' if m['ml_C_active'] else 'gated off'}  "
            f"({m['n_train']} train / {m['n_test']} test)",
            "ok" if m["ml_T_active"] else "warn")

        def fmt(x):
            return f'{x:.0f}' if x == x else "—"
        self.k["t15"].set(fmt(m["T_hit_15C"]) + "%", f'phys {fmt(m["T_hit_15C_phys"])}%')
        self.k["tmae"].set(f'{m["T_MAE_C"]:.1f}' if m["T_MAE_C"] == m["T_MAE_C"] else "—", "hybrid")
        self.k["c02"].set(fmt(m["C_hit_002"]) + "%", f'phys {fmt(m["C_hit_002_phys"])}%')
        self.k["cmae"].set(f'{m["C_MAE"]:.3f}' if m["C_MAE"] == m["C_MAE"] else "—", "hybrid")

        self.fig.clear()
        a1, a2 = self.fig.subplots(1, 2)
        style_axes(a1); style_axes(a2)

        a1.set_title("Temperature — predicted vs actual")
        lo = float(np.nanmin([p["T_true_C"].min(), p["T_pred_C"].min()])) - 10
        hi = float(np.nanmax([p["T_true_C"].max(), p["T_pred_C"].max()])) + 10
        a1.plot([lo, hi], [lo, hi], color=T.TEXT_MUT, ls="--")
        a1.scatter(p["T_true_C"], p["T_phys_C"], color=T.SCRAP_COL, marker="x", s=40, label="physics")
        a1.scatter(p["T_true_C"], p["T_pred_C"], color=T.MOLTEN, s=45, label="hybrid")
        a1.set_xlabel("actual °C"); a1.set_ylabel("predicted °C")
        a1.legend(fontsize=7, labelcolor=T.TEXT, facecolor=T.BG_RAISED)

        a2.set_title("Test-set temperature error")
        a2.bar(p["heat"] - 0.2, p["T_pred_C"] - p["T_true_C"], width=0.4, color=T.MOLTEN, label="hybrid")
        a2.bar(p["heat"] + 0.2, p["T_phys_C"] - p["T_true_C"], width=0.4, color=T.SCRAP_COL, label="physics")
        a2.axhspan(-15, 15, color=T.GREEN, alpha=0.10)
        a2.set_xlabel("test heat"); a2.set_ylabel("pred − actual °C")
        a2.legend(fontsize=7, labelcolor=T.TEXT, facecolor=T.BG_RAISED)
        self.fig.tight_layout()
        self.canvas.draw()


# ════════════════════════════════════════════════════════════════════════════
# 6 · Drift Monitor
# ════════════════════════════════════════════════════════════════════════════
class DriftMonitorTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ctl = tk.Frame(self, bg=T.BG_PANEL); ctl.pack(fill="x", padx=6, pady=4)
        ttk.Button(ctl, text="Check cached data", command=self.check_cached).pack(side="left", padx=4)
        self.s_n = slider(ctl, "Live heats", 30, 80, 50)
        self.s_n.pack(side="left", fill="x", expand=True, padx=4)
        self.s_reg = slider(ctl, "Regime change at heat", 15, 60, 40)
        self.s_reg.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(ctl, text="Generate live (slow)", command=self.run).pack(side="left", padx=4)

        tk.Label(self, text="A 60-heat dataset (copper regime change at heat 40) is "
                 "pre-computed and checks instantly. Live generation runs the physics "
                 "simulator (~3–4 s/heat) as an explicit action.",
                 bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 8), anchor="w",
                 wraplength=1100, justify="left").pack(fill="x", padx=6)

        self.pill = Pill(self, "loading cached data…", "warn"); self.pill.pack(anchor="w", padx=6, pady=2)
        self.k_row, self.k = kpi_row(self, [
            ("psi", "Max PSI"), ("ref", "Reference heats"), ("rec", "Recent heats")])
        self.k_row.pack(fill="x", padx=6, pady=2)

        self.fig, self.canvas = mpl(self, figsize=(11, 4.8))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=4)
        self.after(300, self.check_cached)

    def check_cached(self):
        df = E.load_cached_dataset()
        if df is None:
            self.pill.set("no cached dataset found — use Generate live", "warn")
            return
        self.app.set_status("checking drift on cached dataset…", T.AMBER)
        cfg = self.app.cfg

        def work():
            dr = E.run_drift(cfg, df, ref_frac=0.5)
            return df, dr

        def done(payload):
            d, dr = payload
            self._plot(d, dr, 40); self.app.set_status("drift check ready", T.GREEN)
        self.app.run_async(work, done)

    def run(self):
        self.app.set_status("simulating heats & checking drift…", T.AMBER)
        self.pill.set("working…", "warn")
        n = int(self.s_n.var.get()); reg = int(self.s_reg.var.get()); cfg = self.app.cfg

        def work():
            df = E.generate_dataset(cfg, n_heats=n, seed=0, regime_change_at=reg)
            dr = E.run_drift(cfg, df, ref_frac=0.5)
            return df, dr

        def done(payload):
            df, dr = payload
            self._plot(df, dr, reg); self.app.set_status("drift check ready", T.GREEN)
        self.app.run_async(work, done)

    def _plot(self, df, dr, reg):
        self.pill.set("DRIFT ALARM — " + ", ".join(dr["reasons"][:2]) if dr["alarm"]
                      else "stable — no significant drift",
                      "bad" if dr["alarm"] else "ok")
        self.k["psi"].set(f'{dr["psi_max"]:.2f}', ">0.25 shift · >0.5 major")
        self.k["ref"].set(f'{dr["n_ref"]}', "baseline")
        self.k["rec"].set(f'{dr["n_recent"]}', "checked")

        self.fig.clear()
        a1, a2 = self.fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.2, 1]})
        style_axes(a1); style_axes(a2)

        a1.set_title("Population drift by feature (PSI)")
        psi = dr["psi_df"].head(12)
        colours = [T.RED if v > 0.5 else T.AMBER if v > 0.25 else T.STEEL for v in psi["PSI"]]
        a1.barh(psi["feature"], psi["PSI"], color=colours)
        a1.axvline(0.25, color=T.AMBER, ls="--", lw=1)
        a1.axvline(0.50, color=T.RED, ls="--", lw=1)
        a1.invert_yaxis(); a1.tick_params(labelsize=7)

        a2.set_title("The variable that moved")
        cu = "charge_Cu_pct" if "charge_Cu_pct" in df else df.columns[0]
        a2.plot(df.index, df[cu], color=T.MOLTEN, marker="o", ms=3)
        a2.axvline(reg, color=T.RED, ls=":", lw=1)
        a2.axvspan(0, dr["n_ref"], color=T.STEEL, alpha=0.08)
        a2.set_xlabel("heat number"); a2.set_ylabel(cu, fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()


# ════════════════════════════════════════════════════════════════════════════
# 7 · Charge-Mix Optimiser
# ════════════════════════════════════════════════════════════════════════════
class ChargeMixTab(ttk.Frame):
    """Charge-mix workbench. Two modes:
       • Optimise — LP finds the least-cost blend meeting aim C and Cu/Sn ceilings
       • Manual   — operator types kg per scrap; we cost it and predict bath chem
    Both use the full 17-stream scrap library; weights and prices are editable."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.mats = E.default_materials()
        self.weight_vars = {}

        # ── mode + targets bar ──
        top = tk.Frame(self, bg=T.BG_PANEL); top.pack(fill="x", padx=6, pady=4)
        self.mode = tk.StringVar(value="optimise")
        tk.Label(top, text="Mode:", bg=T.BG_PANEL, fg=T.TEXT_MUT,
                 font=(T.FONT, 9)).pack(side="left")
        for txt, val in [("Optimise (least cost)", "optimise"), ("Manual (operator sets kg)", "manual")]:
            tk.Radiobutton(top, text=txt, variable=self.mode, value=val,
                           bg=T.BG_PANEL, fg=T.TEXT, selectcolor=T.BG_INPUT,
                           activebackground=T.BG_PANEL, activeforeground=T.MOLTEN_HI,
                           font=(T.FONT, 9), command=self._on_mode).pack(side="left", padx=4)

        tgt = tk.Frame(self, bg=T.BG_PANEL); tgt.pack(fill="x", padx=6, pady=2)
        self.s_target = slider(tgt, "Target liquid (t)", 4, 14, 12, fmt="{:.1f}")
        self.s_target.pack(side="left", fill="x", expand=True, padx=3)
        self.s_clo = slider(tgt, "Min C (%)", 0.0, 0.5, 0.10, fmt="{:.2f}")
        self.s_clo.pack(side="left", fill="x", expand=True, padx=3)
        self.s_chi = slider(tgt, "Max C (%)", 0.1, 1.0, 0.40, fmt="{:.2f}")
        self.s_chi.pack(side="left", fill="x", expand=True, padx=3)
        self.s_cu = slider(tgt, "Cu ceiling (%)", 0.08, 0.50, 0.20, fmt="{:.2f}")
        self.s_cu.pack(side="left", fill="x", expand=True, padx=3)
        self.s_sn = slider(tgt, "Sn ceiling (%)", 0.01, 0.10, 0.03, fmt="{:.3f}")
        self.s_sn.pack(side="left", fill="x", expand=True, padx=3)
        self.solve_btn = ttk.Button(tgt, text="Solve", command=self.run)
        self.solve_btn.pack(side="left", padx=6)

        self.pill = Pill(self, "choose a mode and press Solve", "warn")
        self.pill.pack(anchor="w", padx=6, pady=2)
        self.k_row, self.k = kpi_row(self, [
            ("cost", "Blend cost ₹/t"), ("energy", "Charge energy"),
            ("cu", "Predicted Cu %"), ("c", "Predicted C %")])
        self.k_row.pack(fill="x", padx=6, pady=2)

        body = tk.Frame(self, bg=T.BG_PANEL); body.pack(fill="both", expand=True, padx=6, pady=4)

        # ── LEFT: the 17-stream scrap library (editable weights in manual mode) ──
        left = section(body, "Scrap library — 17 streams (price ₹/kg · assays wt%)")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        cols = ("name", "price", "Fe", "Cu", "Sn", "C", "kg")
        self.mat_tv = ttk.Treeview(left, columns=cols, show="headings", height=17)
        heads = [("name", "Material", 130), ("price", "₹/kg", 48), ("Fe", "Fe%", 45),
                 ("Cu", "Cu%", 52), ("Sn", "Sn%", 52), ("C", "C%", 45), ("kg", "kg (manual)", 78)]
        for c, t, w in heads:
            self.mat_tv.heading(c, text=t); self.mat_tv.column(c, width=w, anchor="center")
        self.mat_tv.pack(fill="both", expand=True)
        self._fill_material_table()
        # double-click a row to set its manual kg
        self.mat_tv.bind("<Double-1>", self._edit_weight)
        tk.Label(left, text="Manual mode: double-click a row to set kg. "
                 "Optimise mode: the solver picks the blend.",
                 bg=T.BG_PANEL, fg=T.TEXT_DIM, font=(T.FONT, 8), anchor="w").pack(fill="x")

        # ── RIGHT: result blend + bath chemistry + shadow price ──
        right = section(body, "Result — blend, bath chemistry, shadow price")
        right.pack(side="left", fill="both", expand=True)
        self.res_tv = ttk.Treeview(right, columns=("mat", "kg", "pct"), show="headings", height=7)
        for c, t, w in [("mat", "Material", 150), ("kg", "kg", 80), ("pct", "% charge", 80)]:
            self.res_tv.heading(c, text=t); self.res_tv.column(c, width=w, anchor="center")
        self.res_tv.pack(fill="x")

        tk.Label(right, text="Predicted bath chemistry", bg=T.BG_PANEL, fg=T.TEXT,
                 font=(T.FONT, 10, "bold"), anchor="w").pack(fill="x", pady=(6, 0))
        self.bath_tv = ttk.Treeview(right, columns=("el", "pct"), show="headings", height=5)
        self.bath_tv.heading("el", text="Element"); self.bath_tv.column("el", width=120, anchor="w")
        self.bath_tv.heading("pct", text="wt %"); self.bath_tv.column("pct", width=100, anchor="e")
        self.bath_tv.pack(fill="x")

        self.shadow_lbl = tk.Label(right, text="", bg=T.BG_PANEL, fg=T.TEXT_MUT,
                                   font=(T.FONT, 9), anchor="w", justify="left", wraplength=380)
        self.shadow_lbl.pack(fill="x", pady=6)

        self._on_mode()
        self.run()

    def _fill_material_table(self):
        self.mat_tv.delete(*self.mat_tv.get_children())
        self._iid_to_name = {}
        for mm in self.mats:
            kg = self.weight_vars.get(mm["name"], 0)
            iid = self.mat_tv.insert("", "end", values=(
                mm["name"], f'{mm["price"]:.0f}', f'{mm["Fe"]*100:.1f}',
                f'{mm["Cu"]*100:.3f}', f'{mm.get("Sn",0)*100:.3f}',
                f'{mm.get("C",0)*100:.2f}', f'{kg:.0f}' if kg else "—"))
            self._iid_to_name[iid] = mm["name"]

    def _edit_weight(self, event):
        if self.mode.get() != "manual":
            return
        iid = self.mat_tv.identify_row(event.y)
        if not iid:
            return
        name = self._iid_to_name.get(iid)
        # simple inline dialog
        dlg = tk.Toplevel(self); dlg.title("Set kg"); dlg.configure(bg=T.BG_PANEL)
        dlg.geometry("260x110")
        tk.Label(dlg, text=f"{name}\nkg for this heat:", bg=T.BG_PANEL, fg=T.TEXT,
                 font=(T.FONT, 10)).pack(pady=8)
        e = tk.Entry(dlg, bg=T.BG_INPUT, fg=T.TEXT, insertbackground=T.TEXT,
                     font=(T.FONT_MONO, 11), width=12, justify="center")
        e.insert(0, str(int(self.weight_vars.get(name, 0)))); e.pack()
        e.focus_set()
        def ok():
            try:
                self.weight_vars[name] = float(e.get())
            except ValueError:
                pass
            dlg.destroy(); self._fill_material_table()
        ttk.Button(dlg, text="Set", command=ok).pack(pady=8)
        e.bind("<Return>", lambda _: ok())

    def _on_mode(self):
        opt = self.mode.get() == "optimise"
        # enable/disable target sliders vs manual entry
        state = "normal" if opt else "disabled"
        for s in (self.s_clo, self.s_chi):
            for ch in s.winfo_children():
                pass  # sliders remain visible; they simply don't apply in manual
        self.solve_btn.config(text="Solve (optimise)" if opt else "Evaluate blend")
        if opt:
            self.pill.set("optimise mode — solver finds least-cost blend", "warn")
        else:
            self.pill.set("manual mode — double-click scrap rows to set kg, then Evaluate", "warn")

    def run(self):
        if self.mode.get() == "optimise":
            self._run_optimise()
        else:
            self._run_manual()

    def _run_optimise(self):
        self.app.set_status("optimising charge mix…", T.AMBER)
        target = self.s_target.var.get(); clo = self.s_clo.var.get()
        chi = self.s_chi.var.get(); cu = self.s_cu.var.get(); sn = self.s_sn.var.get()
        cfg = self.app.cfg; mats = self.mats

        def work():
            return E.solve_charge_mix(cfg, mats, target, {"C": (clo, chi)},
                                      cu_limit=cu, tramp_limits={"Sn": sn})

        def done(payload):
            res, shadow, rows = payload
            self._clear_result()
            if not getattr(res, "feasible", False):
                self.pill.set("infeasible — widen C window or raise a ceiling", "bad")
                self.shadow_lbl.config(text=getattr(res, "message", ""))
                self.app.set_status("charge-mix infeasible", T.RED)
                return
            self.pill.set("feasible — least-cost compliant blend", "ok")
            bath = getattr(res, "predicted_bath_pct", {})
            self.k["cost"].set(f'₹{res.cost_INR_per_t_liquid:,.0f}', "of liquid")
            self.k["energy"].set(f'{getattr(res,"energy_kWh",0):,.0f}', "kWh")
            self.k["cu"].set(f'{bath.get("Cu",0):.3f}', f'≤ {cu:.2f}')
            self.k["c"].set(f'{bath.get("C",0):.3f}', f'{clo:.2f}–{chi:.2f}')
            for r in rows:
                self.res_tv.insert("", "end", values=(r["Material"], f'{r["kg"]:.0f}', f'{r["% of charge"]:.1f}'))
            self._fill_bath(bath)
            cu_sh = shadow.get("Cu")
            if cu_sh and abs(cu_sh) > 1:
                per = abs(cu_sh) / 100.0
                self.shadow_lbl.config(
                    text=f"Copper ceiling shadow price ≈ ₹{per:,.0f}/t of liquid per 0.01% "
                         f"relaxed. Relaxing to {cu+0.01:.2f}% would save ≈ ₹{per:,.0f}/t — "
                         f"worth checking whether the spec is real or habit.")
            else:
                self.shadow_lbl.config(
                    text="Copper ceiling is not binding at this optimum — the cheapest "
                         "blend already sits under it. Tighten the ceiling to see the "
                         "optimiser pay for cleaner scrap.")
            self.app.set_status("charge-mix optimised", T.GREEN)
        self.app.run_async(work, done)

    def _run_manual(self):
        weights = {k: v for k, v in self.weight_vars.items() if v > 0}
        if not weights:
            self.pill.set("no kg set — double-click scrap rows to enter weights", "warn")
            return
        self.app.set_status("evaluating manual blend…", T.AMBER)
        cfg = self.app.cfg; mats = self.mats

        def work():
            return E.evaluate_manual_mix(cfg, mats, weights)

        def done(m):
            self._clear_result()
            if not m.get("feasible"):
                self.pill.set("nothing to evaluate", "warn"); return
            self.pill.set("manual blend evaluated — compare with the optimiser", "ok")
            self.k["cost"].set(f'₹{m["cost_INR_per_t_liquid"]:,.0f}', f'{m["liquid_t"]:.1f} t liquid')
            self.k["energy"].set(f'{m["energy_kWh"]:,.0f}', "kWh")
            bath = m["predicted_bath_pct"]
            self.k["cu"].set(f'{bath.get("Cu",0):.3f}', "tramp")
            self.k["c"].set(f'{bath.get("C",0):.3f}', "carbon")
            total = sum(weights.values())
            for name, kg in sorted(weights.items(), key=lambda kv: -kv[1]):
                self.res_tv.insert("", "end", values=(name, f'{kg:.0f}', f'{100*kg/total:.1f}'))
            self._fill_bath(bath)
            self.shadow_lbl.config(
                text=f"Operator blend: {total:,.0f} kg charged → {m['liquid_t']:.1f} t liquid "
                     f"at ₹{m['cost_INR_per_t_liquid']:,.0f}/t. Switch to Optimise to see the "
                     f"least-cost blend meeting the same ceilings.")
            self.app.set_status("manual blend evaluated", T.GREEN)
        self.app.run_async(work, done)

    def _clear_result(self):
        self.res_tv.delete(*self.res_tv.get_children())
        self.bath_tv.delete(*self.bath_tv.get_children())

    def _fill_bath(self, bath):
        self.bath_tv.delete(*self.bath_tv.get_children())
        order = ["C", "Si", "Mn", "Cr", "Cu", "Sn", "Fe"]
        for el in order:
            if el in bath and bath[el] > 1e-6:
                self.bath_tv.insert("", "end", values=(el, f'{bath[el]:.4f}'))


# ════════════════════════════════════════════════════════════════════════════
# 8 · Economics
# ════════════════════════════════════════════════════════════════════════════
class EconomicsTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        ctl = tk.Frame(self, bg=T.BG_PANEL); ctl.pack(fill="x", padx=6, pady=4)
        self.s_out = slider(ctl, "Annual output (t/yr)", 5000, 200000, 40000, fmt="{:.0f}")
        self.s_out.pack(side="left", fill="x", expand=True, padx=4)
        self.s_save = slider(ctl, "SEC saving (kWh/t)", 10, 100, 40)
        self.s_save.pack(side="left", fill="x", expand=True, padx=4)
        self.s_price = slider(ctl, "Licence (₹ lakh)", 5, 40, 20)
        self.s_price.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(ctl, text="Compute", command=self.run).pack(side="left", padx=6)

        self.k_row, self.k = kpi_row(self, [
            ("save", "Annual saving"), ("pay", "Payback"), ("co2", "CO₂ avoided"), ("head", "Headroom left")])
        self.k_row.pack(fill="x", padx=6, pady=2)

        self.note = tk.Label(self, text="", bg=T.BG_PANEL, fg=T.TEXT_MUT,
                             font=(T.FONT, 9), anchor="w", justify="left", wraplength=1000)
        self.note.pack(fill="x", padx=6, pady=4)

        self.tv = ttk.Treeview(self, columns=("out", "s30", "s50", "s80"), show="headings", height=4)
        for c, t in [("out", "Annual output"), ("s30", "30 kWh/t"), ("s50", "50 kWh/t"), ("s80", "80 kWh/t")]:
            self.tv.heading(c, text=t); self.tv.column(c, anchor="center", width=180)
        self.tv.pack(fill="x", padx=6, pady=4)

        self.eng_tv = ttk.Treeview(self, columns=("metric", "value"), show="headings", height=8)
        self.eng_tv.heading("metric", text="Engine economics metric"); self.eng_tv.column("metric", width=320, anchor="w")
        self.eng_tv.heading("value", text="value"); self.eng_tv.column("value", width=180, anchor="e")
        self.eng_tv.pack(fill="both", expand=True, padx=6, pady=4)
        self.run()

    def run(self):
        cfg = self.app.cfg
        summ = E.config_summary(cfg)
        tariff = summ["Tariff (₹/kWh)"]; grid_ef = summ["Grid EF (tCO₂/MWh)"]
        base = summ["Baseline SEC (kWh/t)"]; floor = E.theoretical_floor_kWh_t(cfg)
        tpy = self.s_out.var.get(); saving = self.s_save.var.get(); price = self.s_price.var.get()

        annual = tpy * saving * tariff
        payback = (price * 1e5) / annual * 12 if annual > 0 else float("inf")
        co2 = tpy * saving / 1000.0 * grid_ef
        self.k["save"].set(f'₹{annual/1e7:.2f} cr', f'at ₹{tariff:.1f}/kWh')
        self.k["pay"].set(f'{payback:.1f} mo', 'energy alone')
        self.k["co2"].set(f'{co2:,.0f} t/yr', f'at {grid_ef:.3f}')
        self.k["head"].set(f'{max(base-saving-floor,0):.0f} kWh/t', f'above {floor:.0f}')

        self.note.config(text=(
            f"At ₹{tariff:.1f}/kWh (mid-band Indian HT industrial). Energy alone — yield, "
            f"alloy and reduced reblows are additional. Simple payback is arithmetic; the "
            f"realised figure is quoted as 4–12 months (sub-6 only for high-utilisation, "
            f"high-tariff plants) because savings ramp up and advisory captures part, not "
            f"all, of the identified gap."))

        self.tv.delete(*self.tv.get_children())
        for o in (30000, 50000, 100000):
            self.tv.insert("", "end", values=(f"{o:,} t/yr",
                           f"₹{o*30*tariff/1e7:.2f} cr", f"₹{o*50*tariff/1e7:.2f} cr",
                           f"₹{o*80*tariff/1e7:.2f} cr"))

        self.eng_tv.delete(*self.eng_tv.get_children())
        try:
            ec = E.economics_summary(cfg, base, base - saving, tpy)
            for kk, vv in ec.items():
                self.eng_tv.insert("", "end", values=(kk, f"{vv:,.2f}" if isinstance(vv, (int, float)) else vv))
        except Exception:
            pass
        self.app.set_status("economics computed", T.GREEN)


# ════════════════════════════════════════════════════════════════════════════
# 9 · Validation
# ════════════════════════════════════════════════════════════════════════════
class ValidationTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        tk.Label(self, text="Validation — verified parameters & live conservation",
                 bg=T.BG_PANEL, fg=T.TEXT, font=(T.FONT, 12, "bold"),
                 anchor="w").pack(fill="x", padx=6, pady=(6, 2))

        # parameter audit table
        aud = section(self, "Parameter audit — verified against the literature (v0.5)")
        aud.pack(fill="x", padx=6, pady=2)
        cols = ("q", "model", "lit", "src")
        self.tv = ttk.Treeview(aud, columns=cols, show="headings", height=8)
        for c, t, w in [("q", "Quantity", 200), ("model", "In model", 140),
                        ("lit", "Literature", 160), ("src", "Source", 220)]:
            self.tv.heading(c, text=t); self.tv.column(c, width=w, anchor="w")
        self.tv.pack(fill="x")

        # live conservation
        cons = section(self, "Live conservation check (fresh heat)")
        cons.pack(fill="both", expand=True, padx=6, pady=6)
        row = tk.Frame(cons, bg=T.BG_PANEL); row.pack(fill="x")
        self.p_ledg = Pill(row, "element ledger", "warn"); self.p_ledg.pack(side="left", padx=4)
        self.p_clo = Pill(row, "first-law", "warn"); self.p_clo.pack(side="left", padx=4)
        self.p_end = Pill(row, "endpoint", "warn"); self.p_end.pack(side="left", padx=4)
        self.p_und = Pill(row, "undissolved", "warn"); self.p_und.pack(side="left", padx=4)
        ttk.Button(row, text="Re-run", command=self.run).pack(side="right", padx=4)

        self.fig, self.canvas = mpl(cons, figsize=(10, 3.2))
        self.canvas.get_tk_widget().pack(fill="both", expand=True, pady=4)

        self._fill_audit()
        self.run()

    def _fill_audit(self):
        summ = E.config_summary(self.app.cfg)
        floor = E.theoretical_floor_kWh_t(self.app.cfg)
        rows = [
            ("Latent heat of fusion", f"{summ['L_fusion (kJ/kg)']:.0f} kJ/kg", "247", "CRC Handbook 104th ed."),
            ("(FeO)+[C]→Fe+CO", "1.39 MJ/kg FeO", "+100 kJ/mol CO", "Turkdogan; Fruehan MSTS"),
            ("FeSi75 heat of solution", "−3511 kJ/kg", "−4681 kJ/kg Si", "Sigworth & Elliott 1974"),
            ("Carburiser heat of solution", "+1883 kJ/kg C", "+22.6 kJ/mol", "graphite dissolution"),
            ("Grid emission factor", f"{summ['Grid EF (tCO₂/MWh)']:.3f} tCO₂/MWh", "0.712", "CEA v21.0, FY2024-25"),
            ("Reversible melting floor", f"{floor:.0f} kWh/t", "practical ≈500", "computed, L_f=247"),
            ("Default tariff", f"₹{summ['Tariff (₹/kWh)']:.1f}/kWh", "₹6.0–8.5 grid", "HT industrial FY25-26"),
            ("Baseline SEC", f"{summ['Baseline SEC (kWh/t)']:.0f} kWh/t", "550–650 scrap IF", "field practice"),
        ]
        for r in rows:
            self.tv.insert("", "end", values=r)

    def run(self):
        self.app.set_status("running validation heat…", T.AMBER)
        cfg = self.app.cfg

        def work():
            specs = [E.AdditionSpec("Lime (92% CaO)", 10, 48),
                     E.AdditionSpec("FeSi75", 45, 15),
                     E.AdditionSpec("Mill scale (FeO)", 60, 150)]
            return E.run_heat(cfg, 12000, dict(E.DEFAULT_CHARGE_COMP), 5200,
                              additions=E.build_additions(specs), dt=2.0)

        def done(res):
            aim = getattr(cfg.plant, "tap_temperature_C", 1620)
            clo = res.energy["residual_pct"]
            self.p_ledg.set(f"element ledger {res.ledger_max_pct:.2f}% < 1%",
                            "ok" if res.ledger_max_pct < 1 else "warn")
            self.p_clo.set(f"first-law {clo:+.1f}%", "ok" if abs(clo) < 5 else "warn")
            hit = abs(res.endpoint["T_C"] - aim) <= 15
            self.p_end.set(f"endpoint {res.endpoint['T_C']:.0f}°C", "ok" if hit else "warn")
            self.p_und.set(f"undissolved {res.undissolved_kg:.0f} kg",
                           "ok" if res.undissolved_kg < 5 else "warn")

            lb = res.ledger_df
            self.fig.clear()
            ax = self.fig.add_subplot(111); style_axes(ax)
            if "closure_pct" in lb.columns:
                idc = "element" if "element" in lb.columns else lb.columns[0]
                ax.bar(lb[idc].astype(str), lb["closure_pct"].abs(), color=T.STEEL)
                ax.axhline(1.0, color=T.AMBER, ls="--")
                ax.set_ylabel("|closure| %"); ax.set_title("Per-element mass-balance closure")
            self.fig.tight_layout(); self.canvas.draw()
            self.app.set_status("validation ready", T.GREEN)
        self.app.run_async(work, done)


# ════════════════════════════════════════════════════════════════════════════
# 10 · About / Details
# ════════════════════════════════════════════════════════════════════════════
class HeatLogTab(ttk.Frame):
    """The audit trail — every event (start, additions, tap) across the session.
    This is the same table that serves as the ML training record and the
    shared-savings evidence, mirroring the HTML console's heat log."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        head = tk.Frame(self, bg=T.BG_PANEL); head.pack(fill="x", padx=6, pady=4)
        tk.Label(head, text="Heat log — audit trail", bg=T.BG_PANEL, fg=T.TEXT,
                 font=(T.FONT, 12, "bold")).pack(side="left")
        ttk.Button(head, text="Export CSV", command=self.export).pack(side="right", padx=4)
        ttk.Button(head, text="Clear", command=self.clear).pack(side="right", padx=4)

        tk.Label(self, text="Every advisory shown, every action taken and every outcome "
                 "lands here — the audit trail, the ML training set and the shared-savings "
                 "evidence are the same table.", bg=T.BG_PANEL, fg=T.TEXT_MUT,
                 font=(T.FONT, 8), anchor="w", wraplength=1100, justify="left").pack(fill="x", padx=6)

        cols = ("clock", "sim_min", "event", "detail")
        self.tv = ttk.Treeview(self, columns=cols, show="headings")
        for c, t, w in [("clock", "Clock", 90), ("sim_min", "Heat min", 80),
                        ("event", "Event", 130), ("detail", "Detail", 640)]:
            self.tv.heading(c, text=t); self.tv.column(c, width=w, anchor="w")
        self.tv.pack(fill="both", expand=True, padx=6, pady=4)
        # colour-tag events
        self.tv.tag_configure("HEAT START", foreground=T.GREEN)
        self.tv.tag_configure("ADDITION", foreground=T.STEEL)
        self.tv.tag_configure("TAP", foreground=T.MOLTEN)

        self.app.register_log_listener(self._on_log)
        for row in self.app.heat_log:
            self._on_log(row)

    def _on_log(self, row):
        self.tv.insert("", "end", values=(row["clock"], row["sim_min"], row["event"],
                                          row["detail"]), tags=(row["event"],))
        self.tv.yview_moveto(1.0)

    def clear(self):
        self.app.heat_log.clear()
        self.tv.delete(*self.tv.get_children())

    def export(self):
        import csv, os, datetime
        path = os.path.join(os.path.expanduser("~"),
                            f"smartmelt_heatlog_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
        try:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["clock", "sim_min", "event", "detail"])
                w.writeheader(); w.writerows(self.app.heat_log)
            self.app.set_status(f"heat log exported → {path}", T.GREEN)
        except Exception as e:
            self.app.set_status(f"export failed: {e}", T.RED)


class SettingsTab(ttk.Frame):
    """Plant / process settings the operator or manager can adjust — tap aim,
    carbon window, rated power, tariff, grid emission factor, baseline SEC.
    Changes update the shared config and are reflected across all tabs."""

    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        tk.Label(self, text="Settings — plant & process configuration", bg=T.BG_PANEL,
                 fg=T.TEXT, font=(T.FONT, 12, "bold"), anchor="w").pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(self, text="These set the aim and economic basis used by the advisory, "
                 "the endpoint checks and the economics. Adjust to match your plant, "
                 "then Apply.", bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 8),
                 anchor="w", wraplength=1100, justify="left").pack(fill="x", padx=6)

        body = tk.Frame(self, bg=T.BG_PANEL); body.pack(fill="x", padx=20, pady=10)
        cfg = self.app.cfg
        self.fields = {}
        specs = [
            ("tap_C", "Tap temperature aim (°C)", getattr(cfg.plant, "tap_temperature_C", 1620)),
            ("aim_clo", "Carbon aim — minimum (%)", getattr(cfg.plant, "aim_C_lo_pct", 0.05)),
            ("aim_chi", "Carbon aim — maximum (%)", getattr(cfg.plant, "aim_C_hi_pct", 0.25)),
            ("rated_kW", "Rated power (kW)", getattr(cfg.electrical, "rated_power_kW", 8000)),
            ("tariff", "Electricity tariff (₹/kWh)", getattr(cfg.economics, "tariff_INR_per_kWh", 7.0)),
            ("grid_ef", "Grid emission factor (tCO₂/MWh)", getattr(cfg.economics, "grid_EF_tCO2_per_MWh", 0.712)),
            ("baseline", "Baseline SEC (kWh/t)", getattr(cfg.economics, "baseline_SEC_kWh_per_t", 600)),
        ]
        for i, (key, label, val) in enumerate(specs):
            tk.Label(body, text=label, bg=T.BG_PANEL, fg=T.TEXT, font=(T.FONT, 10),
                     anchor="w").grid(row=i, column=0, sticky="w", pady=4, padx=(0, 16))
            e = tk.Entry(body, bg=T.BG_INPUT, fg=T.TEXT, insertbackground=T.TEXT,
                         font=(T.FONT_MONO, 10), width=14)
            e.insert(0, str(val)); e.grid(row=i, column=1, sticky="w", pady=4)
            self.fields[key] = e

        btns = tk.Frame(self, bg=T.BG_PANEL); btns.pack(fill="x", padx=20, pady=6)
        ttk.Button(btns, text="Apply settings", command=self.apply).pack(side="left")
        self.status = tk.Label(btns, text="", bg=T.BG_PANEL, fg=T.GREEN, font=(T.FONT, 9))
        self.status.pack(side="left", padx=12)

        # plant selector mirror
        tk.Label(self, text="Active plant configuration: " + app.plant_var.get(),
                 bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 9), anchor="w").pack(fill="x", padx=6, pady=(10, 0))

    def apply(self):
        cfg = self.app.cfg
        try:
            cfg.plant.tap_temperature_C = float(self.fields["tap_C"].get())
            if hasattr(cfg.plant, "aim_C_lo_pct"):
                cfg.plant.aim_C_lo_pct = float(self.fields["aim_clo"].get())
                cfg.plant.aim_C_hi_pct = float(self.fields["aim_chi"].get())
            cfg.electrical.rated_power_kW = float(self.fields["rated_kW"].get())
            cfg.economics.tariff_INR_per_kWh = float(self.fields["tariff"].get())
            cfg.economics.grid_EF_tCO2_per_MWh = float(self.fields["grid_ef"].get())
            cfg.economics.baseline_SEC_kWh_per_t = float(self.fields["baseline"].get())
            self.status.config(text="applied — advisory & economics updated", fg=T.GREEN)
            self.app.log_event("SETTINGS", "operator updated plant/process settings")
            self.app.notify_spec_changed()   # refresh dependent tabs
        except ValueError as e:
            self.status.config(text=f"invalid value: {e}", fg=T.RED)


class AboutTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        txt = tk.Text(self, bg=T.BG_INPUT, fg=T.TEXT, font=(T.FONT, 10),
                      wrap="word", padx=16, pady=14, relief="flat",
                      insertbackground=T.TEXT)
        sb = ttk.Scrollbar(self, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y"); txt.pack(fill="both", expand=True)

        # styling tags
        txt.tag_configure("h1", font=(T.FONT, 16, "bold"), foreground=T.MOLTEN, spacing3=8)
        txt.tag_configure("h2", font=(T.FONT, 12, "bold"), foreground=T.MOLTEN_HI, spacing1=10, spacing3=4)
        txt.tag_configure("b", font=(T.FONT, 10, "bold"), foreground=T.TEXT)
        txt.tag_configure("mut", foreground=T.TEXT_MUT)
        txt.tag_configure("mono", font=(T.FONT_MONO, 9), foreground=T.STEEL)

        def h1(s): txt.insert("end", s + "\n", "h1")
        def h2(s): txt.insert("end", s + "\n", "h2")
        def p(s): txt.insert("end", s + "\n")
        def mut(s): txt.insert("end", s + "\n", "mut")
        def mono(s): txt.insert("end", s + "\n", "mono")

        h1("SmartMelt Studio")
        p("Hybrid physics + machine-learning melt optimisation for induction, arc and "
          "basic-oxygen steelmaking. This desktop application is a full operator and "
          "manager console over the validated smartmelt engine. It is advisory-only — "
          "it reads and computes, and never writes to any control system.")
        mut(f"Engine version {E.VERSION}.  Plant identities are anonymised (Industry-X = "
            "MSME IF pilot, Industry-Y = integrated BOF).")

        h2("What each tab does")
        p("• Operator Console — build an ADDITION SCHEDULE (any flux, ferro-alloy or "
          "recarburiser, any mass, any time — multiple entries, as in real practice), then "
          "watch the coloured furnace fill with streaming KPIs (bath °C, carbon, melted %, "
          "SEC, slag FeO, B2, P, S, useful power), a live temperature/chemistry trend, and "
          "tap-readiness advice. SPEED control (⏸ / 1× real-time / 10× / 60×) plays the heat "
          "back at the pace you choose.")
        p("• Process Trajectory — the full six-panel physics of one heat: temperatures, "
          "inventories & dissolution, bath chemistry, slag & basicity, heat flows, and "
          "energy vs the theoretical floor.")
        p("• Physics & Energy — the heat-flow ledger, first-law conservation audit, and "
          "an energy split from grid input to tapped steel.")
        p("• Virtual Sensor (EKF) — an Extended Kalman Filter tracks bath temperature on "
          "a deliberately mismatched plant, assimilating only occasional immersion dips, "
          "and converges the hidden furnace efficiency within one heat.")
        p("• Machine Learning — a hybrid endpoint model: physics predicts, a GP residual "
          "head corrects it, and it gates itself off until it can prove out-of-time "
          "improvement. Shows the ML lift over physics and parity plots.")
        p("• Drift Monitor — population-stability (PSI) detection that flags when incoming "
          "scrap or practice shifts, so the model widens its uncertainty rather than guessing.")
        p("• Charge-Mix — a 17-stream scrap library (shredded, HMS 1/2, bushling, DRI, HBI, "
          "pig iron, turnings, cast, tin-plate, rail/rebar crop, Cr-alloy and more) with two "
          "modes: OPTIMISE finds the least-cost blend meeting the aim carbon window and the "
          "copper AND tin ceilings and reports the ceiling's shadow price in ₹/t; MANUAL lets "
          "the operator type kg per scrap and see the cost and predicted bath chemistry — so a "
          "hand-built charge can be compared against the optimiser's.")
        p("• Economics — savings, payback and CO₂ from a measured SEC reduction, at the "
          "corrected tariff and grid emission factor.")
        p("• Validation — the verified-parameter audit and a live conservation check.")

        h2("The engine behind the GUI")
        p("Every panel calls the real smartmelt package — no physics is re-implemented in "
          "the interface. The modules in use:")
        mono("physics.py    first-principles furnace model (mass, energy, kinetics, refractory)")
        mono("thermo.py     Wagner activities, equilibria, theoretical energy floor")
        mono("ekf.py        Extended Kalman virtual temperature sensor")
        mono("ml.py         hybrid GP-residual + GBM endpoint model, drift monitor")
        mono("chargemix.py  least-cost charge LP with tramp shadow prices")
        mono("mpc.py        receding-horizon power / tap-time advice")
        mono("advisory.py   bilingual traffic-light operator guidance")
        mono("simulator.py  virtual plant for rehearsal & ML data generation")
        mono("metrics.py    hit-rates, PSI, economics    calibrate.py  per-plant calibration")

        h2("Verified parameters (v0.5 literature pass)")
        p("Four physical constants were corrected against the metallurgical literature:")
        mono("latent heat of fusion   272 → 247 kJ/kg           CRC Handbook 104th ed.")
        mono("(FeO)+[C]→Fe+CO         1.89 → 1.39 MJ/kg FeO      Turkdogan; Fruehan MSTS")
        mono("FeSi75 heat of solution −1150 → −3511 kJ/kg        Sigworth & Elliott 1974")
        mono("carburiser              +2500 → +1883 kJ/kg C      graphite dissolution")
        mono("grid emission factor    0.82 → 0.712 tCO₂/MWh      CEA v21.0, FY2024-25")
        p("C_to_CO and Fe_to_FeO are coupled by Hess's law and enforced in the test suite: "
          "their difference is the enthalpy of (FeO)+[C]→Fe+CO, ≈ +100 kJ/mol CO. The "
          "reversible melting minimum is ≈381 kWh/t; the practical floor for a real "
          "coreless furnace is ≈500 kWh/t.")

        h2("Commercial figures (rebanded to verified Indian market data)")
        mono("tariff            ₹6.0–8.5/kWh grid  (₹5.0–6.5 open access/captive)")
        mono("avoidable energy  30–80 kWh/t")
        mono("payback           4–12 months (sub-6 only high-util/high-tariff)")
        mono("licence           ₹15–25 lakh per furnace")
        p("Endpoint accuracy is competitive with published Level-2/ML melt models; a "
          "tolerance is always quoted with the achieved hit-rate for the specific plant.")

        h2("How to run")
        mono("python -m gui.app        (from the smartmelt_model folder)")
        p("Requires Python 3.9+ with numpy, pandas, scipy, scikit-learn and matplotlib. "
          "Tkinter ships with standard Python. No server, no browser, no internet needed — "
          "suitable for an offline shop-floor edge PC.")

        mut("\nFigures are indicative until sized against a plant's audited baseline. "
            "This tool supports operators and managers; it does not replace metallurgical "
            "judgement or plant safety systems.")

        txt.configure(state="disabled")
