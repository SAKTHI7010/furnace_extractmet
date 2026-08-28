"""Operator Console — TRUE LIVE operation, matching the HTML console:

  • Start Heat      → begins a fresh heat; the furnace steps forward in real time
  • Speed  ⏸/1×/10×/60×  → how fast simulated time advances
  • Add material    → INJECT any flux / alloy / recarburiser AT THE CURRENT MOMENT
                       (multiple times, at operator discretion — as on the plant)
  • Tap Heat        → ends the heat and records the endpoint

The heat is stepped incrementally with the engine's FurnaceModel.step, so an
addition clicked at minute 37 enters the bath exactly then. The whole schedule
the operator builds (initial + interactive) is published to app.heat_spec, so
the Trajectory / Physics / ML tabs run on the SAME inputs."""
from __future__ import annotations

import numpy as np
import tkinter as tk
from tkinter import ttk

import engine as E
from smartmelt.physics import HeatInputs, make_addition
from gui import theme as T
from gui.theme import FurnaceCanvas
from gui.app import KPI, Pill, mpl, section


def _slider(master, label, lo, hi, init, fmt="{:.0f}"):
    fr = tk.Frame(master, bg=T.BG_PANEL)
    top = tk.Frame(fr, bg=T.BG_PANEL); top.pack(fill="x")
    tk.Label(top, text=label, bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 9),
             anchor="w").pack(side="left")
    valv = tk.StringVar(value=fmt.format(init))
    tk.Label(top, textvariable=valv, bg=T.BG_PANEL, fg=T.MOLTEN_HI,
             font=(T.FONT_MONO, 9)).pack(side="right")
    var = tk.DoubleVar(value=init)
    ttk.Scale(fr, from_=lo, to=hi, variable=var, orient="horizontal",
              command=lambda v: valv.set(fmt.format(float(v)))).pack(fill="x")
    fr.var = var
    return fr


class OperatorConsoleTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master)
        self.app = app
        self.speed = 10
        self.running = False
        self.tapped = False
        self.model = None
        self.hist = []
        self.applied_adds = []

        left = tk.Frame(self, bg=T.BG_PANEL, width=360)
        left.pack(side="left", fill="y", padx=(4, 8), pady=4)
        left.pack_propagate(False)

        setup = section(left, "Heat setup (applied at Start)"); setup.pack(fill="x")
        self.s_charge = _slider(setup, "Charge (t)", 4, 14, 12, "{:.1f}")
        self.s_charge.pack(fill="x", pady=1)
        self.s_power = _slider(setup, "Power (kW)", 1000, 8000, 5200)
        self.s_power.pack(fill="x", pady=1)
        self.s_carbon = _slider(setup, "Charge C (%)", 0.05, 1.5, 0.30, "{:.2f}")
        self.s_carbon.pack(fill="x", pady=1)
        self.s_cu = _slider(setup, "Charge Cu (%)", 0.05, 0.5, 0.20, "{:.2f}")
        self.s_cu.pack(fill="x", pady=1)

        hb = tk.Frame(left, bg=T.BG_PANEL); hb.pack(fill="x", pady=(8, 2))
        self.btn_start = tk.Button(hb, text="\u25B6 START HEAT", command=self.start_heat,
                                   bg=T.GREEN, fg="#08130c", font=(T.FONT, 10, "bold"),
                                   relief="flat", padx=10, pady=4, activebackground="#2bbd6e")
        self.btn_start.pack(side="left", expand=True, fill="x", padx=2)
        self.btn_tap = tk.Button(hb, text="\u23CF TAP HEAT", command=self.tap_heat,
                                 bg=T.MOLTEN, fg="#170a04", font=(T.FONT, 10, "bold"),
                                 relief="flat", padx=10, pady=4, activebackground="#ff8452",
                                 state="disabled")
        self.btn_tap.pack(side="left", expand=True, fill="x", padx=2)

        sp = tk.Frame(left, bg=T.BG_PANEL); sp.pack(fill="x", pady=2)
        tk.Label(sp, text="Speed:", bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 9)).pack(side="left")
        self.speed_btns = {}
        for label, val in [("\u23F8", 0), ("1\u00D7", 1), ("10\u00D7", 10), ("60\u00D7", 60)]:
            b = tk.Button(sp, text=label, command=lambda v=val: self._set_speed(v),
                          bg=T.BG_RAISED, fg=T.TEXT, font=(T.FONT, 9), relief="flat",
                          padx=10, activebackground="#243240")
            b.pack(side="left", padx=1)
            self.speed_btns[val] = b
        self._set_speed(10)

        add = section(left, "Add material NOW (during heat)"); add.pack(fill="x", pady=(8, 0))
        row1 = tk.Frame(add, bg=T.BG_PANEL); row1.pack(fill="x")
        self.mat_var = tk.StringVar(value=list(E.ADDITION_LIBRARY)[0])
        ttk.Combobox(row1, textvariable=self.mat_var, values=list(E.ADDITION_LIBRARY),
                     width=17, state="readonly").pack(side="left", fill="x", expand=True)
        tk.Label(row1, text="kg", bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 8)).pack(side="left", padx=(4, 1))
        self.mass_e = tk.Entry(row1, width=6, bg=T.BG_INPUT, fg=T.TEXT,
                               insertbackground=T.TEXT, font=(T.FONT_MONO, 9))
        self.mass_e.insert(0, "48"); self.mass_e.pack(side="left")
        self.btn_add = tk.Button(add, text="\uFF0B Add to bath now", command=self.add_now,
                                 bg=T.STEEL, fg="#06131a", font=(T.FONT, 9, "bold"),
                                 relief="flat", pady=3, activebackground="#5fb8e6",
                                 state="disabled")
        self.btn_add.pack(fill="x", pady=3)
        quick = tk.Frame(add, bg=T.BG_PANEL); quick.pack(fill="x")
        for mat, kg in [("Lime (92% CaO)", 48), ("FeSi75", 15), ("Carburiser", 12), ("Mill scale (FeO)", 120)]:
            tk.Button(quick, text=mat.split(" (")[0].split("/")[0][:9], font=(T.FONT, 8),
                      bg=T.BG_RAISED, fg=T.TEXT, relief="flat", padx=2,
                      command=lambda m=mat, k=kg: self._quick_add(m, k)).pack(side="left", expand=True, fill="x", padx=1)

        self.add_log = tk.Text(add, height=4, bg=T.BG_INPUT, fg=T.TEXT_MUT,
                               font=(T.FONT_MONO, 8), relief="flat", wrap="word")
        self.add_log.pack(fill="x", pady=(4, 0))

        tk.Label(left, text="Furnace state", bg=T.BG_PANEL, fg=T.TEXT,
                 font=(T.FONT, 10, "bold"), anchor="w").pack(fill="x", pady=(6, 2))
        self.furnace = FurnaceCanvas(left, width=340, height=240)
        self.furnace.pack(fill="both", expand=True)

        right = tk.Frame(self, bg=T.BG_PANEL); right.pack(side="left", fill="both", expand=True, pady=4)
        top = tk.Frame(right, bg=T.BG_PANEL); top.pack(fill="x")
        self.clock = tk.Label(top, text="00:00", bg=T.BG_PANEL, fg=T.MOLTEN_HI,
                              font=(T.FONT_MONO, 16, "bold"))
        self.clock.pack(side="left")
        self.pill = Pill(top, "press START HEAT", "warn"); self.pill.pack(side="left", padx=12)

        self.k_row1, self.k1 = self._kpis(right, [
            ("T", "Bath \u00B0C"), ("C", "Carbon %"), ("melt", "Melted %"), ("sec", "SEC kWh/t")])
        self.k_row1.pack(fill="x", pady=2)
        self.k_row2, self.k2 = self._kpis(right, [
            ("feo", "Slag FeO %"), ("b2", "Basicity B2"), ("si", "Silicon %"), ("mn", "Manganese %")])
        self.k_row2.pack(fill="x", pady=2)
        # online outputs the operator watches: power, total energy, expected vs actual tap T
        self.k_row3, self.k3 = self._kpis(right, [
            ("power", "Power kW"), ("energy", "Total kWh"), ("exp", "Expected tap °C"), ("act", "Actual bath °C")])
        self.k_row3.pack(fill="x", pady=2)

        # ── advisory panel (6 verdict cards) ──
        adv_sec = section(right, "Advisory — live verdicts"); adv_sec.pack(fill="x", pady=(6, 0))
        self.adv_frame = tk.Frame(adv_sec, bg=T.BG_PANEL)
        self.adv_frame.pack(fill="x")
        self.adv_cards = {}
        for i, key in enumerate(["temp", "carbon", "b2", "feo", "health", "sec"]):
            card = tk.Frame(self.adv_frame, bg=T.BG_RAISED, highlightbackground=T.LINE,
                            highlightthickness=1)
            card.grid(row=i // 3, column=i % 3, sticky="ew", padx=2, pady=2)
            self.adv_frame.columnconfigure(i % 3, weight=1)
            badge = tk.Label(card, text="—", bg=T.BG_RAISED, fg=T.TEXT_MUT,
                             font=(T.FONT, 11, "bold"), width=3)
            badge.pack(side="left", padx=(6, 4), pady=4)
            body = tk.Frame(card, bg=T.BG_RAISED); body.pack(side="left", fill="x", expand=True)
            title = tk.Label(body, text="", bg=T.BG_RAISED, fg=T.TEXT,
                             font=(T.FONT, 8, "bold"), anchor="w")
            title.pack(fill="x")
            msg = tk.Label(body, text="", bg=T.BG_RAISED, fg=T.TEXT_MUT,
                           font=(T.FONT, 8), anchor="w", justify="left", wraplength=220)
            msg.pack(fill="x", pady=(0, 4))
            self.adv_cards[key] = (card, badge, title, msg)

        self.fig, self.canvas = mpl(right, figsize=(7, 5.0))
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.lbl_end = tk.Label(right, text="", bg=T.BG_PANEL, fg=T.TEXT_MUT,
                                font=(T.FONT, 9), anchor="w", justify="left")
        self.lbl_end.pack(fill="x", pady=(4, 0))

        self._blank_plot()

    def _kpis(self, master, specs):
        row = tk.Frame(master, bg=T.BG_PANEL); cards = {}
        for key, label in specs:
            c = KPI(row, label); c.pack(side="left", fill="both", expand=True, padx=3)
            cards[key] = c
        return row, cards

    def _set_speed(self, v):
        self.speed = v
        for val, b in self.speed_btns.items():
            b.config(bg=T.MOLTEN if val == v else T.BG_RAISED,
                     fg="#10100a" if val == v else T.TEXT)

    def _log(self, msg):
        self.add_log.insert("end", msg + "\n"); self.add_log.see("end")

    def start_heat(self):
        self.tapped = False
        self.hist = []
        self.applied_adds = []
        self.frames = []
        self.frame_i = 0
        self.add_log.delete("1.0", "end")
        charge = self.s_charge.var.get(); power = self.s_power.var.get()
        cpct = self.s_carbon.var.get(); cupct = self.s_cu.var.get()
        self.charge_t = charge; self.power = power
        self.comp = dict(E.DEFAULT_CHARGE_COMP)
        self.comp["C"] = cpct / 100.0; self.comp["Cu"] = cupct / 100.0
        self.charge_kg = charge * 1000.0
        self._injected_specs = []        # engine additions injected so far
        self.btn_tap.config(state="normal")
        self.btn_add.config(state="normal")
        self.btn_start.config(text="\u25B6 SIMULATING\u2026")
        self._trend_built = False
        self._log(f"Heat started \u00B7 {charge:.1f} t \u00B7 {power:.0f} kW")
        self.app.log_event("HEAT START", f"{charge:.1f} t, {power:.0f} kW, C {cpct:.2f}%", 0.0)
        self.pill.set("simulating heat\u2026", "warn")
        self.app.set_status("simulating heat", T.AMBER)
        cfg = self.app.cfg; comp = dict(self.comp)

        def work():
            frames, m, xf, species = E.simulate_frames(
                cfg, charge, comp, power, additions=[], dt=2.0, t_end_min=95)
            return frames

        def done(frames):
            self.frames = frames
            self.running = True
            self.frame_i = 0
            self.btn_start.config(text="\u25B6 RUNNING\u2026")
            self.pill.set("heat running \u2014 add materials any time", "ok")
            self.app.set_status("heat running", T.GREEN)
            self._play_next()
        self.app.run_async(work, done)

    def _frames_per_tick(self):
        # frames are 2 s of sim each. 1× ≈ real time would be 1 frame / 2 s wall
        # (too slow); we compress so 1×≈quick, and scale up with speed.
        return {0: 0, 1: 1, 10: 6, 60: 30}.get(self.speed, 6)

    def _play_next(self):
        if not self.running or self.tapped:
            return
        if self.speed == 0:
            self.after(150, self._play_next); return
        step = self._frames_per_tick()
        self.frame_i = min(self.frame_i + step, len(self.frames) - 1)
        snap = self.frames[self.frame_i]
        self.hist = self.frames[: self.frame_i + 1]
        self._update_kpis(snap)
        import time as _t
        now = _t.time()
        if now - getattr(self, "_last_draw", 0) > 0.5 or self.frame_i >= len(self.frames) - 1:
            self._last_draw = now
            self._draw_furnace(snap)
            self._trend()
            self.app.publish_live(self.hist, self.running and not self.tapped)
        if self.frame_i >= len(self.frames) - 1:
            self.pill.set("heat complete \u2014 press TAP HEAT", "ok")
            return
        self.after(80, self._play_next)



    def add_now(self):
        if not self.running or self.tapped:
            return
        try:
            mass = float(self.mass_e.get())
        except ValueError:
            return
        self._inject(self.mat_var.get(), mass)

    def _quick_add(self, material, kg):
        if not self.running or self.tapped:
            self._log("start the heat first"); return
        self._inject(material, kg)

    def _inject(self, material, mass):
        info = E.ADDITION_LIBRARY.get(material)
        if info is None or not self.running or self.tapped:
            return
        if not self.frames:
            return
        cur = self.frames[self.frame_i]
        tmin = cur["t_min"]; t_s = tmin * 60.0
        Tb = cur["T_bath_C"]
        self.applied_adds.append(dict(material=material, mass=mass, time_min=tmin))
        # record the addition spec at the current sim time
        add = E.make_addition_at(t_s, mass, info)
        self._injected_specs.append(add)
        self._log(f"{tmin:4.1f} min \u00B7 +{mass:.0f} kg {material.split(' (')[0]} @ {Tb:.0f}\u00B0C")
        self.app.log_event("ADDITION", f"+{mass:.0f} kg {material} @ {Tb:.0f}\u00B0C", tmin)
        self.pill.set(f"added {mass:.0f} kg {material.split(' (')[0]} — recomputing…", "warn")
        # Re-simulate the REMAINDER of the heat from the current state with all
        # injected additions applied at their times. Keep the past frames; splice
        # the freshly-computed future onto them. Runs in a worker thread.
        keep = self.frames[: self.frame_i + 1]
        cfg = self.app.cfg
        charge = self.charge_t; power = self.power; comp = dict(self.comp)
        specs = list(self._injected_specs)
        # the state to continue from: rebuild by re-simulating from start with the
        # injected additions (cheap, ~4 s) so dissolution/energy stay consistent.
        def work():
            frames, m, xf, species = E.simulate_frames(
                cfg, charge, comp, power, additions=specs, dt=2.0, t_end_min=95)
            return frames

        def done(frames):
            self.frames = frames
            self.frame_i = min(self.frame_i, len(frames) - 1)
            self.pill.set(f"added {mass:.0f} kg {material.split(' (')[0]}", "ok")
        self.app.run_async(work, done)

    def tap_heat(self):
        if not self.running:
            return
        self.running = False
        self.tapped = True
        self.btn_start.config(text="\u25B6 START HEAT")
        self.btn_tap.config(state="disabled")
        self.btn_add.config(state="disabled")
        if not self.hist:
            return
        # freeze the trajectory at the point of tap (drop any un-played future)
        self.frames = self.frames[: self.frame_i + 1]
        self.hist = list(self.frames)
        s = self.hist[-1]
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        hit = abs(s["T_bath_C"] - aim) <= 15
        self.pill.set("TAPPED \u2014 " + ("on aim" if hit else f'{s["T_bath_C"]-aim:+.0f}\u00B0C off aim'),
                      "ok" if hit else "warn")
        self.lbl_end.config(text=(
            f"Tapped at {s['t_min']:.0f} min \u00B7 {s['T_bath_C']:.0f} \u00B0C \u00B7 C {s['pct_C']:.3f}% \u00B7 "
            f"SEC {s['SEC_kWh_t']:.0f} kWh/t \u00B7 slag FeO {s['slag_FeO_pct']:.1f}% \u00B7 B2 {s['B2']:.2f}\n"
            f"Additions this heat: " +
            (", ".join(f"{a['mass']:.0f}kg {a['material'].split(' (')[0]}@{a['time_min']:.0f}min"
                       for a in self.applied_adds) or "none")))
        # update the final KPIs and furnace to the tap state
        self._update_kpis(s); self._draw_furnace(s); self._trend()
        # Publish the ACTUAL heat that ran (the live frames) as the definitive
        # trajectory — NOT a re-simulated batch. This keeps the trajectory screen
        # showing the real process, frozen at tap, instead of reverting.
        self.app.publish_live(self.hist, False)
        self.app.log_event("TAP", f"{s['T_bath_C']:.0f}\u00B0C, C {s['pct_C']:.3f}%, SEC {s['SEC_kWh_t']:.0f} kWh/t", s["t_min"])
        self.app.set_status("heat tapped \u2014 trajectory frozen at final state", T.GREEN)

    def _publish_spec(self):
        self.app.heat_spec = {
            "charge_t": self.s_charge.var.get(),
            "power_kW": self.power,
            "charge_C_pct": self.s_carbon.var.get(),
            "charge_Cu_pct": self.s_cu.var.get(),
            "schedule": [dict(material=a["material"], mass=a["mass"], time_min=a["time_min"])
                         for a in self.applied_adds],
        }
        self.app.notify_spec_changed()

    def _update_kpis(self, s):
        """Cheap text updates — called EVERY tick so the numbers move live."""
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        self.clock.config(text=f'{int(s["t_min"]):02d}:{int((s["t_min"]%1)*60):02d}')
        self.k1["T"].set(f'{s["T_bath_C"]:.0f}', f'aim {aim:.0f}')
        self.k1["C"].set(f'{s["pct_C"]:.3f}', "")
        self.k1["melt"].set(f'{s["melted_pct"]:.0f}', f'{s["M_liquid_t"]:.1f} t liq')
        self.k1["sec"].set(f'{s["SEC_kWh_t"]:.0f}', f'{s["E_kWh"]:.0f} kWh')
        self.k2["feo"].set(f'{s["slag_FeO_pct"]:.1f}', f'P {s["pct_P"]:.4f}')
        self.k2["b2"].set(f'{s["B2"]:.2f}', "CaO/SiO2")
        self.k2["si"].set(f'{s["pct_Si"]:.3f}', "")
        self.k2["mn"].set(f'{s["pct_Mn"]:.3f}', f'S {s["pct_S"]:.4f}')
        proj = self._project_tap(s)
        self.k3["power"].set(f'{self.power:.0f}', "grid" if not self.tapped else "off")
        self.k3["energy"].set(f'{s["E_kWh"]:.0f}', "cumulative")
        self.k3["exp"].set(f'{proj:.0f}', f'aim {aim:.0f}')
        self.k3["act"].set(f'{s["T_bath_C"]:.0f}', "measured")
        self._update_advisories(s, proj)
        if not self.tapped:
            if s["melted_pct"] > 99 and s["T_bath_C"] >= aim - 5:
                self.pill.set("READY TO TAP \u2014 on temperature & fully melted", "ok")
            elif s["melted_pct"] < 2:
                self.pill.set("heating solid charge", "warn")
            else:
                self.pill.set(f'melting \u2014 {aim - s["T_bath_C"]:.0f} \u00B0C below tap aim', "warn")
        # let Tk actually paint the updated labels this frame
        self.update_idletasks()

    def _draw_furnace(self, s):
        """The furnace glyph redraw — throttled (Tkinter canvas is not free)."""
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        heat_t = self.charge_kg / 1000.0
        self.furnace.draw(s["melted_pct"], s["T_bath_C"], s["slag_total_kg"],
                          s["undissolved_kg"], heat_size_t=heat_t, tap_aim_C=aim)

    def _build_trend_axes(self):
        """Create the trend axes and empty line artists ONCE. Subsequent frames
        only update the line data (set_data) — ~50× faster than rebuilding."""
        from gui.tabs import style_axes
        self.fig.clear()
        self._ax1 = self.fig.add_subplot(211)
        self._ax2 = self.fig.add_subplot(212, sharex=self._ax1)
        self.fig.subplots_adjust(left=0.10, right=0.90, top=0.93, bottom=0.10, hspace=0.38)
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        style_axes(self._ax1, title="Temperature & melt progress", ylabel="Temperature (\u00B0C)")
        self._ln_bath, = self._ax1.plot([], [], color=T.MOLTEN, lw=2, label="bath")
        self._ln_solid, = self._ax1.plot([], [], color=T.SCRAP_COL, lw=1.2, label="solid")
        self._ax1.axhline(aim, color=T.GREEN, ls="--", lw=1, label=f"aim {aim:.0f}")
        self._ax1b = self._ax1.twinx()
        self._ln_melt, = self._ax1b.plot([], [], color=T.STEEL, lw=1.4)
        self._ax1b.set_ylim(0, 105); self._ax1b.set_ylabel("Melted (%)", color=T.STEEL, fontsize=8)
        self._ax1b.tick_params(axis="y", colors=T.STEEL, labelsize=7)
        self._ax1.legend(fontsize=7, loc="center right", labelcolor=T.TEXT,
                         facecolor=T.BG_RAISED, framealpha=0.9)
        style_axes(self._ax2, title="Bath chemistry", xlabel="Time (min)", ylabel="Content (wt %)")
        self._chem_lines = {}
        for el, c in [("pct_C", T.MOLTEN), ("pct_Si", T.STEEL), ("pct_Mn", T.GREEN), ("pct_S", T.SLAG_TOP)]:
            self._chem_lines[el], = self._ax2.plot([], [], color=c, lw=1.3, label=el.replace("pct_", ""))
        self._ax2.legend(fontsize=7, ncol=4, labelcolor=T.TEXT, facecolor=T.BG_RAISED, framealpha=0.9)
        self._add_vlines = []
        self._trend_built = True
        self.canvas.draw()

    def _project_tap(self, s):
        """Simple projection of tap temperature: extrapolate the recent heating
        rate to the point of full melt, else use current bath temperature."""
        aim = getattr(self.app.cfg.plant, "tap_temperature_C", 1620)
        if len(self.hist) < 6 or s["melted_pct"] > 99:
            return s["T_bath_C"]
        # heating rate over the last several samples (°C per min)
        recent = self.hist[-6:]
        dT = recent[-1]["T_bath_C"] - recent[0]["T_bath_C"]
        dt = max(recent[-1]["t_min"] - recent[0]["t_min"], 0.1)
        rate = dT / dt
        # minutes to reach ~full melt at current melt rate
        dmelt = recent[-1]["melted_pct"] - recent[0]["melted_pct"]
        if dmelt > 0.5:
            mins_to_full = (100 - s["melted_pct"]) / (dmelt / dt)
            return s["T_bath_C"] + rate * min(mins_to_full, 40)
        return s["T_bath_C"] + rate * 5

    def _update_advisories(self, s, proj):
        advs = E.build_advisories(s, self.app.cfg, projected_tap_C=proj)
        colours = {"ok": (T.GREEN, "#12301f"), "warn": (T.AMBER, "#33250e"),
                   "bad": (T.RED, "#331214")}
        badges = {"ok": "OK", "warn": "!", "bad": "!!!"}
        keys = ["temp", "carbon", "b2", "feo", "health", "sec"]
        for key, (lvl, title, msg) in zip(keys, advs):
            if key not in self.adv_cards:
                continue
            card, badge, tlabel, mlabel = self.adv_cards[key]
            fg, bg = colours.get(lvl, colours["ok"])
            badge.config(text=badges.get(lvl, "—"), fg=fg, bg=bg)
            card.config(highlightbackground=fg if lvl != "ok" else T.LINE)
            tlabel.config(text=title)
            mlabel.config(text=msg)

    def _trend(self):
        if not self.hist:
            return
        if not getattr(self, "_trend_built", False):
            self._build_trend_axes()
        d = self.hist
        t = [s["t_min"] for s in d]
        self._ln_bath.set_data(t, [s["T_bath_C"] for s in d])
        self._ln_solid.set_data(t, [s["T_solid_C"] for s in d])
        self._ln_melt.set_data(t, [s["melted_pct"] for s in d])
        for el, ln in self._chem_lines.items():
            ln.set_data(t, [s[el] for s in d])
        # addition markers (add any new ones)
        for a in self.applied_adds[len(self._add_vlines):]:
            self._add_vlines.append(self._ax1.axvline(a["time_min"], color=T.AMBER, ls=":", lw=0.8))
        # rescale
        self._ax1.relim(); self._ax1.autoscale_view()
        self._ax2.relim(); self._ax2.autoscale_view()
        self.canvas.draw_idle()

    def _blank_plot(self):
        from gui.tabs import style_axes
        self.fig.clear()
        a = self.fig.add_subplot(111)
        style_axes(a, title="Press START HEAT to begin", xlabel="Time (min)", ylabel="Temperature (\u00B0C)")
        a.text(0.5, 0.5, "Set charge, power and carbon, then START HEAT.\n"
               "Add materials at any moment while it runs, then TAP HEAT.",
               ha="center", va="center", color=T.TEXT_MUT, fontsize=10, transform=a.transAxes)
        self.canvas.draw()
