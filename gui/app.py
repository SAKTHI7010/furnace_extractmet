"""
SmartMelt Studio — comprehensive desktop GUI (Tkinter + Matplotlib).

A full operator + manager console over the validated `smartmelt` engine
(physics core, thermo, EKF virtual sensor, hybrid ML, charge-mix LP, MPC,
drift monitor). Runs as a native desktop window — no server, no browser.

Launch:
    python -m gui.app
or:
    python gui/app.py

Tabs
    Operator Console   live heat: coloured furnace, streaming KPIs, tap advice
    Process Trajectory six-panel physics of one heat
    Physics & Energy   heat-flow ledger, conservation audit, energy split
    Virtual Sensor     EKF tracking a mismatched plant from immersion dips
    Machine Learning   hybrid endpoint model, physics-vs-ML lift, parity plots
    Drift Monitor      PSI population-drift alarms
    Charge-Mix         least-cost blend + copper shadow price
    Economics          savings / payback / CO2
    Validation         verified-parameter audit + live conservation
    About / Details    package information, model equations, sources
"""
from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

import numpy as np

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# make the package + bridge importable no matter how launched
_HERE = Path(__file__).resolve()
_ROOT = _HERE.parents[1]                 # .../smartmelt_model
for p in (_ROOT, _ROOT / "app" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import engine as E                         # the proven engine bridge
from gui import theme as T
from gui.theme import FurnaceCanvas

matplotlib.rcParams.update({
    "figure.facecolor": T.BG_PANEL, "axes.facecolor": "#0f1418",
    "axes.edgecolor": T.LINE, "axes.labelcolor": T.TEXT_MUT,
    "text.color": T.TEXT, "xtick.color": T.TEXT_MUT, "ytick.color": T.TEXT_MUT,
    "grid.color": "#20262c", "font.size": 8, "axes.titlesize": 9,
    "axes.titlecolor": T.TEXT, "figure.dpi": 100,
})


# ════════════════════════════════════════════════════════════════════════════
# small reusable widgets
# ════════════════════════════════════════════════════════════════════════════
class KPI(tk.Frame):
    """A labelled value card."""
    def __init__(self, master, label, value="—", sub=""):
        super().__init__(master, bg=T.BG_RAISED, highlightbackground=T.LINE,
                         highlightthickness=1)
        self.lab = tk.Label(self, text=label.upper(), bg=T.BG_RAISED,
                            fg=T.TEXT_MUT, font=(T.FONT, 8), anchor="w")
        self.lab.pack(fill="x", padx=10, pady=(8, 0))
        self.val = tk.Label(self, text=value, bg=T.BG_RAISED, fg=T.MOLTEN,
                            font=(T.FONT, 20, "bold"), anchor="w")
        self.val.pack(fill="x", padx=10)
        self.sub = tk.Label(self, text=sub, bg=T.BG_RAISED, fg=T.TEXT_MUT,
                            font=(T.FONT, 8), anchor="w")
        self.sub.pack(fill="x", padx=10, pady=(0, 8))

    def set(self, value=None, sub=None):
        if value is not None:
            self.val.config(text=value)
        if sub is not None:
            self.sub.config(text=sub)


class Pill(tk.Label):
    """A status chip."""
    COLOURS = {"ok": (T.GREEN, "#12301f"), "warn": (T.AMBER, "#33250e"),
               "bad": (T.RED, "#331214")}

    def __init__(self, master, text="", kind="ok"):
        fg, bg = self.COLOURS.get(kind, (T.GREEN, "#12301f"))
        super().__init__(master, text=text, bg=bg, fg=fg,
                         font=(T.FONT, 9, "bold"), padx=10, pady=3)

    def set(self, text, kind="ok"):
        fg, bg = self.COLOURS.get(kind, (T.GREEN, "#12301f"))
        self.config(text=text, fg=fg, bg=bg)


def mpl(master, figsize=(7, 4)):
    """Create an embedded matplotlib figure + canvas, return (fig, canvas)."""
    fig = Figure(figsize=figsize, facecolor=T.BG_PANEL)
    canvas = FigureCanvasTkAgg(fig, master=master)
    canvas.get_tk_widget().configure(bg=T.BG_PANEL, highlightthickness=0)
    return fig, canvas


def section(master, title):
    """A titled panel frame."""
    outer = tk.Frame(master, bg=T.BG_PANEL)
    tk.Label(outer, text=title, bg=T.BG_PANEL, fg=T.TEXT,
             font=(T.FONT, 11, "bold"), anchor="w").pack(fill="x", pady=(0, 4))
    return outer


# ════════════════════════════════════════════════════════════════════════════
# main application
# ════════════════════════════════════════════════════════════════════════════
class SmartMeltGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SmartMelt Studio — melt optimisation (physics + ML)")
        self.geometry("1280x820")
        self.minsize(1080, 720)
        self.configure(bg=T.BG_DEEP)

        self._init_style()

        # shared state
        self.configs = E.available_configs()
        self.plant_var = tk.StringVar(
            value="if_msme_12t" if "if_msme_12t" in self.configs else
            (list(self.configs)[0] if self.configs else ""))
        self.cfg = E.get_config(self.plant_var.get()) if self.configs else None

        # caches so tabs don't recompute needlessly
        self._heat = None
        self._dataset = None

        # SHARED heat specification — set by the Operator Console, consumed by the
        # Trajectory / Physics / ML tabs so every view reflects the SAME heat:
        # the operator's actual charge, power and full addition schedule.
        self.heat_spec = {
            "charge_t": 12.0,
            "power_kW": 5200.0,
            "charge_C_pct": 0.30,
            "charge_Cu_pct": 0.20,
            "schedule": [
                dict(material="Lime (92% CaO)", mass=48, time_min=8),
                dict(material="FeSi75", mass=15, time_min=42),
                dict(material="Carburiser", mass=12, time_min=48),
                dict(material="Mill scale (FeO)", mass=120, time_min=58),
            ],
        }
        # tabs can register a callback to refresh when the spec changes
        self._spec_listeners = []
        # LIVE history from the running console heat, for cross-tab live plots
        self.live_hist = None            # list of snapshot dicts while a heat runs
        self.live_running = False
        self._live_listeners = []
        # heat log — the audit trail of every event across the session
        self.heat_log = []               # list of dicts: time, event, detail
        self._log_listeners = []

        self._build_header()
        self._build_tabs()

    def register_spec_listener(self, fn):
        """Trajectory/Physics/ML register here; called when the operator's heat
        specification changes so they re-run on the same inputs."""
        self._spec_listeners.append(fn)

    def register_live_listener(self, fn):
        """Tabs that show LIVE trajectories register here; called each console
        frame with the growing history so they animate alongside the console."""
        self._live_listeners.append(fn)

    def register_log_listener(self, fn):
        self._log_listeners.append(fn)

    def log_event(self, event, detail="", sim_min=None):
        """Append an event to the heat log (audit trail / ML training record)."""
        import datetime
        row = {
            "clock": datetime.datetime.now().strftime("%H:%M:%S"),
            "sim_min": f"{sim_min:.1f}" if sim_min is not None else "",
            "event": event,
            "detail": detail,
        }
        self.heat_log.append(row)
        for fn in list(self._log_listeners):
            try:
                fn(row)
            except Exception:
                pass

    def publish_live(self, hist, running):
        """Called by the Operator Console every frame to share the live heat."""
        self.live_hist = hist
        self.live_running = running
        for fn in list(self._live_listeners):
            try:
                fn(hist, running)
            except Exception:
                pass

    def notify_spec_changed(self):
        self._spec_heat_cache = None          # force a fresh run on the new spec
        for fn in list(self._spec_listeners):
            try:
                fn()
            except Exception:
                pass

    def build_spec_additions(self):
        """Return engine addition objects for the current shared schedule."""
        specs = [E.AdditionSpec(a["material"], a["time_min"], a["mass"])
                 for a in self.heat_spec["schedule"]]
        return E.build_additions(specs)

    def run_spec_heat(self):
        """Run one heat using the shared operator spec (used by Trajectory/Physics).
        The result is cached and keyed on the spec so multiple tabs share a single
        simulation rather than each re-running it (which would contend on the GIL)."""
        s = self.heat_spec
        key = (s["charge_t"], s["power_kW"], s["charge_C_pct"], s["charge_Cu_pct"],
               tuple((a["material"], a["mass"], round(a["time_min"], 1)) for a in s["schedule"]))
        cached = getattr(self, "_spec_heat_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        comp = dict(E.DEFAULT_CHARGE_COMP)
        comp["C"] = s["charge_C_pct"] / 100.0
        comp["Cu"] = s["charge_Cu_pct"] / 100.0
        res = E.run_heat(self.cfg, s["charge_t"] * 1000.0, comp, s["power_kW"],
                         additions=self.build_spec_additions(), dt=2.0)
        self._spec_heat_cache = (key, res)
        return res

    # ── styling ─────────────────────────────────────────────────────────────
    def _init_style(self):
        s = ttk.Style(self)
        try:
            s.theme_use("clam")
        except tk.TclError:
            pass
        s.configure("TNotebook", background=T.BG_DEEP, borderwidth=0)
        s.configure("TNotebook.Tab", background=T.BG_PANEL, foreground=T.TEXT_MUT,
                    padding=(14, 7), font=(T.FONT, 9))
        s.map("TNotebook.Tab",
              background=[("selected", T.BG_RAISED)],
              foreground=[("selected", T.MOLTEN)])
        s.configure("TFrame", background=T.BG_PANEL)
        s.configure("TLabel", background=T.BG_PANEL, foreground=T.TEXT)
        s.configure("Treeview", background=T.BG_INPUT, fieldbackground=T.BG_INPUT,
                    foreground=T.TEXT, borderwidth=0, font=(T.FONT_MONO, 9),
                    rowheight=22)
        s.configure("Treeview.Heading", background=T.BG_RAISED, foreground=T.TEXT,
                    font=(T.FONT, 9, "bold"))
        s.map("Treeview", background=[("selected", "#243240")])
        s.configure("TButton", background=T.BG_RAISED, foreground=T.TEXT,
                    borderwidth=1, font=(T.FONT, 9))
        s.map("TButton", background=[("active", "#243240")],
              foreground=[("active", T.MOLTEN_HI)])
        s.configure("Horizontal.TScale", background=T.BG_PANEL)
        s.configure("TCombobox", fieldbackground=T.BG_INPUT, background=T.BG_RAISED,
                    foreground=T.TEXT, arrowcolor=T.TEXT)

    # ── header ──────────────────────────────────────────────────────────────
    def _build_header(self):
        bar = tk.Frame(self, bg=T.BG_PANEL, height=52)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)
        tk.Label(bar, text="  🔥 SmartMelt Studio", bg=T.BG_PANEL, fg=T.TEXT,
                 font=(T.FONT, 15, "bold")).pack(side="left", padx=6)
        tk.Label(bar, text=f"engine v{E.VERSION} · advisory-only",
                 bg=T.BG_PANEL, fg=T.TEXT_MUT, font=(T.FONT, 9)).pack(side="left")

        # plant selector
        tk.Label(bar, text="Plant:", bg=T.BG_PANEL, fg=T.TEXT_MUT,
                 font=(T.FONT, 9)).pack(side="left", padx=(24, 4))
        cb = ttk.Combobox(bar, textvariable=self.plant_var,
                          values=list(self.configs), width=16, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", self._on_plant_change)

        self.status = tk.Label(bar, text="ready", bg=T.BG_PANEL, fg=T.GREEN,
                               font=(T.FONT, 9))
        self.status.pack(side="right", padx=12)

    def _on_plant_change(self, *_):
        self.cfg = E.get_config(self.plant_var.get())
        self._heat = None
        self._dataset = None
        self.set_status(f"plant → {self.plant_var.get()}", T.STEEL)

    def set_status(self, text, colour=None):
        self.status.config(text=text, fg=colour or T.GREEN)
        self.update_idletasks()

    # ── tabs ────────────────────────────────────────────────────────────────
    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=6, pady=6)

        from gui.tabs import (
            TrajectoryTab, PhysicsEnergyTab,
            VirtualSensorTab, MachineLearningTab, DriftMonitorTab,
            ChargeMixTab, EconomicsTab, ValidationTab, AboutTab,
            HeatLogTab, SettingsTab,
        )
        from gui.console_tab import OperatorConsoleTab
        self.tabs = {}
        for name, cls in [
            ("Operator Console", OperatorConsoleTab),
            ("Process Trajectory", TrajectoryTab),
            ("Physics & Energy", PhysicsEnergyTab),
            ("Virtual Sensor", VirtualSensorTab),
            ("Machine Learning", MachineLearningTab),
            ("Drift Monitor", DriftMonitorTab),
            ("Charge-Mix", ChargeMixTab),
            ("Economics", EconomicsTab),
            ("Heat Log", HeatLogTab),
            ("Settings", SettingsTab),
            ("Validation", ValidationTab),
            ("About / Details", AboutTab),
        ]:
            frame = cls(self.nb, self)
            self.nb.add(frame, text=name)
            self.tabs[name] = frame

    # ── background-thread helper (keeps the UI responsive) ──────────────────
    def run_async(self, work, on_done, on_error=None):
        """Run `work()` in a thread; deliver the result on the main loop via a
        queue that the main thread polls. Using a queue (rather than calling
        .after() from the worker thread) keeps all Tk calls on the main thread,
        which Tkinter requires."""
        import queue
        if not hasattr(self, "_result_q"):
            self._result_q = queue.Queue()
            self._poll_results()

        def worker():
            try:
                result = work()
                self._result_q.put(("ok", result, on_done, on_error))
            except Exception as exc:  # noqa
                self._result_q.put(("err", (exc, traceback.format_exc()),
                                    on_done, on_error))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_results(self):
        """Main-thread poller that dispatches finished background work."""
        import queue
        try:
            while True:
                kind, payload, on_done, on_error = self._result_q.get_nowait()
                if kind == "ok":
                    try:
                        on_done(payload)
                    except Exception as exc:  # noqa
                        self._default_error((exc, traceback.format_exc()))
                else:
                    (on_error or self._default_error)(payload)
        except queue.Empty:
            pass
        self.after(80, self._poll_results)

    def _default_error(self, err):
        exc, tb = err
        self.set_status(f"error: {exc}", T.RED)
        messagebox.showerror("SmartMelt", f"{type(exc).__name__}: {exc}")


def main():
    app = SmartMeltGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
