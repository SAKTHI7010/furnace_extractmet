# SmartMelt Studio — desktop GUI

A comprehensive **native desktop** application (Tkinter + Matplotlib) over the
validated `smartmelt` engine. No server, no browser, no ports, no internet —
it opens as an ordinary window and runs offline, suitable for a shop-floor edge PC.

## Run

From the `smartmelt_model` folder:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pyyaml
python run_gui.py
```

Tkinter ships with standard Python (Windows/macOS). On some Linux distributions
you may need `sudo apt-get install python3-tk`.

## What's inside

```
gui/
  app.py        main window, theme, plant selector, background-thread helper
  tabs.py       all ten tabs
  theme.py      colours + the coloured furnace canvas (native Tk drawing)
  cache/        pre-computed EKF result + dataset so heavy tabs load instantly
run_gui.py      launcher
```

Every tab calls the real `smartmelt` package through the proven engine bridge
(`app/lib/engine.py`); no physics is re-implemented in the GUI. The modules in
use: `physics`, `thermo`, `ekf`, `ml`, `chargemix`, `mpc`, `advisory`,
`simulator`, `metrics`, `calibrate`, plus the plant configs in `configs/`.

## The ten tabs

1. **Operator Console** — run a heat and watch the **coloured furnace** fill
   (molten metal colour shifts with bath temperature; slag cap, solid scrap,
   floating undissolved flux, refractory lining and copper coil all drawn to
   scale), with streaming KPIs (bath °C, carbon, melted %, SEC, slag FeO, B2,
   P, S), a live temperature trend, tap-readiness status and an advisory note.
   Press **Play** to animate, or drag the time slider.
2. **Process Trajectory** — the full six-panel physics of one heat.
3. **Physics & Energy** — heat-flow ledger, first-law conservation audit, and
   the grid-to-steel energy split.
4. **Virtual Sensor (EKF)** — the Kalman filter tracks a mismatched plant from
   immersion dips and converges the hidden efficiency. Loads a pre-computed
   result instantly; a live run (~1 min) is an explicit button.
5. **Machine Learning** — the hybrid endpoint model with physics-vs-ML lift and
   parity plots. Trains on a pre-computed dataset instantly; live generation is
   an explicit button.
6. **Drift Monitor** — PSI population-drift alarms on the same dataset.
7. **Charge-Mix** — least-cost compliant blend + copper shadow price.
8. **Economics** — savings, payback and CO₂ at the corrected tariff/grid factor.
9. **Validation** — verified-parameter audit + live conservation check.
10. **About / Details** — full package information, module list, verified
    parameters and sources.

## Performance note

Two computations are inherently heavy: the EKF (finite-difference Jacobians over
a 34-state model) and virtual-plant dataset generation (~3–4 s per heat). Both
ship **pre-computed** so those tabs open instantly. Running them live is offered
as a clearly-labelled button, and always runs on a background thread so the
window stays responsive.

## Notes

- Advisory-only: the app reads and computes, and never writes to any control system.
- Plant identities are anonymised (Industry-X = MSME IF pilot, Industry-Y = BOF).
- Figures are indicative until sized against a plant's audited baseline.
