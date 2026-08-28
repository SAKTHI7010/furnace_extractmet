# SmartMelt Studio — Streamlit application

A comprehensive operator + manager GUI over the validated `smartmelt` engine
(physics core, EKF virtual sensor, hybrid ML endpoint model, charge-mix LP,
MPC, drift monitor). Advisory-only, bilingual-ready, edge-deployable.

## What's inside

```
app/
  Home.py                         entry point + plant selector
  requirements.txt
  lib/
    engine.py                     thin cached bridge to the real smartmelt package
    ui.py                         theme, KPI cards, Plotly styling, coloured furnace SVG
  pages/
    1_Operator_Console.py         LIVE heat: coloured furnace, streaming KPIs, tap advice
    2_Process_Trajectory.py       six-panel physics trajectory of one heat
    3_Physics_Energy.py           heat-flow ledger, conservation audit, energy waterfall
    4_Virtual_Sensor_EKF.py       EKF tracking a mismatched plant from immersion dips
    5_Machine_Learning.py         hybrid endpoint model, physics-vs-ML lift, parity plots
    6_Drift_Monitor.py            PSI population-drift alarms
    7_Charge_Mix_Optimiser.py     least-cost blend + copper shadow price
    8_Economics_Validation.py     savings/payback/CO2 + the parameter-audit validation window
```

Every page calls the real engine through `lib/engine.py`; no physics is
re-implemented in the UI. The modules it touches — `physics`, `thermo`, `ekf`,
`ml`, `chargemix`, `mpc`, `advisory`, `simulator`, `metrics`, `calibrate` — and
the plant config files under `configs/` are all wired in.

## Run locally

From the repository root (the folder that contains both `smartmelt/` and `app/`):

```bash
pip install -r app/requirements.txt
streamlit run app/Home.py
```

Then open http://localhost:8501. Pick a plant in the sidebar; the choice applies
to every page.

## The coloured furnace view

The Operator Console renders a live coreless-induction cross-section that shows,
in real furnace colours: molten metal (colour shifts with bath temperature),
the slag cap, remaining solid scrap, floating undissolved flux lumps, the
refractory lining and the copper coil — alongside a streaming KPI grid (bath
temperature, carbon, melted %, SEC, slag FeO, B2, P, S) and a traffic-light tap
readiness call. Press **▶ Play heat** to animate, or scrub the time slider.

## Performance note

The Machine Learning and Drift pages generate their datasets by running the full
physics simulator (~3–4 s per heat), so results are cached with
`@st.cache_data`. Start with a small number of heats and increase once you've
seen it work. The single-heat pages (Operator Console, Trajectory, Physics,
Validation) are near-instant.

## Deploying to the web

Because it's a standard Streamlit multipage app, any of these work:

- **Streamlit Community Cloud** — push the repo to GitHub, point the app at
  `app/Home.py`, and set `app/requirements.txt` as the requirements file. The
  `smartmelt/` package and `configs/` must be in the same repo (they are).
- **Docker / any VM** — `pip install -r app/requirements.txt` then
  `streamlit run app/Home.py --server.port $PORT --server.address 0.0.0.0`.
- **Behind a reverse proxy** — set `--server.baseUrlPath` if hosting under a
  subpath; enable `--server.enableCORS false --server.enableXsrfProtection true`
  as appropriate for your deployment.

For a shop-floor deployment the same app runs on a local edge PC with no
internet — Streamlit serves on the plant LAN, and no data leaves the site.

## Notes

- Plant identities are anonymised (Industry-X = MSME IF pilot, Industry-Y =
  integrated BOF).
- Figures are indicative until sized against a plant's audited baseline.
- The app is advisory-only: it reads and computes, and never writes to any
  control system.
