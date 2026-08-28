# SmartMelt Studio — smooth Streamlit replica of `run_gui.py`

This release preserves the original 12-tab desktop workflow and the SmartMelt physics engine, while correcting browser-specific rerun, calculation, clipping and responsiveness problems.

## Main corrections in this release

### No full-screen blanking during live operation

The Operator Console is divided into three independent Streamlit fragments:

- controls and calculation status: 800 ms refresh;
- furnace state: 400 ms refresh;
- clock, KPIs, advisories and live trends: 400 ms refresh.

Operator actions use fragment-scoped reruns only. The full application is not rerun when a heat starts, a speed changes, a material is added or a heat is tapped. Streamlit's stale-element fade is also disabled, so existing values remain visible while a fragment updates.

### Material additions no longer restart the entire heat

Every live frame now retains:

- the complete thermodynamic state vector;
- the exact undissolved-addition pool;
- the corresponding process timestamp.

When an operator adds lime, FeSi, carburiser, mill scale or another material, the numerical model resumes from that exact frame and recalculates only the remaining trajectory. Regression testing confirms zero numerical difference from a full minute-zero recalculation.

### Calculation isolated from the web interface

Heat calculations run in a separate short-lived Python process. The Streamlit server therefore remains responsive even on a modest single-PC deployment. A compact amber banner reports calculation activity while the existing furnace, KPIs and trends stay on screen.

### Clearer and safer layout

- Current tracked Streamlit tabs provide full tab names and lazy execution.
- The tab row scrolls horizontally on narrow displays rather than clipping labels.
- KPI and advisory cards grow with their text instead of hiding it.
- Button labels wrap cleanly; “Carburiser” and “Mill scale” are no longer truncated.
- Slider labels, status text, logs and advisory messages have increased contrast and line spacing.
- The workspace is responsive up to 1440 px and includes a compact narrow-screen mode.

### Furnace-state parity retained

`app/furnace_renderer.py` remains a literal SVG translation of `gui/theme.py::FurnaceCanvas.draw()`:

- same 340 × 240 coordinate system;
- same liquid-steel level equation;
- same slag thickness and slag-level equations;
- same temperature-dependent molten-metal colour;
- same scrap and flux positions and layer order;
- smooth 360 ms interpolation between browser updates.

## Run on Windows

Double-click:

```text
run_streamlit.bat
```

The launcher creates `.venv`, upgrades the required packages—including Streamlit 1.60 or newer—and starts the application.

Manual commands:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade -r requirements.txt
python -m streamlit run streamlit_app.py
```

## Run on Linux or macOS

```bash
./run_streamlit.sh
```

Manual commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

## Docker

```bash
docker compose up --build
```

## Verification

```bash
python -m pytest tests -q
```

The release includes conservation, thermodynamic, furnace-parity, UI-contract and exact-continuation tests.
