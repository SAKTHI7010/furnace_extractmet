# Build manifest — SmartMelt Streamlit Smooth Final

## Primary entrypoints

- `streamlit_app.py` — local/Docker Streamlit entrypoint.
- `app/Home.py` — Streamlit Community Cloud wrapper.
- `run_streamlit.bat` — one-click Windows environment and launcher.
- `run_streamlit.sh` — Linux/macOS environment and launcher.
- `run_gui.py` — retained native Tkinter application.

## Live-console performance modules

- `app/exact_tabs.py` — isolated fragments, checkpoint splicing and operator workflow.
- `app/background_jobs.py` — non-blocking child-process job interface.
- `app/sim_worker.py` — isolated numerical worker.
- `app/lib/engine.py` — exact state/pool checkpoint and continuation support.
- `app/furnace_renderer.py` — native-equivalent furnace SVG.
- `app/exact_ui.py` — stable dark layout, anti-flicker and text-clarity rules.

## Required runtime

- Python 3.11 recommended.
- Streamlit 1.60 or newer, below major version 2.
- Dependencies listed in root `requirements.txt`.

## Verification command

```bash
PYTHONPATH=app/lib:. python -m pytest tests -q
```
