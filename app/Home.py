"""Streamlit Community Cloud entrypoint wrapper."""
from pathlib import Path
exec((Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8"),
     {"__name__": "__main__", "__file__": str(Path(__file__).resolve().parents[1] / "streamlit_app.py")})
