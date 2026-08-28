#!/usr/bin/env python3
"""
SmartMelt Studio — desktop GUI launcher.

Run this from the smartmelt_model folder:
    python run_gui.py

It opens a native desktop window (Tkinter) with the full operator/manager
console over the validated smartmelt engine. No server, no browser, no
internet required.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for p in (_ROOT, _ROOT / "app" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if __name__ == "__main__":
    try:
        from gui.app import main
    except ModuleNotFoundError as e:
        print("Missing dependency:", e)
        print("Install requirements first:")
        print("    pip install numpy pandas scipy scikit-learn matplotlib pyyaml")
        print("(Tkinter ships with standard Python. On some Linux distros: "
              "sudo apt-get install python3-tk)")
        sys.exit(1)
    main()
