"""Child-process entrypoint for one SmartMelt live heat calculation."""
from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app" / "lib"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import engine as E


def main(input_path: str, output_path: str, error_path: str) -> int:
    try:
        with open(input_path, "rb") as fh:
            payload = pickle.load(fh)
        frames, states, pools = E.simulate_frames_live(
            payload["cfg_obj"], payload["charge_t"], payload["comp"],
            payload["power_kW"], additions=payload.get("additions") or [], dt=2.0,
            t_end_min=95.0, from_state=payload.get("from_state"),
            from_pool=payload.get("from_pool"), t0_s=payload.get("t0_s", 0.0),
            cooperative=False,
        )
        bundle = {"frames": frames, "states": states, "pools": pools}
        tmp = str(output_path) + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        Path(tmp).replace(output_path)
        return 0
    except Exception:
        Path(error_path).write_text(traceback.format_exc(), encoding="utf-8")
        return 1


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: sim_worker.py INPUT.pkl OUTPUT.pkl ERROR.txt")
    raise SystemExit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
