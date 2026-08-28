# SmartMelt smooth Streamlit validation report

## Numerical engine

- Original public `simulate_frames()` return contract retained.
- Original SmartMelt physics, reaction, energy and advisory equations retained.
- Exact live checkpoints added outside the governing equations.
- Continuation after an addition matches a full recalculation to below `1e-10` for bath temperature, melt progress, chemistry, slag inventory, undissolved mass and cumulative energy.

## Responsiveness

- CPU-heavy calculations execute in an isolated Python child process.
- No Streamlit calls are made from the numerical worker.
- Operator actions call `st.rerun(scope="fragment")`, not a full-app rerun.
- Three small fragments replace the previous monolithic 200 ms fragment.
- Existing elements remain fully opaque while Streamlit marks them stale.
- Live trend remains SVG-based; no Matplotlib image is regenerated in the Operator Console.

## Layout

- Tracked/lazy `st.tabs` preserves the 12-tab order and executes only the active page.
- Tab labels scroll instead of being compressed.
- KPI, advisory, log and button text is allowed to wrap and grow.
- Furnace geometry and process colours retain native-GUI parity.

## Automated result

`27 passed` in the clean source tree, covering:

- thermodynamic and conservation contracts;
- IF and EAF model behaviour;
- dissolution and charge-mix logic;
- furnace geometry and colour parity;
- exact state continuation;
- isolated background-process wiring;
- fragment-only live reruns;
- deployment dependencies and package structure.
