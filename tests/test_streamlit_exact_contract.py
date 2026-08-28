from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_single_window_entrypoint_and_exact_tab_order():
    text = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    expected = [
        "Operator Console", "Process Trajectory", "Physics & Energy",
        "Virtual Sensor", "Machine Learning", "Drift Monitor", "Charge-Mix",
        "Economics", "Heat Log", "Settings", "Validation", "About / Details",
    ]
    positions = [text.index(f'"{name}"') for name in expected]
    assert positions == sorted(positions)
    assert "st.tabs(names" in text
    assert 'on_change="rerun"' in text
    assert "if tab.open" in text
    assert "render_operator_console" in text


def test_operator_workflow_is_live_and_not_checkbox_schedule_only():
    text = (ROOT / "app" / "exact_tabs.py").read_text(encoding="utf-8")
    required = [
        "start_simulation_job", "E.make_addition_at", "_start_heat", "_inject",
        "_tap_heat", "op_speed", "START HEAT", "TAP HEAT",
        "Add to bath now", "build_advisories", "furnace_svg",
    ]
    for token in required:
        assert token in text


def test_live_console_never_forces_a_full_app_rerun():
    text = (ROOT / "app" / "exact_tabs.py").read_text(encoding="utf-8")
    live = text[text.index("def _render_operator_controls"):text.index("# ────────────────────────────────────────────────────────────────────────────\n# Process Trajectory")]
    assert 'st.rerun(scope="fragment")' in live
    assert "st.rerun()" not in live
    assert '@st.fragment(run_every="400ms")' in live
    assert '@st.fragment(run_every="800ms")' in live


def test_addition_uses_exact_continuation_not_minute_zero_restart():
    text = (ROOT / "app" / "exact_tabs.py").read_text(encoding="utf-8")
    for token in ["from_state", "from_pool", "cut_i", "prefix_frames", "np.concatenate"]:
        assert token in text
    engine = (ROOT / "app" / "lib" / "engine.py").read_text(encoding="utf-8")
    assert "def simulate_frames_live" in engine
    assert "states[j]" in engine and "pools[j]" in engine


def test_cpu_heavy_heat_runs_in_an_isolated_process():
    jobs = (ROOT / "app" / "background_jobs.py").read_text(encoding="utf-8")
    worker = (ROOT / "app" / "sim_worker.py").read_text(encoding="utf-8")
    assert "subprocess.Popen" in jobs
    assert "start_simulation_job" in jobs
    assert "simulate_frames_live" in worker
    assert "OMP_NUM_THREADS" in jobs


def test_stale_screen_dimming_and_text_clipping_are_disabled():
    css = (ROOT / "app" / "exact_ui.py").read_text(encoding="utf-8")
    for token in ['[data-stale="true"]', "opacity:1", "overflow-wrap:anywhere",
                  "white-space:normal", ".calc-banner", "max-width:1440px"]:
        assert token in css


def test_all_native_tabs_have_streamlit_renderers():
    text = (ROOT / "app" / "exact_tabs.py").read_text(encoding="utf-8")
    renderers = [
        "render_operator_console", "render_trajectory", "render_physics",
        "render_ekf", "render_ml", "render_drift", "render_charge_mix",
        "render_economics", "render_heat_log", "render_settings",
        "render_validation", "render_about",
    ]
    for fn in renderers:
        assert f"def {fn}" in text


def test_deployment_dependencies_cover_streamlit_cache_and_matplotlib():
    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for name in ["streamlit>=1.60", "matplotlib", "numpy", "pandas", "pyarrow",
                 "scipy", "scikit-learn", "pyyaml"]:
        assert name in req


def test_no_streamlit_multipage_sidebar_competes_with_native_tab_order():
    assert not (ROOT / "app" / "pages").exists()
    assert (ROOT / "legacy_streamlit_pages").exists()


def test_reference_desktop_screen_and_deployment_files_present():
    for rel in ["REFERENCE_run_gui_initial.png", "Dockerfile",
                "docker-compose.yml", ".streamlit/config.toml",
                "README_STREAMLIT_EXACT.md"]:
        assert (ROOT / rel).exists(), rel
