/**
 * SmartMelt Studio — Main Frontend Application
 * Handles: tab navigation, simulation polling, settings, charge-mix LP,
 * heat-log, ML predictions, drift monitoring, economics, validation, and
 * the live Three.js furnace + ApexCharts integration.
 *
 * HTML ID Conventions (matching index.html):
 *  Tabs:      data-tab="operator-console" / pane-operator-console
 *  Buttons:   op-btn-start, op-btn-tap, op-btn-add-material
 *  KPI cards: kpi-bath-temp, kpi-carbon, kpi-melted, kpi-sec, kpi-power, kpi-energy
 *  Sliders:   op-charge, op-power, op-carbon, op-copper
 *  Charts:    chart-live-trend, chart-traj-temp … chart-traj-energy,
 *             chart-phys-heatflows … chart-ekf-params, chart-ml-scatter, chart-drift-psi …
 *  Furnace:   furnace-3d-container, furnace-temp-badge
 *  Log:       op-log-box, log-heats-tbody, log-events-tbody
 */
(function () {
  'use strict';

  // ═══════════════════════════════════════════════════════════════
  //  APPLICATION STATE
  // ═══════════════════════════════════════════════════════════════
  const state = {
    simRunning: false,
    simJobId: null,
    pollTimer: null,
    heatCounter: 1,
    heatLog: [],
    speedMultiplier: 10,

    settings: {
      tap_aim_C: 1620,
      c_lo: 0.05,
      c_hi: 0.25,
      power_kW: 8000,
      tariff_per_kwh: 7.0,
      emission_factor: 0.712,
      baseline_sec: 600,
      furnace_capacity_t: 12.0
    },

    snapshot: {
      t_sec: 0,
      bath_temp_C: null,
      carbon_pct: null,
      silicon_pct: null,
      manganese_pct: null,
      slag_feo_pct: null,
      basicity: null,
      melted_pct: 0,
      melted_t: 0,
      power_kW: 0,
      energy_kwh: 0,
      sec: null,
      expected_tap_C: null,
      drift_C: 0,
      efficiency_pct: 0,
      slag_kg: 0,
      undissolved_kg: 0,
      ekf_confidence: 82,
      energy_breakdown: { melt: 0, superheat: 0, loss_wall: 0, loss_top: 0 }
    },

    chargeParams: {
      charge_t: 12.0,
      power_kW: 5200,
      carbon_pct: 0.30,
      copper_pct: 0.20
    },

    mlParams: {
      train_fraction: 0.70,
      n_heats: 40
    },

    driftParams: {
      n_heats: 50,
      regime_heat: 40
    },

    ekfParams: {
      eta: 0.90,
      ua: 1.35,
      dips: 3
    },

    mixParams: {
      target_t: 12.0,
      c_lo: 0.10,
      c_hi: 0.40,
      cu_ceiling: 0.20,
      sn_ceiling: 0.10
    },

    ecoParams: {
      annual_output_t: 40000,
      energy_saving_kwh_t: 40,
      licence_cost_lakh: 20
    }
  };

  // ═══════════════════════════════════════════════════════════════
  //  TAB NAVIGATION
  // ═══════════════════════════════════════════════════════════════
  function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        const panel = document.getElementById(`pane-${target}`);
        if (panel) panel.classList.add('active');
        onTabActivated(target);
      });
    });
  }

  function onTabActivated(tabName) {
    switch (tabName) {
      case 'process-trajectory': fetchAndRenderTrajectory(); break;
      case 'physics-energy':     fetchAndRenderPhysics(); break;
      case 'virtual-sensor':     initEKFTab(); break;
      case 'machine-learning':   initMLTab(); break;
      case 'drift-monitor':      initDriftTab(); break;
      case 'charge-mix':         initChargeMixTab(); break;
      case 'economics':          initEconomicsTab(); break;
      case 'heat-log':           /* already live */ break;
      case 'settings':           populateSettingsForm(); break;
      case 'validation':         fetchAndRenderValidation(); break;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  OPERATOR CONSOLE — SLIDERS
  // ═══════════════════════════════════════════════════════════════
  function bindSlider(sliderId, labelId, formatter) {
    const el = document.getElementById(sliderId);
    const lbl = document.getElementById(labelId);
    if (!el || !lbl) return;
    const update = () => { lbl.textContent = formatter(parseFloat(el.value)); };
    el.addEventListener('input', update);
    update();
  }

  function initOperatorSliders() {
    bindSlider('op-charge', 'op-charge-val', v => `${v.toFixed(1)} t`);
    bindSlider('op-power',  'op-power-val',  v => `${v.toFixed(0)} kW`);
    bindSlider('op-carbon', 'op-carbon-val', v => `${v.toFixed(2)} %`);
    bindSlider('op-copper', 'op-copper-val', v => `${v.toFixed(2)} %`);
  }

  // ═══════════════════════════════════════════════════════════════
  //  OPERATOR CONSOLE — CONTROLS
  // ═══════════════════════════════════════════════════════════════
  function bindOperatorControls() {
    const startBtn = document.getElementById('op-btn-start');
    const tapBtn   = document.getElementById('op-btn-tap');
    const addBtn   = document.getElementById('op-btn-add-material');

    if (startBtn) startBtn.addEventListener('click', onStartSim);
    if (tapBtn)   tapBtn.addEventListener('click', onTapHeat);
    if (addBtn)   addBtn.addEventListener('click', onAddMaterial);

    // Quick-add buttons
    document.querySelectorAll('.btn-quick-add').forEach(b => {
      b.addEventListener('click', () => {
        const matSel = document.getElementById('op-material-select');
        const massSel = document.getElementById('op-material-mass');
        if (matSel) matSel.value = b.dataset.material;
        if (massSel) massSel.value = b.dataset.mass;
      });
    });

    // Playback speed buttons
    document.querySelectorAll('.play-speed-btn').forEach(b => {
      b.addEventListener('click', () => {
        state.speedMultiplier = parseInt(b.dataset.speed, 10);
        document.querySelectorAll('.play-speed-btn').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
      });
    });
  }

  // ═══════════════════════════════════════════════════════════════
  //  SIMULATION LIFECYCLE
  // ═══════════════════════════════════════════════════════════════
  async function onStartSim() {
    if (state.simRunning) return;

    const chargeT   = parseFloat(document.getElementById('op-charge')?.value  || 12.0);
    const powerKW   = parseFloat(document.getElementById('op-power')?.value   || 5200);
    const carbonPct = parseFloat(document.getElementById('op-carbon')?.value  || 0.30);
    const copperPct = parseFloat(document.getElementById('op-copper')?.value  || 0.20);

    state.chargeParams = { charge_t: chargeT, power_kW: powerKW, carbon_pct: carbonPct, copper_pct: copperPct };

    // Show calc banner
    showCalcBanner(true, 'Preparing heat trajectory…');
    clearSnapshot();
    clearCharts();
    ThreeFurnace.reset();

    try {
      const data = await callApi('/api/operator/start', {
        charge_t: chargeT,
        power_kW: powerKW,
        C_pct: carbonPct,
        Cu_pct: copperPct
      });
      if (!data || data.error) throw new Error(data?.error || 'Server error');

      state.simJobId = (data.status === 'success') ? 'active-heat' : null;
      if (!state.simJobId) throw new Error('Server returned no success status');
      state.simRunning = true;
      state._totalFrames = data.n_frames || 0;

      setStatusPill('running', 'Running…');
      showCalcBanner(false);
      setButtonsForRunning(true);
      logEvent('HEAT START', `Charge: ${chargeT} t · Power: ${powerKW} kW · C: ${carbonPct}%`);
      // frames are pre-computed; playback is driven by frame index
      startPolling();

    } catch (e) {
      showCalcBanner(false);
      setStatusPill('error', 'Error: ' + e.message);
      showToast('Simulation failed to start: ' + e.message, 'error');
    }
  }

  function startPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollSnapshot, 1500);
  }

  async function pollSnapshot() {
    if (!state.simRunning) return;
    try {
      const res = await callApi('/api/status', {});
      if (!res) return;

      if (res.snapshot) {
        Object.assign(state.snapshot, res.snapshot);
        applySnapshotToUI(state.snapshot);
      }

      if (res.log_entry)  logEvent('SIM', res.log_entry);

      const advRes = await callApi('/api/advisory/evaluate', {});
      if (advRes && advRes.advisories) renderAdvisories(advRes.advisories);

      const valRes = await callApi('/api/validation/run', { snapshot: state.snapshot });
      if (valRes) {
        const ledgerPill = document.getElementById('op-ledger-pill');
        const closurePill = document.getElementById('op-closure-pill');
        if (ledgerPill) {
          ledgerPill.style.display = '';
          ledgerPill.textContent = `Ledger: ${(valRes.element_ledger_pct || 100).toFixed(1)}%`;
          ledgerPill.className = `status-pill ${valRes.element_ledger_pct > 99 ? 'status-ok' : 'status-warn'}`;
        }
        if (closurePill) {
          closurePill.style.display = '';
          closurePill.textContent = `Closure: ${(valRes.first_law_pct || 100).toFixed(1)}%`;
          closurePill.className = `status-pill ${valRes.first_law_pct > 99 ? 'status-ok' : 'status-warn'}`;
        }
      }

      // Derive snapshot from frames array
      if (res.op_frames && res.op_frames.length > 0) {
        const frames = res.op_frames;
        // Advance frame index
        state._frameIndex = Math.min((state._frameIndex || 0) + state.speedMultiplier, frames.length - 1);
        const f = frames[state._frameIndex];
        const snap = {
          t_sec: (f.t_min || 0) * 60,
          bath_temp_C: f.T_bath_C,
          carbon_pct: f.pct_C,
          silicon_pct: f.pct_Si,
          manganese_pct: f.pct_Mn,
          slag_feo_pct: f.slag_FeO_pct,
          basicity: f.B2,
          melted_pct: f.melted_pct != null ? f.melted_pct : 0,
          melted_t: f.M_liquid_t || (f.m_liquid_kg || 0) / 1000,
          power_kW: f.Q_useful_kW || 0,
          energy_kwh: f.E_kWh || 0,
          sec: f.SEC_kWh_t,
          expected_tap_C: f.T_bath_C,
          slag_kg: (f.slag_FeO_kg || 0) + (f.slag_CaO_kg || 0),
          undissolved_kg: f.undissolved_kg || f.m_undissolved_kg || 0,
          // raw frame for advisory
          T_bath_C: f.T_bath_C,
          pct_C: f.pct_C,
          pct_Si: f.pct_Si,
          pct_Mn: f.pct_Mn,
          pct_S: f.pct_S,
          slag_FeO_pct: f.slag_FeO_pct,
          B2: f.B2,
          SEC_kWh_t: f.SEC_kWh_t,
          Q_useful_kW: f.Q_useful_kW,
        };
        Object.assign(state.snapshot, snap);
        applySnapshotToUI(state.snapshot);
      }

      // Heat is complete when we've played through all frames
      const totalFrames = state._totalFrames || (res.op_frames ? res.op_frames.length : 0);
      const framesDone = totalFrames > 0 && state._frameIndex >= totalFrames - 1;
      if (res.op_complete || framesDone) {
        state.simRunning = false;
        clearInterval(state.pollTimer);
        setStatusPill('ready', 'Ready to Tap ▶');
        document.getElementById('op-btn-tap')?.removeAttribute('disabled');
        showToast('Heat complete — tap when ready', 'success');
        logEvent('HEAT READY', 'Simulation converged. Tap to transfer.');
      }

    } catch (e) {
      console.warn('Poll error:', e.message);
    }
  }

  function applySnapshotToUI(snap) {
    // ── KPI Cards
    const aimC = state.settings.tap_aim_C;
    setKPI('kpi-bath-temp',    snap.bath_temp_C, v => `${v.toFixed(0)} °C`,   `aim ${aimC} °C`, 'kpi-bath-temp-sub');
    setKPI('kpi-carbon',       snap.carbon_pct,  v => `${v.toFixed(3)} %`,    'wt %',           'kpi-carbon-sub');
    setKPI('kpi-melted',       snap.melted_pct,  v => `${v.toFixed(1)} %`,    `${(snap.melted_t||0).toFixed(2)} t liq`, 'kpi-melted-sub');
    setKPI('kpi-sec',          snap.sec,         v => `${v.toFixed(0)}`,      `${(snap.energy_kwh||0).toFixed(0)} kWh`,  'kpi-sec-sub');
    setKPI('kpi-slag-feo',     snap.slag_feo_pct,v => `${v.toFixed(1)} %`,    'wt %',           'kpi-slag-feo-sub');
    setKPI('kpi-basicity',     snap.basicity,    v => `${v.toFixed(2)}`,       'CaO / SiO₂',    'kpi-basicity-sub');
    setKPI('kpi-silicon',      snap.silicon_pct, v => `${v.toFixed(3)} %`,    'wt %',           'kpi-silicon-sub');
    setKPI('kpi-manganese',    snap.manganese_pct,v => `${v.toFixed(3)} %`,   'wt %',           'kpi-manganese-sub');
    setKPI('kpi-power',        snap.power_kW,    v => `${v.toFixed(0)}`,      'kW',             'kpi-power-sub');
    setKPI('kpi-energy',       snap.energy_kwh,  v => `${v.toFixed(1)}`,      'cumulative kWh', 'kpi-energy-sub');
    setKPI('kpi-expected-tap', snap.expected_tap_C, v => `${v.toFixed(0)} °C`, `aim ${aimC} °C`, 'kpi-expected-tap-sub');
    setKPI('kpi-actual-bath',  snap.bath_temp_C, v => `${v.toFixed(0)} °C`,  'measured',        'kpi-actual-bath-sub');

    // ── Sim clock
    setText('op-clock', formatTime(snap.t_sec));

    // ── Colour-code bath temp
    if (snap.bath_temp_C != null) {
      const tempEl = document.getElementById('kpi-bath-temp');
      if (tempEl) {
        const frac = Math.max(0, Math.min(1, (snap.bath_temp_C - 1000) / 700));
        tempEl.style.color = `hsl(${20 - frac * 20}deg, 90%, ${55 + frac * 15}%)`;
      }
    }

    // ── 3D furnace
    ThreeFurnace.update(
      snap.melted_pct || 0,
      snap.bath_temp_C || 30,
      snap.slag_kg || 0,
      snap.undissolved_kg || 0,
      state.settings.tap_aim_C
    );

    // ── Live trend chart
    updateLiveTrendChart(snap);
  }

  // ── Live trend line chart on Operator Console
  let liveTrendChart = null;
  const liveBufs = { temp: [], melted: [], power: [] };
  let liveTick = 0;

  function updateLiveTrendChart(snap) {
    liveTick = snap.t_sec || liveTick + 1;

    const pushPoint = (buf, val) => {
      buf.push({ x: Math.round(liveTick / 60), y: val != null ? Math.round(val * 10) / 10 : null });
      if (buf.length > 80) buf.shift();
    };

    pushPoint(liveBufs.temp, snap.bath_temp_C);
    pushPoint(liveBufs.melted, snap.melted_pct);

    if (!liveTrendChart) {
      const el = document.getElementById('chart-live-trend');
      if (!el) return;
      liveTrendChart = new ApexCharts(el, {
        chart: {
          type: 'line', height: '100%', background: 'transparent',
          foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif',
          toolbar: { show: false },
          animations: { enabled: true, easing: 'linear', dynamicAnimation: { speed: 700 } }
        },
        theme: { mode: 'dark' },
        series: [
          { name: 'Temp (°C)', data: [] },
          { name: 'Melted %', data: [] }
        ],
        stroke: { curve: 'smooth', width: [2.5, 1.5], dashArray: [0, 4] },
        colors: ['#f6ad55', '#9f7aea'],
        annotations: {
          yaxis: [{ y: state.settings.tap_aim_C, borderColor: '#fc8181', strokeDashArray: 4, label: { text: 'Tap Aim', style: { color: '#fc8181', background: 'transparent' } } }]
        },
        xaxis: { type: 'numeric', labels: { formatter: v => `${v}min` } },
        yaxis: [
          { title: { text: '°C', style: { fontSize: '10px', color: '#64748b' } }, min: 0 },
          { opposite: true, min: 0, max: 100, title: { text: '%', style: { fontSize: '10px', color: '#64748b' } } }
        ],
        tooltip: { theme: 'dark', style: { fontSize: '11px', fontFamily: 'Inter,sans-serif' } },
        grid: { borderColor: '#1e2c40', yaxis: { lines: { show: true } }, xaxis: { lines: { show: false } } },
        legend: { show: true, position: 'top' }
      });
      liveTrendChart.render();
    } else {
      liveTrendChart.updateSeries([
        { data: [...liveBufs.temp] },
        { data: [...liveBufs.melted] }
      ]);
    }
  }

  async function onTapHeat() {
    if (!state.simJobId) { showToast('No active heat to tap.', 'warn'); return; }
    const snap = state.snapshot;
    // First call server tap endpoint, get cut_i from current frame index
    const cutI = state._frameIndex || 0;
    await callApi('/api/operator/tap', { cut_i: cutI });

    const heat = {
      id: `H-${String(state.heatCounter++).padStart(4, '0')}`,
      timestamp: new Date().toLocaleTimeString(),
      tap_temp: snap.bath_temp_C?.toFixed(0) ?? '--',
      delta_t:  snap.bath_temp_C != null ? (snap.bath_temp_C - state.settings.tap_aim_C).toFixed(0) : '--',
      carbon:   snap.carbon_pct?.toFixed(3) ?? '--',
      delta_c:  '--',
      silicon:  snap.silicon_pct?.toFixed(3) ?? '--',
      manganese:snap.manganese_pct?.toFixed(3) ?? '--',
      sulphur:  '--',
      tap_mass: (snap.melted_t || state.chargeParams.charge_t * 0.96).toFixed(2),
      sec:      snap.sec?.toFixed(0) ?? '--',
      avg_power:snap.power_kW?.toFixed(0) ?? '--',
      adds_kg:  '--',
      dips:     '--',
      duration: (snap.t_sec / 60).toFixed(1),
      result:   snap.bath_temp_C >= state.settings.tap_aim_C - 10 ? 'PASS' : 'REVIEW'
    };

    state.heatLog.unshift(heat);
    appendHeatLogRow(heat);
    logEvent('HEAT TAP', `Heat ${heat.id} tapped at ${heat.tap_temp}°C · SEC: ${heat.sec} kWh/t`);
    resetSim();
    showToast(`Heat ${heat.id} tapped successfully`, 'success');
  }

  async function onAddMaterial() {
    if (!state.simRunning) { showToast('Start a heat first.', 'warn'); return; }
    const mat   = document.getElementById('op-material-select')?.value;
    const massKg = parseFloat(document.getElementById('op-material-mass')?.value || 48);
    if (!mat) return;

    await callApi('/api/operator/inject', { cut_i: state._frameIndex || 0, material: mat, mass: massKg });
    logEvent('ADD MATERIAL', `${massKg} kg ${mat} added to bath`);
    showToast(`${massKg} kg ${mat} added`, 'info');
  }

  function resetSim() {
    state.simRunning = false;
    state.simJobId = null;
    state._frameIndex = 0;
    if (state.pollTimer) clearInterval(state.pollTimer);
    clearSnapshot();
    ThreeFurnace.reset();
    clearCharts();
    setStatusPill('idle', 'Press START HEAT');
    showCalcBanner(false);
    setButtonsForRunning(false);
  }

  function clearSnapshot() {
    Object.assign(state.snapshot, {
      t_sec: 0, bath_temp_C: null, carbon_pct: null, silicon_pct: null,
      manganese_pct: null, slag_feo_pct: null, basicity: null,
      melted_pct: 0, melted_t: 0, power_kW: 0, energy_kwh: 0,
      sec: null, expected_tap_C: null, drift_C: 0, efficiency_pct: 0,
      slag_kg: 0, undissolved_kg: 0
    });
    // Reset KPI displays
    ['kpi-bath-temp','kpi-carbon','kpi-melted','kpi-sec','kpi-slag-feo',
     'kpi-basicity','kpi-silicon','kpi-manganese','kpi-power','kpi-energy',
     'kpi-expected-tap','kpi-actual-bath'].forEach(id => {
      const el = document.getElementById(id);
      if (el) { el.textContent = '—'; el.style.color = ''; }
    });
    setText('op-clock', '00:00');
  }

  function clearCharts() {
    liveBufs.temp = []; liveBufs.melted = []; liveBufs.power = [];
    liveTick = 0;
    if (liveTrendChart) { liveTrendChart.updateSeries([{data:[]},{data:[]}]); }
  }

  // ═══════════════════════════════════════════════════════════════
  //  ADVISORIES
  // ═══════════════════════════════════════════════════════════════
  function renderAdvisories(advisories) {
    const container = document.getElementById('op-advisory-container');
    if (!container) return;
    if (!advisories || advisories.length === 0) {
      container.innerHTML = '<div class="advisory-placeholder">No active advisories.</div>';
      return;
    }
    container.innerHTML = advisories.map(adv => `
      <div class="advisory-item ${adv.level || 'info'}">
        <span class="adv-dot">${advDot(adv.level)}</span>
        <div>
          <div class="adv-title">${adv.title}</div>
          <div class="adv-msg">${adv.message}</div>
        </div>
      </div>`).join('');
  }

  function advDot(level) {
    return level === 'critical' ? '🔴' : level === 'warning' ? '🟡' : '🟢';
  }

  // ═══════════════════════════════════════════════════════════════
  //  PROCESS TRAJECTORY TAB
  // ═══════════════════════════════════════════════════════════════
  let trajCharts = {};
  async function fetchAndRenderTrajectory() {
    const res = await callApi('/api/trajectory/run', {});
    if (!res) return;
    setText('traj-desc', res.description || 'Trajectory loaded from simulation data.');

    // KPIs from server response format
    const k = res.kpis || {};
    setKPI('traj-kpi-temp',   k.tap_temp,   v => `${v.toFixed(0)} °C`, `aim ${k.tap_temp_aim || 1620} °C`, 'traj-kpi-temp-sub');
    setKPI('traj-kpi-carbon', k.carbon,     v => `${v.toFixed(3)} %`);
    setKPI('traj-kpi-time',   k.tap_min,    v => `${v.toFixed(1)} min`);
    setKPI('traj-kpi-sec',    k.sec,        v => `${v.toFixed(0)} kWh/t`, `floor ${(k.sec_floor||381).toFixed(0)} kWh/t`, 'traj-kpi-sec-sub');
    setKPI('traj-kpi-ledger', k.ledger ? parseFloat(k.ledger) : null, v => `${v.toFixed(2)} %`);
    setText('traj-desc', res.description || 'Trajectory from operator spec or live heat.');

    // Convert pandas dataframe rows → ApexCharts series
    if (res.data) {
      const d = res.data; // array of row objects
      // 1. Temperatures — bath, solid, hot-face + tap aim line
      const tempSeries = [
        { name: 'Bath',        data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.T_bath_C   != null ? +r.T_bath_C.toFixed(1) : null })) },
        { name: 'Solid Chg',  data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.T_solid_C  != null ? +r.T_solid_C.toFixed(1) : null })) },
        { name: 'Hot Face',   data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.T_hotface_C != null ? +r.T_hotface_C.toFixed(1) : null })) }
      ];
      renderTimeseriesChart('chart-traj-temp', tempSeries, 'Temperatures (°C)', '#f6ad55', true,
        [{ y: res.kpis?.tap_temp_aim || 1620, color: '#68d391', label: 'tap aim', dash: true }]);

      // 2. Inventories & dissolution — solid t, liquid t on left; undissolved kg on right
      const invLeft = [
        { name: 'Solid (t)',  data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.M_solid_t  != null ? +r.M_solid_t.toFixed(3) : null })) },
        { name: 'Liquid (t)', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.M_liquid_t != null ? +r.M_liquid_t.toFixed(3) : null })) }
      ];
      const invRight = { name: 'Undissolved (kg)', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: (r.undissolved_kg || r.m_undissolved_kg || 0) })) };
      Charts.renderDualAxisChart('chart-traj-inventories', invLeft[0], invRight, 'Mass (t)', 'Undissolved (kg)', '#c8855a', '#4fa8d8', 'Inventories & Dissolution', invLeft[1]);

      // 3. Bath composition (C, Si, Mn, S)
      const chemSeries = ['C','Si','Mn','S'].map(el => ({
        name: el, data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r[`pct_${el}`] != null ? +r[`pct_${el}`].toFixed(5) : null }))
      }));
      renderTimeseriesChart('chart-traj-chemistry', chemSeries, 'Bath Composition (wt%)', '#68d391', true);

      // 4. Slag chemistry & basicity — FeO % left, B2 right
      const slag1 = { name: 'FeO %',     data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.slag_FeO_pct != null ? +r.slag_FeO_pct.toFixed(2) : null })) };
      const slag2 = { name: 'B2 (CaO/SiO₂)', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.B2 != null ? +r.B2.toFixed(3) : null })) };
      Charts.renderDualAxisChart('chart-traj-slag', slag1, slag2, 'FeO (%)', 'Basicity B2', '#fc8181', '#4fa8d8', 'Slag Chemistry & Basicity');

      // 5. Heat-flow breakdown — wall loss, radiation, bath→scrap, chemistry
      const flowSeries = [
        { name: 'lining loss', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_wall_kW != null ? +r.Q_wall_kW.toFixed(0) : null })) },
        { name: 'radiation',   data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_rad_kW != null ? +r.Q_rad_kW.toFixed(0) : null })) },
        { name: 'chemical',    data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_chem_kW != null ? +r.Q_chem_kW.toFixed(0) : null })) },
      ];
      renderTimeseriesChart('chart-traj-heatflows', flowSeries, 'Heat-Flow Breakdown (kW)', '#9f7aea', true);

      // 6. Energy & SEC — energy kWh left, SEC kWh/t right with floor line
      const en1 = { name: 'Energy (kWh)', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.E_kWh != null ? +r.E_kWh.toFixed(1) : null })) };
      const en2 = { name: 'SEC (kWh/t)', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.SEC_kWh_t != null ? +r.SEC_kWh_t.toFixed(1) : null })) };
      Charts.renderDualAxisChart('chart-traj-energy', en1, en2, 'Energy (kWh)', 'SEC (kWh/t)', '#c8855a', '#f6ad55', 'Energy & Specific Consumption',
        null, res.kpis?.sec_floor || 381);
    }
  }

  function renderTimeseriesChart(elementId, series, title, defaultColor, multiSeries = false, yAnnotations = []) {
    const el = document.getElementById(elementId);
    if (!el || !series) return;
    el.innerHTML = '';

    const apexSeries = multiSeries && Array.isArray(series)
      ? series
      : [{ name: title, data: Array.isArray(series) ? series : [] }];

    // Build y-axis annotation lines
    const annotations = { yaxis: [] };
    if (yAnnotations && yAnnotations.length) {
      yAnnotations.forEach(a => {
        annotations.yaxis.push({
          y: a.y, borderColor: a.color || '#68d391',
          strokeDashArray: a.dash ? 6 : 0,
          label: a.label ? { text: a.label, style: { color: a.color || '#68d391', background: 'transparent', fontSize: '10px' } } : undefined
        });
      });
    }

    const chart = new ApexCharts(el, {
      chart: {
        type: 'line', height: '100%', background: 'transparent',
        foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif',
        toolbar: { show: false },
        animations: { enabled: false }
      },
      theme: { mode: 'dark' },
      series: apexSeries,
      stroke: { curve: 'smooth', width: 2 },
      colors: multiSeries ? undefined : [defaultColor],
      xaxis: { type: 'numeric', labels: { formatter: v => `${(+v).toFixed(0)}min` } },
      yaxis: { labels: { style: { fontSize: '10px' } } },
      annotations,
      title: { text: title, style: { fontSize: '12px', color: '#a0aec0' } },
      tooltip: { theme: 'dark', style: { fontSize: '11px', fontFamily: 'Inter,sans-serif' }, shared: true },
      grid: { borderColor: '#1e2c40' },
      legend: { show: multiSeries, position: 'top', fontSize: '10px' }
    });
    chart.render();
  }

  // ═══════════════════════════════════════════════════════════════
  //  PHYSICS & ENERGY TAB
  // ═══════════════════════════════════════════════════════════════
  async function fetchAndRenderPhysics() {
    const res = await callApi('/api/physics/run', {});
    if (!res) return;

    const k = res.kpis || {};
    setKPI('phys-kpi-ledger',  k.ledger_max,         v => `${v.toFixed(2)} %`);
    setKPI('phys-kpi-closure', k.first_law_closure,  v => `${(+v).toFixed(1)} %`);
    setKPI('phys-kpi-sec',     k.final_sec,           v => `${v.toFixed(0)} kWh/t`, `floor ${(k.sec_floor||381).toFixed(0)} kWh/t`, 'phys-kpi-sec-sub');
    setKPI('phys-kpi-useful',  k.useful_fraction,    v => `${v.toFixed(1)} %`);

    if (res.data) {
      const d = res.data;
      // Heat-flow breakdown — matches legacy Physics page exactly
      const flowSeries = [
        { name: 'useful (to metal)', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_useful_kW != null ? +r.Q_useful_kW.toFixed(0) : null })) },
        { name: 'lining loss',       data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_wall_kW  != null ? +r.Q_wall_kW.toFixed(0)  : null })) },
        { name: 'radiation',         data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_rad_kW   != null ? +r.Q_rad_kW.toFixed(0)   : null })) },
        { name: 'chemistry',         data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_chem_kW  != null ? +r.Q_chem_kW.toFixed(0)  : null })) },
        { name: 'off-gas',           data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.Q_offgas_kW != null ? +r.Q_offgas_kW.toFixed(0) : null })) },
      ];
      renderTimeseriesChart('chart-phys-heatflows', flowSeries, 'Heat-Flow Breakdown (kW)', '#9f7aea', true);

      // Cumulative energy vs useful melt energy
      renderTimeseriesChart('chart-phys-energy', [
        { name: 'Grid Input (kWh)', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.E_kWh != null ? +r.E_kWh.toFixed(1) : null })) }
      ], 'Cumulative Energy (kWh)', '#f6ad55', true);
      renderTimeseriesChart('chart-phys-rates', [
        { name: 'C', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.rate_C != null ? +r.rate_C.toFixed(5) : null })) },
        { name: 'Si', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.rate_Si != null ? +r.rate_Si.toFixed(5) : null })) },
        { name: 'Mn', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: r.rate_Mn != null ? +r.rate_Mn.toFixed(5) : null })) },
      ], 'Element Reaction Rates', '#68d391', true);
    }
    if (res.waterfall) renderWaterfallChart('chart-phys-waterfall', res.waterfall);
    if (res.energy_audit) {
      const tbody = document.getElementById('phys-audit-tbody');
      if (tbody) {
        const rows = res.energy_audit_table || Object.entries(res.energy_audit)
          .filter(([,v]) => typeof v === 'number').map(([k, v]) => ({ component: k, energy_kWh: v }));
        tbody.innerHTML = rows.map(r => `<tr><td>${r.component || r[0]}</td><td>${(+(r.energy_kWh || r[1] || 0)).toFixed(1)}</td></tr>`).join('');
      }
    }
  }

  function renderWaterfallChart(elementId, data) {
    const el = document.getElementById(elementId);
    if (!el || !data) return;
    el.innerHTML = '';
    const chart = new ApexCharts(el, {
      chart: { type: 'bar', height: '100%', background: 'transparent', foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif', toolbar: { show: false } },
      theme: { mode: 'dark' },
      series: [{ name: 'kWh/t', data: data.values }],
      xaxis: { categories: data.labels, labels: { style: { fontSize: '10px' } } },
      colors: ['#4fa8d8','#f6ad55','#fc8181','#9f7aea','#68d391'],
      plotOptions: { bar: { distributed: true, borderRadius: 4, columnWidth: '60%' } },
      title: { text: 'Energy Waterfall (kWh/t)', style: { fontSize: '12px', color: '#a0aec0' } },
      tooltip: { theme: 'dark' },
      grid: { borderColor: '#1e2c40' },
      legend: { show: false }
    });
    chart.render();
  }

  // ═══════════════════════════════════════════════════════════════
  //  VIRTUAL SENSOR / EKF TAB
  // ═══════════════════════════════════════════════════════════════
  function initEKFTab() {
    bindSlider('ekf-eta',  'ekf-eta-val',  v => v.toFixed(2));
    bindSlider('ekf-ua',   'ekf-ua-val',   v => v.toFixed(2));
    bindSlider('ekf-dips', 'ekf-dips-val', v => `${v}`);

    const runBtn = document.getElementById('ekf-btn-run');
    if (runBtn && !runBtn._bound) {
      runBtn._bound = true;
      runBtn.addEventListener('click', runEKF);
    }
    // Load default / cached result
    runEKF(true);
  }

  async function runEKF(cached = false) {
    const eta  = parseFloat(document.getElementById('ekf-eta')?.value  || 0.90);
    const ua   = parseFloat(document.getElementById('ekf-ua')?.value   || 1.35);
    const dips = parseInt(document.getElementById('ekf-dips')?.value   || 3, 10);

    const endpoint = cached ? '/api/ekf/default' : '/api/ekf/run';
    const body = cached ? {} : { true_eta: eta, true_ua: ua, ndips: dips };
    const res = await callApi(endpoint, body);
    if (!res) return;

    const k = res.kpis || {};
    setKPI('ekf-kpi-error', k.final_error,   v => `${v.toFixed(1)} °C`);
    setKPI('ekf-kpi-eta',   k.eta_converged, v => v.toFixed(3));
    setKPI('ekf-kpi-sigma', k.sigma_end,     v => `${v.toFixed(2)} °C`);
    setKPI('ekf-kpi-dips',  k.dips_count,    v => `${v}`);

    if (res.df) {
      const d = res.df;
      const tempSeries = [
        { name: '±2σ Band', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: [+(r.T_est_C||0) - 2*(r.sigma_T||0), +(r.T_est_C||0) + 2*(r.sigma_T||0)] })) },
        { name: 'EKF Est.', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: +(r.T_est_C||0).toFixed(2) })) },
        { name: 'True Temp', data: d.map(r => ({ x: +(r.t_min||0).toFixed(2), y: +(r.T_true_C||0).toFixed(2) })) },
        { name: 'Dips', data: d.filter(r => r.dip_measured).map(r => ({ x: +(r.t_min||0).toFixed(2), y: +(r.T_true_C||0).toFixed(2) })) }
      ];
      Charts.renderConfidenceBandChart('chart-ekf-temp', tempSeries, 'Bath temperature — truth vs EKF estimate');
      if (res.theta_path) {
        const tp = res.theta_path;
        const etaSeries = [
          { name: 'η̂ Electrical', data: tp.eta_electrical?.map((v,i) => ({ x: i, y: +v.toFixed(4) })) || [] },
          { name: 'UÂ Lining', data: tp.UA_lining_scale?.map((v,i) => ({ x: i, y: +v.toFixed(4) })) || [] }
        ];
        Charts.renderConfidenceBandChart('chart-ekf-params', etaSeries, 'Tracked parameters converging to truth');
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  MACHINE LEARNING TAB
  // ═══════════════════════════════════════════════════════════════
  function initMLTab() {
    bindSlider('ml-split',  'ml-split-val',  v => v.toFixed(2));
    bindSlider('ml-nheats', 'ml-nheats-val', v => `${v}`);

    const cachedBtn = document.getElementById('ml-btn-train-cached');
    const liveBtn   = document.getElementById('ml-btn-train-live');
    if (cachedBtn && !cachedBtn._bound) {
      cachedBtn._bound = true;
      cachedBtn.addEventListener('click', () => runMLTrain(true));
    }
    if (liveBtn && !liveBtn._bound) {
      liveBtn._bound = true;
      liveBtn.addEventListener('click', () => runMLTrain(false));
    }
    // Auto-load cached
    runMLTrain(true);
  }

  async function runMLTrain(useCached) {
    const trainFrac = parseFloat(document.getElementById('ml-split')?.value  || 0.70);
    const nHeats    = parseInt(document.getElementById('ml-nheats')?.value   || 40, 10);

    const res = await callApi('/api/ml/train', { split: trainFrac, n_heats: nHeats, live: !useCached });
    if (!res) return;

    const m = res.metrics || {};
    const pill = document.getElementById('ml-pill-status');
    if (pill) {
      const maturity = m.maturity || 'Insufficient';
      const tActive  = m.ml_T_active ? 'T-ML active' : 'T-ML gated off';
      const cActive  = m.ml_C_active ? 'C-ML active' : 'C-ML gated off';
      pill.innerHTML = `maturity: ${maturity} &middot; ${tActive} &middot; ${cActive} (${m.n_train || 0} train / ${m.n_test || 0} test)`;
      pill.className = `status-pill ${m.ml_T_active ? 'status-ok' : 'status-warn'}`;
    }

    setKPI('ml-kpi-t-hit', m.T_hit_15C,      v => `${v.toFixed(0)} %`, m.T_hit_15C_phys != null ? `phys ${m.T_hit_15C_phys.toFixed(0)}%` : 'phys —%', 'ml-kpi-t-hit-sub');
    setKPI('ml-kpi-t-mae', m.T_MAE_C,        v => `${v.toFixed(1)} °C`);
    setKPI('ml-kpi-c-hit', m.C_hit_002,      v => `${v.toFixed(0)} %`, m.C_hit_002_phys != null ? `phys ${m.C_hit_002_phys.toFixed(0)}%` : 'phys —%', 'ml-kpi-c-hit-sub');
    setKPI('ml-kpi-c-mae', m.C_MAE,          v => `${v.toFixed(3)} %`);

    if (res.pred_df && res.pred_df.length) {
      const scatter = [{ name: 'Physics', data: res.pred_df.map(r => [+(r.T_true_C||0).toFixed(1), +(r.T_phys_C||0).toFixed(1)]) },
                       { name: 'Hybrid',  data: res.pred_df.map(r => [+(r.T_true_C||0).toFixed(1), +(r.T_pred_C||0).toFixed(1)]) }];
      Charts.renderAnnotatedScatterChart('chart-ml-scatter', scatter, 'Temperature — predicted vs actual');

      const errTPhys = { name: 'Physics', data: res.pred_df.map(r => ({ x: Math.round(r.heat||0), y: +(r.T_phys_C - r.T_true_C).toFixed(1) })) };
      const errTPred = { name: 'Hybrid', data: res.pred_df.map(r => ({ x: Math.round(r.heat||0), y: +(r.T_pred_C - r.T_true_C).toFixed(1) })) };
      Charts.renderGroupedBarChart('chart-ml-error', [errTPhys, errTPred], 'Test-set temperature error');
    }
  }

  function renderScatterChart(elementId, data, title) {
    const el = document.getElementById(elementId);
    if (!el || !data) return;
    el.innerHTML = '';
    const chart = new ApexCharts(el, {
      chart: { type: 'scatter', height: '100%', background: 'transparent', foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif', toolbar: { show: false } },
      theme: { mode: 'dark' },
      series: data,
      xaxis: { title: { text: 'Actual (°C)', style: { fontSize: '10px', color: '#64748b' } } },
      yaxis: { title: { text: 'Predicted (°C)', style: { fontSize: '10px', color: '#64748b' } } },
      colors: ['#f6ad55', '#4fa8d8'],
      title: { text: title, style: { fontSize: '12px', color: '#a0aec0' } },
      tooltip: { theme: 'dark' },
      grid: { borderColor: '#1e2c40' },
      legend: { show: true }
    });
    chart.render();
  }

  // ═══════════════════════════════════════════════════════════════
  //  DRIFT MONITOR TAB
  // ═══════════════════════════════════════════════════════════════
  function initDriftTab() {
    bindSlider('drift-nheats', 'drift-nheats-val', v => `${v}`);
    bindSlider('drift-regime', 'drift-regime-val', v => `${v}`);

    const cachedBtn = document.getElementById('drift-btn-cached');
    const liveBtn   = document.getElementById('drift-btn-live');
    if (cachedBtn && !cachedBtn._bound) {
      cachedBtn._bound = true;
      cachedBtn.addEventListener('click', () => runDrift(true));
    }
    if (liveBtn && !liveBtn._bound) {
      liveBtn._bound = true;
      liveBtn.addEventListener('click', () => runDrift(false));
    }
    runDrift(true);
  }

  async function runDrift(useCached) {
    const nHeats      = parseInt(document.getElementById('drift-nheats')?.value || 50, 10);
    const regimeHeat  = parseInt(document.getElementById('drift-regime')?.value  || 40, 10);

    const res = await callApi('/api/drift/run', { n_heats: nHeats, reg: regimeHeat, live: !useCached });
    if (!res) return;

    const pill = document.getElementById('drift-alarm-pill');
    if (pill) {
      pill.innerHTML = res.alarm ? 'DRIFT ALARM &mdash; ' + (res.reasons || []).slice(0, 2).join(', ') : 'stable &mdash; no significant drift';
      pill.className = `status-pill ${res.alarm ? 'status-bad' : 'status-ok'}`;
    }

    setKPI('drift-kpi-psi',    res.psi_max,   v => v.toFixed(2));
    setKPI('drift-kpi-ref',    res.n_ref,     v => `${v}`);
    setKPI('drift-kpi-recent', res.n_recent,  v => `${v}`);

    if (res.psi_df) {
      const psiData = res.psi_df.map(r => ({ x: r.feature, y: +(r.psi||0).toFixed(4) }));
      Charts.renderHorizontalBarChart('chart-drift-psi', psiData, 'Population drift by feature (PSI)');
    }
    if (res.tracking) {
      const trackSeries = [
        { name: res.tracking.feature, data: res.tracking.x.map((x, i) => ({ x, y: +(res.tracking.y[i]||0).toFixed(4) })) }
      ];
      // add vertical annotation hack by rendering an extra point series
      renderTimeseriesChart('chart-drift-tracking', trackSeries, 'The variable that moved', '#4fa8d8', true);
    }
    const reasonsDiv = document.getElementById('drift-reasons-list');
    if (reasonsDiv) {
      reasonsDiv.innerHTML = (res.reasons || []).map(r => `<div>• ${r}</div>`).join('');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  CHARGE-MIX TAB
  // ═══════════════════════════════════════════════════════════════
  function initChargeMixTab() {
    bindSlider('mix-target', 'mix-target-val', v => `${v.toFixed(1)} t`);
    bindSlider('mix-clo',    'mix-clo-val',    v => `${v.toFixed(2)} %`);
    bindSlider('mix-chi',    'mix-chi-val',    v => `${v.toFixed(2)} %`);
    bindSlider('mix-cu',     'mix-cu-val',     v => `${v.toFixed(2)} %`);
    bindSlider('mix-sn',     'mix-sn-val',     v => `${v.toFixed(3)} %`);

    const solveBtn = document.getElementById('mix-btn-solve');
    if (solveBtn && !solveBtn._bound) {
      solveBtn._bound = true;
      solveBtn.addEventListener('click', solveMix);
    }

    // Load scrap library table
    loadScrapLibrary();
    // Auto-solve with defaults
    solveMix();
  }

  async function loadScrapLibrary() {
    const res = await callApi('/api/chargemix/library', {});
    const tbody = document.getElementById('mix-scrap-tbody');
    if (!tbody) return;
    const grades = res?.grades || defaultScrapGrades();
    tbody.innerHTML = grades.map(g => `
      <tr>
        <td>${g.name}</td>
        <td>${g.price_per_kg}</td>
        <td>${g.fe_pct}</td>
        <td>${g.cu_pct}</td>
        <td>${g.sn_pct}</td>
        <td>${g.c_pct}</td>
        <td><input type="number" class="control-input mix-manual-kg" data-grade="${g.name}" value="${g.default_kg || 0}" min="0" style="width:80px"></td>
      </tr>`).join('');
  }

  async function solveMix() {
    const targetT    = parseFloat(document.getElementById('mix-target')?.value || 12.0);
    const cLo        = parseFloat(document.getElementById('mix-clo')?.value    || 0.10);
    const cHi        = parseFloat(document.getElementById('mix-chi')?.value    || 0.40);
    const cuCeiling  = parseFloat(document.getElementById('mix-cu')?.value     || 0.20);
    const snCeiling  = parseFloat(document.getElementById('mix-sn')?.value     || 0.10);
    const mode       = document.querySelector('input[name="mix-mode"]:checked')?.value || 'optimise';

    // Collect manual kg if in manual mode
    const manualKg = {};
    document.querySelectorAll('.mix-manual-kg').forEach(inp => {
      manualKg[inp.dataset.grade] = parseFloat(inp.value || 0);
    });

    const res = await callApi('/api/chargemix/solve', {
      target: targetT, clo: cLo, chi: cHi,
      cu: cuCeiling, sn: snCeiling,
      mode, manual_weights: manualKg,
      mats: null  // use server default
    });

    if (!res) {
      // Fallback in-browser LP
      const mix = fallbackChargeMixLP(targetT, cLo, cHi, cuCeiling, snCeiling);
      renderMixResults(mix, targetT);
      return;
    }

    const pill = document.getElementById('mix-pill-status');
    if (pill) {
      pill.textContent = res.status_text || 'solved';
      pill.className = `status-pill ${res.feasible ? 'status-ok' : 'status-warn'}`;
    }

    setKPI('mix-kpi-cost',   res.cost_per_t,   v => `₹${v.toLocaleString('en-IN', {maximumFractionDigits:0})}`);
    setKPI('mix-kpi-energy', res.energy_kWh,   v => `${v.toFixed(0)} kWh`);
    const bath = res.predicted_bath || {};
    setKPI('mix-kpi-cu',     bath.Cu, v => `${v.toFixed(3)} %`, `ceiling ${cuCeiling}%`, 'mix-kpi-cu-sub');
    setKPI('mix-kpi-c',      bath.C,  v => `${v.toFixed(3)} %`, `limit: ${cLo}–${cHi}%`, 'mix-kpi-c-sub');

    // rows from server
    if (res.rows) {
      const recipe = res.rows.map(r => ({ material: r.name || r.material, mass_kg: r.kg || r.mass_kg || 0, share_pct: r.share_pct || 0 }));
      renderMixRecipe(recipe);
      renderMixChemistry(bath);
    }

    const shadowBox = document.getElementById('mix-shadow-box');
    if (shadowBox) shadowBox.textContent = res.shadow_price_note || '';
  }

  function renderMixRecipe(recipe) {
    const tbody = document.getElementById('mix-recipe-tbody');
    if (!tbody || !recipe) return;
    tbody.innerHTML = recipe.map(r => `
      <tr>
        <td>${r.material}</td>
        <td>${r.mass_kg.toFixed(1)}</td>
        <td>${r.share_pct.toFixed(1)}%</td>
      </tr>`).join('');
  }

  function renderMixChemistry(chem) {
    const tbody = document.getElementById('mix-chemistry-tbody');
    if (!tbody || !chem) return;
    tbody.innerHTML = Object.entries(chem).map(([el, val]) =>
      `<tr><td>${el}</td><td>${typeof val === 'number' ? val.toFixed(4) : val}</td></tr>`
    ).join('');
  }

  function renderMixResults(mix, totalT) {
    const totalKg = totalT * 1000;
    const totalCost = mix.reduce((s, m) => s + m.kg * m.price_per_kg, 0);
    setKPI('mix-kpi-cost',   totalCost / totalT,   v => `₹${v.toLocaleString('en-IN', {maximumFractionDigits:0})}`);
    const recipe = mix.map(m => ({ material: m.name, mass_kg: m.kg, share_pct: m.kg / totalKg * 100 }));
    renderMixRecipe(recipe);
  }

  function fallbackChargeMixLP(targetT, cLo, cHi, cuCeiling, snCeiling) {
    const grades = defaultScrapGrades();
    const kg = targetT * 1000;
    // Simple two-component: HMS + Pig Iron to hit mid of C range
    const targetC = (cLo + cHi) / 2;
    const hms = grades.find(g => g.name === 'HMS 1&2') || grades[0];
    const pig = grades.find(g => g.name === 'Pig Iron') || grades[1];
    const pigFrac = Math.max(0, Math.min(0.3, (targetC - hms.c_pct) / (pig.c_pct - hms.c_pct)));
    return [
      { ...hms, kg: kg * (1 - pigFrac - 0.05) },
      { ...pig, kg: kg * pigFrac },
      { ...grades.find(g => g.name === 'Returns') || grades[4], kg: kg * 0.05 }
    ];
  }

  function defaultScrapGrades() {
    return [
      { name:'HMS 1&2',    price_per_kg:28, fe_pct:98.0, cu_pct:0.15, sn_pct:0.010, c_pct:0.25, default_kg:0 },
      { name:'Pig Iron',   price_per_kg:42, fe_pct:94.5, cu_pct:0.02, sn_pct:0.005, c_pct:4.00, default_kg:0 },
      { name:'Shredded',   price_per_kg:26, fe_pct:97.5, cu_pct:0.20, sn_pct:0.020, c_pct:0.15, default_kg:0 },
      { name:'Cast Iron',  price_per_kg:35, fe_pct:93.0, cu_pct:0.05, sn_pct:0.010, c_pct:3.20, default_kg:0 },
      { name:'Returns',    price_per_kg:24, fe_pct:99.0, cu_pct:0.10, sn_pct:0.005, c_pct:0.18, default_kg:0 }
    ];
  }

  // ═══════════════════════════════════════════════════════════════
  //  ECONOMICS TAB
  // ═══════════════════════════════════════════════════════════════
  function initEconomicsTab() {
    bindSlider('eco-output',  'eco-output-val',  v => `${v.toLocaleString('en-IN')} t`);
    bindSlider('eco-saving',  'eco-saving-val',  v => `${v} kWh/t`);
    bindSlider('eco-licence', 'eco-licence-val', v => `${v} Lakh`);

    const computeBtn = document.getElementById('eco-btn-compute');
    if (computeBtn && !computeBtn._bound) {
      computeBtn._bound = true;
      computeBtn.addEventListener('click', computeEconomics);
    }
    computeEconomics();
  }

  async function computeEconomics() {
    const annualT   = parseFloat(document.getElementById('eco-output')?.value  || 40000);
    const savingKwh = parseFloat(document.getElementById('eco-saving')?.value  || 40);
    const licenceLk = parseFloat(document.getElementById('eco-licence')?.value || 20);
    const tariff    = state.settings.tariff_per_kwh;
    const ef        = state.settings.emission_factor;

    const res = await callApi('/api/economics/compute', { tpy: annualT, saving: savingKwh, price_lakh: licenceLk });

    // Fallback client-side calculation
    const r = res || computeEconomicsLocal(annualT, savingKwh, licenceLk, tariff, ef);

    // Server returns annual_saving_cr, payback_months, co2_avoided, headroom
    setKPI('eco-kpi-saving',   r.annual_saving_cr,  v => `₹${v.toFixed(2)} Cr`, `at ₹${r.tariff||7}/kWh`, 'eco-kpi-saving-sub');
    setKPI('eco-kpi-payback',  r.payback_months,    v => `${v.toFixed(1)} mo`);
    setKPI('eco-kpi-co2',      r.co2_avoided,       v => `${v.toLocaleString('en-IN',{maximumFractionDigits:0})}`, 'tCO₂ per year', 'eco-kpi-co2-sub');
    setKPI('eco-kpi-headroom', r.headroom,          v => `${v.toFixed(0)} kWh/t`, 'above floor', 'eco-kpi-headroom-sub');

    renderScenariosTable(annualT, tariff);
    renderDetailedTable(r);
  }

  function computeEconomicsLocal(annualT, savingKwhT, licenceLakh, tariff, ef) {
    const annualSavKwh = annualT * savingKwhT;
    const annualSavRs  = annualSavKwh * tariff;
    const annualSavCr  = annualSavRs / 1e7;
    const licenceRs    = licenceLakh * 1e5;
    const payback      = licenceRs / annualSavRs * 12;
    const co2          = annualSavKwh / 1000 * ef;
    const headroom     = (state.settings.baseline_sec || 600) - 381;
    return { annual_saving: annualSavCr, payback_months: payback, co2_abated_t: co2, headroom_kwh_t: headroom };
  }

  function renderScenariosTable(baseT, tariff) {
    const tbody = document.getElementById('eco-scenarios-tbody');
    if (!tbody) return;
    const outputs = [baseT * 0.5, baseT, baseT * 2, baseT * 4].map(t => Math.round(t));
    const savings = [30, 50, 80];
    tbody.innerHTML = outputs.map(t =>
      `<tr><td>${t.toLocaleString('en-IN')} t</td>${savings.map(s =>
        `<td>₹${(t * s * tariff / 1e7).toFixed(2)} Cr`
      ).join('')}</tr>`
    ).join('');
  }

  function renderDetailedTable(r) {
    const tbody = document.getElementById('eco-detailed-tbody');
    if (!tbody) return;
    let rows = [];
    if (r.detailed_rows) {
      rows = r.detailed_rows;
    } else {
      rows = [
        ['Annual Energy Saving', r.annual_saving != null ? `₹${r.annual_saving.toFixed(2)} Cr/yr` : '—'],
        ['Payback Period', r.payback_months != null ? `${r.payback_months.toFixed(1)} months` : '—'],
        ['CO₂ Abated', r.co2_abated_t != null ? `${r.co2_abated_t.toFixed(0)} tCO₂/yr` : '—'],
        ['Headroom Above Floor', r.headroom_kwh_t != null ? `${r.headroom_kwh_t.toFixed(0)} kWh/t` : '—']
      ];
    }
    tbody.innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
  }

  // ═══════════════════════════════════════════════════════════════
  //  HEAT LOG TAB
  // ═══════════════════════════════════════════════════════════════
  function appendHeatLogRow(heat) {
    const tbody = document.getElementById('log-heats-tbody');
    if (!tbody) return;
    // Remove placeholder
    const placeholder = tbody.querySelector('.table-placeholder');
    if (placeholder) placeholder.parentElement.remove();

    const row = tbody.insertRow(0);
    const cells = [
      heat.id, heat.timestamp, heat.tap_temp, heat.delta_t,
      heat.carbon, heat.delta_c, heat.silicon, heat.manganese,
      heat.sulphur, heat.tap_mass, heat.sec, heat.avg_power,
      heat.adds_kg, heat.dips, heat.duration, heat.result
    ];
    cells.forEach(v => {
      const td = row.insertCell(-1);
      td.textContent = v;
      if (v === 'PASS') td.style.color = '#68d391';
      if (v === 'REVIEW') td.style.color = '#f6ad55';
    });
    while (tbody.rows.length > 30) tbody.deleteRow(tbody.rows.length - 1);
  }

  function logEvent(type, detail) {
    const tbody = document.getElementById('log-events-tbody');
    if (!tbody) return;
    const placeholder = tbody.querySelector('.table-placeholder');
    if (placeholder) placeholder.parentElement.remove();

    const row = tbody.insertRow(0);
    [state.heatCounter, new Date().toLocaleTimeString(), type, detail].forEach(v => {
      const td = row.insertCell(-1);
      td.textContent = v;
    });

    // Also append to op-log-box
    const logBox = document.getElementById('op-log-box');
    if (logBox) {
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.textContent = `[${new Date().toLocaleTimeString()}] ${type}: ${detail}`;
      logBox.prepend(entry);
      while (logBox.children.length > 30) logBox.removeChild(logBox.lastChild);
    }
  }

  function bindHeatLogExport() {
    const btn = document.getElementById('log-btn-export');
    if (btn) btn.addEventListener('click', exportHeatLogCSV);
    const clearBtn = document.getElementById('log-btn-clear');
    if (clearBtn) clearBtn.addEventListener('click', () => {
      state.heatLog = [];
      const tbody = document.getElementById('log-heats-tbody');
      if (tbody) tbody.innerHTML = '<tr><td colspan="16" class="table-placeholder">No tapped heats recorded this session...</td></tr>';
    });
  }

  function exportHeatLogCSV() {
    if (!state.heatLog.length) { showToast('No heats to export.', 'warn'); return; }
    const headers = ['Heat No','Time','Tap Temp (°C)','ΔT','C %','ΔC','Si %','Mn %','S %','Tap Mass (t)','SEC (kWh/t)','Avg Power (kW)','Adds (kg)','Dips','Duration (min)','Result'];
    const rows = state.heatLog.map(h =>
      [h.id,h.timestamp,h.tap_temp,h.delta_t,h.carbon,h.delta_c,h.silicon,h.manganese,h.sulphur,h.tap_mass,h.sec,h.avg_power,h.adds_kg,h.dips,h.duration,h.result]
    );
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `SmartMelt_HeatLog_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    showToast('Heat log exported as CSV', 'success');
  }

  // ═══════════════════════════════════════════════════════════════
  //  SETTINGS TAB
  // ═══════════════════════════════════════════════════════════════
  function populateSettingsForm() {
    const map = {
      'set-tap-aim':  'tap_aim_C',
      'set-c-lo':     'c_lo',
      'set-c-hi':     'c_hi',
      'set-power':    'power_kW',
      'set-tariff':   'tariff_per_kwh',
      'set-ef':       'emission_factor',
      'set-baseline': 'baseline_sec'
    };
    Object.entries(map).forEach(([inputId, key]) => {
      const el = document.getElementById(inputId);
      if (el) el.value = state.settings[key];
    });
    updateSettingsSummaryTable();
  }

  function updateSettingsSummaryTable() {
    const tbody = document.getElementById('settings-summary-tbody');
    if (!tbody) return;
    const labels = {
      tap_aim_C: 'Tap Temperature Aim (°C)',
      c_lo: 'Min Carbon Aim (%)',
      c_hi: 'Max Carbon Aim (%)',
      power_kW: 'Rated Power (kW)',
      tariff_per_kwh: 'Electricity Tariff (₹/kWh)',
      emission_factor: 'Grid Emission Factor (tCO₂/MWh)',
      baseline_sec: 'Baseline SEC (kWh/t)'
    };
    tbody.innerHTML = Object.entries(labels).map(([k, label]) =>
      `<tr><td>${label}</td><td>${state.settings[k]}</td></tr>`
    ).join('');
  }

  function bindSettingsForm() {
    const form = document.getElementById('settings-form');
    if (!form) return;
    form.addEventListener('submit', async e => {
      e.preventDefault();
      const map = {
        'set-tap-aim':  'tap_aim_C',
        'set-c-lo':     'c_lo',
        'set-c-hi':     'c_hi',
        'set-power':    'power_kW',
        'set-tariff':   'tariff_per_kwh',
        'set-ef':       'emission_factor',
        'set-baseline': 'baseline_sec'
      };
      Object.entries(map).forEach(([inputId, key]) => {
        const el = document.getElementById(inputId);
        if (el) state.settings[key] = parseFloat(el.value) || el.value;
      });
      await callApi('/api/settings/apply', {
      tap: state.settings.tap_aim_C,
      clo: state.settings.c_lo,
      chi: state.settings.c_hi,
      rated: state.settings.power_kW,
      tariff: state.settings.tariff_per_kwh,
      ef: state.settings.emission_factor,
      baseline: state.settings.baseline_sec
    });
      updateSettingsSummaryTable();
      showToast('Settings applied', 'success');
    });
  }

  // ═══════════════════════════════════════════════════════════════
  //  VALIDATION TAB
  // ═══════════════════════════════════════════════════════════════
  async function fetchAndRenderValidation() {
    const rerunBtn = document.getElementById('val-btn-rerun');
    if (rerunBtn && !rerunBtn._bound) {
      rerunBtn._bound = true;
      rerunBtn.addEventListener('click', () => runValidationAudit());
    }
    await loadAuditTable();
    await runValidationAudit();
  }

  async function loadAuditTable() {
    const res = await callApi('/api/validation/run', { snapshot: state.snapshot });
    const tbody = document.getElementById('val-audit-tbody');
    if (!tbody) return;
    const rows = res?.audit_rows || defaultAuditRows();
    tbody.innerHTML = rows.map(r =>
      `<tr><td>${r.quantity}</td><td>${r.in_model}</td><td>${r.literature}</td><td>${r.source}</td></tr>`
    ).join('');
  }

  function defaultAuditRows() {
    return [
      { quantity:'Latent heat Fe', in_model:'247 kJ/kg', literature:'247 kJ/kg', source:'CRC 104th ed.' },
      { quantity:'Cp liquid Fe',   in_model:'0.824 kJ/kg·K', literature:'~0.82 kJ/kg·K', source:'Iida & Guthrie 1988' },
      { quantity:'SEC floor',      in_model:'381 kWh/t', literature:'381 kWh/t', source:'First-Law calc.' },
      { quantity:'Grid EF',        in_model:'0.712 tCO₂/MWh', literature:'0.712 tCO₂/MWh', source:'CEA DB v21.0' }
    ];
  }

  async function runValidationAudit() {
    const res = await callApi('/api/validation/run', { snapshot: state.snapshot });

    const pills = document.getElementById('val-pills-container');
    if (!pills) return;

    const r = res || {
      element_ledger_pct: null,
      first_law_pct: null,
      endpoint_C: null,
      undissolved_kg: null
    };

    pills.innerHTML = `
      <span class="status-pill ${(r.element_ledger_pct || 100) > 99 ? 'status-ok' : 'status-warn'}">
        element ledger ${r.element_ledger_pct?.toFixed(2) ?? '--'} %
      </span>
      <span class="status-pill ${(r.first_law_pct || 100) >= 95 ? 'status-ok' : 'status-warn'}">
        first-law ${r.first_law_pct?.toFixed(2) ?? '--'} %
      </span>
      <span class="status-pill ${r.endpoint_C != null && Math.abs(r.endpoint_C - state.settings.tap_aim_C) <= 15 ? 'status-ok' : 'status-warn'}">
        endpoint ${r.endpoint_C?.toFixed(0) ?? '--'} °C
      </span>
      <span class="status-pill ${(r.undissolved_kg || 0) < 5 ? 'status-ok' : 'status-warn'}">
        undissolved ${r.undissolved_kg?.toFixed(1) ?? '--'} kg
      </span>`;

    if (res?.closure_series) renderTimeseriesChart('chart-val-closure', res.closure_series, 'First-Law Closure (%)', '#68d391');
    if (res?.ledger_df) {
      const elements = res.ledger_df.map(r => ({ x: r.element, y: r.closure_pct }));
      Charts.renderHorizontalBarChart('chart-val-elements', elements, 'Per-Element Mass Closure (%)');
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  API HELPER
  // ═══════════════════════════════════════════════════════════════
  async function callApi(endpoint, body) {
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {})
      });
      if (!res.ok) { console.warn(`${endpoint} → HTTP ${res.status}`); return null; }
      return res.json();
    } catch (e) {
      // Silently fail if server not up — UI still works via fallbacks
      console.debug(`${endpoint} offline: ${e.message}`);
      return null;
    }
  }

  // ═══════════════════════════════════════════════════════════════
  //  UI UTILITIES
  // ═══════════════════════════════════════════════════════════════
  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function setKPI(id, value, formatter, subText, subId) {
    const el = document.getElementById(id);
    if (el) el.textContent = value != null ? formatter(value) : '—';
    if (subId && subText) setText(subId, subText);
  }

  function formatTime(sec) {
    if (sec == null) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2,'0')}:${String(s).padStart(2,'0')}`;
  }

  function setStatusPill(cls, text) {
    const el = document.getElementById('op-status-pill');
    if (el) {
      el.className = `status-pill status-${cls}`;
      el.textContent = text;
    }
  }

  function setButtonsForRunning(running) {
    const startBtn = document.getElementById('op-btn-start');
    const tapBtn   = document.getElementById('op-btn-tap');
    const addBtn   = document.getElementById('op-btn-add-material');
    if (startBtn) startBtn.disabled = running;
    if (tapBtn)   { if (!running) tapBtn.setAttribute('disabled',''); }
    if (addBtn)   { if (running) addBtn.removeAttribute('disabled'); else addBtn.setAttribute('disabled',''); }
  }

  function showCalcBanner(visible, title = '') {
    const banner = document.getElementById('op-calc-banner');
    if (!banner) return;
    banner.style.display = visible ? 'flex' : 'none';
    if (title) setText('op-calc-title', title);
  }

  // Session clock in header
  let sessionSeconds = 0;
  function startSessionClock() {
    setInterval(() => {
      sessionSeconds++;
      // nothing needs updating; op-clock is driven by sim time
    }, 1000);
  }

  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span><button class="toast-close" onclick="this.parentElement.remove()">×</button>`;
    container.appendChild(toast);
    setTimeout(() => { if (toast.parentElement) toast.remove(); }, 4500);
  }

  // Language toggle
  function initLanguageToggle() {
    document.getElementById('btn-lang-en')?.addEventListener('click', () => setLanguage('en'));
    document.getElementById('btn-lang-hi')?.addEventListener('click', () => setLanguage('hi'));
  }

  function setLanguage(lang) {
    document.querySelectorAll('.tab-en').forEach(el => el.style.display = lang === 'en' ? '' : 'none');
    document.querySelectorAll('.tab-hi').forEach(el => el.style.display = lang === 'hi' ? '' : 'none');
    document.getElementById('btn-lang-en')?.classList.toggle('active', lang === 'en');
    document.getElementById('btn-lang-hi')?.classList.toggle('active', lang === 'hi');
  }

  // Plant selector populate
  async function initPlantSelector() {
    const sel = document.getElementById('plant-select');
    if (!sel) return;
    const res = await callApi('/api/configs', {});
    const configs = res?.configs || ['if_msme_12t'];
    const plants = configs.map(id => ({ id, name: id.replace(/_/g,' ').toUpperCase() }));
    sel.innerHTML = plants.map(p => `<option value="${p.id}">${p.name}</option>`).join('');
    sel.addEventListener('change', async () => {
      await callApi('/api/change_plant', { plant: sel.value });
      const hchip = document.getElementById('hchip-plant');
      if (hchip) hchip.innerHTML = `Plant: <b>${sel.options[sel.selectedIndex].text}</b>`;
      showToast(`Switched to ${sel.options[sel.selectedIndex].text}`, 'info');
    });
    // Update header chip
    const hchip = document.getElementById('hchip-plant');
    if (hchip && plants.length > 0) hchip.innerHTML = `Plant: <b>${plants[0].name}</b>`;
  }

  // ═══════════════════════════════════════════════════════════════
  //  KEYBOARD SHORTCUTS
  // ═══════════════════════════════════════════════════════════════
  function initKeyboard() {
    document.addEventListener('keydown', e => {
      if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); if (!state.simRunning) onStartSim(); }
      if (e.ctrlKey && e.key === 'r')     { e.preventDefault(); resetSim(); }
    });
  }

  // ═══════════════════════════════════════════════════════════════
  //  BOOT
  // ═══════════════════════════════════════════════════════════════
  async function init() {
    initTabs();
    initOperatorSliders();
    bindOperatorControls();
    bindSettingsForm();
    bindHeatLogExport();
    initKeyboard();
    initLanguageToggle();
    await initPlantSelector();

    // Init Three.js
    if (typeof ThreeFurnace !== 'undefined') ThreeFurnace.init('furnace-3d-container');

    // Welcome advisory
    renderAdvisories([{
      level: 'info',
      title: 'SmartMelt Studio Ready',
      message: 'Configure charge parameters and press ▶ START HEAT to begin melt simulation.'
    }]);

    // Load initial settings from server
    const settingsRes = await callApi('/api/settings/get', {});
    if (settingsRes?.editable) {
      const e = settingsRes.editable;
      state.settings.tap_aim_C       = e.tap_temperature_C  || state.settings.tap_aim_C;
      state.settings.c_lo            = e.aim_C_lo_pct        || state.settings.c_lo;
      state.settings.c_hi            = e.aim_C_hi_pct        || state.settings.c_hi;
      state.settings.power_kW        = e.rated_power_kW      || state.settings.power_kW;
      state.settings.tariff_per_kwh  = e.tariff_INR_per_kWh  || state.settings.tariff_per_kwh;
      state.settings.emission_factor = e.grid_EF_tCO2_per_MWh || state.settings.emission_factor;
      state.settings.baseline_sec    = e.baseline_SEC_kWh_per_t || state.settings.baseline_sec;
    }

    startSessionClock();
    console.log('[SmartMelt] Init complete ✓');
  }

  // Defer until DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
