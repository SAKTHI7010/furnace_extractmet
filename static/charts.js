/**
 * SmartMelt Studio — Client-Side Charts (ApexCharts)
 * All live-updating charts rendered without a backend call.
 * Charts are created once, then updated via the public API.
 */
(function () {
  'use strict';

  const CHART_THEME = {
    background: 'transparent',
    foreColor: '#a0aec0',
    fontFamily: 'Inter, sans-serif',
    fontSize: '11px',
    stroke: { curve: 'smooth', width: 2 },
    grid: {
      borderColor: '#1e2c40',
      xaxis: { lines: { show: false } },
      yaxis: { lines: { show: true } }
    }
  };

  function darkTooltip() {
    return {
      theme: 'dark',
      style: { fontSize: '11px', fontFamily: 'Inter, sans-serif' }
    };
  }

  function baseLineOptions(yLabel, color = '#4fa8d8') {
    return {
      chart: {
        type: 'line',
        height: '100%',
        background: CHART_THEME.background,
        foreColor: CHART_THEME.foreColor,
        fontFamily: CHART_THEME.fontFamily,
        toolbar: { show: false },
        animations: { enabled: true, easing: 'linear', dynamicAnimation: { speed: 700 } },
        sparkline: { enabled: false }
      },
      theme: { mode: 'dark' },
      stroke: { curve: 'smooth', width: 2.5 },
      colors: [color],
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid,
      xaxis: {
        type: 'numeric',
        labels: { show: true, formatter: v => `${v}s` },
        axisBorder: { show: false }
      },
      yaxis: {
        labels: { show: true, style: { fontSize: '10px' } },
        title: { text: yLabel, style: { fontSize: '10px', color: '#64748b' } }
      },
      legend: { show: false }
    };
  }

  // ----- Chart instances -----
  let tempChart = null;
  let powerChart = null;
  let meltProgressChart = null;
  let driftChart = null;
  let efficiencyChart = null;
  let energyGauge = null;

  // Rolling data buffers (max 60 points)
  const MAX_PTS = 60;
  const dataBufs = {
    temp: [],
    setpoint: [],
    power: [],
    meltPct: [],
    drift: [],
    efficiency: []
  };
  let tick = 0;

  function appendPoint(buf, value) {
    buf.push({ x: tick, y: Math.round(value * 10) / 10 });
    if (buf.length > MAX_PTS) buf.shift();
  }

  // ── Temperature trend chart
  function initTempChart(elementId) {
    const options = {
      ...baseLineOptions('Temperature (°C)', '#f6ad55'),
      series: [
        { name: 'Bath Temp', data: [] },
        { name: 'Setpoint', data: [] }
      ],
      colors: ['#f6ad55', '#4fa8d8'],
      stroke: { curve: 'smooth', width: [2.5, 1.5], dashArray: [0, 5] },
      annotations: {
        yaxis: [{ y: 1620, borderColor: '#fc8181', label: { text: 'Tap Aim', style: { color: '#fc8181', background: 'transparent' } } }]
      }
    };
    tempChart = new ApexCharts(document.getElementById(elementId), options);
    tempChart.render();
  }

  // ── Power chart
  function initPowerChart(elementId) {
    const options = {
      ...baseLineOptions('Power (kW)', '#68d391'),
      series: [{ name: 'Power (kW)', data: [] }],
      chart: {
        ...baseLineOptions('Power (kW)', '#68d391').chart,
        type: 'area'
      },
      fill: {
        type: 'gradient',
        gradient: {
          shadeIntensity: 1,
          opacityFrom: 0.4,
          opacityTo: 0.0,
          stops: [0, 100]
        }
      }
    };
    powerChart = new ApexCharts(document.getElementById(elementId), options);
    powerChart.render();
  }

  // ── Melt progress chart (area)
  function initMeltChart(elementId) {
    const options = {
      ...baseLineOptions('Melted (%)', '#9f7aea'),
      series: [{ name: 'Melted %', data: [] }],
      chart: {
        ...baseLineOptions('Melted (%)', '#9f7aea').chart,
        type: 'area'
      },
      fill: {
        type: 'gradient',
        gradient: {
          shadeIntensity: 1,
          opacityFrom: 0.35,
          opacityTo: 0.0,
          stops: [0, 100]
        }
      },
      yaxis: { ...baseLineOptions('Melted (%)', '#9f7aea').yaxis, min: 0, max: 100 }
    };
    meltProgressChart = new ApexCharts(document.getElementById(elementId), options);
    meltProgressChart.render();
  }

  // ── Drift chart
  function initDriftChart(elementId) {
    const options = {
      ...baseLineOptions('Drift (°C)', '#fc8181'),
      series: [
        { name: 'Drift', data: [] }
      ],
      annotations: {
        yaxis: [
          { y: 5, y2: -5, fillColor: '#68d391', opacity: 0.08 }
        ]
      }
    };
    driftChart = new ApexCharts(document.getElementById(elementId), options);
    driftChart.render();
  }

  // ── Efficiency donut (radial)
  function initEfficiencyGauge(elementId) {
    const options = {
      chart: {
        type: 'radialBar',
        height: '100%',
        background: 'transparent',
        foreColor: '#a0aec0',
        fontFamily: 'Inter, sans-serif',
        toolbar: { show: false }
      },
      theme: { mode: 'dark' },
      series: [0],
      plotOptions: {
        radialBar: {
          startAngle: -120,
          endAngle: 120,
          hollow: { margin: 0, size: '65%' },
          track: { background: '#1e2c40', strokeWidth: '100%' },
          dataLabels: {
            name: { show: true, color: '#64748b', fontSize: '11px' },
            value: {
              show: true,
              color: '#f6ad55',
              fontSize: '22px',
              fontWeight: 700,
              formatter: v => `${v}%`
            }
          }
        }
      },
      labels: ['Efficiency'],
      colors: ['#f6ad55'],
      stroke: { lineCap: 'round' }
    };
    efficiencyGauge = new ApexCharts(document.getElementById(elementId), options);
    efficiencyGauge.render();
  }

  // ── Energy breakdown bar chart
  function initEnergyChart(elementId) {
    const options = {
      chart: {
        type: 'bar',
        height: '100%',
        background: 'transparent',
        foreColor: '#a0aec0',
        fontFamily: 'Inter, sans-serif',
        toolbar: { show: false },
        animations: { enabled: true, easing: 'easeinout', speed: 600 }
      },
      theme: { mode: 'dark' },
      series: [{ name: 'kWh/t', data: [0, 0, 0, 0] }],
      xaxis: {
        categories: ['Melt', 'Superheat', 'Loss Wall', 'Loss Top'],
        labels: { style: { fontSize: '10px' } }
      },
      yaxis: { labels: { show: false } },
      colors: ['#4fa8d8', '#f6ad55', '#fc8181', '#9f7aea'],
      plotOptions: {
        bar: {
          distributed: true,
          horizontal: false,
          columnWidth: '60%',
          borderRadius: 4
        }
      },
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid,
      legend: { show: false }
    };
    energyGauge = new ApexCharts(document.getElementById(elementId), options);
    energyGauge.render();
  }

  // ── Heat Log table helpers
  function buildHeatLogRow(heat) {
    const tbody = document.querySelector('#heat-log-table tbody');
    if (!tbody) return;
    const row = tbody.insertRow(0);
    row.className = 'heat-log-row';
    const cols = [heat.id, heat.timestamp, heat.charge, heat.tap_temp, heat.duration, heat.energy, heat.yield_pct];
    cols.forEach(v => {
      const td = row.insertCell(-1);
      td.textContent = v;
    });
    // Keep max 25 rows
    while (tbody.rows.length > 25) tbody.deleteRow(tbody.rows.length - 1);
  }

  // ── Charge-Mix Gantt / bar
  function renderChargeMixBar(mix) {
    const el = document.getElementById('charge-mix-bar');
    if (!el) return;
    el.innerHTML = '';
    const total = mix.reduce((s, m) => s + m.kg, 0);
    mix.forEach(m => {
      const w = (m.kg / total * 100).toFixed(1);
      const bar = document.createElement('div');
      bar.className = 'mix-bar-segment';
      bar.style.width = `${w}%`;
      bar.style.background = m.color;
      bar.title = `${m.name}: ${m.kg.toFixed(1)} kg`;
      bar.innerHTML = `<span class="mix-label">${m.name} ${w}%</span>`;
      el.appendChild(bar);
    });
  }

  // ── Pareto chart for ML feature importance
  function renderParetoChart(elementId, labels, values) {
    const existing = document.getElementById(elementId);
    if (!existing) return;
    const sorted = labels.map((l, i) => ({ l, v: values[i] }))
      .sort((a, b) => b.v - a.v);
    const options = {
      chart: {
        type: 'bar',
        height: '100%',
        background: 'transparent',
        foreColor: '#a0aec0',
        fontFamily: 'Inter, sans-serif',
        toolbar: { show: false }
      },
      theme: { mode: 'dark' },
      series: [{ name: 'Importance', data: sorted.map(x => +(x.v * 100).toFixed(2)) }],
      xaxis: { categories: sorted.map(x => x.l), labels: { style: { fontSize: '10px' } } },
      yaxis: { labels: { formatter: v => `${v}%` } },
      colors: ['#4fa8d8'],
      plotOptions: { bar: { borderRadius: 4, columnWidth: '55%' } },
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid
    };
    // Destroy previous instance if rerendering
    if (window._paretoChart) { window._paretoChart.destroy(); }
    window._paretoChart = new ApexCharts(existing, options);
    window._paretoChart.render();
  }

  // ── Economics waterfall
  function renderEconomicsChart(elementId, data) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const options = {
      chart: {
        type: 'bar',
        height: '100%',
        background: 'transparent',
        foreColor: '#a0aec0',
        fontFamily: 'Inter, sans-serif',
        toolbar: { show: false }
      },
      theme: { mode: 'dark' },
      series: [{ name: '₹/heat', data: data.values }],
      xaxis: { categories: data.labels, labels: { style: { fontSize: '10px' } } },
      colors: data.colors || ['#68d391'],
      plotOptions: {
        bar: {
          borderRadius: 4,
          columnWidth: '60%',
          distributed: true
        }
      },
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid,
      legend: { show: false }
    };
    if (window._econChart) { window._econChart.destroy(); }
    window._econChart = new ApexCharts(el, options);
    window._econChart.render();
  }

  // ── Main update function called by app.js every poll cycle
  function updateLiveCharts(snap) {
    tick = snap.t_sec || tick + 1;

    if (tempChart) {
      appendPoint(dataBufs.temp, snap.bath_temp_C);
      appendPoint(dataBufs.setpoint, snap.setpoint_C || snap.bath_temp_C);
      tempChart.updateSeries([
        { data: [...dataBufs.temp] },
        { data: [...dataBufs.setpoint] }
      ]);
    }

    if (powerChart) {
      appendPoint(dataBufs.power, snap.power_kW);
      powerChart.updateSeries([{ data: [...dataBufs.power] }]);
    }

    if (meltProgressChart) {
      appendPoint(dataBufs.meltPct, snap.melted_pct);
      meltProgressChart.updateSeries([{ data: [...dataBufs.meltPct] }]);
    }

    if (driftChart && typeof snap.drift_C !== 'undefined') {
      appendPoint(dataBufs.drift, snap.drift_C);
      driftChart.updateSeries([{ data: [...dataBufs.drift] }]);
    }

    if (efficiencyGauge && typeof snap.efficiency_pct !== 'undefined') {
      efficiencyGauge.updateSeries([Math.round(snap.efficiency_pct)]);
    }

    if (energyGauge && snap.energy_breakdown) {
      const b = snap.energy_breakdown;
      energyGauge.updateSeries([{
        data: [
          +(b.melt || 0).toFixed(1),
          +(b.superheat || 0).toFixed(1),
          +(b.loss_wall || 0).toFixed(1),
          +(b.loss_top || 0).toFixed(1)
        ]
      }]);
    }
  }

  function clearHistories() {
    Object.keys(dataBufs).forEach(k => { dataBufs[k] = []; });
    tick = 0;
  }

  // ── New Specialized Charts
  function renderDualAxisChart(elementId, series1, series2, title1, title2, color1, color2, chartTitle, extraSeries, floorValue) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const seriesList = extraSeries ? [series1, extraSeries, series2] : [series1, series2];
    const colors = extraSeries ? [color1, '#9f7aea', color2] : [color1, color2];

    const yaxes = [
      { seriesName: series1.name, title: { text: title1, style: { fontSize: '10px', color: color1 } }, labels: { style: { fontSize: '10px' } } },
      ...(extraSeries ? [{ seriesName: extraSeries.name, title: { text: title1, style: { fontSize: '10px' } }, labels: { style: { fontSize: '10px' } }, show: false }] : []),
      { opposite: true, seriesName: series2.name, title: { text: title2, style: { fontSize: '10px', color: color2 } }, labels: { style: { fontSize: '10px' } } }
    ];

    const annotations = {};
    if (floorValue != null) {
      annotations.yaxis = [{
        y: floorValue,
        yAxisIndex: yaxes.length - 1,
        borderColor: '#68d391',
        strokeDashArray: 6,
        label: { text: `floor ${floorValue}`, style: { color: '#68d391', background: 'transparent', fontSize: '10px' } }
      }];
    }

    const options = {
      chart: { type: 'line', height: '100%', background: 'transparent', foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif', toolbar: { show: false }, animations: { enabled: false } },
      theme: { mode: 'dark' },
      series: seriesList,
      colors,
      stroke: { curve: 'smooth', width: 2 },
      xaxis: { type: 'numeric', labels: { formatter: v => `${(+v).toFixed(0)}min` } },
      yaxis: yaxes,
      annotations,
      title: { text: chartTitle, style: { fontSize: '12px', color: '#a0aec0' } },
      tooltip: { ...darkTooltip(), shared: true },
      grid: CHART_THEME.grid,
      legend: { show: true, position: 'top', fontSize: '10px' }
    };
    if (el._chart) el._chart.destroy();
    el._chart = new ApexCharts(el, options);
    el._chart.render();
  }

  function renderHorizontalBarChart(elementId, data, title) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const options = {
      chart: { type: 'bar', height: '100%', background: 'transparent', foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif', toolbar: { show: false } },
      theme: { mode: 'dark' },
      series: [{ name: title, data: data }],
      plotOptions: { bar: { horizontal: true, borderRadius: 4, colors: {
        ranges: [{from: -100, to: 0.25, color: '#68d391'}, {from: 0.25, to: 100, color: '#fc8181'}]
      } } },
      title: { text: title, style: { fontSize: '12px', color: '#a0aec0' } },
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid
    };
    if (el._chart) el._chart.destroy();
    el._chart = new ApexCharts(el, options);
    el._chart.render();
  }

  function renderGroupedBarChart(elementId, data, title) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const options = {
      chart: { type: 'bar', height: '100%', background: 'transparent', foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif', toolbar: { show: false } },
      theme: { mode: 'dark' },
      series: data,
      plotOptions: { bar: { horizontal: false, borderRadius: 4, columnWidth: '50%', dataLabels: { position: 'top' } } },
      colors: ['#4fa8d8', '#f6ad55', '#fc8181', '#68d391', '#9f7aea'],
      title: { text: title, style: { fontSize: '12px', color: '#a0aec0' } },
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid,
      legend: { show: true }
    };
    if (el._chart) el._chart.destroy();
    el._chart = new ApexCharts(el, options);
    el._chart.render();
  }

  function renderAnnotatedScatterChart(elementId, data, title) {
    const el = document.getElementById(elementId);
    if (!el) return;
    // Add ideal parity line
    let min = 9999, max = -9999;
    data.forEach(s => {
      s.data.forEach(p => {
        if (p[0] < min) min = p[0]; if (p[0] > max) max = p[0];
        if (p[1] < min) min = p[1]; if (p[1] > max) max = p[1];
      });
    });
    const parityLine = { name: 'Ideal', type: 'line', data: [[min, min], [max, max]] };
    
    const options = {
      chart: { type: 'scatter', height: '100%', background: 'transparent', foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif', toolbar: { show: false } },
      theme: { mode: 'dark' },
      series: [...data.map(d => ({...d, type: 'scatter'})), parityLine],
      stroke: { width: [0, 0, 2], dashArray: [0, 0, 5] },
      colors: ['#f6ad55', '#4fa8d8', '#a0aec0'],
      title: { text: title, style: { fontSize: '12px', color: '#a0aec0' } },
      xaxis: { title: { text: 'Actual', style: { fontSize: '10px' } }, type: 'numeric' },
      yaxis: { title: { text: 'Predicted', style: { fontSize: '10px' } } },
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid,
      legend: { show: true }
    };
    if (el._chart) el._chart.destroy();
    el._chart = new ApexCharts(el, options);
    el._chart.render();
  }

  function renderConfidenceBandChart(elementId, data, title) {
    const el = document.getElementById(elementId);
    if (!el) return;
    const options = {
      chart: { type: 'line', height: '100%', background: 'transparent', foreColor: '#a0aec0', fontFamily: 'Inter,sans-serif', toolbar: { show: false } },
      theme: { mode: 'dark' },
      series: data,
      stroke: { curve: 'smooth', width: [0, 2, 0, 2], dashArray: [0, 0, 0, 5] },
      colors: ['#2b6cb0', '#f6ad55', '#fc8181', '#68d391'],
      fill: { type: ['solid', 'solid', 'solid', 'solid'], opacity: [0.15, 1, 1, 1] },
      title: { text: title, style: { fontSize: '12px', color: '#a0aec0' } },
      xaxis: { type: 'numeric', labels: { formatter: v => `${v}min` } },
      tooltip: darkTooltip(),
      grid: CHART_THEME.grid,
      legend: { show: true }
    };
    if (el._chart) el._chart.destroy();
    el._chart = new ApexCharts(el, options);
    el._chart.render();
  }

  // Public API
  window.Charts = {
    initTempChart,
    initPowerChart,
    initMeltChart,
    initDriftChart,
    initEfficiencyGauge,
    initEnergyChart,
    renderChargeMixBar,
    renderParetoChart,
    renderEconomicsChart,
    renderDualAxisChart,
    renderHorizontalBarChart,
    renderGroupedBarChart,
    renderAnnotatedScatterChart,
    renderConfidenceBandChart,
    updateLiveCharts,
    buildHeatLogRow,
    clearHistories
  };
})();
