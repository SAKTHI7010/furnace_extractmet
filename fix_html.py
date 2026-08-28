import re

with open('static/index.html', 'r', encoding='utf-8') as f:
    html = f.read()

vs_pattern = r'<section class="tab-pane" id="pane-virtual-sensor">.*?</section>'
vs_new = """<section class="tab-pane" id="pane-virtual-sensor">
        <div class="kpi-summary-row" style="margin-bottom: 12px; grid-template-columns: 1fr 1fr 1fr 1.5fr;">
          <div class="slider-group">
            <div class="slider-header">
              <label for="ekf-eta" style="color: #9aa4af; font-size: 11px;">True &eta; electrical</label>
              <span id="ekf-eta-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">0.90</span>
            </div>
            <input type="range" id="ekf-eta" min="0.80" max="1.0" step="0.01" value="0.90" class="slider-input">
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="ekf-ua" style="color: #9aa4af; font-size: 11px;">True wall-loss scale</label>
              <span id="ekf-ua-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">1.35</span>
            </div>
            <input type="range" id="ekf-ua" min="0.80" max="1.80" step="0.05" value="1.35" class="slider-input">
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="ekf-dips" style="color: #9aa4af; font-size: 11px;">Immersion dips</label>
              <span id="ekf-dips-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">3</span>
            </div>
            <input type="range" id="ekf-dips" min="1" max="6" step="1" value="3" class="slider-input">
          </div>
          <div class="btn-group" style="align-self: end; height: 32px;">
            <button id="ekf-btn-run" class="btn btn-secondary" style="width: 100%; height: 100%;">Run live (~1 min)</button>
          </div>
        </div>
        <div class="kpi-summary-row" style="grid-template-columns: repeat(4, 1fr);">
          <div class="kpi-card-mini">
            <div class="lab">FINAL ERROR °C</div>
            <div class="val" id="ekf-kpi-error" style="color: #ff6a34;">—</div>
            <div class="sub">est - truth</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">H ELECTRICAL</div>
            <div class="val" id="ekf-kpi-eta" style="color: #f6ad55;">—</div>
            <div class="sub">converged</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">&Sigma;_T END °C</div>
            <div class="val" id="ekf-kpi-sigma" style="color: #f6ad55;">—</div>
            <div class="sub">uncertainty</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">DIPS USED</div>
            <div class="val" id="ekf-kpi-dips" style="color: #f6ad55;">—</div>
            <div class="sub">measurements</div>
          </div>
        </div>
        <div class="sensor-charts-grid" style="grid-template-columns: 1.5fr 1fr; margin-top: 12px;">
          <div class="glass-card grid-chart-card">
            <div id="chart-ekf-temp" class="app-chart"></div>
          </div>
          <div class="glass-card grid-chart-card">
            <div id="chart-ekf-params" class="app-chart"></div>
          </div>
        </div>
      </section>"""
html = re.sub(vs_pattern, vs_new, html, flags=re.DOTALL)

ml_pattern = r'<section class="tab-pane" id="pane-machine-learning">.*?</section>'
ml_new = """<section class="tab-pane" id="pane-machine-learning">
        <div class="kpi-summary-row" style="margin-bottom: 12px; grid-template-columns: 1fr 1.5fr 1fr 1.5fr;">
          <div class="slider-group">
            <div class="slider-header">
              <label for="ml-split" style="color: #9aa4af; font-size: 11px;">Train fraction</label>
              <span id="ml-split-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">0.70</span>
            </div>
            <input type="range" id="ml-split" min="0.50" max="0.85" step="0.01" value="0.70" class="slider-input">
          </div>
          <div class="btn-group" style="align-self: end; height: 32px;">
            <button id="ml-btn-train-cached" class="btn btn-secondary" style="width: 100%; height: 100%;">Train on cached data</button>
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="ml-nheats" style="color: #9aa4af; font-size: 11px;">Live heats</label>
              <span id="ml-nheats-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">40</span>
            </div>
            <input type="range" id="ml-nheats" min="20" max="80" step="1" value="40" class="slider-input">
          </div>
          <div class="btn-group" style="align-self: end; height: 32px;">
            <button id="ml-btn-train-live" class="btn btn-secondary" style="width: 100%; height: 100%;">Generate live (slow)</button>
          </div>
        </div>
        <div style="margin-bottom: 12px; font-size: 11px; color: #9aa4af;">
            The hybrid model combines the physical model with a Gaussian-process residual head. Physics predicts, ML corrects, and gates itself off until it proves out-of-time improvement.
            <div class="status-pill status-warn" id="ml-pill-status" style="display: inline-block; margin-left: 8px; font-size: 11px;">maturity: Insufficient &middot; T-ML gated off &middot; C-ML gated off (21 train / 9 test)</div>
        </div>
        <div class="kpi-summary-row" style="grid-template-columns: repeat(4, 1fr);">
          <div class="kpi-card-mini">
            <div class="lab">T HIT &plusmn;15°C</div>
            <div class="val" id="ml-kpi-t-hit" style="color: #f6ad55;">—</div>
            <div class="sub" id="ml-kpi-t-hit-sub">phys — %</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">T MAE °C</div>
            <div class="val" id="ml-kpi-t-mae" style="color: #ff6a34;">—</div>
            <div class="sub">hybrid</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">C HIT &plusmn;0.02%</div>
            <div class="val" id="ml-kpi-c-hit" style="color: #f6ad55;">—</div>
            <div class="sub" id="ml-kpi-c-hit-sub">phys — %</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">C MAE %</div>
            <div class="val" id="ml-kpi-c-mae" style="color: #ff6a34;">—</div>
            <div class="sub">hybrid</div>
          </div>
        </div>
        <div class="sensor-charts-grid" style="grid-template-columns: 1fr 1fr; margin-top: 12px;">
          <div class="glass-card grid-chart-card">
            <div id="chart-ml-scatter" class="app-chart"></div>
          </div>
          <div class="glass-card grid-chart-card">
            <div id="chart-ml-error" class="app-chart"></div>
          </div>
        </div>
      </section>"""
html = re.sub(ml_pattern, ml_new, html, flags=re.DOTALL)

drift_pattern = r'<section class="tab-pane" id="pane-drift-monitor">.*?</section>'
drift_new = """<section class="tab-pane" id="pane-drift-monitor">
        <div class="kpi-summary-row" style="margin-bottom: 12px; grid-template-columns: 1fr 1fr 1fr 1.5fr;">
          <div class="btn-group" style="align-self: end; height: 32px;">
            <button id="drift-btn-cached" class="btn btn-secondary" style="width: 100%; height: 100%;">Check cached data</button>
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="drift-nheats" style="color: #9aa4af; font-size: 11px;">Live heats</label>
              <span id="drift-nheats-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">50</span>
            </div>
            <input type="range" id="drift-nheats" min="30" max="80" step="1" value="50" class="slider-input">
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="drift-regime" style="color: #9aa4af; font-size: 11px;">Regime change at heat</label>
              <span id="drift-regime-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">40</span>
            </div>
            <input type="range" id="drift-regime" min="15" max="60" step="1" value="40" class="slider-input">
          </div>
          <div class="btn-group" style="align-self: end; height: 32px;">
            <button id="drift-btn-live" class="btn btn-secondary" style="width: 100%; height: 100%;">Generate live (slow)</button>
          </div>
        </div>
        <div style="margin-bottom: 12px; font-size: 11px; color: #9aa4af;">
            A pre-computed dataset checks instantly. Live generation runs the same physics simulator and introduces a copper regime change at the selected heat.
            <div class="status-pill status-bad" id="drift-alarm-pill" style="display: inline-block; margin-left: 8px; font-size: 11px; background: #e5484d33; color: #fc8181; border: 1px solid #fc818155;">DRIFT ALARM &mdash; PSI meas_T_C=0.740</div>
        </div>
        <div class="kpi-summary-row" style="grid-template-columns: repeat(3, 1fr);">
          <div class="kpi-card-mini">
            <div class="lab">MAX PSI</div>
            <div class="val" id="drift-kpi-psi" style="color: #ff6a34;">—</div>
            <div class="sub">>0.25 shift &middot; >0.5 major</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">REFERENCE HEATS</div>
            <div class="val" id="drift-kpi-ref" style="color: #f6ad55;">—</div>
            <div class="sub">baseline</div>
          </div>
          <div class="kpi-card-mini">
            <div class="lab">RECENT HEATS</div>
            <div class="val" id="drift-kpi-recent" style="color: #f6ad55;">—</div>
            <div class="sub">checked</div>
          </div>
        </div>
        <div class="sensor-charts-grid" style="grid-template-columns: 1fr 1fr; margin-top: 12px;">
          <div class="glass-card grid-chart-card">
            <div id="chart-drift-psi" class="app-chart"></div>
          </div>
          <div class="glass-card grid-chart-card">
            <div id="chart-drift-tracking" class="app-chart"></div>
          </div>
        </div>
      </section>"""
html = re.sub(drift_pattern, drift_new, html, flags=re.DOTALL)

mix_pattern = r'<section class="tab-pane" id="pane-charge-mix">.*?</section>'
mix_new = """<section class="tab-pane" id="pane-charge-mix">
        <div class="kpi-summary-row" style="margin-bottom: 12px; grid-template-columns: auto 1fr 1fr 1fr 1fr 1fr 1.5fr;">
          <div class="mode-selection" style="align-self: center; font-size: 11px; margin-right: 12px;">
            <span style="display: block; color: #9aa4af; margin-bottom: 4px;">Mode:</span>
            <label class="radio-container" style="margin-bottom: 4px; display: block;">
              <input type="radio" name="mix-mode" value="optimise" checked>
              <span class="radio-text">Optimise (least cost)</span>
            </label>
            <label class="radio-container" style="display: block;">
              <input type="radio" name="mix-mode" value="manual">
              <span class="radio-text">Manual (operator sets kg)</span>
            </label>
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="mix-target" style="color: #9aa4af; font-size: 11px;">Target liquid (t)</label>
              <span id="mix-target-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">12.00</span>
            </div>
            <input type="range" id="mix-target" min="4.0" max="14.0" step="0.1" value="12.0" class="slider-input">
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="mix-clo" style="color: #9aa4af; font-size: 11px;">Min C (%)</label>
              <span id="mix-clo-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">0.10</span>
            </div>
            <input type="range" id="mix-clo" min="0.0" max="0.5" step="0.01" value="0.10" class="slider-input">
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="mix-chi" style="color: #9aa4af; font-size: 11px;">Max C (%)</label>
              <span id="mix-chi-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">0.40</span>
            </div>
            <input type="range" id="mix-chi" min="0.1" max="1.0" step="0.01" value="0.40" class="slider-input">
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="mix-cu" style="color: #9aa4af; font-size: 11px;">Cu ceiling (%)</label>
              <span id="mix-cu-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">0.20</span>
            </div>
            <input type="range" id="mix-cu" min="0.08" max="0.5" step="0.01" value="0.20" class="slider-input">
          </div>
          <div class="slider-group">
            <div class="slider-header">
              <label for="mix-sn" style="color: #9aa4af; font-size: 11px;">Sn ceiling (%)</label>
              <span id="mix-sn-val" class="slider-val" style="color: #f6ad55; font-size: 13px;">0.10</span>
            </div>
            <input type="range" id="mix-sn" min="0.01" max="0.10" step="0.001" value="0.10" class="slider-input">
          </div>
          <div class="btn-group" style="align-self: end; height: 32px;">
            <button id="mix-btn-solve" class="btn btn-secondary" style="width: 100%; height: 100%;">Solve</button>
          </div>
        </div>
        <div class="sensor-charts-grid" style="grid-template-columns: 1fr 1fr; margin-top: 12px;">
          <div class="glass-card">
            <h3 class="card-title" id="mix-table-title" style="font-size: 12px;">Scrap library &mdash; 17 streams (price ₹/kg &middot; assays wt%)</h3>
            <div class="table-container">
              <table id="mix-scrap-table" class="app-table">
                <thead>
                  <tr>
                    <th>Material</th>
                    <th>₹/kg</th>
                    <th>Fe%</th>
                    <th>Cu%</th>
                    <th>Sn%</th>
                    <th>C%</th>
                    <th id="mix-th-kg" style="display:none;">kg (manual)</th>
                  </tr>
                </thead>
                <tbody id="mix-scrap-tbody">
                </tbody>
              </table>
            </div>
          </div>
          <div class="glass-card">
            <div style="display: flex; align-items: center; margin-bottom: 12px;">
              <h3 class="card-title" style="font-size: 12px; margin-right: 8px; margin-bottom: 0;">Result &mdash; blend, bath chemistry, shadow price</h3>
              <div class="status-pill status-ok" id="mix-pill-status" style="font-size: 11px;">feasible &mdash; least-cost compliant blend</div>
            </div>
            <div class="kpi-summary-row" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 12px;">
              <div class="kpi-card-mini">
                <div class="lab">BLEND COST ₹/T</div>
                <div class="val" id="mix-kpi-cost" style="color: #f6ad55;">—</div>
                <div class="sub">of liquid</div>
              </div>
              <div class="kpi-card-mini">
                <div class="lab">CHARGE ENERGY</div>
                <div class=\"val\" id="mix-kpi-energy" style="color: #ff6a34;">—</div>
                <div class="sub">kWh</div>
              </div>
              <div class="kpi-card-mini">
                <div class="lab">PREDICTED CU %</div>
                <div class="val" id="mix-kpi-cu" style="color: #f6ad55;">—</div>
                <div class="sub" id="mix-kpi-cu-sub">&le;0.20</div>
              </div>
              <div class="kpi-card-mini">
                <div class="lab">PREDICTED C %</div>
                <div class="val" id="mix-kpi-c" style="color: #f6ad55;">—</div>
                <div class="sub" id="mix-kpi-c-sub">0.10 - 0.40</div>
              </div>
            </div>
            <div class="table-container" style="height: 180px;">
              <table class="app-table" id="mix-recipe-table">
                <thead>
                  <tr>
                    <th>Material</th>
                    <th>kg</th>
                    <th>% of charge</th>
                  </tr>
                </thead>
                <tbody id="mix-recipe-tbody">
                </tbody>
              </table>
            </div>
            <div class="table-container" style="height: 120px; margin-top: 12px;">
              <table class="app-table" id="mix-chemistry-table">
                <thead>
                  <tr>
                    <th>Element</th>
                    <th>wt %</th>
                  </tr>
                </thead>
                <tbody id="mix-chemistry-tbody">
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>"""
html = re.sub(mix_pattern, mix_new, html, flags=re.DOTALL)

with open('static/index.html', 'w', encoding='utf-8') as f:
    f.write(html)
