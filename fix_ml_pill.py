import re
with open('static/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

pattern = r"const pill = document\.getElementById\('ml-pill-status'\);\s*if \(pill\) \{.*?\n\s*\}"
new_block = """const pill = document.getElementById('ml-pill-status');
    if (pill) {
      const maturity = m.maturity || 'Insufficient';
      const tActive  = m.ml_T_active ? 'T-ML active' : 'T-ML gated off';
      const cActive  = m.ml_C_active ? 'C-ML active' : 'C-ML gated off';
      pill.innerHTML = `maturity: ${maturity} &middot; ${tActive} &middot; ${cActive} (${m.n_train || 0} train / ${m.n_test || 0} test)`;
      pill.className = `status-pill ${m.ml_T_active ? 'status-ok' : 'status-warn'}`;
    }"""

js = re.sub(pattern, new_block, js, flags=re.DOTALL)

with open('static/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
