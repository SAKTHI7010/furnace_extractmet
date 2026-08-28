import re
with open('static/app.js', 'r', encoding='utf-8') as f:
    js = f.read()

pattern = r"const pill = document\.getElementById\('drift-alarm-pill'\);\s*if \(pill\) \{.*?\n\s*\}"
new_block = """const pill = document.getElementById('drift-alarm-pill');
    if (pill) {
      pill.innerHTML = res.alarm ? 'DRIFT ALARM &mdash; ' + (res.reasons || []).slice(0, 2).join(', ') : 'stable &mdash; no significant drift';
      pill.className = `status-pill ${res.alarm ? 'status-bad' : 'status-ok'}`;
    }"""

js = re.sub(pattern, new_block, js, flags=re.DOTALL)

with open('static/app.js', 'w', encoding='utf-8') as f:
    f.write(js)
