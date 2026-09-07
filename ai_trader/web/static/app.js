/* AI-Trader browser system — SPA logic (no dependencies). */
(function () {
  'use strict';

  /* ============================== helpers ============================== */
  const $ = (sel, el) => (el || document).querySelector(sel);
  const $$ = (sel, el) => Array.from((el || document).querySelectorAll(sel));

  function esc(v) {
    if (v === null || v === undefined) return '—';
    return String(v).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }
  function num(v, d) {
    if (v === null || v === undefined || isNaN(Number(v))) return '—';
    return Number(v).toLocaleString('en-US', { maximumFractionDigits: d === undefined ? 4 : d });
  }
  function signed(v, d) {
    if (v === null || v === undefined || isNaN(Number(v))) return '—';
    const n = Number(v);
    return (n >= 0 ? '+' : '') + n.toFixed(d === undefined ? 2 : d);
  }
  function pct(v, d) {
    if (v === null || v === undefined || isNaN(Number(v))) return '—';
    return (Number(v) * 100).toFixed(d === undefined ? 1 : d) + '%';
  }
  function clsFor(v) {
    if (v === null || v === undefined || isNaN(Number(v))) return 'v-muted';
    const n = Number(v);
    return n > 0 ? 'v-green' : n < 0 ? 'v-red' : 'v-muted';
  }

  function toast(msg, kind) {
    const box = $('#toasts');
    const el = document.createElement('div');
    el.className = 'toast ' + (kind || '');
    el.textContent = msg;
    box.appendChild(el);
    setTimeout(() => { el.style.opacity = '0'; setTimeout(() => el.remove(), 300); }, 4200);
  }

  async function api(path, opts) {
    opts = opts || {};
    opts.headers = Object.assign({ 'Content-Type': 'application/json' }, opts.headers || {});
    let res;
    try {
      res = await fetch(path, opts);
    } catch (e) {
      throw new Error('Cannot reach server (' + e.message + '). Is it running?');
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* non-JSON */ }
    if (!res.ok) {
      const msg = (data && (data.detail || data.message)) || ('HTTP ' + res.status);
      throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
    }
    return data;
  }

  function setBusy(btn, busy, label) {
    if (!btn) return;
    if (busy) {
      btn.dataset.label = btn.innerHTML;
      btn.innerHTML = '<span class="spinner"></span>' + esc(label || 'Working…');
      btn.disabled = true;
    } else {
      btn.innerHTML = btn.dataset.label || btn.innerHTML;
      btn.disabled = false;
    }
  }

  function loadingBox(text) {
    return '<div class="card loading-box"><span class="spinner" style="border-top-color:var(--accent)"></span> ' + esc(text || 'Working…') + '</div>';
  }

  /* ============================== router ============================== */
  const VIEW_SUBS = {
    analyze: 'Run the full decision engine on any instrument.',
    scenarios: 'Scripted textbook markets — prove the TRADE and NO TRADE paths.',
    backtest: 'Replay history and measure the edge (background jobs).',
    instruments: 'Every instrument the engine knows about.',
    settings: 'Live engine configuration (thresholds, scoring, risk).',
  };
  const VIEW_TITLES = { analyze: 'Analyze', scenarios: 'Scenarios', backtest: 'Backtest', instruments: 'Instruments', settings: 'Settings' };

  function showView(name) {
    $$('.nav-btn').forEach((b) => b.classList.toggle('active', b.dataset.view === name));
    $$('.view').forEach((v) => v.classList.toggle('active', v.id === 'view-' + name));
    $('#view-title').textContent = VIEW_TITLES[name] || name;
    $('#view-sub').textContent = VIEW_SUBS[name] || '';
    if (window.innerWidth <= 760) $('#sidebar').classList.add('hidden');
    if (name === 'instruments') loadInstruments();
    if (name === 'settings') loadSettings();
    if (name === 'backtest') refreshJobs(false);
  }
  $$('.nav-btn').forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));
  $('#menu-btn').addEventListener('click', () => $('#sidebar').classList.toggle('hidden'));
  $('#side-toggle').addEventListener('click', () => $('#sidebar').classList.toggle('hidden'));

  /* ============================== health ============================== */
  async function checkHealth() {
    const pill = $('#health-pill');
    try {
      const h = await api('/api/health');
      pill.innerHTML = '<span class="dot ok"></span><span>Engine online</span>';
      pill.title = 'Uptime ' + h.uptime_s + 's';
    } catch (e) {
      pill.innerHTML = '<span class="dot bad"></span><span>Offline</span>';
    }
  }

  /* ============================== shared widgets ============================== */
  const ALL_TFS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'];
  function buildTfPills(el, defaults) {
    el.innerHTML = '';
    ALL_TFS.forEach((tf) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'tf-pill' + (defaults.indexOf(tf) >= 0 ? ' on' : '');
      b.textContent = tf;
      b.dataset.tf = tf;
      b.addEventListener('click', () => b.classList.toggle('on'));
      el.appendChild(b);
    });
  }
  function selectedTfs(el) {
    const out = $$('.tf-pill.on', el).map((b) => b.dataset.tf);
    return out.length ? out : ['15m', '1h', '4h', '1d'];
  }

  let INSTRUMENTS = [];
  async function loadSymbolList() {
    try {
      const data = await api('/api/instruments');
      INSTRUMENTS = data.instruments || [];
      const dl = $('#symbol-list');
      dl.innerHTML = '';
      INSTRUMENTS.forEach((i) => {
        const o = document.createElement('option');
        o.value = i.symbol;
        o.label = i.name;
        dl.appendChild(o);
      });
    } catch (e) { /* table view will surface errors */ }
  }

  /* ============================== decision renderer ============================== */
  function verdictClass(decision) {
    const d = String(decision || '').toUpperCase();
    if (d === 'TRADE') return 'trade';
    if (d === 'WAIT') return 'wait';
    return 'no_trade';
  }
  function verdictLabel(decision) {
    const d = String(decision || '').toUpperCase();
    if (d === 'TRADE') return '◈ TRADE';
    if (d === 'WAIT') return '◷ WAIT';
    return '✕ NO TRADE';
  }

  function breakdownRows(breakdown) {
    if (!breakdown) return '';
    let items = [];
    if (Array.isArray(breakdown)) {
      items = breakdown.map((c) => ({ name: c.name || c.component || '?', points: c.points, max: c.max_points || c.max }));
    } else if (typeof breakdown === 'object') {
      Object.keys(breakdown).forEach((k) => {
        const v = breakdown[k];
        if (v && typeof v === 'object') items.push({ name: k, points: v.points, max: v.max_points || v.max });
        else items.push({ name: k, points: v, max: null });
      });
    }
    if (!items.length) return '<p class="muted small">No breakdown available.</p>';
    return items.map((c) => {
      const p = Number(c.points) || 0, m = Number(c.max) || 0;
      const w = m > 0 ? Math.max(0, Math.min(100, (p / m) * 100)) : 0;
      return '<div class="score-row"><span class="nm">' + esc(String(c.name).replace(/_/g, ' ')) +
        '</span><span class="bar-track"><span class="bar-fill" style="display:block;width:' + w.toFixed(1) + '%"></span></span>' +
        '<span class="vl">' + num(p, 1) + (m ? ' / ' + num(m, 1) : '') + '</span></div>';
    }).join('');
  }

  function listHtml(items, emptyText) {
    if (!items || !items.length) return '<p class="muted small">' + esc(emptyText || 'None.') + '</p>';
    return '<ul>' + items.map((x) => '<li>' + esc(typeof x === 'string' ? x : JSON.stringify(x)) + '</li>').join('') + '</ul>';
  }

  function accHtml(title, inner, open) {
    return '<details class="acc"' + (open ? ' open' : '') + '><summary>' + esc(title) +
      '</summary><div class="acc-body">' + inner + '</div></details>';
  }

  function kvHtml(obj) {
    if (!obj || typeof obj !== 'object') return '<p class="muted small">—</p>';
    return Object.keys(obj).map((k) => {
      let v = obj[k];
      if (v && typeof v === 'object') v = JSON.stringify(v);
      return '<div class="kv"><span class="k">' + esc(k) + '</span><span class="v">' + esc(v) + '</span></div>';
    }).join('');
  }

  function candidatesTable(cands) {
    if (!cands || !cands.length) return '<p class="muted">No strategy produced a signal on this market.</p>';
    const rows = cands.map((c) => {
      const dir = String(c.direction || '-').toUpperCase();
      const tag = dir === 'LONG' || dir === 'BUY' ? 'buy' : dir === 'SHORT' || dir === 'SELL' ? 'sell' : 'neutral';
      return '<tr><td><b>' + esc(c.strategy || '?') + '</b></td>' +
        '<td><span class="tag ' + tag + '">' + esc(dir) + '</span></td>' +
        '<td class="num"><b>' + num(c.score, 1) + '</b></td>' +
        '<td class="num">' + (c.rr_tp2 !== undefined && c.rr_tp2 !== null ? '1:' + num(c.rr_tp2, 1) : '—') + '</td>' +
        '<td>' + esc(outcome) + '</td></tr>';
    }).join('');
    return '<div class="table-wrap"><table class="tbl"><thead><tr><th>Strategy</th><th>Dir</th><th class="num">Score</th><th class="num">R:R TP2</th><th>Outcome</th></tr></thead><tbody>' +
      rows + '</tbody></table></div>';
  }

  function renderDecision(container, payload, chartOpts) {
    const d = payload.decision || {};
    const plan = d.plan || null;
    const regime = d.regime || null;
    const meta = payload.meta || {};
    const vc = verdictClass(d.decision);

    let html = '';
    html += '<div class="verdict ' + vc + '"><div class="badge">' + verdictLabel(d.decision) + '</div>' +
      '<div class="v-msg"><b>' + esc(d.instrument || '') + '</b> — ' + esc(d.message || '') + '</div>' +
      '<div class="v-meta">' + esc(d.created_at || '') +
      (meta.source ? '<br>source: ' + esc(meta.source) + ' · seed ' + esc(meta.seed) : '') +
      (meta.scenario ? '<br>scenario: ' + esc(meta.scenario) : '') + '</div></div>';

    // stat cards
    const biasTag = d.bias === 'BUY' ? 'buy' : d.bias === 'SELL' ? 'sell' : 'neutral';
    html += '<div class="stats">';
    html += '<div class="stat"><div class="k">Bias</div><div class="v"><span class="tag ' + biasTag + '">' + esc(d.bias || '—') + '</span></div></div>';
    html += '<div class="stat"><div class="k">Regime</div><div class="v" style="font-size:14px">' + esc(regime ? regime.primary : 'UNKNOWN') + '</div>' +
      '<div class="s">' + (regime ? 'confidence ' + pct(regime.confidence, 0) : '—') + '</div></div>';
    if (plan) {
      html += '<div class="stat"><div class="k">Score</div><div class="v v-blue">' + num(plan.score, 0) + '<span class="s"> /100</span></div><div class="s">' + esc(plan.grade || '') + '</div></div>';
      html += '<div class="stat"><div class="k">R:R (TP2)</div><div class="v v-purple">1:' + num(plan.rr, 2) + '</div><div class="s">' + esc(plan.strategy || '') + '</div></div>';
      html += '<div class="stat"><div class="k">Position</div><div class="v">' + num(plan.risk ? plan.risk.lots : null, 2) + '<span class="s"> lots</span></div><div class="s">risk ' + num(plan.risk ? plan.risk.risk_pct : null, 2) + '%</div></div>';
      html += '<div class="stat"><div class="k">Entry</div><div class="v" style="font-size:14px">' + num(plan.entry_low) + ' – ' + num(plan.entry_high) + '</div><div class="s">' + esc(plan.entry_type || '') + ' · ' + esc(plan.timeframe || '') + '</div></div>';
    } else {
      html += '<div class="stat"><div class="k">Candidates</div><div class="v">' + ((d.candidates || []).length) + '</div><div class="s">strategies evaluated</div></div>';
      html += '<div class="stat"><div class="k">Reasons</div><div class="v">' + ((d.reasons || []).length) + '</div><div class="s">blocking / notes</div></div>';
    }
    html += '</div>';

    // price chart
    const chartId = 'ch-' + Math.random().toString(36).slice(2, 8);
    html += '<div class="card chart-card"><h2>Price Chart <span class="count" id="' + chartId + '-note"></span></h2>' +
      '<div style="position:relative"><canvas class="candles" id="' + chartId + '"></canvas><div class="chart-tip"></div></div>' +
      '<div class="chart-legend"><span><span class="sw" style="background:#34d17b"></span>Entry zone</span>' +
      '<span><span class="sw" style="background:#ff5d5d"></span>Stop loss</span>' +
      '<span><span class="sw" style="background:#4da3ff"></span>TP1 / TP2</span>' +
      '<span><span class="sw" style="background:#a371f7"></span>TP3</span>' +
      '<span class="muted">drag to pan · scroll to zoom · hover for OHLC</span></div></div>';

    // plan + score columns
    html += '<div class="cols">';
    if (plan) {
      const r = plan.risk || {};
      html += '<div class="card"><h2>Trade Plan</h2>' +
        '<div class="kv"><span class="k">Direction</span><span class="v"><span class="tag ' + (String(plan.direction).toLowerCase()) + '">' + esc(plan.direction) + '</span></span></div>' +
        '<div class="kv"><span class="k">Entry zone</span><span class="v v-green">' + num(plan.entry_low) + ' – ' + num(plan.entry_high) + '</span></div>' +
        '<div class="kv"><span class="k">Stop loss</span><span class="v v-red">' + num(plan.stop) + '</span></div>' +
        '<div class="kv"><span class="k">TP1 / TP2 / TP3</span><span class="v v-blue">' + num(plan.tp1) + ' / ' + num(plan.tp2) + ' / ' + num(plan.tp3) + '</span></div>' +
        '<div class="kv"><span class="k">Stop distance</span><span class="v">' + num(r.stop_distance) + ' (' + num(r.stop_distance_atr, 2) + ' ATR)</span></div>' +
        '<div class="kv"><span class="k">Position size</span><span class="v">' + num(r.lots, 2) + ' lots (' + num(r.units, 0) + ' units)</span></div>' +
        '<div class="kv"><span class="k">Risk</span><span class="v">' + num(r.risk_pct, 2) + '% = ' + num(r.risk_amount, 2) + '</span></div>' +
        '<div class="kv"><span class="k">Strategy</span><span class="v">' + esc(plan.strategy) + ' · ' + esc(plan.timeframe) + '</span></div>' +
        '<div class="kv"><span class="k">Regime</span><span class="v">' + esc(regime ? regime.primary : 'UNKNOWN') + (regime ? ' (' + pct(regime.confidence, 0) + ')' : '') + '</span></div>' +
        (plan.warnings && plan.warnings.length ? '<div style="margin-top:10px">' + plan.warnings.map((w) => '<div class="chip">⚠ ' + esc(w) + '</div>').join(' ') + '</div>' : '') +
        '</div>';
      html += '<div class="card"><h2>Confluence Score — ' + num(plan.score, 0) + '/100</h2>' + breakdownRows(plan.breakdown) + '</div>';
    } else {
      html += '<div class="card"><h2>Why no trade?</h2>' + listHtml(d.reasons, 'No reasons recorded.') + '</div>';
      html += '<div class="card"><h2>Regime detail</h2>' + (regime ? '<p class="muted small">' + esc(regime.explanation || '') + '</p>' +
        '<div class="chips">' + (regime.allowed_families || []).map((f) => '<span class="chip">' + esc(f) + '</span>').join('') + '</div>' +
        ((regime.secondary && regime.secondary.length) ? '<p class="small" style="margin-top:8px">Overlays: ' + regime.secondary.map(esc).join(', ') + '</p>' : '')
        : '<p class="muted">Unknown.</p>') + '</div>';
    }
    html += '</div>';

    // factors / candidates
    html += '<div class="cols">';
    if (plan && plan.factors) html += '<div class="card"><h2>Confluence Factors</h2>' + listHtml(plan.factors) + '</div>';
    html += '<div class="card"><h2>Candidates Evaluated <span class="count">' + (d.candidates || []).length + '</span></h2>' + candidatesTable(d.candidates) + '</div>';
    if (!(plan && plan.factors)) html += '<div class="card"><h2>Multi-timeframe Context</h2>' + mtfSummary(d.context && d.context.mtf) + '</div>';
    html += '</div>';

    // explanation accordions
    if (plan && plan.explanation) {
      const e = plan.explanation;
      html += '<div class="card"><h2>Explanation</h2>' +
        accHtml('Why this trade?', listHtml(e.why_this_trade), true) +
        accHtml('Why now?', listHtml(e.why_now)) +
        accHtml('Invalidation', listHtml(e.what_invalidates)) +
        accHtml('What could make it fail?', listHtml(e.what_could_make_it_fail)) +
        accHtml('No-trade conditions', listHtml(e.no_trade_conditions)) +
        (plan.invalidation ? '<p class="small muted">Invalidation: ' + esc(plan.invalidation) + '</p>' : '') +
        '</div>';
    }
    if (plan && plan.conflicts && plan.conflicts.length) {
      html += '<div class="card"><h2>Conflicts noted</h2>' + listHtml(plan.conflicts) + '</div>';
    }

    // raw terminal-style text
    html += '<div class="card"><h2>Full Report (text)</h2>' +
      accHtml('Show plain-text report', '<pre class="raw">' + esc(payload.text || '') + '</pre>', false) +
      '<p class="hint">' + esc(d.disclaimer || '') + '</p></div>';

    container.innerHTML = html;

    // load chart candles
    loadDecisionChart(chartId, d, plan, chartOpts || {});
  }

  async function loadDecisionChart(chartId, d, plan, opts) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    const params = new URLSearchParams({
      symbol: opts.symbol || d.instrument || 'XAUUSD',
      timeframe: opts.timeframe || '15m',
      limit: String(opts.limit || 300),
      source: opts.source || 'synthetic',
      seed: String(opts.seed !== undefined ? opts.seed : 1),
    });
    if (opts.scenario) { params.set('source', 'scenario'); params.set('scenario', opts.scenario); }
    if (opts.data_dir) params.set('data_dir', opts.data_dir);
    try {
      const data = await api('/api/candles?' + params.toString());
      const note = document.getElementById(chartId + '-note');
      if (note) note.textContent = data.symbol + ' · ' + data.timeframe + ' · ' + data.count + ' bars' + (data.note ? ' · ' + data.note : '');
      const ov = {};
      if (plan) {
        ov.entryLow = plan.entry_low; ov.entryHigh = plan.entry_high;
        ov.stop = plan.stop; ov.tp1 = plan.tp1; ov.tp2 = plan.tp2; ov.tp3 = plan.tp3;
        ov.direction = String(plan.direction || '').toLowerCase();
      }
      window.CandleChart.render(canvas, data.candles, ov);
    } catch (e) {
      window.CandleChart.render(canvas, [], {});
      const note = document.getElementById(chartId + '-note');
      if (note) note.textContent = 'chart unavailable: ' + e.message;
    }
  }

  /* ============================== analyze view ============================== */
  buildTfPills($('#an-tfs'), ['15m', '1h', '4h', '1d']);
  $('#an-source').addEventListener('change', (e) => {
    const hints = {
      synthetic: 'Synthetic data proves the plumbing, never the edge. For real analysis use CSV history or TwelveData.',
      csv: 'Reads {SYMBOL}_{tf}.csv from the data/ folder (e.g. XAUUSD_15m.csv). Higher timeframes resample automatically.',
      twelvedata: 'Live market data via TwelveData — requires AITRADER_TWELVEDATA_API_KEY in the server environment.',
    };
    $('#an-hint').textContent = hints[e.target.value] || '';
  });
  $('#an-run').addEventListener('click', async () => {
    const btn = $('#an-run'), box = $('#an-result');
    const body = {
      symbol: $('#an-symbol').value.trim() || 'XAUUSD',
      source: $('#an-source').value,
      seed: parseInt($('#an-seed').value, 10) || 0,
      timeframes: selectedTfs($('#an-tfs')),
      no_news_penalty: $('#an-nonews').checked,
    };
    setBusy(btn, true, 'Analyzing…');
    box.innerHTML = loadingBox('Running decision engine on ' + body.symbol + '…');
    try {
      const payload = await api('/api/analyze', { method: 'POST', body: JSON.stringify(body) });
      renderDecision(box, payload, { symbol: body.symbol, source: body.source, seed: body.seed });
      const vd = String(payload.decision.decision || '');
      toast(body.symbol + ': ' + vd.replace('_', ' '), vd === 'TRADE' ? 'ok' : '');
      box.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      box.innerHTML = '<div class="card"><h2>Analysis failed</h2><p class="v-red">' + esc(e.message) + '</p></div>';
      toast(e.message, 'err');
    } finally {
      setBusy(btn, false);
    }
  });

  /* ============================== scenarios view ============================== */
  let SCENARIOS = [], SEL_SCN = 'textbook_long';
  async function loadScenarios() {
    try {
      const data = await api('/api/scenarios');
      SCENARIOS = data.scenarios || [];
      const grid = $('#scn-grid');
      grid.innerHTML = '';
      SCENARIOS.forEach((s) => {
        const el = document.createElement('div');
        el.className = 'scn-card' + (s.id === SEL_SCN ? ' sel' : '');
        el.innerHTML = '<h3>' + esc(s.title) + '</h3><p>' + esc(s.description) + '</p>' +
          '<div class="row"><span class="tag info">' + esc(s.symbol) + '</span><span class="tag ' +
          (s.expected.indexOf('NO TRADE') >= 0 ? 'neutral' : 'purple') + '">' + esc(s.expected) + '</span></div>';
        el.addEventListener('click', () => { SEL_SCN = s.id; loadScenarios(); });
        grid.appendChild(el);
      });
      const sel = SCENARIOS.find((s) => s.id === SEL_SCN);
      const tg = $('#scn-toggles');
      tg.innerHTML = '';
      if (!sel || !sel.toggles.length) {
        tg.innerHTML = '<span class="muted small">This scenario has no switchable components.</span>';
      } else {
        sel.toggles.forEach((t) => {
          const b = document.createElement('button');
          b.type = 'button';
          b.className = 'tf-pill';
          b.textContent = '✓ ' + t;
          b.dataset.comp = t;
          b.title = 'Active — click to switch this component OFF';
          b.addEventListener('click', () => {
            const off = b.classList.toggle('on');
            b.textContent = (off ? '✕ ' : '✓ ') + t;
            b.title = off ? 'Switched OFF — click to re-enable' : 'Active — click to switch this component OFF';
          });
          tg.appendChild(b);
        });
      }
    } catch (e) {
      $('#scn-grid').innerHTML = '<p class="v-red">' + esc(e.message) + '</p>';
    }
  }
  $('#scn-run').addEventListener('click', async () => {
    const btn = $('#scn-run'), box = $('#scn-result');
    const without = $$('#scn-toggles .tf-pill.on').map((b) => b.dataset.comp);
    const seed = parseInt($('#scn-seed').value, 10) || 0;
    setBusy(btn, true, 'Running…');
    box.innerHTML = loadingBox('Running scenario ' + SEL_SCN + '…');
    try {
      const payload = await api('/api/scenario', { method: 'POST', body: JSON.stringify({ name: SEL_SCN, seed, without }) });
      renderDecision(box, payload, { scenario: SEL_SCN, seed, source: 'scenario' });
      const vd = String(payload.decision.decision || '');
      toast(SEL_SCN + ': ' + vd.replace('_', ' '), vd === 'TRADE' ? 'ok' : '');
      box.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      box.innerHTML = '<div class="card"><h2>Scenario failed</h2><p class="v-red">' + esc(e.message) + '</p></div>';
      toast(e.message, 'err');
    } finally {
      setBusy(btn, false);
    }
  });

  /* ============================== backtest view ============================== */
  buildTfPills($('#bt-tfs'), ['15m', '1h', '4h', '1d']);
  function syncBtFields() {
    const src = $('#bt-source').value;
    $$('.bt-only-scenario').forEach((el) => { el.style.display = src === 'scenario' ? '' : 'none'; });
    $$('.bt-only-synthetic').forEach((el) => { el.style.display = src === 'synthetic' ? '' : 'none'; });
  }
  $('#bt-source').addEventListener('change', syncBtFields);
  syncBtFields();

  function metricStats(m) {
    m = m || {};
    const card = (k, v, cls, s) => '<div class="stat"><div class="k">' + k + '</div><div class="v ' + (cls || '') + '">' + v + '</div>' + (s ? '<div class="s">' + s + '</div>' : '') + '</div>';
    return '<div class="stats">' +
      card('Trades', num(m.trades, 0), '', 'win rate ' + pct(m.win_rate)) +
      card('Net P&L', signed(m.net_pnl) + ' <span class="s">(' + signed(m.return_pct) + '%)</span>', clsFor(m.net_pnl), 'expectancy ' + signed(m.expectancy_r) + ' R') +
      card('Profit factor', m.profit_factor === null || m.profit_factor === undefined ? '—' : num(m.profit_factor, 2), '', 'payoff ' + num(m.avg_rr_realised, 2)) +
      card('Max drawdown', (m.max_drawdown_pct !== null && m.max_drawdown_pct !== undefined ? num(m.max_drawdown_pct, 1) + '%' : '—'), 'v-red', num(m.max_drawdown_r, 1) + ' R') +
      card('Sharpe / trade', num(m.sharpe_per_trade, 2), '', 't-stat ' + num(m.t_stat, 2)) +
      card('Avg bars held', num(m.avg_bars_held, 1), '', 'max loss streak ' + num(m.max_consecutive_losses, 0)) +
      '</div>';
  }

  function equitySvg(curve) {
    if (!curve || curve.length < 2) return '<p class="muted small">Not enough trades for an equity curve.</p>';
    const W = 900, H = 200, P = 30;
    const vals = curve.map((p) => p.equity);
    let lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi === lo) { hi += 1; lo -= 1; }
    const X = (i) => P + (i / (curve.length - 1)) * (W - 2 * P);
    const Y = (v) => (H - P) - ((v - lo) / (hi - lo)) * (H - 2 * P);
    let d = '';
    curve.forEach((p, i) => { d += (i ? 'L' : 'M') + X(i).toFixed(1) + ' ' + Y(p.equity).toFixed(1) + ' '; });
    const up = vals[vals.length - 1] >= vals[0];
    const col = up ? '#34d17b' : '#ff5d5d';
    return '<svg viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
      '<defs><linearGradient id="eqg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="' + col + '" stop-opacity=".35"/><stop offset="1" stop-color="' + col + '" stop-opacity="0"/></linearGradient></defs>' +
      '<path d="' + d + 'L' + X(curve.length - 1).toFixed(1) + ' ' + (H - P) + ' L' + X(0).toFixed(1) + ' ' + (H - P) + ' Z" fill="url(#eqg)"/>' +
      '<path d="' + d + '" fill="none" stroke="' + col + '" stroke-width="2" vector-effect="non-scaling-stroke"/>' +
      '<text x="' + (P + 4) + '" y="18" fill="#8b98ab" font-size="13">' + num(hi, 0) + '</text>' +
      '<text x="' + (P + 4) + '" y="' + (H - P + 16) + '" fill="#8b98ab" font-size="13">' + num(lo, 0) + '</text>' +
      '</svg>';
  }

  function tradesTable(trades) {
    if (!trades || !trades.length) return '<p class="muted">No closed trades.</p>';
    const rows = trades.map((t) => '<tr><td class="num">' + esc(t.id) + '</td>' +
      '<td>' + esc((t.exit_time || '').slice(0, 16).replace('T', ' ')) + '</td>' +
      '<td><span class="tag ' + String(t.direction || '').toLowerCase() + '">' + esc(t.direction) + '</span></td>' +
      '<td>' + esc(t.strategy) + '</td>' +
      '<td class="num">' + num(t.entry) + '</td><td class="num">' + num(t.exit) + '</td>' +
      '<td class="num ' + clsFor(t.pnl) + '"><b>' + signed(t.pnl) + '</b></td>' +
      '<td class="num ' + clsFor(t.r) + '">' + signed(t.r) + '</td>' +
      '<td>' + esc(t.exit_reason) + '</td></tr>').join('');
    return '<div class="table-wrap"><table class="tbl"><thead><tr><th class="num">#</th><th>Exit</th><th>Dir</th><th>Strategy</th><th class="num">Entry</th><th class="num">Exit</th><th class="num">P&L</th><th class="num">R</th><th>Exit reason</th></tr></thead><tbody>' + rows + '</tbody></table></div>';
  }

  function chipsHtml(obj) {
    if (!obj || !Object.keys(obj).length) return '<p class="muted small">None.</p>';
    return '<div class="chips">' + Object.keys(obj).map((k) => '<span class="chip">' + esc(k) + ' <b>' + esc(obj[k]) + '</b></span>').join('') + '</div>';
  }

  function renderBacktestResult(box, payload) {
    const parts = payload.mode === 'split'
      ? [['In-sample', payload.in_sample], ['Out-of-sample', payload.out_of_sample]]
      : [['Result', payload]];
    let html = '';
    if (payload.mode === 'split') {
      html += '<div class="card"><h2>IS → OOS degradation: ' + (payload.degradation === null || payload.degradation === undefined ? 'n/a' : num(payload.degradation, 2)) + '</h2>' +
        '<p class="muted small">' + esc(payload.notes || '') + '</p></div>';
    }
    parts.forEach(([title, part]) => {
      const r = part.result || {};
      const m = r.metrics || {};
      html += '<div class="card"><h2>' + esc(title) + ' — ' + esc(r.symbol || '') + ' <span class="count">' + esc(r.label || '') + ' · ' + esc(r.data_source || '') + '</span></h2>' +
        '<p class="muted small">' + esc(r.first_ts || '') + ' → ' + esc(r.last_ts || '') + ' · ' + num(r.bars_processed, 0) + ' bars · ' + num(r.decisions, 0) + ' decisions · runtime ' + num(r.runtime_s, 1) + 's</p>' +
        metricStats(m) +
        '<h2 style="margin-top:6px">Equity Curve</h2><div class="eq-wrap">' + equitySvg(r.equity_curve) + '</div>' +
        '</div>';
      html += '<div class="cols"><div class="card"><h2>Why no trade?</h2>' + chipsHtml(r.block_reasons) + '</div>' +
        '<div class="card"><h2>Regimes seen</h2>' + chipsHtml(r.regime_counts) + '</div></div>';
      html += '<div class="card"><h2>Trades <span class="count">' + (r.trades || []).length + '</span></h2>' + tradesTable(r.trades) + '</div>';
      if (r.by_strategy && Object.keys(r.by_strategy).length) {
        const rows = Object.keys(r.by_strategy).map((k) => {
          const s = r.by_strategy[k];
          return '<tr><td><b>' + esc(k) + '</b></td><td class="num">' + num(s.trades, 0) + '</td><td class="num">' + pct(s.win_rate) + '</td><td class="num">' + signed(s.expectancy_r) + '</td><td class="num ' + clsFor(s.net_pnl) + '">' + signed(s.net_pnl) + '</td></tr>';
        }).join('');
        html += '<div class="card"><h2>By Strategy</h2><div class="table-wrap"><table class="tbl"><thead><tr><th>Strategy</th><th class="num">Trades</th><th class="num">Win rate</th><th class="num">Exp (R)</th><th class="num">Net P&L</th></tr></thead><tbody>' + rows + '</tbody></table></div></div>';
      }
      if (r.walk_forward) {
        const w = r.walk_forward;
        html += '<div class="card"><h2>Walk-forward</h2>' +
          '<div class="kv"><span class="k">OOS expectancy</span><span class="v">' + signed(w.oos_expectancy_r) + ' R (' + num(w.oos_trades, 0) + ' trades)</span></div>' +
          '<div class="kv"><span class="k">IS expectancy</span><span class="v">' + signed(w.is_expectancy_r) + ' R</span></div>' +
          '<div class="kv"><span class="k">Degradation</span><span class="v">' + (w.degradation === null || w.degradation === undefined ? 'n/a' : num(w.degradation, 2)) + '</span></div>' +
          '<div class="kv"><span class="k">Positive folds</span><span class="v">' + num(w.positive_folds, 0) + '</span></div>' +
          listHtml(w.notes, '') + '</div>';
      }
      if (r.overfit) {
        const o = r.overfit;
        html += '<div class="card"><h2>Overfit Report — ' + esc(o.verdict || '') + '</h2>' +
          '<div class="kv"><span class="k">Trades / parameter</span><span class="v">' + num(o.trades_per_parameter, 1) + ' (' + num(o.trades, 0) + ' trades, ~' + num(o.free_parameters, 0) + ' params)</span></div>' +
          listHtml(o.checks, 'No checks.') + '</div>';
      }
      if (r.warnings && r.warnings.length) html += '<div class="card"><h2>Warnings</h2>' + listHtml(r.warnings) + '</div>';
      html += '<div class="card"><h2>Full Report (text)</h2>' + accHtml('Show plain-text report', '<pre class="raw">' + esc(part.text || '') + '</pre>', false) + '</div>';
    });
    box.innerHTML = html;
  }

  let POLL_TIMER = null;
  function stopPoll() { if (POLL_TIMER) { clearInterval(POLL_TIMER); POLL_TIMER = null; } }

  async function refreshJobs(auto) {
    try {
      const data = await api('/api/backtest/jobs');
      const el = $('#bt-jobs');
      if (!data.jobs.length) { el.innerHTML = '<p class="muted">No backtests yet.</p>'; return; }
      el.innerHTML = '';
      data.jobs.forEach((j) => {
        const d = document.createElement('div');
        d.className = 'job';
        const pctDone = (j.progress && j.progress.pct) || 0;
        d.innerHTML = '<span class="st ' + esc(j.status) + '">' + esc(j.status) + '</span>' +
          '<span class="nm">' + esc(j.label || j.id) + '</span>' +
          ((j.status === 'running' || j.status === 'queued')
            ? '<div class="prog"><div style="width:' + pctDone + '%"></div></div><span class="small muted">' + pctDone + '%</span>'
            : '<span class="small muted">' + esc(j.id) + '</span>') +
          (j.error ? '<span class="small v-red">' + esc(j.error) + '</span>' : '') +
          (j.status === 'done' ? '<button class="btn sm ghost" data-job="' + esc(j.id) + '">View</button>' : '');
        const vb = d.querySelector('[data-job]');
        if (vb) vb.addEventListener('click', () => viewJob(j.id));
        el.appendChild(d);
      });
      if (auto === undefined) auto = true;
      const active = data.jobs.some((j) => j.status === 'running' || j.status === 'queued');
      if (active && !POLL_TIMER && auto) {
        POLL_TIMER = setInterval(async () => {
          await refreshJobs(false);
          const d2 = await api('/api/backtest/jobs').catch(() => null);
          const still = d2 && d2.jobs.some((j) => j.status === 'running' || j.status === 'queued');
          if (!still) { stopPoll(); refreshJobs(false); }
        }, 1500);
      }
      if (!active) stopPoll();
    } catch (e) { /* ignore transient */ }
  }

  async function viewJob(jobId) {
    const box = $('#bt-result');
    box.innerHTML = loadingBox('Loading backtest result…');
    try {
      const job = await api('/api/backtest/jobs/' + encodeURIComponent(jobId));
      if (job.status !== 'done' || !job.result) {
        box.innerHTML = '<div class="card"><p class="v-red">Job ' + esc(job.status) + (job.error ? ': ' + esc(job.error) : '') + '</p></div>';
        return;
      }
      renderBacktestResult(box, job.result);
      box.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (e) {
      box.innerHTML = '<div class="card"><p class="v-red">' + esc(e.message) + '</p></div>';
    }
  }

  $('#bt-run').addEventListener('click', async () => {
    const btn = $('#bt-run');
    const eq = $('#bt-equity').value.trim();
    const body = {
      symbol: $('#bt-symbol').value.trim() || 'XAUUSD',
      source: $('#bt-source').value,
      direction: $('#bt-direction').value,
      resolve: $('#bt-resolve').value,
      bars: parseInt($('#bt-bars').value, 10) || 600,
      seed: parseInt($('#bt-seed').value, 10) || 0,
      timeframes: selectedTfs($('#bt-tfs')),
      every: parseInt($('#bt-every').value, 10) || 1,
      equity: eq === '' ? null : parseFloat(eq),
      commission: parseFloat($('#bt-commission').value) || 0,
      split: $('#bt-split').value.trim() || null,
      walk_forward: $('#bt-wf').checked,
      no_news_penalty: $('#bt-nonews').checked,
    };
    setBusy(btn, true, 'Starting…');
    try {
      const started = await api('/api/backtest', { method: 'POST', body: JSON.stringify(body) });
      toast('Backtest started: ' + started.label, 'ok');
      await refreshJobs(true);
      // poll this job until done, then render
      const poll = setInterval(async () => {
        try {
          const job = await api('/api/backtest/jobs/' + encodeURIComponent(started.job_id));
          await refreshJobs(false);
          if (job.status === 'done') {
            clearInterval(poll);
            renderBacktestResult($('#bt-result'), job.result);
            $('#bt-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
            toast('Backtest complete', 'ok');
          } else if (job.status === 'error') {
            clearInterval(poll);
            $('#bt-result').innerHTML = '<div class="card"><h2>Backtest failed</h2><p class="v-red">' + esc(job.error || 'unknown') + '</p></div>';
            toast(job.error || 'Backtest failed', 'err');
          }
        } catch (e) { /* keep polling */ }
      }, 1500);
    } catch (e) {
      toast(e.message, 'err');
    } finally {
      setBusy(btn, false);
    }
  });

  /* ============================== instruments view ============================== */
  let INST_LOADED = false;
  async function loadInstruments() {
    if (!INSTRUMENTS.length) await loadSymbolList();
    const tbody = $('#inst-table tbody');
    const clsSel = $('#inst-class');
    if (!INST_LOADED) {
      const classes = Array.from(new Set(INSTRUMENTS.map((i) => i.asset_class))).sort();
      classes.forEach((c) => {
        const o = document.createElement('option');
        o.value = c; o.textContent = c;
        clsSel.appendChild(o);
      });
      INST_LOADED = true;
    }
    function draw() {
      const q = $('#inst-search').value.trim().toLowerCase();
      const ac = clsSel.value;
      const rows = INSTRUMENTS.filter((i) =>
        (!ac || i.asset_class === ac) &&
        (!q || i.symbol.toLowerCase().includes(q) || i.name.toLowerCase().includes(q)));
      $('#inst-count').textContent = '· ' + rows.length + ' shown';
      tbody.innerHTML = rows.map((i) => '<tr><td><b>' + esc(i.symbol) + '</b>' +
        (i.aliases && i.aliases.length ? ' <span class="muted small">' + esc(i.aliases.join(', ')) + '</span>' : '') + '</td>' +
        '<td>' + esc(i.name) + '</td>' +
        '<td><span class="tag info">' + esc(i.asset_class) + '</span>' + (i.is_synthetic ? ' <span class="tag purple">synthetic</span>' : '') + '</td>' +
        '<td class="num">' + num(i.digits, 0) + '</td>' +
        '<td class="num">' + num(i.lot_size) + '</td>' +
        '<td class="num">' + num(i.min_lot) + '</td>' +
        '<td class="num">' + num(i.spread_points) + '</td></tr>').join('') ||
        '<tr><td colspan="7" class="muted">No instruments match.</td></tr>';
    }
    $('#inst-search').oninput = draw;
    clsSel.onchange = draw;
    draw();
  }

  /* ============================== settings view ============================== */
  let SETTINGS_LOADED = false;
  async function loadSettings() {
    if (SETTINGS_LOADED) return;
    const wrap = $('#settings-wrap');
    wrap.innerHTML = '<p class="muted">Loading…</p>';
    try {
      const s = await api('/api/settings');
      let html = '<div class="grid2">';
      html += '<div class="card"><h2>Environment</h2>' +
        '<div class="kv"><span class="k">environment</span><span class="v">' + esc(s.environment) + '</span></div>' +
        '<div class="kv"><span class="k">execution mode</span><span class="v"><span class="tag warn">' + esc(s.execution_mode) + '</span></span></div>' +
        '<div class="kv"><span class="k">default provider</span><span class="v">' + esc(s.default_provider) + '</span></div>' +
        '<div class="kv"><span class="k">data dir</span><span class="v">' + esc(s.data_dir) + '</span></div></div>';
      html += '<div class="card"><h2>Decision Thresholds</h2>' + kvHtml(s.thresholds) + '</div>';
      html += '<div class="card"><h2>Scoring Weights (total 100)</h2>' +
        Object.keys(s.scoring_weights || {}).map((k) => {
          const v = s.scoring_weights[k];
          return '<div class="score-row"><span class="nm">' + esc(k.replace(/_/g, ' ')) + '</span>' +
            '<span class="bar-track"><span class="bar-fill" style="display:block;width:' + v + '%"></span></span>' +
            '<span class="vl">' + num(v, 0) + '</span></div>';
        }).join('') + '</div>';
      html += '<div class="card"><h2>Risk Limits</h2>' + kvHtml({
        'account equity': s.risk.account_equity,
        'max risk / trade': s.risk.max_risk_per_trade_pct + ' %',
        'max daily loss': s.risk.max_daily_loss_pct + ' %',
        'max weekly loss': s.risk.max_weekly_loss_pct + ' %',
        'max drawdown': s.risk.max_drawdown_pct + ' %',
        'max open positions': s.risk.max_open_positions,
        'max consecutive losses': s.risk.max_consecutive_losses,
        'allow martingale': s.risk.allow_martingale,
      }) + '</div>';
      html += '</div>';
      html += '<div class="card"><h2>Regime → Allowed Strategy Families</h2><div class="table-wrap"><table class="tbl"><thead><tr><th>Regime</th><th>Allowed families</th></tr></thead><tbody>' +
        Object.keys(s.regime_map || {}).map((r) => '<tr><td><b>' + esc(r) + '</b></td><td>' +
          ((s.regime_map[r] || []).map((f) => '<span class="tag info">' + esc(f) + '</span>').join(' ') || '<span class="muted">none — no entries</span>') + '</td></tr>').join('') +
        '</tbody></table></div></div>';
      html += '<div class="card"><h2>News Settings</h2>' + kvHtml(s.news) + '</div>';
      wrap.innerHTML = html;
      SETTINGS_LOADED = true;
    } catch (e) {
      wrap.innerHTML = '<div class="card"><p class="v-red">' + esc(e.message) + '</p></div>';
    }
  }

  /* ============================== init ============================== */
  checkHealth();
  setInterval(checkHealth, 15000);
  loadSymbolList();
  loadScenarios();
  loadScoreWeights();
})();
