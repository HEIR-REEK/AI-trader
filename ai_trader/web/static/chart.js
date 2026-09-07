/* AI-Trader lightweight candlestick chart — zero dependencies.
 * Features: candles + volume, entry/SL/TP overlays, crosshair, drag-pan, wheel-zoom.
 * Usage: CandleChart.render(canvas, candles, overlays)
 *   candles: [{time, open, high, low, close, volume}]
 *   overlays: {entryLow, entryHigh, stop, tp1, tp2, tp3, direction: 'long'|'short'|null}
 */
(function (global) {
  'use strict';

  const UP = '#26c281', DOWN = '#ff5d5d';
  const GRID = 'rgba(35,44,58,.9)', AXIS_TXT = '#8b98ab';

  function cssVar(name, fallback) {
    try {
      const v = getComputedStyle(document.documentElement).getPropertyValue(name);
      return (v && v.trim()) || fallback;
    } catch (e) { return fallback; }
  }

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(280, rect.width), h = canvas.clientHeight || 380;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w, h };
  }

  function fmtPrice(p, digits) {
    if (p === null || p === undefined || isNaN(p)) return '—';
    if (digits === undefined) {
      const a = Math.abs(p);
      digits = a >= 1000 ? 2 : a >= 100 ? 2 : a >= 1 ? 4 : 5;
    }
    return Number(p).toFixed(digits);
  }

  function fmtTime(epoch) {
    const d = new Date(epoch * 1000);
    const p = (n) => String(n).padStart(2, '0');
    return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())} ${p(d.getUTCHours())}:${p(d.getUTCMinutes())}`;
  }

  function render(canvas, candles, overlays) {
    overlays = overlays || {};
    if (!candles || !candles.length) {
      const { ctx, w, h } = setupCanvas(canvas);
      ctx.fillStyle = AXIS_TXT;
      ctx.font = '13px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('No candle data', w / 2, h / 2);
      return;
    }

    // view state: last N bars visible
    const state = { end: candles.length, count: Math.min(candles.length, 140), hover: -1 };
    canvas._chartState = state;

    const PAD_R = 74, PAD_B = 24, PAD_T = 12, PAD_L = 8;

    function draw() {
      const { ctx, w, h } = setupCanvas(canvas);
      const W = w - PAD_R - PAD_L, H = h - PAD_T - PAD_B;
      const n = Math.max(10, Math.min(state.count, state.end));
      const start = Math.max(0, state.end - n);
      const end = Math.min(candles.length, state.end);
      const view = candles.slice(start, end);
      if (!view.length) return;

      let lo = Infinity, hi = -Infinity, vmax = 0;
      view.forEach((c) => {
        lo = Math.min(lo, c.low); hi = Math.max(hi, c.high);
        vmax = Math.max(vmax, c.volume || 0);
      });
      // include overlay levels in range
      ['entryLow', 'entryHigh', 'stop', 'tp1', 'tp2', 'tp3'].forEach((k) => {
        const v = overlays[k];
        if (typeof v === 'number' && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      });
      const pad = (hi - lo) * 0.08 || 1;
      lo -= pad; hi += pad;

      const volH = Math.round(H * 0.14);
      const priceH = H - volH - 8;
      const y = (p) => PAD_T + (1 - (p - lo) / (hi - lo)) * priceH;
      const bw = W / view.length;
      const x = (i) => PAD_L + bw * (i + 0.5);

      ctx.clearRect(0, 0, w, h);

      // horizontal grid + price labels
      ctx.font = '11px ui-monospace, monospace';
      ctx.textBaseline = 'middle';
      const rows = 6;
      for (let g = 0; g <= rows; g++) {
        const p = lo + (hi - lo) * (g / rows);
        const yy = Math.round(y(p)) + 0.5;
        ctx.strokeStyle = GRID;
        ctx.beginPath(); ctx.moveTo(PAD_L, yy); ctx.lineTo(PAD_L + W, yy); ctx.stroke();
        ctx.fillStyle = AXIS_TXT;
        ctx.fillText(fmtPrice(p), PAD_L + W + 8, yy);
      }
      // vertical grid + time labels
      const step = Math.max(1, Math.floor(view.length / 5));
      ctx.textAlign = 'center';
      for (let i = 0; i < view.length; i += step) {
        const xx = Math.round(x(i)) + 0.5;
        ctx.strokeStyle = GRID;
        ctx.beginPath(); ctx.moveTo(xx, PAD_T); ctx.lineTo(xx, PAD_T + H); ctx.stroke();
        ctx.fillStyle = AXIS_TXT;
        const d = new Date(view[i].time * 1000);
        const lbl = `${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')} ${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`;
        ctx.fillText(lbl, x(i), PAD_T + H + 12);
      }
      ctx.textAlign = 'left';

      // entry zone shading
      if (typeof overlays.entryLow === 'number' && typeof overlays.entryHigh === 'number') {
        const y1 = y(overlays.entryHigh), y2 = y(overlays.entryLow);
        ctx.fillStyle = overlays.direction === 'short' ? 'rgba(255,93,93,.10)' : 'rgba(52,209,123,.10)';
        ctx.fillRect(PAD_L, y1, W, Math.max(2, y2 - y1));
      }

      // volume bars
      view.forEach((c, i) => {
        if (!vmax) return;
        const vh = (c.volume / vmax) * volH;
        ctx.fillStyle = c.close >= c.open ? 'rgba(38,194,129,.35)' : 'rgba(255,93,93,.35)';
        const bx = x(i) - bw * 0.35;
        ctx.fillRect(bx, PAD_T + priceH + 8 + (volH - vh), Math.max(1, bw * 0.7), vh);
      });

      // candles
      view.forEach((c, i) => {
        const col = c.close >= c.open ? UP : DOWN;
        ctx.strokeStyle = col;
        ctx.fillStyle = col;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x(i), y(c.high));
        ctx.lineTo(x(i), y(c.low));
        ctx.stroke();
        const bTop = y(Math.max(c.open, c.close));
        const bBot = y(Math.min(c.open, c.close));
        const bwid = Math.max(1, Math.min(bw * 0.7, 18));
        ctx.fillRect(x(i) - bwid / 2, bTop, bwid, Math.max(1, bBot - bTop));
      });

      // overlay lines
      const lines = [
        ['entryLow', 'Entry', '#34d17b', 'dashed'],
        ['entryHigh', 'Entry', '#34d17b', 'dashed'],
        ['stop', 'SL', '#ff5d5d', 'solid'],
        ['tp1', 'TP1', '#4da3ff', 'dotted'],
        ['tp2', 'TP2', '#4da3ff', 'dotted'],
        ['tp3', 'TP3', '#a371f7', 'dotted'],
      ];
      const drawn = {};
      lines.forEach(([key, label, color, style]) => {
        const v = overlays[key];
        if (typeof v !== 'number' || !isFinite(v)) return;
        const yy = Math.round(y(v)) + 0.5;
        if (yy < PAD_T - 20 || yy > PAD_T + priceH + 20) return;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.2;
        ctx.setLineDash(style === 'dashed' ? [6, 4] : style === 'dotted' ? [2, 3] : []);
        ctx.beginPath(); ctx.moveTo(PAD_L, yy); ctx.lineTo(PAD_L + W, yy); ctx.stroke();
        ctx.setLineDash([]);
        if (!drawn[label + yy]) {
          drawn[label + yy] = 1;
          ctx.fillStyle = color;
          ctx.font = 'bold 11px ui-monospace, monospace';
          ctx.fillText(`${label} ${fmtPrice(v)}`, PAD_L + 6, yy - 9);
          ctx.font = '11px ui-monospace, monospace';
        }
      });

      // last price tag
      const last = view[view.length - 1];
      const ly = y(last.close);
      if (ly >= PAD_T && ly <= PAD_T + priceH) {
        const col = last.close >= last.open ? UP : DOWN;
        ctx.fillStyle = col;
        const tag = fmtPrice(last.close);
        ctx.font = 'bold 11px ui-monospace, monospace';
        const tw = ctx.measureText(tag).width + 12;
        ctx.fillRect(PAD_L + W + 2, ly - 10, Math.min(tw, PAD_R - 4), 20);
        ctx.fillStyle = '#07090d';
        ctx.fillText(tag, PAD_L + W + 8, ly);
        ctx.font = '11px ui-monospace, monospace';
      }

      // crosshair
      if (state.hover >= 0 && state.hover < view.length) {
        const i = state.hover, c = view[i];
        const xx = Math.round(x(i)) + 0.5, yy = Math.round(y(c.close)) + 0.5;
        ctx.strokeStyle = 'rgba(139,152,171,.6)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(xx, PAD_T); ctx.lineTo(xx, PAD_T + H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PAD_L, yy); ctx.lineTo(PAD_L + W, yy); ctx.stroke();
        ctx.setLineDash([]);
        const tip = canvas.parentElement.querySelector('.chart-tip');
        if (tip) {
          tip.style.display = 'block';
          tip.textContent = `${fmtTime(c.time)}\nO ${fmtPrice(c.open)}  H ${fmtPrice(c.high)}\nL ${fmtPrice(c.low)}  C ${fmtPrice(c.close)}`;
          const rect = canvas.getBoundingClientRect();
          const mx = state.mouseX || 0;
          tip.style.left = Math.min(rect.width - 190, mx + 18) + 'px';
          tip.style.top = '14px';
        }
      } else {
        const tip = canvas.parentElement.querySelector('.chart-tip');
        if (tip) tip.style.display = 'none';
      }

      // stash geometry for events
      canvas._geom = { PAD_L, W, start, viewLen: view.length, bw };
    }

    function barAt(clientX) {
      const rect = canvas.getBoundingClientRect();
      const g = canvas._geom;
      if (!g) return -1;
      const i = Math.floor((clientX - rect.left - g.PAD_L) / g.bw);
      return i >= 0 && i < g.viewLen ? i : -1;
    }

    // interactions (bind once)
    if (!canvas._bound) {
      canvas._bound = true;
      let dragging = false, dragX = 0, dragEnd = 0;
      canvas.addEventListener('mousemove', (e) => {
        const st = canvas._chartState;
        if (dragging) {
          const g = canvas._geom;
          if (g && g.bw > 0) {
            const dx = Math.round((dragX - e.clientX) / g.bw);
            st.end = Math.max(10, Math.min(candles.length, dragEnd + dx));
          }
        } else {
          st.hover = barAt(e.clientX);
          const rect = canvas.getBoundingClientRect();
          st.mouseX = e.clientX - rect.left;
        }
        draw();
      });
      canvas.addEventListener('mouseleave', () => {
        dragging = false;
        const st = canvas._chartState;
        if (st) { st.hover = -1; }
        draw();
      });
      canvas.addEventListener('mousedown', (e) => {
        dragging = true; dragX = e.clientX;
        dragEnd = canvas._chartState ? canvas._chartState.end : candles.length;
      });
      window.addEventListener('mouseup', () => { dragging = false; });
      canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const st = canvas._chartState;
        const factor = e.deltaY > 0 ? 1.15 : 0.87;
        st.count = Math.max(20, Math.min(candles.length, Math.round(st.count * factor)));
        draw();
      }, { passive: false });
      window.addEventListener('resize', () => { if (canvas.isConnected) draw(); });
    }

    draw();
  }

  global.CandleChart = { render, fmtPrice, fmtTime };
})(window);
