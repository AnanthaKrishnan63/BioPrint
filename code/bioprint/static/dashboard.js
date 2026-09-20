// dashboard.js — live view of login attempts. OWNER: Agent D (UI).
//
// The six signals (keystroke, pointer, bot, device, keypad, keypad_motor) are shown
// side by side and never added together: a bot flag and an unfamiliar rhythm are
// different accusations, "different device" is advisory context for the behaviour
// verdict rather than a verdict itself, and the keypad's two halves are deliberately
// split — the cognitive half crosses device classes, the motor half does not.
// A signal a given attempt never produced is drawn as "not measured", not as zero.
import {
  $, el, gauge, miniBars, badge, svgIcon, decisionInfo, markNav, featureLabel,
  fmt, fmtValue, relTime, direction, applyTheme, SIGNAL_LABEL, SIGNAL_BLURB, SIGNAL_VAR,
} from './ui.js';

const POLL_MS = 2000;
const NS = 'http://www.w3.org/2000/svg';
const SERIES = ['keystroke', 'pointer', 'bot', 'device', 'keypad', 'keypad_motor', 'pointer_neural'];
// Bars in the attempts table: the three that every attempt has, plus whatever
// else that attempt actually carries. Six bars on every row would be noise.
const MINI_BASE = ['keystroke', 'pointer', 'bot'];
const miniNames = (signals = []) =>
  MINI_BASE.concat(['device', 'keypad', 'keypad_motor', 'pointer_neural'].filter((n) => signals.some((s) => s.name === n)));
// The chart labels each line at its right-hand end, so those names must be short.
const CHART_LABEL = { keypad: 'Keypad', keypad_motor: 'Keypad move' };
const chartLabel = (n) => CHART_LABEL[n] || SIGNAL_LABEL[n] || n;

// The device gauge (device agent) extends ui.js's tables here rather than editing
// them: gauge() and the legend read SIGNAL_LABEL/SIGNAL_VAR by signal name.
SIGNAL_LABEL.device = 'Device';
SIGNAL_BLURB.device = 'Bits of browser identity that differ from enrollment: fonts, GPU, screen, time zone… A browser update is under the limit; another laptop is far over.';
SIGNAL_VAR.device = 'var(--series-4)';

const userInput = $('user');
const followBtn = $('follow');
const pauseBtn = $('pause');
const live = $('live');
const liveText = $('live-text');
const tip = $('tip');

const params = new URLSearchParams(location.search);
let user = params.get('user') || localStorage.getItem('bioprint.user') || '';
let selectedId = Number(params.get('attempt')) || null;
let follow = !selectedId;
let paused = false;
let attempts = [];
let lastKey = '';
let seen = new Set();
let first = true;
// /api/users/<name> for whoever the shown attempt belongs to: enrollment state
// and habits, which are per account rather than per attempt.
let status = null;
let statusUser = '';
let statusAt = 0;

userInput.value = user;
followBtn.setAttribute('aria-pressed', String(follow));
markNav('dashboard');
applyTheme();

// ---------------------------------------------------------------- data
async function poll() {
  if (paused) return;
  const q = new URLSearchParams({ limit: '25' });
  if (user) q.set('user', user);
  try {
    const res = await fetch(`/api/attempts?${q}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const rows = await res.json();
    live.className = 'live';
    liveText.textContent = `live · every ${POLL_MS / 1000}s`;
    const key = JSON.stringify(rows.map((r) => [r.id, r.label]));
    if (key !== lastKey) {
      lastKey = key;
      attempts = rows;
      render();
    } else {
      // times are relative; keep them honest without a full re-render
      for (const node of document.querySelectorAll('[data-when]')) node.textContent = relTime(node.dataset.when);
    }
  } catch (err) {
    live.className = 'live err';
    liveText.textContent = `server unreachable (${err.message})`;
  }
}

const STATUS_TTL_MS = 10_000;

/** Account habits, refreshed lazily: they change at enrollment speed, not poll speed. */
async function refreshStatus(name) {
  if (!name) {
    if (status) { status = null; statusUser = ''; renderHabits(); }
    return;
  }
  if (name === statusUser && Date.now() - statusAt < STATUS_TTL_MS) return;
  statusUser = name;
  statusAt = Date.now();
  let data = null;
  try {
    const res = await fetch(`/api/users/${encodeURIComponent(name)}`);
    data = res.ok ? await res.json() : null;
  } catch {
    data = null; /* the attempts poll already reports an unreachable server */
  }
  if (name !== statusUser) return; // a later name won the race
  status = data;
  renderHabits();
}

/** Pills under the verdict: what this account has taught BioPrint so far. */
function renderHabits() {
  const box = $('habits');
  if (!box) return;
  box.replaceChildren();
  const st = status;
  if (!st || !st.username) return;
  const add = (text, title) => box.append(el('span', { className: 'pill', title }, text));
  add(`${st.enroll_count ?? 0} of ${st.enroll_target ?? 10} typing reps`,
    'password repetitions that fitted the rhythm model');
  if (st.pointer_enrolled) add('pointer profile ✓', 'enough clicked repetitions to fit a pointer model');
  // Keypad habits (keypad agent): how far the scrambled-keypad enrollment got,
  // and on what kind of device — a desktop template cannot score a phone.
  if (Number.isFinite(st.keypad_target)) {
    const n = st.keypad_count || 0;
    add(st.keypad_enrolled ? `keypad ✓ ${n} of ${st.keypad_target}` : `keypad ${n} of ${st.keypad_target}`,
      'scrambled-keypad captchas solved at enrollment');
  }
  if (st.keypad_device_class && st.keypad_device_class !== 'unknown') {
    add(`keypad taught on ${st.keypad_device_class}`,
      'the device class the keypad profile was enrolled on; the movement half only compares within it');
  }
  const h = st.habits;
  if (h && h.keypad_runs) {
    add(`keypad ${Math.round(h.keypad_mean_duration_ms || 0)} ms per run`, 'mean time from layout shown to the last tap');
    const slips = h.keypad_errors || 0;
    add(`${slips} keypad slip${slips === 1 ? '' : 's'}`, 'wrong digits tapped and backspaced across enrollment');
  }
  if (h && h.submissions) {
    add(`corrects ${Math.round((h.correction_rate || 0) * 100)}%`, 'share of submissions where the password was fixed mid-way');
    add(`wrong password ${Math.round((h.wrong_password_rate || 0) * 100)}%`, 'share of submissions with the wrong password');
  }
}

async function setLabel(id, label) {
  try {
    await fetch(`/api/attempts/${id}/label`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    });
  } catch {
    /* the next poll shows the truth */
  }
  lastKey = '';
  poll();
}

const current = () => attempts.find((a) => a.id === selectedId) || attempts[0] || null;

// ---------------------------------------------------------------- hero
function renderVerdict(a) {
  const box = $('verdict');
  box.replaceChildren();
  if (!a) {
    box.append(el('p', { className: 'empty' }, 'No attempts yet. Sign in on the login page and this fills in.'));
    return;
  }
  const info = decisionInfo(a.decision);
  const head = el('div', { className: `verdict-badge ${info.tone}` }, svgIcon(info.icon), info.word.toUpperCase());
  const when = el('span', { 'data-when': a.created_at }, relTime(a.created_at));
  const subline = el('div', { className: 'verdict-sub' }, `${a.username} · attempt #${a.id} · `, when);
  const stats = el(
    'div',
    { className: 'verdict-stats' },
    el('span', { className: 'pill', title: 'server-side scoring time' }, `${fmt(a.latency_ms, 1)} ms decision`),
    a.label ? el('span', { className: 'pill' }, `labelled ${a.label}`) : null,
    follow ? el('span', { className: 'pill' }, 'following latest') : el('span', { className: 'pill' }, 'pinned'),
  );
  const reasons = el('ul', {}, (a.reasons || []).map((r) => el('li', {}, r)));
  box.append(head, subline, stats, reasons.childElementCount ? reasons : null);
}

function contribTable(sig) {
  const rows = (sig.contributions || [])
    .slice()
    .sort((x, y) => Math.abs(y.deviation) - Math.abs(x.deviation))
    .slice(0, 4);
  if (!rows.length) return null;
  const max = Math.max(...rows.map((r) => Math.abs(r.deviation)), 1e-9);
  const table = el('table', { className: 'contrib' });
  table.append(
    el(
      'thead',
      {},
      el('tr', {}, el('th', {}, 'What moved it'), el('th', { className: 'num' }, 'this time'), el('th', { className: 'num' }, 'usual'), el('th', {}, 'shift')),
    ),
  );
  const body = el('tbody');
  for (const c of rows) {
    const up = c.value > c.expected;
    const bar = el('span', { className: 'shift' });
    bar.style.width = `${Math.max(6, (Math.abs(c.deviation) / max) * 60)}px`;
    bar.style.background = sig.flagged ? 'var(--bad)' : 'var(--line-strong)';
    body.append(
      el(
        'tr',
        {},
        el('td', { title: c.feature }, featureLabel(c.feature)),
        el('td', { className: `num ${up ? 'up' : 'down'}` }, fmtValue(c.feature, c.value)),
        el('td', { className: 'num' }, fmtValue(c.feature, c.expected)),
        el('td', {}, bar, ' ', direction(c.feature, up)),
      ),
    );
  }
  table.append(body);
  return table;
}

/** Device contributions are "attribute differs, weighing N bits", not observed-vs-usual
 *  numbers, so they get their own table: one row per mismatched attribute. */
function deviceTable(sig) {
  const rows = (sig.contributions || []).slice().sort((x, y) => y.deviation - x.deviation);
  if (!rows.length) return null;
  const max = Math.max(...rows.map((r) => r.deviation), 1e-9);
  const table = el('table', { className: 'contrib' });
  table.append(el('thead', {}, el('tr', {}, el('th', {}, 'What differs'), el('th', { className: 'num' }, 'bits'), el('th', {}, 'weight'))));
  const body = el('tbody');
  for (const c of rows) {
    const bar = el('span', { className: 'shift' });
    bar.style.width = `${Math.max(6, (c.deviation / max) * 60)}px`;
    bar.style.background = sig.flagged ? 'var(--bad)' : 'var(--line-strong)';
    body.append(el('tr', {},
      el('td', { title: c.feature }, featureLabel(c.feature)),
      el('td', { className: 'num up' }, fmt(c.deviation)),
      el('td', {}, bar)));
  }
  table.append(body);
  return table;
}

function renderSignals(a) {
  const box = $('signals');
  box.replaceChildren();
  for (const name of SERIES) {
    const sig = (a && (a.signals || []).find((s) => s.name === name)) || {
      name, available: false, score: 0, threshold: 1, flagged: false, reasons: ['not measured for this attempt'],
    };
    const available = sig.available !== false;
    const card = el('div', { className: 'signal-card' });
    card.append(gauge(sig));
    if (available) {
      const unit = name === 'device' ? ' bits' : '';
      card.append(el('div', { className: 'big' }, fmt(sig.score), el('small', {}, `${unit} of limit ${fmt(sig.threshold)}`)));
    }
    card.append(el('p', { className: 'gauge-val', style: 'margin:0' }, SIGNAL_BLURB[name]));
    if (sig.reasons && sig.reasons.length) {
      card.append(el('ul', {}, sig.reasons.map((r) => el('li', {}, r))));
    }
    const t = name === 'device' ? deviceTable(sig) : contribTable(sig);
    if (t) card.append(t);
    box.append(card);
  }
}

// ---------------------------------------------------------------- chart
const s = (tag, attrs = {}) => {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
};

function renderChart(rows) {
  const wrap = $('chart-wrap');
  for (const old of wrap.querySelectorAll('svg, .empty')) old.remove();
  const legend = $('legend');
  legend.replaceChildren(
    ...SERIES.map((n) => {
      const dot = el('i');
      dot.style.background = SIGNAL_VAR[n];
      return el('span', {}, dot, SIGNAL_LABEL[n]);
    }),
    el('span', {}, el('i', { style: 'background:var(--ink)' }), 'its limit (1.0)'),
  );
  const data = rows.slice().reverse().filter((r) => (r.signals || []).length);
  if (!data.length) {
    wrap.append(el('p', { className: 'empty' }, 'Scored attempts appear here — one column per attempt.'));
    return;
  }

  const W = 900, H = 260, L = 44, R = 118, T = 16, B = 34;
  const ratios = data.flatMap((r) => (r.signals || []).filter((x) => x.available !== false).map((x) => (Number(x.score) || 0) / (Number(x.threshold) || 1)));
  const cap = Math.max(2, Math.min(3.2, Math.max(...ratios, 0) * 1.15));
  const x = (i) => (data.length === 1 ? (L + W - R) / 2 : L + (i * (W - L - R)) / (data.length - 1));
  const y = (v) => H - B - (Math.min(v, cap) / cap) * (H - T - B);

  const svg = s('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Per-signal score divided by its limit, for each recent attempt. 1.0 is the limit; above it the signal is flagged.' });

  // y grid
  for (const v of [0, 1, 2, 3].filter((v) => v <= cap)) {
    svg.append(s('line', { x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: v === 1 ? 'var(--ink)' : 'var(--line)', 'stroke-width': v === 1 ? 2 : 1, 'stroke-dasharray': v === 1 ? '6 5' : '' }));
    const t = s('text', { x: L - 10, y: y(v) + 4, 'text-anchor': 'end', fill: 'var(--muted)', 'font-size': 13 });
    t.textContent = v.toFixed(1);
    svg.append(t);
  }
  const labels = [{ name: 'limit', x: W - R + 8, y: y(1) + 4, muted: true }];
  for (const name of SERIES) {
    const pts = data
      .map((r, i) => {
        const sig = (r.signals || []).find((x) => x.name === name);
        if (!sig || sig.available === false) return null;
        const ratio = (Number(sig.score) || 0) / (Number(sig.threshold) || 1);
        return { i, r, sig, ratio, cx: x(i), cy: y(ratio), over: ratio > cap };
      })
      .filter(Boolean);
    if (!pts.length) continue;
    svg.append(s('polyline', {
      points: pts.map((p) => `${p.cx},${p.cy}`).join(' '),
      fill: 'none', stroke: SIGNAL_VAR[name], 'stroke-width': 2, 'stroke-linejoin': 'round', opacity: .55,
    }));
    for (const p of pts) {
      const mark = p.over
        ? s('polygon', { points: `${p.cx},${p.cy - 7} ${p.cx - 6},${p.cy + 4} ${p.cx + 6},${p.cy + 4}` })
        : s('circle', { cx: p.cx, cy: p.cy, r: p.r.id === selectedId ? 7 : 5 });
      mark.setAttribute('fill', SIGNAL_VAR[name]);
      mark.setAttribute('stroke', 'var(--surface)');
      mark.setAttribute('stroke-width', 2);
      mark.setAttribute('tabindex', '0');
      mark.setAttribute('role', 'button');
      mark.setAttribute('aria-label', `${SIGNAL_LABEL[name]} ${fmt(p.ratio, 2)} times its limit on attempt ${p.r.id} (${p.r.decision})`);
      mark.style.cursor = 'pointer';
      const show = (ev) => showTip(ev, `#${p.r.id} ${p.r.decision} · ${SIGNAL_LABEL[name]} ${fmt(p.sig.score)} / limit ${fmt(p.sig.threshold)}`);
      mark.addEventListener('pointerenter', show);
      mark.addEventListener('focus', show);
      mark.addEventListener('pointerleave', hideTip);
      mark.addEventListener('blur', hideTip);
      mark.addEventListener('click', () => select(p.r.id));
      svg.append(mark);
    }
    const last = pts[pts.length - 1];
    labels.push({ name, x: last.cx + 10, y: last.cy + 4 });
  }

  // Direct labels, nudged apart so two series that finish at the same height stay legible.
  labels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < labels.length; i++) {
    if (labels[i].y - labels[i - 1].y < 17) labels[i].y = labels[i - 1].y + 17;
  }
  for (const l of labels) {
    const t = s('text', { x: l.x, y: Math.min(l.y, H - B), fill: l.muted ? 'var(--muted)' : 'var(--ink-2)', 'font-size': 13, 'font-weight': 600 });
    t.textContent = l.name === 'limit' ? 'limit' : chartLabel(l.name);
    svg.append(t);
  }

  // x labels: first, last and the selected attempt
  data.forEach((r, i) => {
    if (i !== 0 && i !== data.length - 1 && r.id !== selectedId) return;
    const t = s('text', { x: x(i), y: H - 10, 'text-anchor': 'middle', fill: r.id === selectedId ? 'var(--ink)' : 'var(--muted)', 'font-size': 13 });
    t.textContent = `#${r.id}`;
    svg.append(t);
  });
  wrap.append(svg);
}

function showTip(ev, text) {
  const wrap = $('chart-wrap');
  const r = ev.currentTarget.getBoundingClientRect();
  const box = wrap.getBoundingClientRect();
  tip.textContent = text;
  tip.hidden = false;
  tip.style.left = `${r.left + r.width / 2 - box.left}px`;
  tip.style.top = `${r.top - box.top}px`;
}
const hideTip = () => { tip.hidden = true; };

// ---------------------------------------------------------------- table
const LABELS = ['genuine', 'impostor', 'bot'];

function renderTable(rows) {
  const wrap = $('table-wrap');
  wrap.replaceChildren();
  $('count').textContent = rows.length ? `· newest first` : '';
  if (!rows.length) {
    wrap.append(el('p', { className: 'empty' }, 'Nothing yet for this account.'));
    return;
  }
  const table = el('table', { className: 'attempts' });
  table.append(
    el(
      'thead',
      {},
      el(
        'tr',
        {},
        el('th', {}, 'When'),
        el('th', {}, 'Account'),
        el('th', {}, 'Decision'),
        el('th', {}, 'Signals vs limit'),
        el('th', { className: 'hide-sm' }, 'Why'),
        el('th', { className: 'hide-sm' }, 'Latency'),
        el('th', {}, 'Ground truth'),
      ),
    ),
  );
  const body = el('tbody');
  for (const a of rows) {
    const tr = el('tr', { tabIndex: 0 });
    tr.setAttribute('aria-selected', String(a.id === (current() || {}).id));
    if (!first && !seen.has(a.id)) tr.classList.add('fresh');
    seen.add(a.id);
    const when = el('span', { 'data-when': a.created_at }, relTime(a.created_at));
    const labelCell = el('div', { className: 'labels' });
    for (const l of LABELS) {
      const on = a.label === l;
      const b = el('button', { type: 'button' }, l);
      b.setAttribute('aria-pressed', String(on));
      b.addEventListener('click', (ev) => {
        ev.stopPropagation();
        setLabel(a.id, on ? null : l);
      });
      labelCell.append(b);
    }
    tr.append(
      el('td', {}, when),
      el('td', {}, a.username),
      el('td', {}, badge(a.decision)),
      el('td', {}, miniBars(a.signals || [], miniNames(a.signals || []))),
      el('td', { className: 'reasons-cell hide-sm' }, (a.reasons || []).slice(0, 2).join('; ')),
      el('td', { className: 'hide-sm' }, `${fmt(a.latency_ms, 1)} ms`),
      el('td', {}, labelCell),
    );
    tr.addEventListener('click', () => select(a.id));
    tr.addEventListener('keydown', (ev) => {
      if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        select(a.id);
      }
    });
    body.append(tr);
  }
  table.append(body);
  wrap.append(table);
}

// ---------------------------------------------------------------- glue
function select(id) {
  selectedId = id;
  follow = false;
  followBtn.setAttribute('aria-pressed', 'false');
  render();
}

function render() {
  if (follow) selectedId = attempts.length ? attempts[0].id : null;
  const a = current();
  refreshStatus(user || (a && a.username) || '');
  renderHabits();
  renderVerdict(a);
  renderSignals(a);
  renderChart(attempts);
  renderTable(attempts);
  first = false;
}

followBtn.addEventListener('click', () => {
  follow = !follow;
  followBtn.setAttribute('aria-pressed', String(follow));
  render();
});
pauseBtn.addEventListener('click', () => {
  paused = !paused;
  pauseBtn.setAttribute('aria-pressed', String(paused));
  pauseBtn.textContent = paused ? 'Resume' : 'Pause';
  live.className = paused ? 'live paused' : 'live';
  liveText.textContent = paused ? 'paused' : `live · every ${POLL_MS / 1000}s`;
  if (!paused) poll();
});

let debounce;
userInput.addEventListener('input', () => {
  clearTimeout(debounce);
  debounce = setTimeout(() => {
    user = userInput.value.trim();
    if (user) localStorage.setItem('bioprint.user', user);
    lastKey = '';
    seen = new Set();
    first = true;
    poll();
  }, 300);
});
addEventListener('resize', () => renderChart(attempts));

render();
poll();
setInterval(poll, POLL_MS);
