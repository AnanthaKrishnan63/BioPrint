// ui.js — presentation helpers shared by the login page and the dashboard.
// OWNER: Agent D (UI). No capture logic here; no network calls here.

export const $ = (id, root = document) => root.getElementById(id);
export const el = (tag, props = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null) continue;
    if (k.includes('-')) node.setAttribute(k, String(v)); // data-*, aria-*
    else node[k] = v;
  }
  for (const k of kids.flat()) if (k != null) node.append(k);
  return node;
};

/** ?theme=light|dark (or a stored choice) forces the palette; otherwise the OS decides. */
export function applyTheme() {
  let t = new URLSearchParams(location.search).get('theme');
  try {
    if (t) localStorage.setItem('bioprint.theme', t);
    else t = localStorage.getItem('bioprint.theme');
  } catch { /* private mode */ }
  if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t;
}

export function svgIcon(name, cls = '') {
  const paths = {
    check: 'M20 6 9 17l-5-5',
    block: 'M12 3l7 3v6c0 4.4-3 8.2-7 9-4-.8-7-4.6-7-9V6l7-3zM9 9l6 6m0-6l-6 6',
    retype: 'M3 12a9 9 0 0 1 15-6.7L21 8M21 4v4h-4M21 12a9 9 0 0 1-15 6.7L3 16M3 20v-4h4',
    info: 'M12 3a9 9 0 1 1 0 18 9 9 0 0 1 0-18zM12 8h.01M11 12h1v5h1',
    fingerprint:
      'M12 3c2.5 0 4.7 1.2 6 3M4 8.5C5.3 5.8 8.4 4 12 4M6.5 19.5c-.9-1.8-1.3-3.6-1.3-5.5 0-3.3 3-6 6.8-6s6.8 2.7 6.8 6c0 1.2-.1 2.3-.4 3.4M9 20.4c-.6-1.6-1-3.4-1-6 0-2.2 1.8-4 4-4s4 1.8 4 4c0 2.6-.4 4.7-1.2 6.6M12 12.4v3.2c0 1.6.2 3.2.7 4.7',
  };
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  if (cls) svg.setAttribute('class', cls);
  const p = document.createElementNS(ns, 'path');
  p.setAttribute('d', paths[name] || paths.info);
  svg.append(p);
  return svg;
}

/** Everything the UI needs to show a decision, in one place. */
export const DECISION = {
  allow: { tone: 'allow', icon: 'check', word: 'Allowed', title: 'Welcome back' },
  block: { tone: 'block', icon: 'block', word: 'Blocked', title: 'Blocked — that is not how this account types' },
  retype: { tone: 'retype', icon: 'retype', word: 'Retype', title: 'Please type your password again' },
  // Not a verdict: the rhythm matched but the device is new, so more typings are asked for.
  step_up: { tone: 'stepup', icon: 'fingerprint', word: 'More rhythm', title: 'Let’s hear a bit more of your rhythm' },
  // Also not a verdict: the rhythm cannot settle it (new device class, or a score
  // too close to its limit), so the scrambled keypad is asked for instead.
  keypad: { tone: 'stepup', icon: 'fingerprint', word: 'Keypad check', title: 'Two quick target checks' },
  wrong_password: { tone: 'other', icon: 'info', word: 'Wrong password', title: 'That password is not right' },
  unknown_user: { tone: 'other', icon: 'info', word: 'No such account', title: 'No account with that name' },
  not_enrolled: { tone: 'other', icon: 'info', word: 'Not enrolled', title: 'This account has not finished enrolling' },
};
export const decisionInfo = (d) =>
  DECISION[d] || { tone: 'other', icon: 'info', word: d || 'unknown', title: d || 'unknown' };

export const SIGNAL_LABEL = {
  keystroke: 'Typing rhythm',
  pointer: 'Pointer movement',
  bot: 'Bot / replay',
  keypad: 'Keypad: search & cadence',
  keypad_motor: 'Keypad: movement',
};
export const SIGNAL_BLURB = {
  keystroke: 'How long each key is held and the gaps between them.',
  pointer: 'How the mouse travelled to the button and clicked it.',
  bot: 'Signs of a script: untrusted events, automation flags, replayed timing.',
  keypad: 'How fast you find and reach each digit on a keypad that is shuffled every time — a cognitive habit, so it compares across any device, phone included.',
  keypad_motor: 'How the pointer or finger actually travels to each key: path, speed, overshoot. Only comparable on the same kind of device as enrollment, so it is advisory.',
};
export const SIGNAL_VAR = {
  keystroke: 'var(--series-1)',
  pointer: 'var(--series-2)',
  bot: 'var(--series-3)',
  keypad: 'var(--series-5)',
  keypad_motor: 'var(--series-6)',
};
/** Column heads for the compact bars in the attempts table. */
export const SIGNAL_SHORT = {
  keystroke: 'Key', pointer: 'Ptr', bot: 'Bot', device: 'Dev', keypad: 'Pad', keypad_motor: 'Mov',
};

// ------------------------------------------------------------------ numbers
export const fmt = (n, digits = 1) =>
  !Number.isFinite(n) ? '—' : Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(digits);

export const isTiming = (feature = '') => /^(H|DD|UD)\./.test(feature) || /_ms$/.test(feature);
export const direction = (feature, up) =>
  isTiming(feature) ? (up ? '▲ slower' : '▼ faster') : up ? '▲ higher' : '▼ lower';

export function fmtValue(feature, v) {
  if (!Number.isFinite(v)) return '—';
  if (/^(H|DD|UD)\./.test(feature)) return `${Math.round(v)} ms`;
  return fmt(v, 2);
}

export const relTime = (iso) => {
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return '';
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 10) return 'just now';
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return new Date(t).toLocaleTimeString();
};

// ------------------------------------------------------------------ feature names
const KEY_WORDS = {
  Space: 'space', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']', Backslash: '\\',
  Semicolon: ';', Quote: "'", Comma: ',', Period: '.', Slash: '/', Backquote: '`',
  ShiftLeft: 'left Shift', ShiftRight: 'right Shift', CapsLock: 'Caps Lock', Enter: 'Enter',
  Backspace: 'Backspace', Tab: 'Tab',
};
export function keyLabel(code = '') {
  if (KEY_WORDS[code]) return KEY_WORDS[code];
  if (code.startsWith('Key')) return code.slice(3);
  if (code.startsWith('Digit')) return code.slice(5);
  if (code.startsWith('Numpad')) return `numpad ${code.slice(6)}`;
  return code;
}
const part = (s) => {
  const i = s.lastIndexOf('#');
  return i < 0 ? { key: keyLabel(s), pos: null } : { key: keyLabel(s.slice(0, i)), pos: Number(s.slice(i + 1)) + 1 };
};

/** "H.KeyT#2" -> "hold of “T” (key 3)". Plain English, for judges and users. */
export function featureLabel(name = '') {
  const [kind, ...rest] = name.split('.');
  if (kind === 'H' && rest.length === 1) {
    const a = part(rest[0]);
    return `hold of “${a.key}”${a.pos ? ` (key ${a.pos})` : ''}`;
  }
  if ((kind === 'DD' || kind === 'UD') && rest.length === 2) {
    const a = part(rest[0]);
    const b = part(rest[1]);
    const how = kind === 'DD' ? 'press to press' : 'release to press';
    return `“${a.key}” → “${b.key}” ${how}${a.pos ? ` (keys ${a.pos}–${b.pos})` : ''}`;
  }
  if (kind === 'pointer' || kind === 'bot') return rest.join(' ').replace(/_/g, ' ');
  return name.replace(/_/g, ' ');
}

// ------------------------------------------------------------------ gauges
/**
 * Score-vs-threshold bar. Signals are drawn one per gauge and never merged.
 * @param {object} sig contracts.SignalResult
 * @param {{compact?: boolean}} opts
 */
export function gauge(sig, { compact = false } = {}) {
  const name = SIGNAL_LABEL[sig.name] || sig.name;
  const available = sig.available !== false;
  const thr = Number(sig.threshold) || 1;
  const score = Number(sig.score) || 0;
  // Fixed scale: threshold sits at 55% of the track, so every gauge is read the
  // same way and a score far past the limit still shows as "off the end".
  const full = thr / 0.55;
  const pct = (v) => Math.max(0, Math.min(100, (v / full) * 100));

  const root = el('div', { className: `gauge${sig.flagged ? ' flagged' : ''}${available ? '' : ' na'}` });
  const swatch = el('span', { className: 'swatch' });
  swatch.style.background = SIGNAL_VAR[sig.name] || 'var(--muted)';
  const top = el(
    'div',
    { className: 'gauge-top' },
    el('span', { className: 'gauge-name' }, swatch, name),
    el('span', { className: 'gauge-val' }, available ? `${fmt(score)} / limit ${fmt(thr)}` : 'no data'),
  );
  const fill = el('div', { className: 'gauge-fill' });
  fill.style.width = available ? `${pct(score)}%` : '0%';
  const mark = el('div', { className: 'gauge-thr', title: `limit ${fmt(thr)}` });
  mark.style.left = `${pct(thr)}%`;
  const track = el('div', { className: 'gauge-track', role: 'img' }, fill, mark);
  track.setAttribute(
    'aria-label',
    available
      ? `${name}: score ${fmt(score)}, limit ${fmt(thr)}, ${sig.flagged ? 'over the limit' : 'within the limit'}`
      : `${name}: no data`,
  );
  root.append(top, track);
  if (!compact) {
    const state = !available ? 'na' : sig.flagged ? 'flag' : 'ok';
    const word = !available ? 'not available' : sig.flagged ? 'over the limit' : 'within the limit';
    root.append(el('span', { className: `gauge-status ${state}` }, `${state === 'flag' ? '▲' : state === 'ok' ? '●' : '○'} ${word}`));
  }
  return root;
}

const MINI_DEFAULT = ['keystroke', 'pointer', 'bot'];

/**
 * Tiny per-signal bars for the attempts table. Still one bar per signal, never
 * merged. `names` lets the caller add the signals that exist on this attempt
 * (device, keypad, keypad_motor) without every row growing to six bars.
 */
export function miniBars(signals = [], names = MINI_DEFAULT) {
  const wrap = el('div', { className: 'minibars' });
  for (const nm of names) {
    const sig = signals.find((s) => s.name === nm);
    const track = el('div', { className: 'gauge-track' });
    const nice = SIGNAL_LABEL[nm] || nm;
    let label = `${nice}: not measured`;
    if (sig) {
      const thr = Number(sig.threshold) || 1;
      const score = Number(sig.score) || 0;
      const full = thr / 0.55;
      const clamp = (v) => `${Math.max(0, Math.min(100, (v / full) * 100))}%`;
      const available = sig.available !== false;
      const fill = el('div', { className: 'gauge-fill' });
      fill.style.width = available ? clamp(score) : '0%';
      fill.style.background = sig.flagged ? 'var(--bad)' : available ? 'var(--good)' : 'transparent';
      const mark = el('div', { className: 'gauge-thr' });
      mark.style.left = clamp(thr);
      track.append(fill, mark);
      label = available
        ? `${nice}: ${fmt(score)} of limit ${fmt(thr)}${sig.flagged ? ', over the limit' : ''}`
        : `${nice}: no data`;
    }
    track.setAttribute('role', 'img');
    track.setAttribute('aria-label', label);
    wrap.append(el('div', { className: 'minibar', title: label }, el('span', {}, SIGNAL_SHORT[nm] || nm.slice(0, 3)), track));
  }
  return wrap;
}

export const pill = (text, title = '') => el('span', { className: 'pill', title }, text);

export function badge(decision) {
  const info = decisionInfo(decision);
  return el('span', { className: `badge ${info.tone}` }, svgIcon(info.icon), info.word);
}

/** Mark the current page in the Demo nav. */
export function markNav(page) {
  for (const a of document.querySelectorAll('.demo-nav a[data-page]')) {
    if (a.dataset.page === page) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  }
}
