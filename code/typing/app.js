// app.js — wires capture and metrics to the page. The throwaway layer:
// when this becomes the real study harness, this file is what gets replaced.

import { createCapture } from './capture.js';
import {
  computeMetrics,
  pairKeystrokes,
  flightTimes,
  estimateTimerResolution,
} from './metrics.js';

const $ = (id) => document.getElementById(id);
const box = $('box');
const status = $('status');

const capture = createCapture(box, render);
box.addEventListener('input', render); // catches deletions, which change netChars

const fmt = (x, digits = 0) =>
  Number.isFinite(x) ? x.toFixed(digits) : '–';

function render() {
  const m = computeMetrics(capture.events, box.value.length);

  $('netWpm').textContent = fmt(m.netWpm);
  $('grossWpm').textContent = fmt(m.grossWpm);
  $('keys').textContent = m.keystrokesGross;
  $('elapsed').textContent = fmt(m.durationMs / 1000, 1);
  $('netChars').textContent = m.netChars;

  $('dwell').textContent = fmt(m.meanDwellMs, 1);
  $('flight').textContent = fmt(m.meanFlightMs, 1);
  $('rollover').textContent = m.negativeFlights;
  $('backspace').textContent = m.backspaceCount;
  $('del').textContent = m.deleteCount;
  $('ratio').textContent = m.netChars ? fmt(m.correctionRatio, 2) : '–';

  const res = estimateTimerResolution(capture.events);
  $('timerRes').textContent = res === null ? '–' : `${res} ms`;

  $('pasteWarning').hidden = !capture.hadPaste;
  renderTable();
}

// The last handful of keystrokes, so the numbers in the research notes stop
// being abstract. Dwell lands around 70-110 ms for most people; flight varies
// far more, and goes negative when you overlap keys.
function renderTable() {
  const pairs = pairKeystrokes(capture.events);
  const flights = flightTimes(pairs);
  const start = Math.max(0, pairs.length - 15);

  const rows = pairs.slice(start).map((p, i) => {
    const idx = start + i;
    const flight = idx > 0 ? flights[idx - 1] : null;
    const cls = flight !== null && flight < 0 ? ' class="neg"' : '';
    const flightCell = flight === null ? '–' : flight.toFixed(1);
    return `<tr><td>${idx + 1}</td><td class="code">${p.code}</td>` +
      `<td>${p.dwell.toFixed(1)}</td><td${cls}>${flightCell}</td></tr>`;
  });

  $('tableBody').innerHTML =
    rows.reverse().join('') ||
    '<tr><td colspan="4" class="empty">start typing</td></tr>';
}

function sessionPayload() {
  return {
    subject: $('subject').value.trim() || null,
    user_agent: navigator.userAgent,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
    timer_res_ms: estimateTimerResolution(capture.events),
    had_paste: capture.hadPaste,
    net_chars: box.value.length,
    metrics: computeMetrics(capture.events, box.value.length),
    events: capture.events,
  };
}

function say(message, ok = true) {
  status.textContent = message;
  status.className = ok ? 'ok' : 'err';
}

$('save').addEventListener('click', async () => {
  if (capture.events.length === 0) return say('nothing to save yet', false);
  try {
    const res = await fetch('/api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sessionPayload()),
    });
    if (!res.ok) throw new Error(`server said ${res.status}`);
    const { id, keystrokes } = await res.json();
    say(`saved session #${id} — ${keystrokes} key events`);
  } catch (err) {
    say(`save failed: ${err.message}`, false);
  }
});

$('export').addEventListener('click', () => {
  const blob = new Blob([JSON.stringify(sessionPayload(), null, 2)], {
    type: 'application/json',
  });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `session-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
});

$('reset').addEventListener('click', () => {
  box.value = '';
  capture.reset();
  say('');
  box.focus();
});

render();
box.focus();
