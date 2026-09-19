// app.js — STUB wiring. OWNER: Agent D (UI). Shows the whole API working end to end.
import { createCapture } from './capture.js';
import { createPointerCapture } from './pointer.js';
import { collectEnv } from './probe.js';

const $ = (id) => document.getElementById(id);
const card = $('card');
let origin = performance.now();
const keys = createCapture(card, { origin });
const pointer = createPointerCapture(card, { origin });
let submitVia = 'unknown';
let submitId = 'login'; // whichever button submits this sample

function rect(id) {
  const r = $(id).getBoundingClientRect();
  return [r.x, r.y, r.width, r.height];
}

async function sample() {
  return {
    keystrokes: keys.events,
    pointer: pointer.events,
    env: await collectEnv(),
    meta: {
      had_paste: keys.hadPaste,
      viewport: `${innerWidth}x${innerHeight}@${devicePixelRatio}`,
      targets: { submit: rect(submitId), password: rect('password'), username: rect('username') },
      submit_via: submitVia,
    },
  };
}

function reset() {
  origin = performance.now();
  keys.reset(origin);
  pointer.reset(origin);
  submitVia = 'unknown';
  $('password').value = '';
  $('password').focus();
}

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  $('out').textContent = JSON.stringify(await res.json(), null, 2);
}

const creds = () => ({ username: $('username').value, password: $('password').value });

$('password').addEventListener('keydown', (e) => { if (e.key === 'Enter') submitVia = 'enter'; });
$('login').addEventListener('pointerdown', () => { submitVia = 'click'; submitId = 'login'; });
$('enroll').addEventListener('pointerdown', () => { submitVia = 'click'; submitId = 'enroll'; });

card.addEventListener('submit', async (e) => {
  e.preventDefault();
  await post('/api/login', { ...creds(), sample: await sample() });
  reset();
});
$('enroll').addEventListener('click', async () => {
  await post('/api/enroll', { ...creds(), sample: await sample() });
  reset();
});
$('register').addEventListener('click', () => post('/api/register', creds()));
