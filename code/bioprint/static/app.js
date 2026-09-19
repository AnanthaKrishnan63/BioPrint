// app.js — the login / enrollment screen. OWNER: Agent D (UI).
//
// One card serves three modes (sign in, create account, enrollment reps). The
// fields and the submit button never move between modes: pointer features compare
// the approach to the button, so the button must sit in the same place every time.
import { createCapture } from './capture.js';
import { createPointerCapture } from './pointer.js';
import { collectEnv } from './probe.js';
import { $, el, gauge, svgIcon, decisionInfo, markNav, applyTheme, fmt } from './ui.js';

applyTheme();

const card = $('card');
const username = $('username');
const password = $('password');
const submit = $('submit');
const hint = $('hint');
const foot = $('foot');
const title = $('card-title');
const sub = $('card-sub');
const result = $('result');

// ---------------------------------------------------------------- capture
let origin = performance.now();
const keys = createCapture(card, { origin, onChange: updateHint });
// 'submit' is this page's button id, so it must be added to the tracked ids.
const pointer = createPointerCapture(card, { origin, ids: ['submit', 'reveal'] });
let submitVia = 'unknown';
let busy = false;

function rect(id) {
  const r = $(id).getBoundingClientRect();
  return [r.x, r.y, r.width, r.height];
}

async function buildSample() {
  return {
    keystrokes: keys.events.slice(),
    pointer: pointer.events.slice(),
    env: await collectEnv(),
    meta: {
      had_paste: keys.hadPaste,
      viewport: `${innerWidth}x${innerHeight}@${devicePixelRatio}`,
      // "submit" is always the button that sent THIS sample. There is only one.
      targets: { submit: rect('submit'), username: rect('username'), password: rect('password') },
      submit_via: submitVia,
    },
  };
}

// The form can be submitted while keys (or the mouse button) are still down: Enter
// fires its submit on keydown, so the last password key's dwell is still open. Wait
// for the matching releases — briefly — before serialising the sample.
const SETTLE_MS = 150;
let buttonHeld = false;

function heldKeyCount() {
  const held = new Set();
  for (const e of keys.events) {
    if (e.type === 'down') held.add(e.code);
    else held.delete(e.code);
  }
  return held.size;
}

function settle(maxMs = SETTLE_MS) {
  if (!heldKeyCount() && !buttonHeld) return Promise.resolve();
  return new Promise((resolve) => {
    const finish = () => {
      clearTimeout(timer);
      document.removeEventListener('keyup', check);
      document.removeEventListener('pointerup', check);
      resolve();
    };
    // Bubble-phase on the document: capture.js (on the card) and pointer.js
    // (capture phase) have both recorded the event by the time this runs.
    const check = () => {
      if (!heldKeyCount() && !buttonHeld) finish();
    };
    const timer = setTimeout(finish, maxMs);
    document.addEventListener('keyup', check);
    document.addEventListener('pointerup', check);
  });
}

function resetCapture({ clearPassword = true } = {}) {
  origin = performance.now();
  keys.reset(origin);
  pointer.reset(origin);
  submitVia = 'unknown';
  buttonHeld = false;
  if (clearPassword) password.value = '';
  updateHint();
}

function updateHint() {
  if (keys.hadPaste) {
    hint.className = 'hint-row warn';
    hint.textContent = 'Pasted — there is no rhythm in a paste, so this attempt gets flagged.';
    return;
  }
  const n = keys.events.filter((e) => e.type === 'down').length;
  hint.className = 'hint-row';
  hint.textContent = n ? `Capturing rhythm — ${n} keystroke${n === 1 ? '' : 's'} so far` : '';
}

// ---------------------------------------------------------------- network
async function api(path, body) {
  const t0 = performance.now();
  let res;
  try {
    res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (err) {
    return { ok: false, status: 0, data: { detail: 'server unreachable' }, rtt: performance.now() - t0 };
  }
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = {};
  }
  return { ok: res.ok, status: res.status, data, rtt: performance.now() - t0 };
}

// ---------------------------------------------------------------- modes
const POINTER_CLICKS = 5; // server needs this many clicked reps before it fits a pointer model
const enrollState = { target: 10, count: 0, clicks: 0, reps: 0, last: null, lastBad: false };
let mode = 'login';
// Who the server says is signed in on this browser (null if nobody). Only used
// for the "Continue" note under the sign-in form; the form itself never changes.
let sessionUser = null;
// A "step_up" decision in progress: the server wants STEP_UP_SAMPLES more typings
// of the password from this same form. Same fields, same button, same place —
// pointer features depend on the geometry not changing. Null when not stepping up.
const STEP_UP_SAMPLES = 3;
let stepUp = null; // { username, samples: [] }

const COPY = {
  login: {
    title: 'Sign in',
    sub: 'Type your password the way you normally do. We compare the rhythm, not just the characters.',
    button: 'Log in',
  },
  register: {
    title: 'Create your account',
    sub: 'Pick a username and password. You will then type that password ten times so we can learn its rhythm — timing only.',
    button: 'Create account',
  },
  enroll: {
    title: 'Teach BioPrint your rhythm',
    sub: 'Type the same password, then click the button — with the mouse, not Enter. We learn the timing and how you reach the button.',
    button: 'Click to save',
  },
};

function setMode(next, { keepResult = false } = {}) {
  mode = next;
  stepUp = null; // switching modes abandons a step-up; the server forgets it in 10 minutes
  const copy = COPY[next];
  title.textContent = copy.title;
  sub.textContent = copy.sub;
  submit.textContent = next === 'enroll' ? enrollButtonLabel() : copy.button;
  username.readOnly = next === 'enroll';
  markNav(next === 'login' ? 'login' : 'enroll');
  // Keep the address bar in step, so the Demo nav links always fire a change.
  const want = next === 'login' ? '#login' : '#enroll';
  if (location.hash !== want) history.replaceState(null, '', location.pathname + location.search + want);
  document.title = next === 'login' ? 'BioPrint — sign in' : 'BioPrint — enrollment';
  if (!keepResult) hideResult();
  renderFoot();
  resetCapture();
  (next === 'enroll' || username.value ? password : username).focus();
}

const enrollButtonLabel = () =>
  `${COPY.enroll.button} ${Math.min(enrollState.count + 1, enrollState.target)} of ${enrollState.target}`;

function renderFoot() {
  foot.replaceChildren();
  if (mode === 'enroll') {
    const bar = el('div', { className: 'progress', role: 'img' });
    bar.setAttribute('aria-label', `${enrollState.count} of ${enrollState.target} repetitions saved`);
    for (let i = 0; i < enrollState.target; i++) {
      bar.append(el('span', { className: i < enrollState.count ? 'done' : i === enrollState.count ? 'current' : '' }));
    }
    const label = el(
      'div',
      { className: 'progress-label' },
      el('span', {}, `${enrollState.count} of ${enrollState.target} saved`),
      el('span', {}, enrollState.clicks >= POINTER_CLICKS
        ? `${enrollState.clicks} clicked ✓`
        : `clicked ${enrollState.clicks} of ${POINTER_CLICKS} needed`),
    );
    const note = el('p', { className: enrollState.lastBad ? 'foot-note err' : 'foot-note', role: enrollState.lastBad ? 'alert' : null },
      enrollState.last || 'Tip: type it straight through — a backspace makes the sample unusable — and finish with a click.');
    foot.append(bar, label, note);
  } else if (mode === 'register') {
    foot.append(
      el('p', { className: 'foot-note' }, 'Already set up? ', el('a', { href: './index.html#login' }, 'Sign in'), '.'),
    );
  } else if (stepUp) {
    const n = stepUp.samples.length;
    const bar = el('div', { className: 'progress', role: 'img' });
    bar.setAttribute('aria-label', `${n} of ${STEP_UP_SAMPLES} extra typings heard`);
    for (let i = 0; i < STEP_UP_SAMPLES; i++) {
      bar.append(el('span', { className: i < n ? 'done' : i === n ? 'current' : '' }));
    }
    const label = el(
      'div',
      { className: 'progress-label' },
      el('span', {}, `${n} of ${STEP_UP_SAMPLES} more typings`),
      el('span', {}, n < STEP_UP_SAMPLES ? 'same password, your usual pace' : 'checking…'),
    );
    foot.append(bar, label, el('p', { className: 'foot-note' },
      'New device, so BioPrint listens a little longer. No code, no other device — just your rhythm.'));
  } else {
    foot.append(
      el('p', { className: 'foot-note' }, 'New here? ', el('a', { href: './index.html#enroll' }, 'Create an account'), ' and teach BioPrint your rhythm.'),
    );
    if (sessionUser) {
      foot.append(
        el('p', { className: 'foot-note' }, `Signed in as ${sessionUser}. `, el('a', { href: './welcome.html' }, 'Continue →')),
      );
    }
  }
}

// ---------------------------------------------------------------- result panel
function hideResult() {
  result.hidden = true;
  result.replaceChildren();
}

function showResult({ tone, icon, heading, body, reasons = [], signals = [], meta = [], actions = [] }) {
  result.className = `result ${tone}`;
  result.hidden = false;
  const head = el(
    'div',
    { className: 'result-head' },
    svgIcon(icon, 'result-icon'),
    el('div', {}, el('h2', {}, heading), body ? el('p', {}, body) : null),
  );
  result.replaceChildren(head);
  if (reasons.length) {
    result.append(el('ul', {}, reasons.map((r) => el('li', {}, r))));
  }
  if (signals.length) {
    const box = el('div', { className: 'sig-mini' });
    for (const s of signals) box.append(gauge(s, { compact: true }));
    result.append(box);
  }
  if (meta.length || actions.length) {
    result.append(el('div', { className: 'meta' }, meta, actions));
  }
  // Nudge the verdict into view if it fell below the fold. 'nearest' scrolls the
  // minimum needed; pointer features are measured against the button's rect at
  // submit time, so a small scroll cannot distort them.
  if (result.getBoundingClientRect().bottom > innerHeight) {
    result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

const pill = (text, title = '') => el('span', { className: 'pill', title }, text);

function renderLogin(data, rtt) {
  const info = decisionInfo(data.decision);
  const user = username.value.trim();
  const bodies = {
    allow: 'Your typing rhythm matched the profile enrolled for this account.',
    block: 'The password was correct, but the behaviour was not. BioPrint blocked this on behaviour alone — no code, no second device.',
    step_up: `Your rhythm matched, but this browser does not look like the one you enrolled on. Type your password ${STEP_UP_SAMPLES} more times below so we can hear more of it — the decision is made from typing alone.`,
    retype: 'There is nothing to compare when the password is corrected mid-way. Type it again, straight through.',
    wrong_password: 'Check the password and try again.',
    unknown_user: 'Create the account first, then enroll your rhythm.',
    not_enrolled: 'This account still needs its enrollment repetitions before behaviour can be checked.',
  };
  const meta = [];
  if (Number.isFinite(data.latency_ms)) meta.push(pill(`decision in ${fmt(data.latency_ms, 1)} ms`, 'server-side scoring time'));
  meta.push(pill(`${Math.round(rtt)} ms round trip`, 'browser to server and back'));
  const actions = [];
  if (data.attempt_id) {
    actions.push(el('a', { href: `./dashboard.html?user=${encodeURIComponent(user)}&attempt=${data.attempt_id}` }, 'See why on the dashboard →'));
  }
  if (data.decision === 'not_enrolled' || data.decision === 'unknown_user') {
    actions.push(el('a', { href: './index.html#enroll' }, 'Go to enrollment →'));
  }
  showResult({
    tone: info.tone,
    icon: info.icon,
    heading: data.decision === 'allow' ? `Welcome back, ${user}` : info.title,
    body: bodies[data.decision] || '',
    // The step-up card stays calm: the device details are on the dashboard.
    reasons: data.decision === 'step_up' ? [] : data.reasons || [],
    signals: (data.signals || []).filter((s) => s.available !== false || s.name === 'pointer'),
    meta,
    actions,
  });
}

// ---------------------------------------------------------------- actions
async function doRegister(creds) {
  const { ok, status, data } = await api('/api/register', creds);
  if (ok) {
    enrollState.target = data.enroll_target || 10;
    enrollState.count = 0;
    enrollState.clicks = 0;
    enrollState.reps = 0;
    enrollState.last = `Account created. Now type that same password ${enrollState.target} times.`;
    localStorage.setItem('bioprint.user', creds.username);
    setMode('enroll');
    return;
  }
  if (status === 409) {
    const res = await fetch(`/api/users/${encodeURIComponent(creds.username)}`);
    const st = res.ok ? await res.json() : null;
    if (st && !st.enrolled) {
      enrollState.target = st.enroll_target || 10;
      enrollState.count = st.enroll_count || 0;
      enrollState.last = 'Picking up where you left off — use the same password.';
      setMode('enroll');
      return;
    }
    showResult({
      tone: 'other',
      icon: 'info',
      heading: 'That username is taken',
      body: 'If the account is yours, sign in instead.',
      actions: [el('a', { href: './index.html#login' }, 'Go to sign in →')],
    });
    return;
  }
  showResult({ tone: 'other', icon: 'info', heading: 'Could not create the account', body: (data && data.detail) || 'Unknown error.' });
}

async function doEnrollRep(creds) {
  const via = submitVia;
  await settle();
  const sample = await buildSample();
  const { ok, data } = await api('/api/enroll', { ...creds, sample });
  resetCapture();
  if (!ok) {
    enrollState.last = (data && data.detail) || 'The server rejected that repetition.';
    renderFoot();
    return;
  }
  enrollState.target = data.target || enrollState.target;
  enrollState.count = data.count;
  enrollState.lastBad = !data.accepted;
  if (data.accepted) {
    enrollState.reps += 1;
    if (via === 'click') enrollState.clicks += 1;
    const left = enrollState.target - data.count;
    enrollState.last = left > 0 ? `Saved. ${left} to go — keep it natural.` : 'Saved — that was the last one.';
    if (via !== 'click') {
      enrollState.last += enrollState.clicks < POINTER_CLICKS
        ? ` Please click the button instead of pressing Enter — the pointer profile needs ${POINTER_CLICKS - enrollState.clicks} more clicked repetition(s).`
        : ' (That one was sent with Enter, so it carries no pointer data.)';
    }
  } else {
    const why = (data.reasons && data.reasons[0]) || 'that repetition could not be used';
    // A rejected repetition must be unmissable: a wrong password here would
    // otherwise look like a saved one.
    enrollState.last = `✕ Not saved — ${why}.`;
  }
  submit.textContent = enrollButtonLabel();
  renderFoot();

  if (data.enrolled) {
    const user = creds.username;
    showResult({
      tone: 'allow',
      icon: 'check',
      heading: 'Your rhythm is enrolled',
      body: `BioPrint fitted a profile for ${user} from ${data.count} repetitions. From now on a login has to match it.`,
      actions: [
        el('a', { href: `./dashboard.html?user=${encodeURIComponent(user)}` }, 'Open the dashboard →'),
      ],
    });
    setMode('login', { keepResult: true }); // never focuses or clicks the button itself
  }
}

// An allow is a real login: the server has set the session cookie. Give the
// verdict card a moment to be seen, then land on the signed-in page.
const WELCOME_DELAY_MS = 1200;
const WELCOME = './welcome.html';

async function doLogin(creds) {
  await settle();
  const sample = await buildSample();
  const { ok, data, rtt } = await api('/api/login', { ...creds, sample });
  resetCapture();
  if (!ok) {
    showResult({ tone: 'other', icon: 'info', heading: 'Could not reach the checker', body: (data && data.detail) || 'Unknown error.' });
    return;
  }
  afterVerdict(creds, data, rtt);
}

// Show a login verdict and act on it: allow lands on the signed-in page, block
// drops the session, step_up starts collecting more typings from this same form.
function afterVerdict(creds, data, rtt) {
  renderLogin(data, rtt);
  if (data.decision === 'allow') {
    sessionUser = creds.username;
    result.append(el('p', { className: 'result-next' }, 'Taking you in…'));
    setTimeout(() => { location.href = WELCOME; }, WELCOME_DELAY_MS);
  } else if (data.decision === 'block') {
    sessionUser = null; // the server cleared the cookie too
    renderFoot();
  } else if (data.decision === 'step_up') {
    stepUp = { username: creds.username, samples: [] };
    submit.textContent = stepUpButtonLabel();
    result.append(el('p', { className: 'result-next', id: 'stepup-progress' }, stepUpLine()));
    renderFoot();
    password.focus();
  }
}

const stepUpLine = () => `${stepUp.samples.length} of ${STEP_UP_SAMPLES} more typings heard`;
const stepUpButtonLabel = () =>
  `Type again ${Math.min(stepUp.samples.length + 1, STEP_UP_SAMPLES)} of ${STEP_UP_SAMPLES}`;

function endStepUp() {
  stepUp = null;
  submit.textContent = COPY.login.button;
  renderFoot();
}

// One more typing for a pending step-up. The first two are only collected; the
// third sends all three to the server, which adds the original attempt's score
// and judges the median of the four.
async function doStepUpRep(creds) {
  await settle();
  const sample = await buildSample();
  resetCapture();
  stepUp.samples.push(sample);
  const line = $('stepup-progress');
  if (line) line.textContent = stepUpLine();
  if (stepUp.samples.length < STEP_UP_SAMPLES) {
    submit.textContent = stepUpButtonLabel();
    renderFoot();
    password.focus();
    return;
  }
  renderFoot();
  const { ok, status, data, rtt } = await api('/api/login/stepup', { ...creds, samples: stepUp.samples });
  if (!ok) {
    endStepUp();
    if (status === 409) {
      showResult({ tone: 'other', icon: 'info', heading: 'That step-up has lapsed',
        body: 'Sign in again and we will pick it up from there.' });
    } else {
      showResult({ tone: 'other', icon: 'info', heading: 'Could not reach the checker', body: (data && data.detail) || 'Unknown error.' });
    }
    return;
  }
  if (data.decision === 'retype' || data.decision === 'wrong_password') {
    // The step-up is still pending on the server: start the three again.
    stepUp.samples = [];
    renderLogin(data, rtt);
    result.append(el('p', { className: 'result-next', id: 'stepup-progress' }, `Start again — ${stepUpLine()}`));
    submit.textContent = stepUpButtonLabel();
    renderFoot();
    password.focus();
    return;
  }
  stepUp = null;
  submit.textContent = COPY.login.button;
  afterVerdict(creds, data, rtt);
}

card.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (busy) return;
  const creds = { username: username.value.trim(), password: password.value };
  if (!creds.username || !creds.password) {
    hint.className = 'hint-row warn';
    hint.textContent = 'Both a username and a password are needed.';
    (creds.username ? password : username).focus();
    return;
  }
  busy = true;
  submit.setAttribute('aria-busy', 'true');
  try {
    if (mode === 'register') await doRegister(creds);
    else if (mode === 'enroll') await doEnrollRep(creds);
    else if (stepUp && stepUp.username === creds.username) await doStepUpRep(creds);
    else {
      if (stepUp) endStepUp(); // a different username is a fresh sign-in
      await doLogin(creds);
    }
  } finally {
    busy = false;
    submit.removeAttribute('aria-busy');
  }
});

// How the sample was submitted: a click on the button, or Enter in a field.
submit.addEventListener('pointerdown', () => { submitVia = 'click'; buttonHeld = true; });
document.addEventListener('pointerup', () => { buttonHeld = false; });
for (const field of [username, password]) {
  field.addEventListener('keydown', (e) => { if (e.key === 'Enter') submitVia = 'enter'; });
}

$('reveal').addEventListener('click', (e) => {
  const on = password.type === 'password';
  password.type = on ? 'text' : 'password';
  e.currentTarget.setAttribute('aria-pressed', String(on));
  e.currentTarget.textContent = on ? 'Hide' : 'Show';
  password.focus();
});

// ---------------------------------------------------------------- routing
function route() {
  const hash = location.hash.replace('#', '');
  if (hash === 'enroll') {
    if (mode !== 'enroll') setMode('register');
  } else if (mode !== 'login') {
    setMode('login');
  }
}
addEventListener('hashchange', route);
// The nav links point at this same page; handle them directly so a click always
// switches mode, even when the hash already matches.
for (const link of document.querySelectorAll('.demo-nav a[data-page]')) {
  if (link.dataset.page === 'dashboard') continue;
  link.addEventListener('click', (e) => {
    e.preventDefault();
    if (link.dataset.page === 'enroll') setMode(mode === 'enroll' ? 'enroll' : 'register');
    else setMode('login');
  });
}

const remembered = localStorage.getItem('bioprint.user');
if (remembered) {
  username.value = remembered;
  // A remembered name is a suggestion: the first click selects it, so typing replaces.
  username.addEventListener('focus', function once() {
    username.select();
    username.removeEventListener('focus', once);
  });
}
setMode(location.hash.replace('#', '') === 'enroll' ? 'register' : 'login');
username.addEventListener('change', () => {
  if (username.value.trim()) localStorage.setItem('bioprint.user', username.value.trim());
});

// Already signed in? Say so under the form. Never auto-redirect: the demo needs
// to log in again and again, and a blocked attempt must be possible from here.
fetch('/api/session')
  .then((r) => (r.ok ? r.json() : null))
  .then((s) => {
    if (!s || !s.username) return;
    sessionUser = s.username;
    if (mode === 'login') renderFoot();
  })
  .catch(() => {});
