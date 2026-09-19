// welcome.js — the signed-in landing page. OWNER: Agent F (session).
//
// Trusts only the server: the name comes from /api/session (the HttpOnly cookie
// is invisible to script), and without a session the page bounces straight back
// to sign-in. The last attempt is fetched separately so the headline shows the
// same numbers the dashboard does.
import { $, decisionInfo, applyTheme, fmt, relTime } from './ui.js';

applyTheme();

const LOGIN = './index.html#login';
const card = $('welcome');

function toLogin() {
  location.replace(LOGIN);
}

async function getJson(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) throw Object.assign(new Error(`${path} → ${res.status}`), { status: res.status });
  return res.json();
}

function keystrokeLine(attempt) {
  const sig = (attempt.signals || []).find((s) => s.name === 'keystroke');
  if (!sig || sig.available === false) return 'not scored';
  return `${fmt(sig.score)} against a limit of ${fmt(sig.threshold)} — ${sig.flagged ? 'over' : 'within'} the limit`;
}

function renderAttempt(attempt) {
  const info = decisionInfo(attempt.decision);
  $('fact-decision').textContent = `${info.word}${attempt.created_at ? ` · ${relTime(attempt.created_at)}` : ''}`;
  $('fact-decision').className = `tone-${info.tone}`;
  $('fact-keystroke').textContent = keystrokeLine(attempt);
  $('fact-latency').textContent = Number.isFinite(attempt.latency_ms) ? `${fmt(attempt.latency_ms, 1)} ms` : '—';
  if (attempt.id) {
    $('to-dashboard').href = `./dashboard.html?user=${encodeURIComponent(attempt.username)}&attempt=${attempt.id}`;
  }
}

async function main() {
  let session;
  try {
    session = await getJson('/api/session');
  } catch {
    toLogin();
    return;
  }
  const user = session.username;
  document.title = `BioPrint — welcome back, ${user}`;
  $('welcome-title').textContent = `Welcome back, ${user}`;
  $('welcome-sub').textContent = 'Your password and your typing rhythm both matched. You are signed in.';
  $('fact-since').textContent = session.since ? relTime(session.since) : '—';
  $('to-dashboard').href = `./dashboard.html?user=${encodeURIComponent(user)}`;
  $('welcome-facts').hidden = false;
  $('welcome-actions').hidden = false;
  card.classList.add('ready');
  card.removeAttribute('aria-busy');

  try {
    const [attempt] = await getJson(`/api/attempts?user=${encodeURIComponent(user)}&limit=1`);
    if (attempt) renderAttempt(attempt);
  } catch {
    $('fact-decision').textContent = 'could not load the last attempt';
  }
}

$('signout').addEventListener('click', async (e) => {
  e.currentTarget.disabled = true;
  try {
    await fetch('/api/logout', { method: 'POST' });
  } catch { /* the cookie is gone either way once we leave */ }
  toLogin();
});

main();
