import { createPointerCapture } from './neural-pointer.js';
const status = document.querySelector('#status');
const start = document.querySelector('#start');
const mount = document.querySelector('#capture');
let widget;
async function request(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  const data = await r.json();
  if (!r.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Could not save this recording. Please try again.');
  return data;
}
function fail(error) {
  widget?.destroy();
  status.textContent = error.message || error;
  start.hidden = false;
}
async function refresh() {
  const s = await request('/api/pointer/status');
  status.textContent = s.enrolled ? `${s.username}: your trained pointer profile is ready.` : `${s.username}: ${s.count} of ${s.target} recordings saved.`;
  start.hidden = s.enrolled;
  start.textContent = `Start recording ${s.count + 1} of ${s.target}`;
}
start.addEventListener('click', async () => {
  start.hidden = true;
  try {
    const challenge = await request('/api/pointer/challenge', { kind: 'enroll' });
    widget?.destroy();
    widget = createPointerCapture(mount, { challenge, onError: fail, onDone: async recording => {
      try {
        await request('/api/pointer/capture', { kind: 'enroll', ...recording });
        mount.replaceChildren();
        await refresh();
      } catch (error) { fail(error); }
    } });
  } catch (error) { fail(error); }
});
refresh().catch(error => {
  status.textContent = error.message;
  const link = document.createElement('a');
  link.href = './index.html#login';
  link.textContent = 'Sign in to enroll your pointer';
  mount.replaceChildren(link);
});
window.addEventListener('pagehide', () => widget?.destroy());
