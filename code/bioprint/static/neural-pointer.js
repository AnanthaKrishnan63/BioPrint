// Capture native mouse events without resampling; the server owns model scoring.
export function createPointerCapture(mount, { challenge, onDone, onError }) {
  const note = document.createElement('p');
  note.setAttribute('role', 'status');
  const stage = document.createElement('div');
  stage.style.cssText = 'position:relative;height:320px;min-width:260px;border:1px solid #aebbc8;border-radius:12px;overflow:hidden;touch-action:none;background:#f4f7fb';
  const target = document.createElement('button');
  target.type = 'button';
  target.textContent = 'Click';
  target.style.cssText = 'position:absolute;width:60px;height:48px;border-radius:24px;background:#2459db;color:white;border:0;cursor:pointer';
  stage.append(target);
  mount.replaceChildren(note, stage);
  let events = [], stopped = false, first = null, timer;
  const started = performance.now();
  function relocate() {
    target.style.left = `${10 + Math.random() * (stage.clientWidth - 80)}px`;
    target.style.top = `${10 + Math.random() * (stage.clientHeight - 68)}px`;
  }
  function destroy() {
    stopped = true;
    clearInterval(timer);
    for (const name of ['pointermove', 'pointerdown', 'pointerup']) stage.removeEventListener(name, capture);
  }
  function renderProgress() {
    if (stopped) return;
    if (first === null) {
      note.textContent = 'Move to the target to start the recording, then follow and click the targets naturally.';
      return;
    }
    const elapsed = performance.now() - started - first;
    const remaining = Math.max(0, Math.ceil((challenge.minimum_ms - elapsed) / 1000));
    const instruction = remaining > 0
      ? `At least ${remaining} more seconds. Follow and click the targets naturally.`
      : events.length < challenge.minimum_points
        ? 'Keep moving and clicking the targets while we collect enough movement.'
        : 'Click the target to finish.';
    note.textContent = `${Math.floor(elapsed / 1000)} seconds recorded · ${events.length} / ${challenge.minimum_points} movements minimum. ${instruction}`;
  }
  function capture(event) {
    if (stopped) return;
    if (event.pointerType !== 'mouse') {
      destroy();
      onError('This profile needs a mouse or trackpad. Touch verification uses the keypad.');
      return;
    }
    const t = performance.now() - started;
    if (first === null) first = t;
    events.push({ type: { pointermove: 'move', pointerdown: 'down', pointerup: 'up' }[event.type],
      t, x: event.clientX, y: event.clientY, pointer_type: 'mouse', trusted: event.isTrusted,
      buttons: event.buttons, target: event.target === target ? 'target' : 'stage' });
    if (events.length >= 20000) {
      destroy();
      onError('This recording filled up. Please start another recording.');
      return;
    }
    const elapsed = t - first;
    const enough = events.length >= challenge.minimum_points && elapsed >= challenge.minimum_ms;
    renderProgress();
    if (enough && event.type === 'pointerup') {
      destroy();
      note.textContent = 'Saving your movement…';
      onDone({ challenge_id: challenge.id, events });
    }
  }
  target.addEventListener('click', relocate);
  for (const name of ['pointermove', 'pointerdown', 'pointerup']) stage.addEventListener(name, capture);
  timer = setInterval(() => {
    if (performance.now() - started > (challenge.expires_in - 5) * 1000) {
      destroy();
      onError('The recording expired. Please start again.');
    }
    renderProgress();
  }, 250);
  renderProgress();
  relocate();
  return { destroy };
}
