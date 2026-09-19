// keypad.js — the scrambled-keypad captcha widget. OWNER: Agent M (UI).
//
// A 3x2 keypad of the digits 0..4 (one blank cell) in a layout the server
// shuffles for every challenge, plus a backspace key in a FIXED position (row 3,
// last column — it is never scrambled, so the reach to it is comparable run to
// run). The user taps a six-digit target. Each tap is a visual search, a reach
// and a press, so one captcha yields six behavioural samples on ANY device,
// touch included — which is the point: the password's rhythm cannot cross from a
// laptop to a phone, but "how long you take to find a 3 and how you travel to
// it" can.
//
// Notes, all deliberate:
//  - timing comes from pointerdown/pointerup, never from click. click is
//    synthesised (and coalesced) by the browser and fires after the press, so
//    its timestamp is not the press.
//  - the entry is NOT auto-corrected. A wrong digit stays red until the user
//    backspaces it, because the correction itself is behaviour we want to keep.
//  - one shared pointer capture per page: pointer.js listens on the document and
//    exposes no way to stop, so creating one per run would leak a listener set
//    per captcha. It is reset() at every render instead, which is exactly what a
//    fresh instance would give us. (Consequence: two keypads must not be alive
//    at the same time. The app never does that.)
//  - every handler is wrapped in the same spirit as capture.js: a capture bug
//    must never wedge the captcha.
//  - KNOWN GAP: the keys are focusable buttons, but Enter/Space fire `click`,
//    which this deliberately ignores, so a keyboard-only user cannot solve a
//    captcha. Giving them a keyboard path would produce runs with no reach and
//    no press duration, which the engine cannot score anyway; the honest fix is
//    a separate accessible route, not a fake one. Not built for the hackathon.
import { el } from './ui.js';
import { createPointerCapture } from './pointer.js';

export const KEYPAD_BACK = -1; // contracts.KEYPAD_BACK
export const KEYPAD_BLANK = -2; // contracts.KEYPAD_BLANK

const FLASH_MS = 340; // green confirmation before onDone, so the user sees it land
const IDS = ['kp-0', 'kp-1', 'kp-2', 'kp-3', 'kp-4', 'kp-5', 'kp-back', 'kp-target'];

/** @type {ReturnType<typeof createPointerCapture>|null} */
let sharedPointer = null;

function pointerFor(root, origin) {
  if (!sharedPointer) sharedPointer = createPointerCapture(root, { origin, ids: IDS });
  else sharedPointer.reset(origin);
  return sharedPointer;
}

const viewportString = () => `${innerWidth}x${innerHeight}@${devicePixelRatio}`;
// Cell rects are client coordinates, so a scroll invalidates them just as a
// rotation does: both belong in the key that decides whether to re-measure.
const frameKey = () => `${viewportString()}+${Math.round(scrollX)},${Math.round(scrollY)}`;

// Event timestamps share performance.now()'s clock. A script-dispatched event can
// carry 0, which would put the tap before the origin: fall back to reading the
// clock, and let `trusted` carry the fact that it was synthetic.
const stamp = (ev) => (Number.isFinite(ev.timeStamp) && ev.timeStamp > 0 ? ev.timeStamp : performance.now());

function rectOf(node) {
  const r = node.getBoundingClientRect();
  return [r.x, r.y, r.width, r.height];
}

/**
 * Render one scrambled-keypad captcha.
 *
 * @param {HTMLElement} mount element to render into (its children are replaced)
 * @param {{challenge: object, env?: object, onDone: (run: object) => void}} opts
 *   challenge — contracts.KeypadChallenge {id, layout, target, cols, rows, expires_at}
 *   env       — probe.js output, copied into the run as-is
 *   onDone    — called once, with a contracts.KeypadRun, when the six digits match
 * @returns {{destroy(): void, focus(): void}}
 */
export function createKeypad(mount, { challenge, env = {}, onDone = () => {} } = {}) {
  // One origin for the whole run: shown_at, every tap and every pointer sample
  // are ms since this instant, exactly like contracts.Sample.
  const origin = performance.now();
  const layout = (challenge.layout || []).slice();
  const target = (challenge.target || []).slice();
  const cols = challenge.cols || 3;
  const cells = layout.length || cols * (challenge.rows || 2);

  const pointer = pointerFor(mount, origin);
  const taps = [];
  const entry = []; // digits tapped so far, wrong ones included
  let shownAt = 0;
  let shownFrame = frameKey();
  let rects = {};
  let done = false;
  let flashTimer = null;

  // ---------------------------------------------------------------- DOM
  const targetBox = el('div', { className: 'kp-target', id: 'kp-target' },
    ...target.map((d) => el('span', { className: 'kp-target-d' }, String(d))));
  targetBox.setAttribute('aria-label', `Enter these digits: ${target.join(', ')}`);

  const slots = target.map(() => el('span', { className: 'kp-slot' }));
  const entryBox = el('div', { className: 'kp-entry', role: 'status' }, ...slots);
  entryBox.setAttribute('aria-live', 'polite');
  entryBox.setAttribute('aria-label', 'Digits entered so far');

  const keys = [];
  const grid = el('div', { className: 'kp-grid', role: 'group' });
  grid.setAttribute('aria-label', 'Scrambled keypad');
  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  for (let i = 0; i < cells; i++) {
    const digit = layout[i] === undefined ? KEYPAD_BLANK : layout[i];
    const blank = digit === KEYPAD_BLANK;
    const key = el('button', {
      type: 'button',
      id: `kp-${i}`,
      className: `kp-key${blank ? ' kp-blank' : ''}`,
      disabled: blank,
    }, blank ? '' : String(digit));
    if (blank) key.setAttribute('aria-hidden', 'true');
    else key.setAttribute('aria-label', `Digit ${digit}`);
    key.dataset.cell = String(i);
    keys.push(key);
    grid.append(key);
  }
  const back = el('button', { type: 'button', id: 'kp-back', className: 'kp-key kp-back' },
    backIcon(), el('span', { className: 'sr-only' }, 'Backspace'));
  back.setAttribute('aria-label', 'Backspace');
  back.dataset.cell = '-1';
  grid.append(back);

  // The challenge id on the root: the only way from outside (a test, a debug
  // session) to tell one rendered captcha from the next.
  const root = el('div', { className: 'kp', 'data-challenge': challenge.id || '' },
    el('p', { className: 'kp-prompt' }, 'Tap these digits, in order'),
    targetBox,
    entryBox,
    grid);
  mount.replaceChildren(root);
  // On a phone the card can sit below the verdict: bring the keys into view
  // before the rects are taken, so the geometry recorded is the geometry tapped.
  mount.scrollIntoView({ block: 'nearest' });

  // ---------------------------------------------------------------- geometry
  function measure() {
    const out = {};
    for (let i = 0; i < cells; i++) out[String(i)] = rectOf(keys[i]);
    out.back = rectOf(back);
    out.target = rectOf(targetBox);
    return out;
  }

  // The layout is "shown" at the first frame after it is in the document: that
  // is when the search for the first digit can start, and the first tap's
  // reaction time is measured from here.
  requestAnimationFrame(() => {
    shownAt = performance.now() - origin;
    shownFrame = frameKey();
    rects = measure();
  });

  // ---------------------------------------------------------------- entry
  function paint() {
    for (let i = 0; i < slots.length; i++) {
      const d = entry[i];
      slots[i].textContent = d === undefined ? '' : String(d);
      slots[i].className = 'kp-slot'
        + (d === undefined ? '' : d === target[i] ? ' filled' : ' bad')
        + (d === undefined && i === entry.length ? ' next' : '');
    }
  }
  paint();

  const complete = () => entry.length === target.length && entry.every((d, i) => d === target[i]);

  function finish() {
    if (done) return;
    done = true;
    // Re-measure if the page scrolled or rotated under us; otherwise keep the
    // rects the user actually aimed at. (A run that scrolls mid-way keeps the
    // geometry of its end, which is what its last taps were aimed at.)
    const cellRects = frameKey() === shownFrame && Object.keys(rects).length ? rects : measure();
    const run = {
      challenge_id: challenge.id,
      layout,
      target,
      cells: cellRects,
      shown_at: shownAt,
      taps: taps.slice(),
      pointer: pointer.events.slice(),
      env,
      viewport: viewportString(),
      completed: true,
    };
    root.classList.add('kp-ok');
    for (const k of keys) k.disabled = true;
    back.disabled = true;
    // taps.slice() copies the array, not the tap objects, so the release of the
    // press that completed the run still lands in `run` during the flash.
    flashTimer = setTimeout(() => onDone(run), FLASH_MS);
  }

  // ---------------------------------------------------------------- taps
  /** The tap that is still waiting for its release, so pointerup can close it. */
  let pending = null;

  function onDown(ev) {
    if (done) return;
    // Only a primary press counts: right-click and a second finger are not taps.
    if (ev.isPrimary === false) return;
    if (ev.pointerType === 'mouse' && ev.button !== 0) return;
    const node = ev.currentTarget;
    const cell = Number(node.dataset.cell);
    const digit = cell < 0 ? KEYPAD_BACK : layout[cell];
    if (digit === KEYPAD_BLANK) return;
    // What the entry needed at this instant; -1 once six digits are down.
    const expected = entry.length < target.length ? target[entry.length] : -1;
    const tap = {
      t_down: stamp(ev) - origin,
      t_up: null,
      cell,
      digit,
      expected,
      correct: digit === expected,
      x: ev.clientX,
      y: ev.clientY,
      pointer_type: ev.pointerType || 'mouse',
      trusted: ev.isTrusted === true,
    };
    taps.push(tap);
    pending = tap;

    if (digit === KEYPAD_BACK) entry.pop();
    else if (entry.length < target.length) entry.push(digit);
    paint();
    node.classList.add('held');
    if (complete()) finish();
  }

  function onUp(ev) {
    if (pending && pending.t_up === null) pending.t_up = stamp(ev) - origin;
    pending = null;
    for (const k of [...keys, back]) k.classList.remove('held');
  }

  const safe = (fn) => (ev) => {
    try {
      fn(ev);
    } catch {
      /* a capture bug must never wedge the captcha */
    }
  };
  const downHandler = safe(onDown);
  const upHandler = safe(onUp);
  const noMenu = (ev) => ev.preventDefault(); // long-press on a phone must not open a menu

  for (const key of [...keys, back]) {
    if (key.disabled) continue;
    key.addEventListener('pointerdown', downHandler);
    key.addEventListener('contextmenu', noMenu);
  }
  // On the document: a press that starts on a key can be released anywhere.
  document.addEventListener('pointerup', upHandler);
  document.addEventListener('pointercancel', upHandler);

  return {
    focus() {
      const first = keys.find((k) => !k.disabled);
      if (first) first.focus({ preventScroll: true });
    },
    destroy() {
      done = true;
      clearTimeout(flashTimer);
      document.removeEventListener('pointerup', upHandler);
      document.removeEventListener('pointercancel', upHandler);
      mount.replaceChildren();
    },
  };
}

function backIcon() {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '2');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS(ns, 'path');
  p.setAttribute('d', 'M21 5H9L3 12l6 7h12zM17 9l-6 6m0-6 6 6');
  svg.append(p);
  return svg;
}
