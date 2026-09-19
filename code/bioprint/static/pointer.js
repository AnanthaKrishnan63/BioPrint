// pointer.js — pointer capture. OWNER: Agent C (pointer).
//
// Records the movement, presses and hovers that lead to the submit click, on the
// same clock as capture.js (one origin, passed in by app.js).
//
// Notes, all deliberate:
//  - listens on the document, not just the card: the approach usually starts
//    outside the card (or on the password field, which is inside it), and
//    clientX/clientY are document-wide anyway. `root` only supplies the document
//    and is kept in the signature for the agreed interface.
//  - getCoalescedEvents() recovers the sub-frame points the compositor merged, so
//    the velocity profile is real and not frame-quantised. Points are then thinned
//    to ~1 per 8 ms (~125 Hz), which is plenty for speed and keeps a login at a
//    few hundred events instead of a few thousand (contracts.MAX_EVENTS is 20k).
//  - enter/leave are synthesised from pointerover/pointerout, because pointerenter
//    does not bubble and we want one listener, not one per field.
//  - every handler is wrapped: capture must never break a login.

const TRACKED = ['username', 'password', 'login', 'enroll', 'register'];
const MIN_MOVE_GAP_MS = 8; // thin moves to ~125 Hz
const MAX_EVENTS = 4000; // hard ceiling; oldest events are dropped first
const TRIM_TO = 3000;

/**
 * @param root element the form lives in (the login card); its document is listened on
 * @param {{origin?: number, ids?: string[]}} opts extra element ids to tag, if Agent D adds fields
 * @returns {{events: object[], reset(origin?: number): void}}
 *   events match contracts.PointerEvent: {type, t, x, y, pointer_type, buttons, target, trusted}
 */
export function createPointerCapture(root, { origin = performance.now(), ids = [] } = {}) {
  /** @type {object[]} */
  const events = [];
  const tracked = new Set([...TRACKED, ...ids]);
  const doc = (root && root.ownerDocument) || document;
  let t0 = origin;
  let lastMoveT = -Infinity;
  let hovered = null; // id currently under the pointer, for synthetic enter/leave

  // Nearest ancestor with an id we care about, else null.
  const idOf = (node) => {
    for (let n = node; n && n.nodeType === 1; n = n.parentElement) {
      if (n.id && tracked.has(n.id)) return n.id;
    }
    return null;
  };

  const push = (type, ev, t, x, y, target) => {
    if (events.length >= MAX_EVENTS) events.splice(0, events.length - TRIM_TO);
    events.push({
      type,
      t: t - t0,
      x,
      y,
      pointer_type: ev.pointerType || 'mouse',
      buttons: ev.buttons | 0,
      target,
      trusted: ev.isTrusted === true,
    });
  };

  const onMove = (ev) => {
    const target = idOf(ev.target);
    // Coalesced points carry their own timestamps; fall back to the event itself.
    let points = [ev];
    if (typeof ev.getCoalescedEvents === 'function') {
      const c = ev.getCoalescedEvents();
      if (c && c.length) points = c;
    }
    for (const p of points) {
      const t = Number.isFinite(p.timeStamp) && p.timeStamp > 0 ? p.timeStamp : ev.timeStamp;
      if (t - lastMoveT < MIN_MOVE_GAP_MS) continue;
      lastMoveT = t;
      push('move', ev, t, p.clientX, p.clientY, target);
    }
  };

  const onDown = (ev) => push('down', ev, ev.timeStamp, ev.clientX, ev.clientY, idOf(ev.target));
  const onUp = (ev) => push('up', ev, ev.timeStamp, ev.clientX, ev.clientY, idOf(ev.target));

  // pointerover/out bubble; derive enter/leave for the tracked elements only.
  const onOver = (ev) => {
    const id = idOf(ev.target);
    if (id === hovered) return;
    if (hovered) push('leave', ev, ev.timeStamp, ev.clientX, ev.clientY, hovered);
    hovered = id;
    if (id) push('enter', ev, ev.timeStamp, ev.clientX, ev.clientY, id);
  };
  const onOut = (ev) => {
    // Leaving to somewhere untracked (or off the window) still ends the hover.
    if (hovered && idOf(ev.relatedTarget) !== hovered) {
      push('leave', ev, ev.timeStamp, ev.clientX, ev.clientY, hovered);
      hovered = null;
    }
  };

  const safe = (fn) => (ev) => {
    try {
      fn(ev);
    } catch {
      /* capture must never throw into the page */
    }
  };

  const opts = { passive: true, capture: true };
  doc.addEventListener('pointermove', safe(onMove), opts);
  doc.addEventListener('pointerdown', safe(onDown), opts);
  doc.addEventListener('pointerup', safe(onUp), opts);
  doc.addEventListener('pointerover', safe(onOver), opts);
  doc.addEventListener('pointerout', safe(onOut), opts);

  return {
    events,
    reset(newOrigin = performance.now()) {
      events.length = 0; // in place, so existing references stay valid
      t0 = newOrigin;
      lastMoveT = -Infinity;
      hovered = null;
    },
  };
}
