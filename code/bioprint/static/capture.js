// capture.js — records key events and nothing else. Forked from code/typing/capture.js.
//
// Differences from the research harness, all agreed for BioPrint:
//  - listens on a whole form, and tags each event with the field it happened in
//  - timestamps share one origin with pointer.js (set by the caller), so
//    keystrokes and pointer events live on one clock
//  - records event.isTrusted, the cheapest bot signal there is
//
// HACKATHON ONLY: event.code is recorded on the password field too, so the stored
// events spell the password. Accepted trade-off for the demo; never ship this.

const FIELDS = new Set(['username', 'password']);

export function createCapture(root, { origin = performance.now(), onChange = () => {} } = {}) {
  /** @type {{code: string, type: 'down'|'up', t: number, field: string, trusted: boolean}[]} */
  const events = [];
  let t0 = origin;
  let hadPaste = false;

  const fieldOf = (target) => (target && FIELDS.has(target.id) ? target.id : 'other');

  const record = (type) => (ev) => {
    // Holding a key makes the OS fire keydown repeatedly. Only the first is a press.
    if (type === 'down' && ev.repeat) return;
    events.push({
      code: ev.code,
      type,
      t: ev.timeStamp - t0,
      field: fieldOf(ev.target),
      trusted: ev.isTrusted,
    });
    onChange();
  };

  // Pasting fires no key events at all. Flag it rather than silently accepting it.
  const notePaste = () => {
    hadPaste = true;
    onChange();
  };

  root.addEventListener('keydown', record('down'));
  root.addEventListener('keyup', record('up'));
  root.addEventListener('paste', notePaste);

  return {
    events,
    get hadPaste() {
      return hadPaste;
    },
    reset(newOrigin = performance.now()) {
      events.length = 0; // in place, so existing references stay valid
      t0 = newOrigin;
      hadPaste = false;
      onChange();
    },
  };
}
