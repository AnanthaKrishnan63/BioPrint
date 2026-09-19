// capture.js — records key events and nothing else.
//
// Deliberately dumb: it does not compute, interpret or filter for meaning. Its
// only job is to produce a faithful event log. That separation is what lets
// metrics.js be tested without a browser, and what lets this file become the
// real research collector later without dragging UI code along with it.
//
// PRIVACY: we store `event.code` (which physical key) and never `event.key`
// (which character), and never the contents of the text box. On a QWERTY layout
// KeyT still implies the letter T, so this is not anonymisation — but it drops
// capitalisation, clipboard content and IME composition, and it is the habit we
// want established before this ever records anyone else's typing.

export function createCapture(element, onChange = () => {}) {
  /** @type {{code: string, type: 'down'|'up', t: number}[]} */
  const events = [];
  let t0 = null;
  let hadPaste = false;

  const record = (type) => (ev) => {
    // Holding a key makes the OS fire keydown repeatedly. Only the first is a press.
    if (type === 'down' && ev.repeat) return;

    // Timestamps are relative to the first event, not to page load, so idle time
    // before the person starts typing never lands in the duration.
    if (t0 === null) t0 = ev.timeStamp;

    events.push({ code: ev.code, type, t: ev.timeStamp - t0 });
    onChange();
  };

  // Pasting fires no key events at all, so a pasted session would show hundreds
  // of characters and zero keystrokes. Flag it rather than silently recording it.
  const notePaste = () => {
    hadPaste = true;
    onChange();
  };

  element.addEventListener('keydown', record('down'));
  element.addEventListener('keyup', record('up'));
  element.addEventListener('paste', notePaste);

  return {
    events,
    get hadPaste() {
      return hadPaste;
    },
    reset() {
      events.length = 0; // in place, so existing references stay valid
      t0 = null;
      hadPaste = false;
      onChange();
    },
  };
}
