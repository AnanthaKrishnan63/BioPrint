// metrics.js — turning raw key events into numbers.
//
// Everything here is a pure function over an array of events shaped like
//   { code: "KeyT", type: "down" | "up", t: 12345.6 }
// No DOM, no globals, so it runs identically in the browser and under `node --test`.

/** Keys that put a character on the screen. Excludes modifiers, navigation and corrections. */
const TEXT_KEY = /^(?:Key[A-Z]|Digit[0-9]|Numpad[0-9]|Space|Minus|Equal|BracketLeft|BracketRight|Backslash|Semicolon|Quote|Comma|Period|Slash|Backquote|IntlBackslash|IntlRo|IntlYen|NumpadAdd|NumpadSubtract|NumpadMultiply|NumpadDivide|NumpadDecimal|NumpadComma)$/;

export function isTextKey(code) {
  return TEXT_KEY.test(code);
}

/**
 * Match each keydown to its keyup.
 *
 * Three things make this less trivial than it looks:
 *  - Holding a key makes the OS fire keydown over and over with no keyup between.
 *    Only the first one is a real press.
 *  - A keyup can arrive with no keydown (the key was already held when the page
 *    gained focus), and a keydown can never be released (focus lost mid-press).
 *    Both are discarded rather than guessed at.
 *  - Keys overlap. Holding Shift across a letter completes the letter first, so
 *    completion order is not press order. We sort by keydown time, which is the
 *    order the person actually typed in.
 *
 * @returns {{code: string, downT: number, upT: number, dwell: number}[]}
 */
export function pairKeystrokes(events) {
  const pending = new Map(); // code -> keydown timestamp
  const pairs = [];

  for (const e of events) {
    if (e.type === 'down') {
      if (!pending.has(e.code)) pending.set(e.code, e.t); // ignore auto-repeat
    } else if (e.type === 'up') {
      if (!pending.has(e.code)) continue; // keyup with no keydown
      const downT = pending.get(e.code);
      pending.delete(e.code);
      pairs.push({ code: e.code, downT, upT: e.t, dwell: e.t - downT });
    }
  }

  pairs.sort((a, b) => a.downT - b.downT);
  return pairs;
}

/**
 * Gaps between consecutive keystrokes: release of one key to depression of the next.
 * Negative values are real — a fast typist presses the next key before letting go
 * of the last. That overlap is called rollover and it is highly personal.
 */
export function flightTimes(pairs) {
  const out = [];
  for (let i = 0; i + 1 < pairs.length; i++) {
    out.push(pairs[i + 1].downT - pairs[i].upT);
  }
  return out;
}

const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

function timeSpan(events) {
  if (events.length === 0) return 0;
  let lo = events[0].t;
  let hi = events[0].t;
  for (const e of events) {
    if (e.t < lo) lo = e.t;
    if (e.t > hi) hi = e.t;
  }
  return hi - lo;
}

/**
 * @param events raw key events
 * @param netChars characters actually surviving in the text box. Passed in rather
 *        than derived, because pasting and selecting produce no key events at all.
 */
export function computeMetrics(events, netChars = 0) {
  const pairs = pairKeystrokes(events);
  const flights = flightTimes(pairs);
  const durationMs = timeSpan(events);
  const minutes = durationMs / 60000;

  const textKeys = pairs.filter((p) => isTextKey(p.code)).length;
  const backspaceCount = pairs.filter((p) => p.code === 'Backspace').length;
  const deleteCount = pairs.filter((p) => p.code === 'Delete').length;
  const keystrokesGross = textKeys + backspaceCount + deleteCount;

  // A "word" is 5 characters by convention, which is what makes WPM comparable
  // across different text. Guard every division: zero elapsed time is normal
  // at the very start of a session.
  return {
    durationMs,
    totalKeys: pairs.length,
    textKeys,
    backspaceCount,
    deleteCount,
    keystrokesGross,
    netChars,
    grossWpm: minutes > 0 ? keystrokesGross / 5 / minutes : 0,
    netWpm: minutes > 0 ? netChars / 5 / minutes : 0,
    correctionRatio: netChars > 0 ? keystrokesGross / netChars : 0,
    meanDwellMs: mean(pairs.map((p) => p.dwell)),
    meanFlightMs: mean(flights),
    negativeFlights: flights.filter((f) => f < 0).length,
  };
}

const gcd = (a, b) => (b === 0 ? a : gcd(b, a % b));

/**
 * Estimate how finely the browser's clock actually ticks.
 *
 * Browsers deliberately blur their timers to block a class of CPU attack called
 * Spectre — roughly 100 microseconds in Chrome, 1 millisecond in Firefox. Since
 * dwell times are only tens of milliseconds, that rounding eats into exactly the
 * measurement we care about. Rather than take anyone's word for it, we find the
 * largest tick every observed timestamp is a multiple of.
 *
 * @returns resolution in ms, or null if there is not enough data to tell.
 */
export function estimateTimerResolution(events) {
  const stamps = [...new Set(events.map((e) => e.t))].sort((a, b) => a - b);
  if (stamps.length < 2) return null;

  // Offsets from the first stamp, so the absolute epoch doesn't dominate the GCD.
  const SCALE = 1e6; // work in integers to keep floating point out of the GCD
  let g = 0;
  for (let i = 1; i < stamps.length; i++) {
    g = gcd(g, Math.round((stamps[i] - stamps[0]) * SCALE));
  }
  return g === 0 ? null : g / SCALE;
}
