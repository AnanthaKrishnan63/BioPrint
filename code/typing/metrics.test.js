// Tests for metrics.js — pure functions over key-event arrays.
// Run: node --test
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  pairKeystrokes,
  flightTimes,
  computeMetrics,
  estimateTimerResolution,
  isTextKey,
} from './metrics.js';

const down = (code, t) => ({ code, type: 'down', t });
const up = (code, t) => ({ code, type: 'up', t });

// ---------------------------------------------------------------- pairing

test('pairs each keydown with its keyup and computes dwell', () => {
  const pairs = pairKeystrokes([down('KeyA', 100), up('KeyA', 190)]);
  assert.equal(pairs.length, 1);
  assert.deepEqual(pairs[0], { code: 'KeyA', downT: 100, upT: 190, dwell: 90 });
});

test('ignores OS auto-repeat: a second keydown with no intervening keyup', () => {
  // Holding a key makes the OS fire keydown repeatedly. Only the first counts.
  const pairs = pairKeystrokes([
    down('KeyE', 100), down('KeyE', 150), down('KeyE', 200), up('KeyE', 250),
  ]);
  assert.equal(pairs.length, 1);
  assert.equal(pairs[0].dwell, 150, 'dwell spans the first down to the up');
});

test('ignores a keyup with no matching keydown', () => {
  // Happens when a key was already held as the page gained focus.
  const pairs = pairKeystrokes([up('KeyQ', 50), down('KeyA', 100), up('KeyA', 180)]);
  assert.equal(pairs.length, 1);
  assert.equal(pairs[0].code, 'KeyA');
});

test('drops an unreleased keydown at the end of the stream', () => {
  // Happens when focus is lost mid-press: the keyup never arrives.
  const pairs = pairKeystrokes([down('KeyA', 100), up('KeyA', 180), down('KeyB', 200)]);
  assert.equal(pairs.length, 1);
  assert.equal(pairs[0].code, 'KeyA');
});

test('orders pairs by keydown time even when keys overlap', () => {
  // Shift held across a letter: ShiftLeft goes down first but comes up last,
  // so completion order is KeyA then ShiftLeft. Press order is the truth.
  const pairs = pairKeystrokes([
    down('ShiftLeft', 100),
    down('KeyA', 150),
    up('KeyA', 220),
    up('ShiftLeft', 260),
  ]);
  assert.deepEqual(pairs.map((p) => p.code), ['ShiftLeft', 'KeyA']);
});

// ---------------------------------------------------------------- flight

test('flight time is the gap from one key releasing to the next depressing', () => {
  const pairs = pairKeystrokes([
    down('KeyA', 100), up('KeyA', 180),
    down('KeyB', 300), up('KeyB', 360),
  ]);
  assert.deepEqual(flightTimes(pairs), [120]);
});

test('flight time is negative when keys overlap (rollover)', () => {
  // A fast typist presses B before releasing A. This is normal and personal.
  const pairs = pairKeystrokes([
    down('KeyA', 100), up('KeyA', 200),
    down('KeyB', 160), up('KeyB', 260),
  ]);
  assert.deepEqual(flightTimes(pairs), [-40]);
});

test('a single keystroke has no flight times', () => {
  assert.deepEqual(flightTimes(pairKeystrokes([down('KeyA', 100), up('KeyA', 180)])), []);
});

// ---------------------------------------------------------------- key classes

test('classifies character-producing keys apart from modifiers and navigation', () => {
  for (const code of ['KeyA', 'Digit7', 'Space', 'Comma', 'Slash', 'Numpad3']) {
    assert.equal(isTextKey(code), true, `${code} should count as text`);
  }
  for (const code of ['ShiftLeft', 'ControlLeft', 'Backspace', 'Delete', 'ArrowUp', 'Tab', 'F5']) {
    assert.equal(isTextKey(code), false, `${code} should not count as text`);
  }
});

// ---------------------------------------------------------------- metrics

test('computes gross and net words per minute', () => {
  // 10 text keys over exactly 60s. WPM divides characters by 5 by convention.
  const events = [];
  for (let i = 0; i < 10; i++) {
    events.push(down('KeyA', i * 6000), up('KeyA', i * 6000 + 80));
  }
  // duration runs from the first event to the last: 9*6000 + 80 = 54080 ms
  const m = computeMetrics(events, 10);
  assert.equal(m.textKeys, 10);
  assert.equal(m.keystrokesGross, 10);
  assert.equal(m.netChars, 10);
  const minutes = 54080 / 60000;
  assert.ok(Math.abs(m.grossWpm - (10 / 5) / minutes) < 1e-9);
  assert.ok(Math.abs(m.netWpm - (10 / 5) / minutes) < 1e-9);
});

test('counts Backspace and Delete separately and keeps them out of the text count', () => {
  const events = [
    down('KeyA', 0), up('KeyA', 80),
    down('Backspace', 200), up('Backspace', 260),
    down('Backspace', 400), up('Backspace', 450),
    down('Delete', 600), up('Delete', 660),
    down('KeyB', 800), up('KeyB', 880),
  ];
  const m = computeMetrics(events, 1);
  assert.equal(m.textKeys, 2, 'only KeyA and KeyB produce characters');
  assert.equal(m.backspaceCount, 2);
  assert.equal(m.deleteCount, 1);
  assert.equal(m.keystrokesGross, 5, 'text keys plus corrections');
});

test('correction ratio is gross keystrokes over surviving characters', () => {
  // Typed 5 characters, deleted 2, 3 survive.
  const events = [];
  for (let i = 0; i < 5; i++) events.push(down('KeyA', i * 100), up('KeyA', i * 100 + 50));
  events.push(down('Backspace', 600), up('Backspace', 650));
  events.push(down('Backspace', 700), up('Backspace', 750));
  const m = computeMetrics(events, 3);
  assert.equal(m.keystrokesGross, 7);
  assert.equal(m.netChars, 3);
  assert.ok(Math.abs(m.correctionRatio - 7 / 3) < 1e-9);
});

test('excludes modifier keys from the character count but keeps them in totalKeys', () => {
  const events = [
    down('ShiftLeft', 0), down('KeyA', 50), up('KeyA', 120), up('ShiftLeft', 160),
  ];
  const m = computeMetrics(events, 1);
  assert.equal(m.totalKeys, 2);
  assert.equal(m.textKeys, 1);
  assert.equal(m.keystrokesGross, 1);
});

test('averages dwell and flight, and counts rollovers', () => {
  const events = [
    down('KeyA', 0), up('KeyA', 100),    // dwell 100
    down('KeyB', 80), up('KeyB', 200),   // dwell 120, flight -20 (rollover)
    down('KeyC', 300), up('KeyC', 380),  // dwell 80,  flight 100
  ];
  const m = computeMetrics(events, 3);
  assert.ok(Math.abs(m.meanDwellMs - 100) < 1e-9);
  assert.ok(Math.abs(m.meanFlightMs - 40) < 1e-9);
  assert.equal(m.negativeFlights, 1);
});

test('returns zeros instead of NaN or Infinity for an empty stream', () => {
  const m = computeMetrics([], 0);
  for (const [key, value] of Object.entries(m)) {
    assert.ok(Number.isFinite(value), `${key} should be finite, got ${value}`);
  }
  assert.equal(m.grossWpm, 0);
  assert.equal(m.correctionRatio, 0);
});

test('returns zeros instead of Infinity when every event shares one timestamp', () => {
  const m = computeMetrics([down('KeyA', 500), up('KeyA', 500)], 1);
  assert.equal(m.durationMs, 0);
  assert.equal(m.grossWpm, 0, 'zero elapsed time must not divide');
  assert.ok(Number.isFinite(m.netWpm));
});

// ---------------------------------------------------------------- timer

test('estimates a 1 ms clock when every timestamp is a whole millisecond', () => {
  const events = [down('KeyA', 100), up('KeyA', 180), down('KeyB', 300), up('KeyB', 361)];
  assert.equal(estimateTimerResolution(events), 1);
});

test('estimates a finer clock from fractional timestamps', () => {
  const events = [down('KeyA', 100.1), up('KeyA', 180.3), down('KeyB', 300.5)];
  assert.ok(estimateTimerResolution(events) < 1);
});

test('reports null resolution when there are too few distinct timestamps', () => {
  assert.equal(estimateTimerResolution([down('KeyA', 100)]), null);
});
