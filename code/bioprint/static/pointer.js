// pointer.js — pointer capture. OWNER: Agent C (pointer).
// STUB: records nothing yet. Agent C implements; the interface stays as below.

/**
 * @param root element to listen on (the login card)
 * @returns {{events: object[], reset(origin?: number): void}}
 *   events match contracts.PointerEvent: {type, t, x, y, pointer_type, buttons, target, trusted}
 */
export function createPointerCapture(root, { origin = performance.now() } = {}) {
  const events = [];
  return {
    events,
    reset(newOrigin = performance.now()) {
      events.length = 0;
    },
  };
}
