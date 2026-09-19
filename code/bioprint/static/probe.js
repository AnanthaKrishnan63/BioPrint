// probe.js — browser environment probe for bot detection. OWNER: Agent B (bot).
// STUB: only navigator.webdriver. Agent B implements; engine/bot.py owns the keys.

/** @returns {Promise<Record<string, unknown>>} sent as Sample.env */
export async function collectEnv() {
  return { webdriver: navigator.webdriver === true };
}
