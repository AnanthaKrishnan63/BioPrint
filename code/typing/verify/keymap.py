"""Translate Linux kernel key names (KEY_A) into DOM event.code names (KeyA).

The browser and the kernel name the same physical key differently, so the two
event streams cannot be lined up until they speak the same language.
"""

from __future__ import annotations

# Keys whose names follow no rule.
IRREGULAR = {
    "KEY_ESC": "Escape",
    "KEY_MINUS": "Minus",
    "KEY_EQUAL": "Equal",
    "KEY_BACKSPACE": "Backspace",
    "KEY_TAB": "Tab",
    "KEY_LEFTBRACE": "BracketLeft",
    "KEY_RIGHTBRACE": "BracketRight",
    "KEY_ENTER": "Enter",
    "KEY_LEFTCTRL": "ControlLeft",
    "KEY_RIGHTCTRL": "ControlRight",
    "KEY_SEMICOLON": "Semicolon",
    "KEY_APOSTROPHE": "Quote",
    "KEY_GRAVE": "Backquote",
    "KEY_LEFTSHIFT": "ShiftLeft",
    "KEY_RIGHTSHIFT": "ShiftRight",
    "KEY_BACKSLASH": "Backslash",
    "KEY_COMMA": "Comma",
    "KEY_DOT": "Period",
    "KEY_SLASH": "Slash",
    "KEY_LEFTALT": "AltLeft",
    "KEY_RIGHTALT": "AltRight",
    "KEY_SPACE": "Space",
    "KEY_CAPSLOCK": "CapsLock",
    "KEY_NUMLOCK": "NumLock",
    "KEY_SCROLLLOCK": "ScrollLock",
    "KEY_LEFTMETA": "MetaLeft",
    "KEY_RIGHTMETA": "MetaRight",
    "KEY_COMPOSE": "ContextMenu",
    "KEY_INSERT": "Insert",
    "KEY_DELETE": "Delete",
    "KEY_HOME": "Home",
    "KEY_END": "End",
    "KEY_PAGEUP": "PageUp",
    "KEY_PAGEDOWN": "PageDown",
    "KEY_UP": "ArrowUp",
    "KEY_DOWN": "ArrowDown",
    "KEY_LEFT": "ArrowLeft",
    "KEY_RIGHT": "ArrowRight",
    "KEY_SYSRQ": "PrintScreen",
    "KEY_PAUSE": "Pause",
    "KEY_102ND": "IntlBackslash",
    "KEY_RO": "IntlRo",
    "KEY_YEN": "IntlYen",
    # numeric keypad
    "KEY_KPASTERISK": "NumpadMultiply",
    "KEY_KPMINUS": "NumpadSubtract",
    "KEY_KPPLUS": "NumpadAdd",
    "KEY_KPDOT": "NumpadDecimal",
    "KEY_KPSLASH": "NumpadDivide",
    "KEY_KPENTER": "NumpadEnter",
    "KEY_KPEQUAL": "NumpadEqual",
    "KEY_KPCOMMA": "NumpadComma",
}


def dom_code(key_name: str) -> str:
    """KEY_A -> KeyA. Unknown keys come back unchanged so they stay visible."""
    if key_name in IRREGULAR:
        return IRREGULAR[key_name]

    body = key_name[4:] if key_name.startswith("KEY_") else key_name

    if len(body) == 1 and body.isalpha():           # KEY_A   -> KeyA
        return f"Key{body.upper()}"
    if len(body) == 1 and body.isdigit():           # KEY_7   -> Digit7
        return f"Digit{body}"
    if body.startswith("F") and body[1:].isdigit(): # KEY_F5  -> F5
        return body
    if body.startswith("KP") and body[2:].isdigit():# KEY_KP3 -> Numpad3
        return f"Numpad{body[2:]}"

    return key_name  # leave it recognisable rather than guessing
