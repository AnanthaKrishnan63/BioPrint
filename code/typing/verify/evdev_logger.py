#!/usr/bin/env python3
"""Log keyboard events straight from the kernel, to check the browser's clock.

The kernel timestamps each key press in the input driver, before X11 and before
the browser have seen it. That makes it the closest thing to the physical key
that software can observe, which is why it serves as the reference here.

Records ONLY which physical key moved and when. Never the character, never the
window, never anything you type. Output is a plain text file you can read.

Needs root, because /dev/input is not world readable:

    sudo $(which python) evdev_logger.py -o kernel.jsonl

Press Ctrl-C to stop. Written line by line, so an interrupt loses nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import evdev
    from evdev import ecodes
except ImportError:
    sys.exit("evdev is not installed. Run: conda run -n bigidea pip install evdev")

from keymap import dom_code

# evdev reports value 2 for OS auto-repeat when a key is held. The browser
# filters those via event.repeat, so we drop them too and stay comparable.
VALUE_TO_TYPE = {1: "down", 0: "up"}


def find_keyboard() -> evdev.InputDevice:
    """Pick the device that looks most like the typing keyboard."""
    visible = evdev.list_devices()
    if not visible:
        # list_devices() silently skips anything unreadable, so an empty list
        # almost always means "not root" rather than "no keyboard attached".
        exists = list(Path("/dev/input").glob("event*"))
        if exists:
            sys.exit(
                f"Found {len(exists)} input devices but cannot read any of them.\n"
                f"/dev/input is root-only. Re-run with:\n\n"
                f"    sudo $(which python) {Path(__file__).name} "
                f"-o kernel.jsonl\n"
            )
        sys.exit("No input devices found under /dev/input at all.")

    candidates = []
    for path in visible:
        try:
            dev = evdev.InputDevice(path)
        except PermissionError:
            sys.exit(
                f"Cannot read {path}. Run this with sudo, using the environment's\n"
                f"python explicitly:  sudo $(which python) {Path(__file__).name}"
            )
        keys = dev.capabilities().get(ecodes.EV_KEY, [])
        # A real keyboard has letters. Power buttons and hotkey pads do not.
        if ecodes.KEY_A in keys and ecodes.KEY_Z in keys and ecodes.KEY_SPACE in keys:
            candidates.append(dev)
        else:
            dev.close()

    if not candidates:
        sys.exit("No keyboard-like device found under /dev/input.")

    # Prefer one that says so in its name; otherwise take the first.
    for dev in candidates:
        if "keyboard" in dev.name.lower():
            chosen = dev
            break
    else:
        chosen = candidates[0]

    for dev in candidates:
        if dev is not chosen:
            dev.close()
    return chosen


def hand_back_ownership(path: Path) -> None:
    """We run as root, so give the file to the user who invoked sudo."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if uid and gid:
        os.chown(path, int(uid), int(gid))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default="kernel.jsonl", type=Path)
    ap.add_argument("-d", "--device", help="e.g. /dev/input/event2; auto-detected if omitted")
    args = ap.parse_args()

    if args.out.exists() and args.out.stat().st_size > 0:
        sys.exit(
            f"{args.out} already exists and is not empty.\n"
            f"Recordings are expensive to reproduce, so this will not overwrite one.\n"
            f"Move it aside or choose another name with -o."
        )

    dev = evdev.InputDevice(args.device) if args.device else find_keyboard()

    print(f"Reading  : {dev.path}  ({dev.name})")
    print(f"Writing  : {args.out}")
    print("\nNow type in the browser page. Press Ctrl-C here when you have saved the session.\n")

    count = 0
    try:
        with args.out.open("w") as fh:
            for event in dev.read_loop():
                if event.type != ecodes.EV_KEY:
                    continue
                kind = VALUE_TO_TYPE.get(event.value)
                if kind is None:  # value 2 == auto-repeat
                    continue

                name = ecodes.KEY.get(event.code, f"UNKNOWN_{event.code}")
                if isinstance(name, list):  # some codes carry aliases
                    name = name[0]

                fh.write(json.dumps({
                    "code": dom_code(name),
                    "type": kind,
                    # seconds since the Unix epoch, stamped in the kernel
                    "t": event.sec + event.usec / 1e6,
                }) + "\n")
                fh.flush()
                count += 1
                print(f"\r{count} events", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        dev.close()
        hand_back_ownership(args.out)
        print(f"\n\nStopped. {count} events written to {args.out}")


if __name__ == "__main__":
    main()
