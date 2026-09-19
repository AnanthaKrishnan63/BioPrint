# Is the browser's clock trustworthy?

Every dwell and flight number rests on one assumption: that `event.timeStamp`
reports when the key was *physically pressed*. This checks it, by logging the
same keystrokes twice and comparing.

The kernel stamps each key inside the input driver, before X11 and before the
browser. That makes it the reference:

```
physical key -> kernel driver -> X11 -> browser -> JS handler
                     ^                                ^
                evdev stamps here            event.timeStamp claims here
```

## What we are looking for

A **constant whole-stream offset** is harmless. If the browser is late by a fixed
15 ms on everything, every interval is still exactly right, because the offset
cancels out of any subtraction.

But that only holds if keydown and keyup are delayed *equally*. If releases are
reported later than presses, every dwell is inflated by the difference and nothing
cancels. So in section 3, the **mean** error matters as much as the spread: a
non-zero mean is a systematic bias in dwell, not noise.

**Jitter** is the danger. If the delay varies from key to key, that variation lands
directly in dwell and flight. The browser's main thread can delay event delivery by
a whole frame (16 ms at 60 Hz) when it is busy, which would swallow a real fraction
of a 90 ms dwell.

**Dropped events** are worse still, and silent.

## Running it

Three terminals, or run the logger in the background.

```bash
conda activate bigidea

# 1. start the kernel logger (needs root, so it will ask for your password)
cd verify
sudo ~/miniconda3/envs/bigidea/bin/python evdev_logger.py -o kernel.jsonl

# 2. in another terminal, start the page
cd ..
uvicorn server:app --reload
```

Now open <http://localhost:8000>, type a couple of sentences, press **Save session**.
Then Ctrl-C the logger and compare:

```bash
cd verify
python compare.py --kernel kernel.jsonl          # uses the newest saved session
```

Type for at least 30 seconds. Drift and jitter both need a decent sample.

## Reading the result

| Section | Question |
|---|---|
| 1. Sequence integrity | Did the browser miss any keys? |
| 2. Per-event timing error | How far off is each event, after removing the constant offset? |
| 3. Dwell time error | The number that matters. Compare it against the mean dwell printed beneath. |
| 4. Flight time error | Same, for the gaps between keys. |
| 5. Clock resolution | How finely each clock ticks. |

The verdict calls dwell error **usable** under 5% of a typical dwell, **marginal**
to 15%, **a problem** beyond that.

## Privacy

The logger records only which physical key moved and when. Never characters,
never window titles, never anything you type. It writes one plain-text file you
can read, and stops when you stop it.

## Why you should not trust this tool on my word

`compare.py` is only useful if it actually detects error. `make_synthetic.py`
builds two streams with a known fault injected, so the tool can be checked against
a known answer:

```bash
python make_synthetic.py --jitter 2 --drift 0.5 --drop 3
python compare.py --kernel synthetic-kernel.jsonl --browser synthetic-browser.json
```

It should report about 3 ms of dwell spread (two independent 2 ms errors combine
to 2 x sqrt(2)), exactly 3 lost events, and about 0.5 ms/s of drift. If it does
not, the tool is broken and its real measurements mean nothing.

Synthetic output is written to `synthetic-*` filenames so it can never overwrite a
real recording, and the logger refuses to write over an existing non-empty file.
