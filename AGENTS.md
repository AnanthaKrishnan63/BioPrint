# Repository Guidelines

## Purpose & Architecture

This project researches browser-based account-holder verification. The implemented typing harness collects events and session metrics; identity features and scoring remain future work. Keep device/context and behavioral signals separate. Use per-user, per-device-class behavioral templates and fuzzy-match fingerprint attributes. Report false rejection rates at fixed low false acceptance rates; respond with additional verification rather than denial.

## Project Structure & Module Organization

- `code/typing/`: `capture.js` records events, `metrics.js` computes pure features, and `app.js` connects them to `index.html` and its inline styles. `server.py` persists sessions in SQLite.
- `code/typing/metrics.test.js`: metrics unit tests.
- `code/typing/verify/`: kernel logging, timing comparison, and synthetic validation.
- `research/`: Obsidian vault root. Start with `research/big_idea/13 Roadmap.md` for priorities and decision gates, `00 Home.md` for rationale, and `10 Basics and Glossary.md` for terminology.

## Build, Test, and Development Commands

Create the environment from the repository root:

```bash
conda env create -f code/typing/environment.yml
conda activate bigidea
cd code/typing
uvicorn server:app --reload
```

Open `http://localhost:8000`. Use the `bigidea` Python environment, never system Python. No build step or formatter/linter is configured.

From `code/typing/`, run `node --test` for all metrics tests or `node --test --test-name-pattern rollover` for a focused check.

## Coding Style & Naming Conventions

Follow existing JavaScript conventions: two-space indentation, ES modules, camelCase, single quotes, and semicolons. Python uses four spaces, snake_case functions, and PascalCase classes. Keep features in DOM-free `metrics.js`.

## Testing Guidelines

Use `node:test` and strict assertions in `*.test.js`; cover duplicate keydowns, missing releases, and overlapping keys. No coverage threshold is configured. After changing comparison logic, run from `code/typing/verify/`:

```bash
python make_synthetic.py --jitter 2 --drift 0.5 --drop 3
python compare.py --kernel synthetic-kernel.jsonl --browser synthetic-browser.json
```

Verify injected faults are detected. Align streams by time, never sequence shape.

## Data & Research Conventions

Record `event.code`, never `event.key` or textarea contents. Preserve raw recordings and derive new metrics from them. Never recreate `sessions.db`; update `METRIC_COLUMNS`, schema, and existing tables through migrations. Use `synthetic-*` filenames for generated recordings.

Research notes use numbered filenames, YAML frontmatter (`title`, `tags`, `updated`), summaries, tables, and related-note wikilinks. Label unverified claims explicitly.

## Commit & Pull Request Guidelines

Git history is unavailable here. Use concise imperative commit subjects. Describe behavior changes and validation in PRs; link relevant issues and include screenshots for UI changes.
