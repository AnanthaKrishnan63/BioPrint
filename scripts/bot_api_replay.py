"""Replay all genuine CMU development records through frozen bot-rule ASGI."""
import asyncio
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.research-deps'), str(ROOT / 'code/bioprint')]
import httpx
from bot_rule_validation import summarize


async def run():
    from research_api import app
    from research_bot_api import router
    if not any(getattr(r, 'path', '').startswith('/api/bot-rules') for r in app.routes):
        app.include_router(router)
    output = ROOT / 'research/benchmarks/bot_rules/api_replay.json'
    if output.exists():
        raise SystemExit('Preserve completed replay')
    offline = json.loads((output.parent / 'results.json').read_text())
    expected = {(r['subject'], r['session'], r['repetition']): r for r in offline['observations']}
    records, max_error, flags_equal, rules_equal = [], 0., True, True
    transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
    async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
        assert (await client.get('/api/bot-rules/samples?split=test')).status_code == 403
        offset = 0
        while True:
            response = await client.get('/api/bot-rules/samples', params={'offset': offset, 'limit': 128})
            response.raise_for_status(); page = response.json()
            if not page['rows']:
                break
            response = await client.post('/api/bot-rules/infer', json={'rows': [
                {'claimed_owner': r['claimed_owner'], 'sample': r['sample']} for r in page['rows']]})
            response.raise_for_status()
            inferred = response.json()['rows']
            assert len(inferred) == len(page['rows'])
            for raw, value in zip(page['rows'], inferred):
                stored = expected[(raw['claimed_owner'], raw['session'], raw['repetition'])]
                max_error = max(max_error, abs(value['score'] - stored['score']))
                flags_equal &= value['flagged'] == stored['flagged']
                rules_equal &= value['rules'] == stored['rules']
                records.append(value)
            offset += len(page['rows'])
            if offset >= page['total']:
                break
    summary = summarize(records)
    assert summary == offline['genuine_dev'] and summary['samples'] == 5100 and summary['flagged'] == 15
    assert max_error == 0 and flags_equal and rules_equal
    report = {'mode': 'isolated ASGI; no listener/database', 'samples_replayed': len(records),
              'every_flag_matches': flags_equal, 'every_rule_list_matches': rules_equal,
              'maximum_score_error': max_error, 'summary': summary, 'test_guard_status': 403,
              'test_accessed': False, 'limitations': offline['limitations']}
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(run())
