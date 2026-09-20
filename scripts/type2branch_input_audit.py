"""Generated-input checks for published preprocessing/CSV interface contracts.

Extracts only the reviewed prepare_sample function from GPLv3 author source;
does not execute its module, load datasets, or initialize training.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / 'research/benchmarks/references/type2branch'
SYNTH = ROOT / 'research/benchmarks/references/type2branch_synthesis'
OUT = ROOT / 'research/benchmarks/type2branch_input_audit_v1'


def reference_prepare():
    tree = ast.parse((REF / 'generate_dataset.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare_sample']
    if len(functions) != 1: raise ValueError('Expected one reference function')
    module = ast.Module(body=functions, type_ignores=[])
    scope = {'np': np}
    exec(compile(module, str(REF / 'generate_dataset.py'), 'exec'), scope)
    return scope['prepare_sample']


def integer_csv_compatibility(text):
    """Structural emulation of the inspected C# reader, not runtime parity."""
    accepted, rejected = 0, 0
    # C# unconditionally skips the first line; treats remaining non-integer
    # values as parse exceptions and silently omits those rows.
    for line in text.splitlines()[1:]:
        if not line.strip(): continue
        fields = line.strip().split(',')
        if len(fields) >= 3 and all(re.fullmatch(r'[+-]?\d+', v.strip()) for v in fields[:3]):
            accepted += 1
        else:
            rejected += 1
    return {'accepted': accepted, 'rejected': rejected}


def main():
    OUT.mkdir(exist_ok=False)
    inputs = [REF / 'generate_dataset.py', REF / 'merge_synth_features.py',
              SYNTH / 'source/KSD-SLD/Datasets/Readers/CsvDatasetReader.cs',
              SYNTH / 'source/KSD-SLD/Program.cs', Path(__file__)]
    plan = {'scope': 'Generated arrays and reviewed source only; no dataset observations',
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    (OUT / 'source.py').write_bytes(Path(__file__).read_bytes())
    prepare = reference_prepare()
    # Synthetic press/release times in milliseconds, with a >30-second gap/hold.
    events = np.array([[0., 120., 65.], [250., 350., 66.], [40000., 75000., 67.]])
    actual = prepare(events, 1000.)
    expected = np.array([[65/255, .12, 0], [66/255, .1, .25], [67/255, 30, 30]])
    np.testing.assert_allclose(actual[:3], expected, atol=1e-15, rtol=0)
    assert actual.shape == (100, 3) and np.count_nonzero(actual[3:]) == 0
    buffer = io.StringIO(); np.savetxt(buffer, actual, delimiter=',', fmt='%s')
    normalized = integer_csv_compatibility(buffer.getvalue())
    assert normalized == {'accepted': 0, 'rejected': 99}
    integer_fixture = integer_csv_compatibility('VK,HT,FT\n65,120,-1\n66,100,250\n')
    assert integer_fixture == {'accepted': 2, 'rejected': 0}
    report = {'status': 'complete_generated_interface_audit',
              'reference_output_shape': list(actual.shape), 'seconds_clip_observed_in_code': 30,
              'keycode_divisor': 255, 'timing_divisor': 1000,
              'normalized_direct_csv_reader_emulation': normalized,
              'integer_header_csv_reader_emulation': integer_fixture,
              'limitations': ['C# parser behavior emulated from source, not executed in .NET.',
                              'Paper clipping10s differs from code30s; fidelity choice unresolved.',
                              'Synthesis mode/context/RNG/residual conversion still need validation.',
                              'No accuracy or full reproduction claim.'],
              'dataset_observations_read': False}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__': main()
