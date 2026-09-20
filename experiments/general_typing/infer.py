"""Score browser-format JSON offline without opening a database or listener.

Input: {"enrollment": [ten Sample objects], "attempt": Sample}.
Use only locally generated trusted joblib artifacts listed in frozen.json.
This command checks behavior, not credentials, bots, or device context.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

BRANCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BRANCH / 'code/bioprint'))
import joblib
from contracts import Sample
from engine.features import needs_retype, password_keystrokes
from engine.general_typing import from_keystrokes, profile, comparison
from engine import features, scorer, aligned_typing


def score_input(payload, artifact):
    samples = [Sample.model_validate(s) for s in payload['enrollment']]
    if len(samples) != 10:
        raise ValueError('Exactly ten enrollment repetitions required by this experiment')
    attempt = Sample.model_validate(payload['attempt'])
    template = [k.code for k in password_keystrokes(samples[0])]
    for sample in [*samples, attempt]:
        if reason := needs_retype(sample, template):
            raise ValueError(reason)
    if artifact.get('schema') == 'aligned-personal-residuals-v1':
        vectors = [features.keystroke_vector(password_keystrokes(s)) for s in samples]
        model_profile = scorer.fit([v.values for v in vectors], vectors[0].names)
        query = features.keystroke_vector(password_keystrokes(attempt))
        x = aligned_typing.comparison(model_profile, [query.values])
    else:
        p = profile([from_keystrokes(password_keystrokes(s)) for s in samples])
        x = comparison(p, [from_keystrokes(password_keystrokes(attempt))])
    model = artifact['model']
    if model is None:
        score = -x[0, -1] if artifact.get('schema') == 'aligned-personal-residuals-v1' else -x[0, :21].clip(max=6).mean()
    elif hasattr(model, 'decision_function'):
        score = model.decision_function(x)[0]
    else:
        score = model.predict_proba(x)[0, 1]
    return {'score': float(score), 'threshold': float(artifact['threshold']),
            'typing_match': bool(score >= artifact['threshold']),
            'scope': 'experimental behavioral score only; not login authorization'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', choices=['pooled/distance', 'pooled/logistic', 'pooled/extra_trees'], required=True)
    parser.add_argument('--representation', choices=['summary', 'aligned'], default='summary')
    args = parser.parse_args()
    results = Path(__file__).parent / ('results' if args.representation == 'summary' else 'aligned_results')
    entry = json.loads((results / 'frozen.json').read_text())[args.model]
    path = results / entry['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('Model artifact checksum mismatch')
    print(json.dumps(score_input(json.loads(args.input.read_text()), joblib.load(path)), indent=2))


if __name__ == '__main__':
    main()
