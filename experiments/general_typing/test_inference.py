import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('general_infer', Path(__file__).with_name('infer.py'))
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)


def sample(codes):
    return {'keystrokes': [event for i, code in enumerate(codes) for event in (
        {'code': code, 'type': 'down', 't': i * 130., 'field': 'password'},
        {'code': code, 'type': 'up', 't': i * 130. + 80., 'field': 'password'})]}


@pytest.mark.parametrize('codes', [['KeyA', 'KeyB', 'KeyC'], ['Digit1', 'KeyX', 'Period', 'KeyZ']])
@pytest.mark.parametrize('schema', ['general-typing-summary-v1', 'aligned-personal-residuals-v1'])
def test_browser_input_for_non_cmu_passwords(codes, schema):
    s = sample(codes)
    result = infer.score_input({'enrollment': [s] * 10, 'attempt': s}, {'model': None, 'threshold': -1., 'schema': schema})
    assert result['typing_match'] and result['score'] == 0.


def test_unusable_input_does_not_receive_score():
    s = sample(['KeyA', 'KeyB', 'KeyC'])
    pasted = {**s, 'meta': {'had_paste': True}}
    with pytest.raises(ValueError, match='pasted'):
        infer.score_input({'enrollment': [s] * 10, 'attempt': pasted}, {'model': None, 'threshold': -1.})
