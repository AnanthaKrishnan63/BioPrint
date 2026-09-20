"""Meaningful policy boundaries and real API checks for experimental branches."""
import pytest
from contracts import SignalResult
from engine import experiment as policy
from helpers import PASSWORD, make_sample
from test_bot import human_env


def signals(typing=.5, device=True, bot=False):
    return [SignalResult(name='keystroke', score=typing or 0., threshold=1., available=typing is not None),
            SignalResult(name='device', available=device is not None, flagged=device is False),
            SignalResult(name='bot', flagged=bot)]


@pytest.mark.parametrize('mode',['typing-stepup','typing-pointer','typing-keypad'])
def test_strict_admission_bot_priority_and_uncertainty(monkeypatch, mode):
    monkeypatch.setattr(policy,'MODE',mode)
    assert policy.decide(signals(),keypad_enrolled=True)[0]=='allow'
    assert policy.decide(signals(bot=True),keypad_enrolled=True)[0]=='block'
    assert policy.decide(signals(typing=3),keypad_enrolled=True)[0]=='block'
    for context in [False,None]:
        assert policy.decide(signals(device=context),keypad_enrolled=True)[0] in ('step_up','keypad')
    assert policy.decide(signals(typing=1.2),keypad_enrolled=True)[0] in ('step_up','keypad')


def test_repeat_typing_cannot_loop_or_accept_absent_evidence():
    assert policy.decide(signals(typing=None),after_step_up=True)[0]=='block'
    assert policy.decide(signals(typing=1.01),after_step_up=True)[0]=='block'
    assert policy.decide(signals(typing=1.),after_step_up=True)[0]=='allow'


@pytest.mark.parametrize('score,decision', [(1.4, 'allow'), (1.7, 'allow'), (1.701, 'block')])
def test_mobile_keypad_uses_full_profile_limit(monkeypatch, score, decision):
    monkeypatch.setattr(policy, 'MODE', 'typing-pointer')
    s = signals(typing=None) + [SignalResult(name='keypad', score=score, threshold=1.7),
                               SignalResult(name='keypad_motor', score=score, threshold=1.7)]
    assert policy.decide_keypad(s, mobile=True)[0] == decision
    assert policy.decide_keypad(s, mobile=False)[0] == 'block'
    s[2].flagged = True
    assert policy.decide_keypad(s, mobile=True)[0] == 'block'


def test_mobile_leeway_never_overrides_missing_evidence_or_bad_typing():
    assert policy.decide_keypad(signals(typing=None), mobile=True)[0] == 'block'
    s = signals(typing=3) + [SignalResult(name='keypad', score=.5, threshold=1.7)]
    assert policy.decide_keypad(s, mobile=True)[0] == 'block'


def test_motor_and_cognitive_branches_use_different_evidence(monkeypatch):
    s = signals(typing=1.2)+[SignalResult(name='keypad',score=.4,threshold=1.),
                           SignalResult(name='keypad_motor',score=1.4,threshold=1.)]
    monkeypatch.setattr(policy,'MODE','typing-pointer')
    assert policy.decide_keypad(s)[0]=='block'
    assert policy.decide_keypad(s,mobile=True)[0]=='allow'
    monkeypatch.setattr(policy,'MODE','typing-keypad')
    assert policy.decide_keypad(s)[0]=='allow'
    s[0].score=3.
    assert policy.decide_keypad(s)[0]=='block'


def test_missing_motor_and_unavailable_keypad_never_pass(monkeypatch):
    monkeypatch.setattr(policy,'MODE','typing-pointer')
    assert policy.decide_keypad(signals())[0]=='block'
    assert policy.ratio(SignalResult(name='keypad',score=0,threshold=0))==float('inf')
    assert policy.ratio(SignalResult(name='keypad',score=float('nan'),threshold=1))==float('inf')


def test_background_requires_checksum(monkeypatch,tmp_path):
    p=tmp_path/'bank.json';p.write_text('{}')
    monkeypatch.setenv('BIOPRINT_BACKGROUND',str(p))
    monkeypatch.delenv('BIOPRINT_BACKGROUND_SHA256',raising=False)
    policy.background.cache_clear()
    try:
        with pytest.raises(RuntimeError,match='checksum'):policy.background()
    finally:policy.background.cache_clear()


def test_real_login_revokes_old_session_while_stepup_pending(client,monkeypatch):
    monkeypatch.delenv('BIOPRINT_BACKGROUND',raising=False)
    policy.background.cache_clear();policy.fitted.cache_clear()
    creds={'username':'policy-check','password':PASSWORD}
    assert client.post('/api/register',json=creds).status_code==200
    env=human_env(fonts=['Arial','Synthetic'],webgl_renderer='NVIDIA GeForce RTX 3060')
    for i in range(11):
        r=client.post('/api/enroll',json=dict(**creds,sample=make_sample(seed=i,env=env))).json()
        assert r['accepted'],r
    import server
    client.cookies.set(server.SESSION_COOKIE,server._session_token(creds['username']))
    changed=human_env(fonts=['Calibri','Segoe UI'],timezone='Europe/London')
    response=client.post('/api/login',json=dict(**creds,sample=make_sample(seed=99,env=changed)))
    r=response.json()
    assert 'Max-Age=0' in response.headers.get('set-cookie','')
    assert r['decision']=='step_up',r
    # Explicitly inspect the deletion header separately: a domainless test cookie
    # may coexist with an origin-bound cookie in httpx's jar.
    assert any('no compatible supervised background' in why for s in r['signals'] if s['name']=='keystroke' for why in s['reasons'])
    client.cookies.clear()
    assert client.get('/api/session').status_code==401
