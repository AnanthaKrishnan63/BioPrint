import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def lan(client, monkeypatch):
    monkeypatch.setenv('LAN_HOST', '192.168.1.20')
    monkeypatch.setenv('LAN_PORT', '8443')
    import lan_gateway
    importlib.reload(lan_gateway)
    gateway = lan_gateway.LanGateway(client.app, 'x' * 48)
    with TestClient(gateway, base_url='https://192.168.1.20:8443') as browser:
        browser.auth = ('participant', 'x' * 48)
        yield browser


def test_invitation_tls_host_and_origin(lan):
    assert lan.get('/').status_code == 200
    assert lan.get('/', auth=('participant', 'wrong')).status_code == 401
    assert lan.get('http://192.168.1.20:8443/').status_code == 400
    assert lan.get('/', headers={'Host': 'attacker.example'}).status_code == 400
    assert lan.post('/api/logout', headers={'Origin': 'https://attacker.example'}).status_code == 403


def test_private_routes_and_proxy_spoofing(lan):
    for path in ['/dashboard.html', '/dashboard.js', '/bioprint.db', '/docs', '/openapi.json', '/server.py']:
        assert lan.get(path, headers={'X-Forwarded-For': '127.0.0.1'}).status_code == 404
    assert lan.get('/api/attempts?user=someone').status_code == 403
    assert lan.post('/api/attempts/1/label', json={'label': 'genuine'}).status_code == 404


def test_enrollment_status_and_request_limits(lan):
    response = lan.post('/api/register', json={'username': 'lan-user', 'password': 'DemoOnly123!'})
    assert response.status_code == 200, response.text
    status = lan.get('/api/users/lan-user')
    assert status.status_code == 200
    assert 'habits' not in status.json()
    assert status.json()['enroll_count'] == 0
    assert lan.post('/api/login', json={}).status_code == 422
    assert lan.post('/api/login', content='x' * 1_048_577, headers={'Content-Type': 'application/json'}).status_code == 413


def test_history_forced_to_signed_in_user_and_secure_cookie(lan, monkeypatch):
    import lan_gateway
    import server
    from fastapi import Response
    # A real signed cookie must scope history regardless of supplied query.
    cookie = server._session_token('alice')
    lan.cookies.set(server.SESSION_COOKIE, cookie)
    captured = {}
    async def fake_app(scope, receive, send):
        captured['query'] = scope['query_string']
        response = Response('ok')
        response.set_cookie('bioprint_session', 'example', httponly=True)
        await response(scope, receive, send)
    lan.app.app = fake_app
    response = lan.get('/api/attempts?user=bob&limit=500')
    assert captured['query'] == b'user=alice&limit=1'
    assert 'Secure' in response.headers['set-cookie']
    assert response.headers['cache-control'] == 'no-store'


def test_rate_limit(lan):
    for _ in range(180):
        assert lan.get('/', auth=('participant', 'wrong')).status_code == 401
    assert lan.get('/').status_code == 429
