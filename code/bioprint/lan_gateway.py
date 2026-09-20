"""Restricted HTTPS entry point for invited LAN participants.

Start with run-lan.sh. The regular localhost app remains available to the host.
"""
import base64
import hmac
import os
import time
from collections import defaultdict, deque
from urllib.parse import urlencode

from starlette.requests import Request
from starlette.responses import JSONResponse

from server import app as application, read_session, user_status


class LanGateway:
    def __init__(self, app, token):
        if len(token) < 24:
            raise ValueError("A random LAN invitation password of at least 24 characters is required")
        self.app = app
        self.credential = base64.b64encode(f"participant:{token}".encode())
        self.requests = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request = Request(scope)

        async def reject(status, message, headers=None):
            response = JSONResponse({'detail': message}, status_code=status, headers=headers)
            await response(scope, receive, send)

        if scope.get('scheme') != 'https':
            return await reject(400, 'HTTPS is required')
        host = request.headers.get('host', '')
        expected_host = f"{os.environ['LAN_HOST']}:{os.environ.get('LAN_PORT', '8443')}"
        if host != expected_host:
            return await reject(400, 'Invalid host')
        now = time.monotonic()
        # Bound both per-client request rates and bookkeeping memory.
        for client, times in list(self.requests.items()):
            if not times or now - times[-1] >= 60:
                del self.requests[client]
        client = scope.get('client', ('unknown', 0))[0]
        if client not in self.requests and len(self.requests) >= 1024:
            return await reject(429, 'Try again later')
        times = self.requests[client]
        while times and now - times[0] >= 60:
            times.popleft()
        if len(times) >= 180:
            return await reject(429, 'Try again in a minute', {'Retry-After': '60'})
        times.append(now)
        auth = request.headers.get('authorization', '').encode()
        if not hmac.compare_digest(auth, b'Basic ' + self.credential):
            return await reject(401, 'Invitation required', {
                'WWW-Authenticate': 'Basic realm="BioPrint invited participants", charset="UTF-8"',
                'Cache-Control': 'no-store',
            })
        origin = request.headers.get('origin')
        if (origin and origin != f'https://{expected_host}') or request.headers.get('sec-fetch-site') == 'cross-site':
            return await reject(403, 'Cross-site requests are disabled')

        path, method = scope['path'], scope['method']
        assets = {'/', '/index.html', '/welcome.html', '/styles.css', '/app.js',
                  '/welcome.js', '/ui.js', '/capture.js', '/pointer.js', '/probe.js', '/keypad.js',
                  '/neural-pointer.js', '/pointer-enroll.html', '/pointer-enroll.js'}
        posts = {'/api/register', '/api/enroll', '/api/login', '/api/login/stepup',
                 '/api/keypad/challenge', '/api/enroll/keypad', '/api/login/keypad', '/api/logout',
                 '/api/pointer/challenge', '/api/pointer/capture'}
        allowed = (method in {'GET', 'HEAD'} and path in assets) or (method == 'POST' and path in posts)
        allowed |= method == 'GET' and path in {'/api/session', '/api/attempts', '/api/pointer/status', '/api/experiment'}
        allowed |= method == 'GET' and path.startswith('/api/users/')
        if not allowed:
            return await reject(404, 'Not available on the participant interface')
        if path == '/api/attempts':
            session = read_session(request)
            if not session:
                return await reject(403, 'Sign in to view your latest attempt')
            scope = dict(scope, query_string=urlencode({'user': session['username'], 'limit': 1}).encode())
        if method == 'POST':
            if path != '/api/logout' and request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return await reject(415, 'JSON required')
            body = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                body.extend(message.get('body', b''))
                if len(body) > (4_194_304 if path == '/api/pointer/capture' else 1_048_576):
                    return await reject(413, 'Request too large')
                if not message.get('more_body', False):
                    break
            async def buffered_receive():
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            receive = buffered_receive

        async def secure_send(message):
            if message['type'] == 'http.response.start':
                headers = []
                for name, value in message.get('headers', []):
                    if name.lower() == b'set-cookie':
                        value += b'; Secure'
                    if name.lower() != b'cache-control':
                        headers.append((name, value))
                headers.extend([(b'cache-control', b'no-store'), (b'x-content-type-options', b'nosniff'),
                                (b'x-frame-options', b'DENY'), (b'referrer-policy', b'no-referrer')])
                message = dict(message, headers=headers)
            await send(message)

        if path.startswith('/api/users/'):
            # Enrollment resumption needs counts, never behavioral statistics.
            from fastapi import HTTPException
            from starlette.concurrency import run_in_threadpool
            try:
                status = await run_in_threadpool(user_status, path[len('/api/users/'):])
            except HTTPException as exc:
                return await reject(exc.status_code, exc.detail)
            fields = {'enrolled', 'enroll_count', 'enroll_target', 'keypad_enrolled', 'keypad_count', 'keypad_target'}
            return await JSONResponse({k: v for k, v in status.items() if k in fields})(scope, receive, secure_send)
        await self.app(scope, receive, secure_send)


def create_app():
    return LanGateway(application, os.environ.get('LAN_INVITE_PASSWORD', ''))
