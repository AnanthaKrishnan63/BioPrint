"""Browser regression: a wrong middle repetition preserves earlier samples.

Mock API responses isolate browser state; server behavior has separate pytest
coverage. Run with bigidea and BIOPRINT_RESEARCH_ROOT set. No listener or DB.
"""
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

root = Path(os.environ['BIOPRINT_RESEARCH_ROOT'])
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True)
    for mode in ['login-control', 'typing-stepup', 'typing-pointer', 'typing-keypad']:
        static = root/'.worktrees'/mode/'code/bioprint/static'
        context = browser.new_context()
        page = context.new_page()
        errors, checks, final = [], [], []
        page.on('pageerror', lambda e: errors.append(str(e)))

        def route_request(route):
            path = urlparse(route.request.url).path
            if path == '/api/session':
                route.fulfill(status=401, json={})
            elif path == '/api/experiment':
                route.fulfill(json={'mode': mode})
            elif path == '/api/login':
                route.fulfill(json={'decision':'step_up', 'signals':[], 'reasons':[], 'attempt_id':1})
            elif path == '/api/login/stepup/check':
                body = route.request.post_data_json
                checks.append(body)
                route.fulfill(json={'accepted':body['password']=='demo-pass', 'reason':'Incorrect password. Repeat this entry.'})
            elif path == '/api/login/stepup':
                final.append(route.request.post_data_json)
                route.fulfill(json={'decision':'block', 'signals':[], 'reasons':[]})
            else:
                file = static/('index.html' if path=='/' else path.lstrip('/'))
                route.fulfill(body=file.read_bytes(), content_type=mimetypes.guess_type(str(file))[0] or 'text/plain')

        page.route('**/*', route_request)
        page.goto('http://bioprint.test/#login')
        page.locator('#username').fill('synthetic-browser')

        def submit(password):
            page.locator('#password').fill('')
            page.locator('#password').press_sequentially(password)
            page.locator('#submit').click()

        submit('demo-pass')
        page.wait_for_function("document.querySelector('#submit').textContent.includes('Type again 1')")
        submit('demo-pass')
        page.wait_for_function("document.querySelector('#submit').textContent.includes('Type again 2')")
        submit('wrong')
        page.wait_for_function("document.querySelector('#hint').textContent.includes('Incorrect password')")
        assert 'Type again 2' in page.locator('#submit').inner_text()
        assert '1 of 3' in page.locator('#stepup-progress').inner_text()
        assert not final
        submit('demo-pass')
        page.wait_for_function("document.querySelector('#submit').textContent.includes('Type again 3')")
        submit('demo-pass')
        page.wait_for_function("document.querySelector('#submit').textContent === 'Log in'")
        assert len(checks)==4 and len(final)==1
        assert final[0]['samples']==[checks[i]['sample'] for i in [0,2,3]]
        assert not errors, errors
        print(mode+': wrong middle entry rejected immediately; final batch retains exactly three correct entries', flush=True)
        context.close()
    browser.close()
