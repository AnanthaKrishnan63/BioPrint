"""Offline Chromium UI smoke checks; API responses are explicit fixtures.

Actual API scoring is covered separately by replay.py. No listening server.
"""
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

root = Path(os.environ['BIOPRINT_RESEARCH_ROOT'])
results = []
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', headless=True)
    for mode in ['login-control','typing-stepup','typing-pointer','typing-keypad']:
        app = root/'.worktrees'/mode/'code/bioprint'
        for mobile in [False, True]:
            context = browser.new_context(viewport={'width':390,'height':844} if mobile else {'width':1280,'height':900},
                                          is_mobile=mobile,has_touch=mobile)
            page = context.new_page()
            errors = []
            page.on('pageerror', lambda e: errors.append(str(e)))
            def handle(route):
                path = urlparse(route.request.url).path
                if path=='/api/experiment':route.fulfill(json={'mode':mode})
                elif path=='/api/session':route.fulfill(status=401,json={'detail':'not signed in'})
                elif path=='/api/login':
                    decision='step_up' if mode=='typing-stepup' else 'keypad'
                    route.fulfill(json={'decision':decision,'reasons':[],'signals':[],'latency_ms':1.,'attempt_id':1})
                elif path=='/api/keypad/challenge':
                    route.fulfill(json={'id':'synthetic-ui','layout':[0,1,2,3,4,-2],'target':[0,1,2,3,4,0],'expires_at':'2099-01-01T00:00:00Z'})
                elif path.startswith('/api/'):
                    route.fulfill(status=400,json={'detail':'unexpected fixture request'})
                else:
                    file=app/'static'/('index.html' if path=='/' else path.lstrip('/'))
                    if file.is_file():route.fulfill(body=file.read_bytes(),content_type=mimetypes.guess_type(str(file))[0] or 'text/plain')
                    else:route.fulfill(status=404,body='not found')
            page.route('**/*',handle)
            page.goto('http://bioprint.test/#login')
            page.locator('#username').fill('synthetic-ui')
            page.locator('#password').fill('synthetic-password')
            page.locator('#reveal').click()
            assert page.locator('#password').get_attribute('type')=='text'
            if mode!='login-control':
                page.locator('#submit').click()
                if mode=='typing-stepup':
                    page.wait_for_function("document.querySelector('#submit').textContent.includes('Type again')",timeout=20000)
                else:
                    page.wait_for_selector('.keypad-on',timeout=20000)
                    page.wait_for_selector('.kp-key',timeout=10000)
                    page.locator('.kp-key').first.click()
            assert not errors,errors
            results.append({'mode':mode,'mobile':mobile,'typing_and_clicks':'passed','stepup_ui':'passed' if mode!='login-control' else 'not exercised','page_errors':errors})
            print(mode, 'mobile' if mobile else 'desktop', 'passed',flush=True)
            context.close()
    browser.close()
(root/'research/benchmarks/login_combinations_v3/browser-smoke.json').write_text(json.dumps({'scope':'UI with mocked API; not biometric validation','results':results},indent=2)+'\n')
