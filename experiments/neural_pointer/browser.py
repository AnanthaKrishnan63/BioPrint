import json, mimetypes
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
root=Path.cwd()
static=root/'.worktrees/typing-pointer/code/bioprint/static'
with sync_playwright() as p:
    browser=p.chromium.launch(executable_path='/usr/bin/google-chrome',headless=True)
    page=browser.new_page(viewport={'width':1000,'height':800})
    errors=[]; recordings=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    def route(r):
        path=urlparse(r.request.url).path
        if path=='/api/pointer/status':
            r.fulfill(json={'username':'synthetic-browser-user','count':len(recordings),'target':3,'enrolled':len(recordings)==3})
        elif path=='/api/pointer/challenge':
            r.fulfill(json={'id':'test-'+str(len(recordings)),'kind':'enroll','minimum_points':20,'minimum_ms':0,'expires_in':180})
        elif path=='/api/pointer/capture':
            recordings.append(r.request.post_data_json)
            r.fulfill(json={'accepted':True,'count':len(recordings),'target':3,'enrolled':len(recordings)==3})
        else:
            file=static/path.lstrip('/')
            r.fulfill(body=file.read_bytes(),content_type=mimetypes.guess_type(str(file))[0] or 'text/plain')
    page.route('**/*',route)
    page.goto('http://bioprint.test/pointer-enroll.html')
    for i in range(3):
        page.locator('#start').click()
        target=page.locator('#capture button')
        target.wait_for()
        box=page.locator('#capture > div').bounding_box()
        page.mouse.move(box['x']+20,box['y']+20)
        page.mouse.move(box['x']+box['width']-20,box['y']+box['height']-20,steps=35)
        target.click()
        page.wait_for_function(f'document.querySelector("#status").textContent.includes("{i+1} of") || document.querySelector("#status").textContent.includes("ready")')
    page.wait_for_function("document.querySelector('#start').hidden")
    assert page.locator('#start').is_hidden()
    assert len(recordings)==3 and all(len(r['events'])>=20 for r in recordings)
    assert all(e['trusted'] and e['pointer_type']=='mouse' for r in recordings for e in r['events'])
    assert not errors,errors
    page.screenshot(path=str(root/'.worktrees/typing-pointer/experiments/results/pointer-enrollment-browser.png'))
    print(json.dumps({'enrollment_recordings':len(recordings),'native_event_counts':[len(r['events']) for r in recordings],'page_errors':errors}))
    page.close()
    page=browser.new_page(viewport={'width':1000,'height':900})
    errors=[]; checked=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    def login_route(r):
        path=urlparse(r.request.url).path
        if path=='/api/session': r.fulfill(status=401,json={})
        elif path=='/api/experiment': r.fulfill(json={'mode':'typing-pointer'})
        elif path=='/api/login': r.fulfill(json={'decision':'keypad','signals':[{'name':'pointer_neural','available':False}],'reasons':[],'attempt_id':1})
        elif path=='/api/pointer/challenge': r.fulfill(json={'id':'verify','kind':'verify','minimum_points':20,'minimum_ms':0,'expires_in':180})
        elif path=='/api/pointer/capture':
            checked.append(r.request.post_data_json)
            r.fulfill(json={'decision':'block','signals':[{'name':'pointer_neural','score':.2,'threshold':.05,'flagged':True}],'reasons':['Pointer movement did not match your profile'],'attempt_id':2})
        else:
            file=static/('index.html' if path=='/' else path.lstrip('/'))
            r.fulfill(body=file.read_bytes(),content_type=mimetypes.guess_type(str(file))[0] or 'text/plain')
    page.route('**/*',login_route)
    page.goto('http://bioprint.test/#login')
    page.locator('#username').fill('synthetic-browser-user')
    page.locator('#password').press_sequentially('test-only')
    page.locator('#submit').click()
    target=page.locator('#keypad-mount button')
    target.wait_for()
    box=page.locator('#keypad-mount > div').bounding_box()
    page.mouse.move(box['x']+20,box['y']+20)
    page.mouse.move(box['x']+box['width']-20,box['y']+box['height']-20,steps=35)
    target.click()
    page.wait_for_function("document.querySelector('#submit').textContent === 'Log in'")
    assert len(checked)==1 and checked[0]['kind']=='verify'
    assert len(checked[0]['events'])>=20 and all(e['trusted'] for e in checked[0]['events'])
    assert not errors,errors
    print(json.dumps({'neural_login_capture':True,'page_errors':errors}))
    browser.close()
