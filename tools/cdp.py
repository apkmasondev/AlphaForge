"""Tiny Chrome DevTools Protocol client for driving the WebView2 UI in tests.
usage: cdp.py eval "<js expression>"   |   cdp.py shot out.png   |   cdp.py click "<css selector>"
"""
import base64, json, sys, urllib.request
import websocket

def target():
    tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9222/json"))
    for t in tabs:
        if t.get("type") == "page":
            return t["webSocketDebuggerUrl"]
    raise SystemExit("no page")

ws = websocket.create_connection(target(), timeout=120, suppress_origin=True)
_id = 0
def call(method, **params):
    global _id
    _id += 1
    ws.send(json.dumps({"id": _id, "method": method, "params": params}))
    while True:
        m = json.loads(ws.recv())
        if m.get("id") == _id:
            if "error" in m: raise SystemExit(m["error"])
            return m["result"]

cmd = sys.argv[1]
if cmd == "eval":
    r = call("Runtime.evaluate", expression=sys.argv[2], awaitPromise=True, returnByValue=True)
    if "exceptionDetails" in r: print("EXC", json.dumps(r["exceptionDetails"])[:2000])
    else: print(json.dumps(r["result"].get("value"), indent=1)[:20000])
elif cmd == "shot":
    r = call("Page.captureScreenshot", format="png")
    open(sys.argv[2], "wb").write(base64.b64decode(r["data"]))
    print(sys.argv[2])
elif cmd == "click":
    js = f"(()=>{{const e=document.querySelector({json.dumps(sys.argv[2])}); if(!e) return 'not found'; e.click(); return 'ok'}})()"
    print(call("Runtime.evaluate", expression=js, returnByValue=True)["result"].get("value"))
