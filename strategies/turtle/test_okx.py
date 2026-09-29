#!/usr/bin/env python3
"""OKX API 连接测试 — 放在 turtle/ 目录下运行."""
import hmac, hashlib, base64, json, time
import urllib.request, ssl

from pathlib import Path
cfg = json.load(open(Path(__file__).resolve().parent / 'config.json'))   # 不管从哪个目录运行都能找到
print(f"apiKey: {cfg['apiKey'][:8]}...{cfg['apiKey'][-4:]}")
print(f"secret: {cfg['secret'][:4]}...{cfg['secret'][-4:]}")
print(f"password: {'*' * len(cfg.get('password', ''))}")
print(f"proxy: {cfg.get('proxy', '无')}")
print()

ts = time.strftime('%Y-%m-%dT%H:%M:%S.000Z', time.gmtime())
path = '/api/v5/account/balance'
msg = ts + 'GET' + path
sig = base64.b64encode(
    hmac.new(cfg['secret'].encode(), msg.encode(), hashlib.sha256).digest()
).decode()

headers = {
    'OK-ACCESS-KEY': cfg['apiKey'],
    'OK-ACCESS-SIGN': sig,
    'OK-ACCESS-TIMESTAMP': ts,
    'OK-ACCESS-PASSPHRASE': cfg.get('password', ''),
    'Content-Type': 'application/json',
}

url = 'https://www.okx.com' + path
print(f'请求: GET {url}')
print(f'时间戳: {ts}')
print()

try:
    req = urllib.request.Request(url, headers=headers)

    handlers = []
    if cfg.get('proxy'):
        handlers.append(urllib.request.ProxyHandler({
            'http': cfg['proxy'],
            'https': cfg['proxy'],
        }))

    ctx = ssl.create_default_context()
    handlers.append(urllib.request.HTTPSHandler(context=ctx))
    opener = urllib.request.build_opener(*handlers)

    resp = opener.open(req, timeout=15)
    body = resp.read().decode()
    data = json.loads(body)

    if data.get('code') == '0':
        print('✓ 连接成功!')
        print(json.dumps(data, indent=2, ensure_ascii=False)[:500])
    else:
        print(f"✗ OKX 返回错误:")
        print(f"  code: {data.get('code')}")
        print(f"  msg:  {data.get('msg')}")

except Exception as e:
    print(f'✗ 请求失败: {type(e).__name__}: {e}')
