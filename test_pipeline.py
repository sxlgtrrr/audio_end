"""测试全管线"""
import requests, json, os, sys, io, numpy as np
sys.path.insert(0, '.')
import config
from data_processor import AudioProcessor
from scipy.io import wavfile as wf

# 1. 说话人列表
print("=== 说话人列表 ===")
r = requests.get('http://127.0.0.1:5000/api/speaker/list')
print(json.dumps(r.json(), ensure_ascii=False))

# 2. 注册说话人
print("\n=== 注册说话人 ===")
proc = AudioProcessor()
fp = f"data/train/happy/{os.listdir('data/train/happy')[0]}"
audio = proc.load_audio(fp)
audio = proc.normalize_audio(audio)
audio_i16 = (audio * 32767).astype(np.int16)
buf = io.BytesIO()
wf.write(buf, config.SAMPLE_RATE, audio_i16)
import base64
b64 = base64.b64encode(buf.getvalue()).decode()
r = requests.post('http://127.0.0.1:5000/api/speaker/register',
    json={'name': 'TestUser', 'audio': b64})
print(r.json())

# 3. 测试情绪识别
print("\n=== 情绪识别测试 ===")
for emo in config.EMOTIONS:
    d = f'data/test/{emo}'
    fs = [f for f in os.listdir(d) if f.endswith('.wav')][:1]
    if not fs: continue
    fp = os.path.join(d, fs[0])
    audio = proc.load_audio(fp)
    audio = proc.normalize_audio(audio)
    audio_i16 = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    wf.write(buf, config.SAMPLE_RATE, audio_i16)
    r = requests.post('http://127.0.0.1:5000/api/analyze', data=buf.getvalue())
    data = r.json()
    if 'error' in data:
        print(f'  ✗ {emo}: ERROR - {data["error"]}')
        continue
    mark = '✓' if data['emotion'] == emo else '✗'
    spkr = data.get('speaker','') or '未知'
    print(f'  {mark} {emo:10s} -> {data["emotion"]:10s} conf={data["confidence"]}% speaker={spkr} tts_rate={data.get("tts_params",{}).get("rate","?")}')

print("\n=== 完成 ===")
