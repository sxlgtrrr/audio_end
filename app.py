"""
语音情感感知助手 - Web 前端
Flask 后端: Whisper 语音识别 + Wav2Vec2 情感识别 + 情感感知回复
"""
import os
import sys
import torch
import numpy as np
import whisper
import pyttsx3
import random
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from scipy.io import wavfile
import io
import struct

import config
from models import get_model

app = Flask(__name__)

# ---- 全局初始化 ----
print("=" * 60)
print("  语音情感感知助手 - Web 版初始化...")
print("=" * 60)

# 1. Whisper (small 模型，中文效果更好)
print("[1/3] 加载 Whisper 语音识别模型 (small)...")
asr_model = whisper.load_model("small")

# 2. Wav2Vec2 情感识别
print("[2/3] 加载 Wav2Vec2 情感识别模型...")
emotion_model = get_model('wav2vec2')
model_path = 'models/best_model_wav2vec2_best.pth'
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=config.DEVICE)
    model_dict = emotion_model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                      if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    emotion_model.load_state_dict(model_dict, strict=False)
    print(f"   ✓ 验证准确率: {ckpt['best_val_acc']:.2f}%")
else:
    print("   ⚠ 未找到训练模型")
emotion_model.eval()

# 3. TTS
print("[3/3] 初始化语音合成...")
tts = pyttsx3.init()
tts.setProperty('rate', 170)
tts.setProperty('volume', 0.9)

print("\n   ✓ 初始化完成，访问 http://localhost:5000\n")

# ---- 情感回复库 ----
EMOTION_RESPONSES = {
    'angry': [
        "我能感受到你的愤怒，让我们冷静下来想一想。",
        "我理解你现在很生气，深呼吸，慢慢说。",
        "情绪激动的时候，不如先休息一下？我在这里陪你。",
    ],
    'disgust': [
        "听起来你对这件事很不满意。",
        "我能理解你的反感，换个角度想想可能会有帮助。",
    ],
    'fear': [
        "别担心，我会一直陪着你。",
        "恐惧有时候是因为未知——说出来会好受一点。",
        "你很安全，我听到了你的不安。",
    ],
    'happy': [
        "听你这么说我也很开心！继续保持好心情！",
        "快乐是会传染的，谢谢你分享这份喜悦。",
        "太好了！今天真是个美好的日子！",
    ],
    'neutral': [
        "收到，请继续说吧。",
        "嗯，我在听。",
        "好的，还有别的想聊的吗？",
    ],
    'sad': [
        "我能感受到你的悲伤，想聊聊发生了什么吗？",
        "难过的时候不用一个人扛，我在这里。",
        "有时候说出来就会好受很多，你愿意试试吗？",
    ],
}

EMOTION_ICONS = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨',
    'happy': '😊', 'neutral': '😐', 'sad': '😢'
}

EMOTION_COLORS = {
    'angry': '#e74c3c', 'disgust': '#8e44ad', 'fear': '#f39c12',
    'happy': '#2ecc71', 'neutral': '#95a5a6', 'sad': '#3498db'
}


def base64_to_audio(b64_string):
    """Base64 WAV -> numpy audio array, resample to 16kHz"""
    wav_bytes = base64.b64decode(b64_string)
    # WAV header is 44 bytes, then PCM 16-bit data
    audio_16k = np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32768.0
    # 如果超过 5 秒，截断
    max_len = config.SAMPLE_RATE * 5
    if len(audio_16k) > max_len:
        audio_16k = audio_16k[:max_len]
    return audio_16k


def audio_to_wav_bytes(audio, sample_rate=16000):
    """numpy audio -> WAV bytes"""
    audio_int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sample_rate, audio_int16)
    return buf.getvalue()


@app.route('/')
def index():
    return render_template('index.html', emotions=config.EMOTIONS)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    audio_b64 = data.get('audio', '')

    if not audio_b64:
        return jsonify({'error': 'No audio data'}), 400

    # 1. 解码音频
    audio = base64_to_audio(audio_b64)

    # 2. Whisper 语音识别
    text = ""
    try:
        result = asr_model.transcribe(audio, language='zh', fp16=False)
        text = result['text'].strip()
    except Exception as e:
        print(f"ASR error: {e}")
        text = "[识别失败]"

    # 3. Wav2Vec2 情感识别
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        outputs = emotion_model(audio_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1).item()
        confidence = probs[0, pred].item()

    emotion = config.EMOTIONS[pred]
    all_probs = {e: float(p) for e, p in zip(config.EMOTIONS, probs[0].cpu().numpy())}

    # 4. 生成回复
    responses = EMOTION_RESPONSES.get(emotion, EMOTION_RESPONSES['neutral'])
    if text and len(text) > 1 and text != "[识别失败]":
        response = f"你说「{text}」——{random.choice(responses)}"
    else:
        response = random.choice(responses)

    return jsonify({
        'text': text,
        'emotion': emotion,
        'emotion_icon': EMOTION_ICONS.get(emotion, ''),
        'emotion_color': EMOTION_COLORS.get(emotion, '#666'),
        'confidence': round(confidence * 100, 1),
        'probabilities': all_probs,
        'response': response,
    })


@app.route('/api/speak', methods=['POST'])
def speak():
    """TTS 朗读回复"""
    data = request.json
    text = data.get('text', '')
    if text:
        tts.say(text)
        tts.runAndWait()
    return jsonify({'ok': True})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
