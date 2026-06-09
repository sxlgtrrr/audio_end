"""
语音情感感知助手 - Flask 后端
接受原始 PCM 音频，返回识别结果
"""
import os
import torch
import numpy as np
import whisper
import pyttsx3
import random
import io
import struct
import tempfile
import soundfile as sf
from flask import Flask, render_template, request, jsonify

import config
from models import get_model

app = Flask(__name__)

# ---- 全局初始化 ----
print("=" * 60)
print("  语音情感感知助手 - Web 版")
print("=" * 60)

print("[1/3] 加载 Whisper (small)...")
asr_model = whisper.load_model("small")

print("[2/3] 加载 Wav2Vec2 情感模型...")
emotion_model = get_model('wav2vec2')
model_path = 'models/best_model_wav2vec2_best.pth'
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=config.DEVICE)
    model_dict = emotion_model.state_dict()
    pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                      if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(pretrained_dict)
    emotion_model.load_state_dict(model_dict, strict=False)
    print(f"   val_acc={ckpt['best_val_acc']:.1f}%")
emotion_model.eval()

print("[3/3] 初始化 TTS...")
tts = pyttsx3.init()
tts.setProperty('rate', 170)
tts.setProperty('volume', 0.9)
print("   就绪: http://localhost:5000\n")

# ---- 情感回复 ----
RESPONSES = {
    'angry': ["我能感受到你的愤怒，深呼吸，慢慢说。","我理解你很生气，我们一起来看看怎么办。","情绪激动时不如先停一下，我在这里。","生气很正常，说出来会好受些。"],
    'disgust': ["听起来你对这件事很不满。","我理解你的反感。","这种感觉确实让人不舒服。"],
    'fear': ["别担心，我在这里陪你。","恐惧有时来自未知——说出来就好了。","你是安全的。"],
    'happy': ["你的快乐感染了我！","太好了！今天真棒！","保持这个状态！"],
    'neutral': ["收到，请继续。","嗯，我在听。","还有想说的吗？"],
    'sad': ["我能感受到你的悲伤，想聊聊吗？","难过不用一个人扛。","说出来会好受很多，你愿意试试吗？","抱抱你，一切都会好起来。"],
}

ICONS = {'angry':'😠','disgust':'🤢','fear':'😨','happy':'😊','neutral':'😐','sad':'😢'}
COLORS = {'angry':'#e74c3c','disgust':'#8e44ad','fear':'#f39c12','happy':'#2ecc71','neutral':'#95a5a6','sad':'#3498db'}


@app.route('/')
def index():
    return render_template('index.html', emotions=config.EMOTIONS)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        raw_bytes = request.get_data()
        if len(raw_bytes) < 1600:
            return jsonify({'error': '音频太短'}), 400

        audio = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        if len(audio) == 0:
            return jsonify({'error': '空音频'}), 400

        # 获取原始采样率，重采样到 16kHz
        src_sr = int(request.args.get('sr', 16000))
        if src_sr != config.SAMPLE_RATE and src_sr > 0:
            from scipy import signal as scipy_signal
            audio = scipy_signal.resample(audio, int(len(audio) * config.SAMPLE_RATE / src_sr))

        # 归一化
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # 截断到 5 秒
        max_len = config.SAMPLE_RATE * 5
        if len(audio) > max_len:
            audio = audio[:max_len]

        # 保存临时 WAV 给 Whisper
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, config.SAMPLE_RATE)
            wav_path = f.name

        # 语音识别
        text = ""
        try:
            result = asr_model.transcribe(wav_path, language='zh', fp16=False)
            text = result['text'].strip()
        except:
            text = "[识别失败]"

        os.unlink(wav_path)

        # 情感识别
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            outputs = emotion_model(audio_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = round(probs[0, pred].item() * 100, 1)

        emotion = config.EMOTIONS[pred]
        all_probs = {e: float(p) for e, p in zip(config.EMOTIONS, probs[0].cpu().numpy())}

        # 生成回复
        pool = RESPONSES.get(emotion, RESPONSES['neutral'])
        if text and len(text) > 1 and text != "[识别失败]":
            response = f"「{text}」——{random.choice(pool)}"
        else:
            response = random.choice(pool)

        return jsonify({
            'text': text,
            'emotion': emotion,
            'icon': ICONS[emotion],
            'color': COLORS[emotion],
            'confidence': conf,
            'probabilities': all_probs,
            'response': response,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/speak', methods=['POST'])
def speak():
    text = request.json.get('text', '')
    if text:
        tts.say(text)
        tts.runAndWait()
    return jsonify({'ok': True})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
