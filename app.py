"""
语音情感感知助手 - Flask 后端
整合: 语音识别(Whisper) + 情感识别(Wav2Vec2) + 说话人识别(MFCC+GMM) + 个性化回复
"""
import os
import torch
import numpy as np
import whisper
import random
import tempfile
from collections import defaultdict
from flask import Flask, render_template, request, jsonify

import config
from models import get_model
from speaker_id import SpeakerRecognitionEngine

app = Flask(__name__)

# ---- 全局初始化 ----
print("=" * 60)
print("  语音情感感知智能助手 - Web 版")
print("=" * 60)

print("[1/4] 加载 Whisper (small)...")
asr_model = whisper.load_model("small")

print("[2/4] 加载 Wav2Vec2 情感模型...")
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

print("[3/4] 加载说话人识别引擎 (MFCC+GMM)...")
speaker_engine = SpeakerRecognitionEngine()

# 每个说话人的情绪历史
speaker_emotion_history = defaultdict(list)


def get_emotion_trend(name):
    """分析说话人近期情绪趋势"""
    history = speaker_emotion_history.get(name, [])
    if len(history) < 2:
        return None
    recent = history[-5:]  # 最近 5 次
    counts = {}
    for e in recent:
        counts[e] = counts.get(e, 0) + 1
    dominant = max(counts, key=counts.get)
    return {
        'dominant': dominant,
        'counts': counts,
        'total': len(recent),
        'today_first': history[-1] if history[-1] != history[-2] else None
    }

print("[4/4] 就绪: http://localhost:5000\n")

# ---- 情感回复 ----
RESPONSES = {
    'angry': ["深呼吸，我在这里陪你。","我能感受到你的愤怒，我们一起来看看怎么办。","情绪激动时不如先停一下。","生气很正常，说出来会好受些。"],
    'disgust': ["听起来你对这件事很不满。","我理解你的反感。","这种感觉确实让人不舒服。"],
    'fear': ["别担心，我在这里陪你。","你很安全，我听到了你的不安。","恐惧有时来自未知——说出来就好了。"],
    'happy': ["你的快乐感染了我！","太好了！今天真棒！","保持这个状态！"],
    'neutral': ["收到，请继续。","嗯，我在听。","还有想说的吗？"],
    'sad': ["我能感受到你的悲伤，想聊聊吗？","难过不用一个人扛。","一切都会好起来的，抱抱你。"],
}

ICONS = {'angry':'😠','disgust':'🤢','fear':'😨','happy':'😊','neutral':'😐','sad':'😢'}
COLORS = {'angry':'#e74c3c','disgust':'#8e44ad','fear':'#f39c12','happy':'#2ecc71','neutral':'#95a5a6','sad':'#3498db'}

# 情绪→TTS 参数 (前端用)
EMOTION_TTS = {
    'angry':  {'rate': 1.15, 'pitch': 1.1,  'volume': 1.0},
    'disgust': {'rate': 0.95, 'pitch': 0.9,  'volume': 0.9},
    'fear':   {'rate': 1.05, 'pitch': 1.15, 'volume': 0.8},
    'happy':  {'rate': 1.2,  'pitch': 1.2,  'volume': 1.1},
    'neutral':{'rate': 1.0,  'pitch': 1.0,  'volume': 1.0},
    'sad':    {'rate': 0.8,  'pitch': 0.85, 'volume': 0.8},
}


@app.route('/')
def index():
    return render_template('index.html', emotions=config.EMOTIONS)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        wav_bytes = request.get_data()
        if len(wav_bytes) < 1000:
            return jsonify({'error': '音频太短'}), 400

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        import librosa
        audio, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, duration=config.DURATION)
        os.unlink(wav_path)

        if len(audio) < 1600:
            return jsonify({'error': '音频太短'}), 400

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        # ---- 1. 语音识别 ----
        text = ""
        try:
            result = asr_model.transcribe(audio, language='zh', fp16=False)
            text = result['text'].strip()
        except:
            text = "[识别失败]"

        # ---- 2. 情感识别 ----
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(config.DEVICE)
        with torch.no_grad():
            outputs = emotion_model(audio_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = probs.argmax(dim=1).item()
            conf = round(probs[0, pred].item() * 100, 1)

        emotion = config.EMOTIONS[pred]
        all_probs = {e: float(p) for e, p in zip(config.EMOTIONS, probs[0].cpu().numpy())}

        # ---- 3. 说话人识别 ----
        speaker, speaker_score = speaker_engine.identify(audio, config.SAMPLE_RATE)
        speaker_conf = None
        if speaker and speaker_score > -200:
            speaker_conf = min(round((speaker_score + 200) / 10, 1), 99.9)
            speaker_emotion_history[speaker].append(emotion)
            if len(speaker_emotion_history[speaker]) > 50:
                speaker_emotion_history[speaker] = speaker_emotion_history[speaker][-50:]

        # ---- 4. 生成回复（个性化 + 趋势感知） ----
        pool = RESPONSES.get(emotion, RESPONSES['neutral'])
        response = ""

        if speaker and len(speaker_emotion_history.get(speaker, [])) >= 2:
            trend = get_emotion_trend(speaker)
            if trend and trend['today_first']:
                # 上一轮情绪和本轮不同 → 趋势感知回复
                prev_emo = speaker_emotion_history[speaker][-2]
                if prev_emo == 'sad' and emotion == 'happy':
                    response = f"你刚才听起来好多了！从之前的{ICONS['sad']} 变成了现在的{ICONS['happy']}，真为你高兴！"
                elif prev_emo == 'happy' and emotion == 'sad':
                    response = f"你之前还很开心，现在听起来有些难过。发生了什么？我愿意听听。"
                elif prev_emo == 'angry' and emotion in ('neutral', 'sad'):
                    response = f"你比起刚才平静了一些。有时候深呼吸一下，情绪就过去了。"

        if not response:
            if text and len(text) > 1 and text != "[识别失败]":
                if speaker:
                    response = f"{speaker}，你说「{text}」——{random.choice(pool)}"
                else:
                    response = f"「{text}」——{random.choice(pool)}"
            else:
                if speaker:
                    response = f"{speaker}，{random.choice(pool)}"
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
            'speaker': speaker,
            'speaker_confidence': speaker_conf,
            'tts_params': EMOTION_TTS.get(emotion, EMOTION_TTS['neutral']),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---- 说话人管理 API ----
@app.route('/api/speaker/register', methods=['POST'])
def register_speaker():
    try:
        data = request.json
        name = data.get('name', '').strip()
        if not name:
            return jsonify({'error': '请输入名字'}), 400

        wav_b64 = data.get('audio', '')
        if not wav_b64:
            return jsonify({'error': '无音频数据'}), 400

        import base64
        wav_bytes = base64.b64decode(wav_b64)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        import librosa
        audio, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, duration=config.DURATION)
        os.unlink(wav_path)

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        success = speaker_engine.register(name, [audio], config.SAMPLE_RATE)
        if success:
            return jsonify({'ok': True, 'name': name, 'total': len(speaker_engine.get_all_speakers())})
        else:
            return jsonify({'error': '注册失败，音频太短'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/speaker/list', methods=['GET'])
def list_speakers():
    speakers = speaker_engine.get_all_speakers()
    result = []
    for s in speakers:
        hist = speaker_emotion_history.get(s, [])
        counts = {}
        for e in hist[-10:]:
            counts[e] = counts.get(e, 0) + 1
        dominant = max(counts, key=counts.get) if counts else None
        result.append({
            'name': s,
            'total': len(hist),
            'dominant': dominant,
            'icon': ICONS.get(dominant, ''),
        })
    return jsonify({'speakers': result})


@app.route('/api/speaker/delete', methods=['POST'])
def delete_speaker():
    name = request.json.get('name', '')
    if name in speaker_emotion_history:
        del speaker_emotion_history[name]
    ok = speaker_engine.delete_speaker(name)
    return jsonify({'ok': ok})


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
