"""
语音情感感知助手 - Demo
整合: 语音识别(Whisper) + 情感识别(Wav2Vec2) + 语音合成(pyttsx3)
"""

import os
import sys
import torch
import numpy as np
import sounddevice as sd
import whisper
import pyttsx3
import random
import config

from data_processor import AudioProcessor
from models import get_model, Wav2Vec2EmotionModel

# ---- 全局初始化 ----
print("=" * 60)
print("  语音情感感知助手 初始化中...")
print("=" * 60)

# 1. 加载 Whisper (默认 tiny，快且够用)
print("[1/3] 加载 Whisper 语音识别模型...")
asr_model = whisper.load_model("tiny")

# 2. 加载 Wav2Vec2 情感识别模型
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
    print("   ⚠ 未找到训练模型，将使用随机权重")
emotion_model.eval()

processor = AudioProcessor()

# 3. 语音合成引擎
print("[3/3] 初始化语音合成...")
tts = pyttsx3.init()
tts.setProperty('rate', 170)
tts.setProperty('volume', 0.9)

# ---- 录音函数 ----
def record_audio(duration=5, sample_rate=16000):
    """录制音频"""
    print(f"\n🎤 开始录音 ({duration}秒)... 请说话:")
    try:
        audio = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        print("   ✓ 录音完成")
        return audio
    except Exception as e:
        print(f"   ⚠ 录音失败: {e}")
        return None


def speech_to_text(audio, sample_rate=16000):
    """Whisper 语音转文字"""
    audio = audio.astype(np.float32)
    result = asr_model.transcribe(audio, language='zh', fp16=False)
    text = result['text'].strip()
    return text if text else "[未识别到语音]"


def recognize_emotion(audio):
    """Wav2Vec2 情感识别"""
    audio = torch.FloatTensor(audio).unsqueeze(0).to(config.DEVICE)
    with torch.no_grad():
        outputs = emotion_model(audio)
        probs = torch.softmax(outputs, dim=1)
        pred = probs.argmax(dim=1).item()
        conf = probs[0, pred].item() * 100
    return config.EMOTIONS[pred], conf, probs[0].cpu().numpy()


# ---- 情感感知回复 ----
EMOTION_RESPONSES = {
    'angry': [
        "我能感受到你的愤怒，让我们冷静下来想一想。",
        "我理解你现在很生气，深呼吸，慢慢说。",
        "情绪激动的时候，不如先休息一下？我在这里陪你。",
    ],
    'disgust': [
        "听起来你对这件事很不满意。",
        "我能理解你的反感，换个角度想想可能会有帮助。",
        "这种感觉确实让人不舒服。",
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
        "我能感受到你的悲伤。你想聊聊发生了什么吗？",
        "难过的时候不用一个人扛，我在这里。",
        "有时候说出来就会好受很多，你愿意试试吗？",
    ],
}


def generate_response(text, emotion, confidence):
    """根据情感和文字生成回复"""
    responses = EMOTION_RESPONSES.get(emotion, EMOTION_RESPONSES['neutral'])
    response = random.choice(responses)

    # 如果识别到了文字，可以做一些简单回显
    if text and text != "[未识别到语音]" and len(text) > 1:
        prefix = {
            'angry': f"你说「{text}」——",
            'sad': f"你说「{text}」——",
            'fear': f"你说「{text}」——",
            'happy': f"你说「{text}」——",
        }.get(emotion, "")
        return f"{prefix}{response}" if prefix else response
    return response


def speak(text):
    """文字转语音"""
    print(f"\n🤖 助手回复: {text}")
    tts.say(text)
    tts.runAndWait()


# ---- 主循环 ----
def main():
    print("\n" + "=" * 60)
    print("  准备就绪！按 Enter 开始录音，输入 q 退出")
    print("  (建议说话内容: 我今天很开心 / 我好难过 / 气死我了)")
    print("=" * 60)

    while True:
        cmd = input("\n按 Enter 开始录音 (q 退出): ").strip()
        if cmd.lower() == 'q':
            print("再见！")
            break

        # 1. 录音
        audio = record_audio(duration=5)
        if audio is None:
            continue

        # 2. 语音识别
        print("🔍 正在识别语音...")
        text = speech_to_text(audio)
        print(f"   识别结果: {text}")

        # 3. 情感识别
        print("🎭 正在分析情感...")
        emotion, confidence, probs = recognize_emotion(audio)
        print(f"   情感: {emotion}")
        print(f"   置信度: {confidence:.1f}%")
        print(f"   分布: " + " | ".join(f"{e}={p*100:.0f}%"
              for e, p in zip(config.EMOTIONS, probs)))

        # 4. 生成回复
        response = generate_response(text, emotion, confidence)

        # 5. 语音回复
        speak(response)


if __name__ == '__main__':
    main()
