"""测试 Flask 管线是否正常"""
import numpy as np
import torch, sys, os
sys.path.insert(0, '.')
import config
from models import get_model
from data_processor import AudioProcessor
from scipy import signal as scipy_signal

print("=== 加载模型 ===")
model = get_model('wav2vec2')
ckpt = torch.load('models/best_model_wav2vec2_best.pth', map_location='cpu')
md = model.state_dict()
pd2 = {k: v for k, v in ckpt['model_state_dict'].items() if k in md and md[k].shape == v.shape}
md.update(pd2)
model.load_state_dict(md, strict=False)
model.eval()

proc = AudioProcessor()

# 测试1: 训练数据Pipeline → 应该正常
print("\n=== 测试1: 训练数据 (librosa loading) ===")
for emo in config.EMOTIONS:
    d = f'data/test/{emo}'
    if os.path.exists(d):
        fs = [f for f in os.listdir(d) if f.endswith('.wav')][:3]
        for f in fs:
            audio = proc.load_audio(os.path.join(d, f))
            audio = proc.normalize_audio(audio)
            t = torch.FloatTensor(audio).unsqueeze(0)
            with torch.no_grad():
                p = torch.softmax(model(t), 1)[0]
                pred = p.argmax().item()
            mark = '✓' if config.EMOTIONS[pred] == emo else '✗'
            print(f'  {mark} {emo:10s} -> {config.EMOTIONS[pred]:10s} (conf={p[pred]:.2f})')
            break
        break  # only one sample per emotion to save time

# 测试2: 模拟Flask管线 (resample 48k->16k)
print("\n=== 测试2: 模拟浏览器录音 (48kHz -> 16kHz resample) ===")
for emo in config.EMOTIONS:
    d = f'data/test/{emo}'
    if os.path.exists(d):
        fs = [f for f in os.listdir(d) if f.endswith('.wav')][:1]
        for f in fs:
            # 模拟：先以48kHz加载，再resample到16k
            import librosa
            audio_48k, _ = librosa.load(os.path.join(d, f), sr=48000, duration=3)
            audio_16k = scipy_signal.resample(audio_48k, int(len(audio_48k)*16000/48000))
            peak = np.max(np.abs(audio_16k))
            if peak > 0: audio_16k = audio_16k / peak
            if len(audio_16k) > 48000: audio_16k = audio_16k[:48000]
            
            t = torch.FloatTensor(audio_16k).unsqueeze(0)
            with torch.no_grad():
                p = torch.softmax(model(t), 1)[0]
                pred = p.argmax().item()
            mark = '✓' if config.EMOTIONS[pred] == emo else '✗'
            print(f'  {mark} {emo:10s} -> {config.EMOTIONS[pred]:10s} (conf={p[pred]:.2f})')

# 测试3: 检查模型分类头bias
print("\n=== 测试3: 检查是否有bias ===")
bias = model.classifier[-1].bias.detach().cpu().numpy()
for i, e in enumerate(config.EMOTIONS):
    print(f'  {e}: bias={bias[i]:.4f}')

# 测试4: 静音测试
print("\n=== 测试4: 静音/噪声 ===")
for label, audio in [
    ("全零", np.zeros(48000)),
    ("噪声", np.random.randn(48000)*0.1),
    ("大噪声", np.random.randn(48000)),
]:
    t = torch.FloatTensor(audio).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(t), 1)[0]
        pred = p.argmax().item()
    print(f'  {label:10s} -> {config.EMOTIONS[pred]:10s} (conf={p[pred]:.2f})')
