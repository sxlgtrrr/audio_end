import torch, numpy as np, sys, os
sys.path.insert(0, '.')
import config
from models import get_model
from data_processor import AudioProcessor

# 1. 加载模型
model = get_model('wav2vec2')
ckpt = torch.load('models/best_model_wav2vec2_best.pth', map_location='cpu')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                  if k in model_dict and model_dict[k].shape == v.shape}
model.load_state_dict(pretrained_dict, strict=False)
model.eval()
print(f'Best val acc: {ckpt["best_val_acc"]:.2f}%')
print(f'Loaded {len(pretrained_dict)}/{len(model_dict)} layers')

# 2. 随机噪声测试
print('\n--- Random noise test ---')
for _ in range(5):
    fake_audio = torch.randn(1, 16000 * 3)
    with torch.no_grad():
        prob = torch.softmax(model(fake_audio), 1)[0]
        pred = prob.argmax().item()
    print(f'  -> {config.EMOTIONS[pred]} (max={prob.max():.3f})')
    # 如果是同一个类每次都出现，说明有 bias

# 3. 测试集文件测试
print('\n--- Test file test ---')
proc = AudioProcessor()
for emo in config.EMOTIONS:
    d = f'data/test/{emo}'
    if os.path.exists(d):
        fs = [f for f in os.listdir(d) if f.endswith('.wav')]
        if fs:
            fp = os.path.join(d, fs[0])
            audio = proc.load_audio(fp)
            audio = proc.normalize_audio(audio)
            t = torch.FloatTensor(audio).unsqueeze(0)
            with torch.no_grad():
                prob = torch.softmax(model(t), 1)[0]
                pred = prob.argmax().item()
                conf = prob[pred].item()
            mark = '✓' if config.EMOTIONS[pred] == emo else '✗'
            print(f'  {mark} {emo:10s} -> {config.EMOTIONS[pred]:10s} (conf={conf:.2f})')

# 4. 检查音频长度是否是 48000
print('\n--- Audio duration check ---')
for emo in config.EMOTIONS:
    d = f'data/test/{emo}'
    if os.path.exists(d):
        fs = [f for f in os.listdir(d) if f.endswith('.wav')][:1]
        for f in fs:
            audio = proc.load_audio(os.path.join(d, f))
            print(f'  {emo}: {len(audio)} samples = {len(audio)/config.SAMPLE_RATE:.1f}s')
