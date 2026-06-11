"""从已有权重继续训练 10 轮"""
import torch, sys, os
sys.path.insert(0, '.')
import config

# 降低 LR 继续微调
config.WAV2VEC2_LR = 5e-6

from data_processor import get_wav2vec2_dataloaders
from train import Wav2Vec2Trainer

t = Wav2Vec2Trainer()
ckpt = torch.load('models/best_model_wav2vec2_best.pth', map_location=config.DEVICE)
t.model.load_state_dict(ckpt['model_state_dict'])
t.best_val_acc = ckpt['best_val_acc']
epoch = ckpt.get('epoch', 0)
print(f'从 epoch {epoch} 继续训练, best_val_acc = {t.best_val_acc:.2f}%')

tl, vl, _ = get_wav2vec2_dataloaders()
t.train(tl, vl, epochs=20)
