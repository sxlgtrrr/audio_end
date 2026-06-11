"""
Wav2Vec2 测试时增强 (TTA) 评估：对同一段音频多种轻微扰动，概率平均后预测。
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import Wav2Vec2EmotionModel
from data_processor import Wav2Vec2EmotionDataset, AudioProcessor


def build_tta_variants(audio, processor):
    """返回若干增强版本（均含原始音频）。"""
    variants = [audio.copy()]
    variants.append(processor.add_noise(audio, noise_factor=0.003))
    variants.append(processor.add_noise(audio, noise_factor=0.006))
    variants.append(processor.time_shift(audio.copy(), shift_max=0.08))
    variants.append(processor.time_shift(audio.copy(), shift_max=0.15))
    try:
        variants.append(processor.pitch_shift(audio.copy(), n_steps=1))
        variants.append(processor.pitch_shift(audio.copy(), n_steps=-1))
    except Exception:
        pass
    return variants


@torch.no_grad()
def evaluate_split(model, dataset, device, use_tta=False):
    model.eval()
    processor = AudioProcessor()
    correct, total = 0, 0
    all_preds, all_labels = [], []

    desc = 'TTA' if use_tta else 'Standard'
    for idx in tqdm(range(len(dataset)), desc=desc):
        label = dataset.labels[idx]
        audio = processor.load_audio(dataset.audio_files[idx])
        audio = processor.normalize_audio(audio)

        if use_tta:
            variants = build_tta_variants(audio, processor)
            probs_sum = None
            for v in variants:
                v = processor.normalize_audio(v)
                x = torch.FloatTensor(v).unsqueeze(0).to(device)
                probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
                probs_sum = probs if probs_sum is None else probs_sum + probs
            pred = int(np.argmax(probs_sum / len(variants)))
        else:
            x = torch.FloatTensor(audio).unsqueeze(0).to(device)
            pred = int(model(x).argmax(dim=1).item())

        correct += int(pred == label)
        total += 1
        all_preds.append(pred)
        all_labels.append(label)

    acc = 100.0 * correct / total
    report = classification_report(all_labels, all_preds, target_names=config.EMOTIONS, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm


def main():
  parser = argparse.ArgumentParser(description='Wav2Vec2 TTA 评估')
  parser.add_argument('--checkpoint', default='./models/best_model_wav2vec2_best.pth')
  parser.add_argument('--split', choices=['val', 'test', 'both'], default='both')
  args = parser.parse_args()

  device = config.DEVICE
  config.WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'
  model = Wav2Vec2EmotionModel().to(device)
  ckpt = torch.load(args.checkpoint, map_location=device)
  model.load_state_dict(ckpt['model_state_dict'])
  print(f"模型: {args.checkpoint}")
  print(f"checkpoint val_acc: {ckpt.get('best_val_acc', 0):.2f}%\n")

  splits = ['val', 'test'] if args.split == 'both' else [args.split]
  results = {}

  for split in splits:
    ds = Wav2Vec2EmotionDataset(mode=split, augment_level='light')
    std_acc, _, _ = evaluate_split(model, ds, device, use_tta=False)
    tta_acc, tta_report, tta_cm = evaluate_split(model, ds, device, use_tta=True)
    results[split] = (std_acc, tta_acc)
    print(f"\n{'='*50}")
    print(f"  {split.upper()} 集")
    print(f"{'='*50}")
    print(f"  标准推理:  {std_acc:.2f}%")
    print(f"  TTA 推理:  {tta_acc:.2f}%  (+{tta_acc - std_acc:.2f}%)")
    print(f"\nTTA 分类报告:\n{tta_report}")
    print(f"混淆矩阵:\n{tta_cm}")

  if len(splits) == 2:
    print(f"\n{'='*50}")
    print("  汇总")
    for s in splits:
      std, tta = results[s]
      print(f"  {s}: {std:.2f}% -> {tta:.2f}% (+{tta - std:.2f}%)")


if __name__ == '__main__':
  main()
