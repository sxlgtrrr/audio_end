"""
多模型概率融合评估（带权重网格搜索）。

用法:
  python eval_ensemble.py                                    # 默认加载全部已训练模型
  python eval_ensemble.py --search                           # 网格搜索最优权重
  python eval_ensemble.py --tta                              # 融合 + TTA
  python eval_ensemble.py --checkpoints a.pth b.pth c.pth    # 指定模型
"""
import argparse
import itertools
import os
import sys

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import (Wav2Vec2EmotionModel, Wav2Vec2AttentionModel,
                    MultiBackboneModel, MultiBackboneMoEModel,
                    CTMWavLMModel, MULTIMODEL_CONFIGS)
from data_processor import Wav2Vec2EmotionDataset, AudioProcessor
from eval_tta import build_tta_variants


# 默认可用模型列表
AVAILABLE_CHECKPOINTS = [
    ('./models/best_model_wav2vec2_best.pth',           'wav2vec2-base'),
    ('./models/best_model_wav2vec2_improved_best.pth',   'wav2vec2-improved'),
    ('./models/best_model_wav2vec2_attn_boost_best.pth', 'wav2vec2-attn_boost'),
    ('./models/best_model_wav2vec2_large_v3_best.pth',   'wav2vec2-large_v3'),
]


def load_model(path, device):
    if not os.path.isfile(path):
        return None, 0.0

    ckpt = torch.load(path, map_location=device)
    mtype = ckpt.get('model_type', '')

    # 判断模型类型
    is_attn = 'attn' in mtype or 'improved' in path or 'attn_boost' in path

    # 检查是否为 MultiBackbone 模型 (hubert / wavlm / xlsr 等)
    model = None
    is_ctm = 'ctm' in mtype or 'CTM' in mtype
    is_moe = 'moe' in mtype or 'MoE' in mtype
    if is_ctm:
        model_cls = CTMWavLMModel
    elif is_moe:
        model_cls = MultiBackboneMoEModel
    else:
        model_cls = MultiBackboneModel
    for key in MULTIMODEL_CONFIGS:
        if key in mtype or key in os.path.basename(path):
            cfg = MULTIMODEL_CONFIGS[key]
            model = model_cls(
                cfg['model_name'], config.NUM_CLASSES,
                hidden_size=cfg['hidden_size']
            ).to(device)
            break

    if model is None:
        # 传统 Wav2Vec2 模型
        if 'large' in path:
            config.WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-large'
        model = Wav2Vec2AttentionModel().to(device) if is_attn else Wav2Vec2EmotionModel().to(device)

    sd = ckpt['model_state_dict']
    model_dict = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in model_dict and model_dict[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)

    config.WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'  # 恢复默认
    model.eval()
    return model, ckpt.get('best_val_acc', 0.0)


@torch.no_grad()
def predict_probs(model, audio_tensor, device):
    return torch.softmax(model(audio_tensor), dim=1).cpu().numpy()[0]


def evaluate_ensemble(models, dataset, device, use_tta=False, weights=None):
    processor = AudioProcessor()
    if weights is None:
        weights = [1.0] * len(models)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()

    correct, total = 0, 0
    all_preds, all_labels = [], []

    desc = 'Ensemble+TTA' if use_tta else 'Ensemble'
    for idx in tqdm(range(len(dataset)), desc=desc):
        label = dataset.labels[idx]
        audio = processor.load_audio(dataset.audio_files[idx])
        audio = processor.normalize_audio(audio)

        variants = build_tta_variants(audio, processor) if use_tta else [audio]

        probs_sum = None
        for v in variants:
            v = processor.normalize_audio(v)
            x = torch.FloatTensor(v).unsqueeze(0).to(device)
            blend = np.zeros(config.NUM_CLASSES, dtype=np.float64)
            for model, w in zip(models, weights):
                blend += predict_probs(model, x, device) * w
            probs_sum = blend if probs_sum is None else probs_sum + blend
        final = probs_sum / len(variants)

        pred = int(np.argmax(final))
        correct += int(pred == label)
        total += 1
        all_preds.append(pred)
        all_labels.append(label)

    acc = 100.0 * correct / total
    report = classification_report(all_labels, all_preds, target_names=config.EMOTIONS, digits=4)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm, all_preds, all_labels


def grid_search_weights(models, dataset, device, n_steps=4):
    """在 val 集上网格搜索 convex 组合权重。
    n_steps: 每个模型的权重采样步数（越大越细）。
    返回 (best_weights, best_acc)。
    """
    n = len(models)
    if n == 1:
        print("  只有一个模型，无需搜索权重")
        return [1.0], 0.0

    # sample all convex combinations of n weights with step size 1/n_steps
    best_weights, best_acc = None, 0
    total_combos = 0

    # 枚举所有 weight = [k0/S, k1/S, ..., kn/S] where sum(ki) = S
    S = n_steps

    def gen_weights(n_models, remaining, path):
        """递归生成 convex 权重组合"""
        if n_models == 1:
            yield path + [remaining]
            return
        for k in range(remaining + 1):
            yield from gen_weights(n_models - 1, remaining - k, path + [k])

    # 简单网格：每个模型权重取 [0, 1/S, 2/S, ..., 1] 的凸组合
    # 为避免枚举爆炸，n>3 时用 random search
    if n <= 3:
        combos = list(gen_weights(n, S, []))
        total_combos = len(combos)
        print(f"  网格搜索 {n} 个模型, {total_combos} 种组合 (steps={S})...")
        for ks in tqdm(combos, desc='Weight search'):
            w = np.array(ks, dtype=np.float64) / S
            acc, _, _, _, _ = evaluate_ensemble(models, dataset, device, weights=w.tolist())
            if acc > best_acc:
                best_acc = acc
                best_weights = w.tolist()
    else:
        # 随机搜索
        n_trials = 200
        print(f"  随机搜索 {n} 个模型, {n_trials} 次采样...")
        for _ in tqdm(range(n_trials), desc='Weight search'):
            raw = np.random.rand(n)
            w = raw / raw.sum()
            acc, _, _, _, _ = evaluate_ensemble(models, dataset, device, weights=w.tolist())
            if acc > best_acc:
                best_acc = acc
                best_weights = w.tolist()

    print(f"  最优权重: {[f'{w:.3f}' for w in best_weights]}  val_acc={best_acc:.2f}%")
    return best_weights, best_acc


def main():
    parser = argparse.ArgumentParser(description='多模型融合评估')
    parser.add_argument('--checkpoints', nargs='+',
                        default=[p for p, _ in AVAILABLE_CHECKPOINTS],
                        help='模型 checkpoint 路径列表')
    parser.add_argument('--weights', nargs='+', type=float, default=None)
    parser.add_argument('--split', choices=['val', 'test', 'both'], default='both')
    parser.add_argument('--tta', action='store_true', help='融合后再做 TTA')
    parser.add_argument('--search', action='store_true', help='在 val 上网格搜索最优权重')
    parser.add_argument('--search-steps', type=int, default=4)
    args = parser.parse_args()

    device = config.DEVICE

    # ---- 加载模型 ----
    models, val_accs, names = [], [], []
    for p in args.checkpoints:
        if not os.path.isfile(p):
            print(f"  跳过缺失: {p}")
            continue
        m, va = load_model(p, device)
        if m is None:
            continue
        models.append(m)
        val_accs.append(va)
        names.append(os.path.basename(p))
        print(f"  加载 {os.path.basename(p)}  (记录 val={va:.2f}%)")

    if len(models) < 2:
        print("至少需要 2 个有效 checkpoint")
        return

    # ---- 权重搜索 ----
    w = args.weights
    if args.search:
        val_ds = Wav2Vec2EmotionDataset(mode='val')
        w, best_val = grid_search_weights(models, val_ds, device, n_steps=args.search_steps)

    if w and len(w) != len(models):
        print("weights 数量与 checkpoints 不一致，使用均等权重")
        w = None

    # ---- 评估 ----
    splits = ['val', 'test'] if args.split == 'both' else [args.split]
    for split in splits:
        ds = Wav2Vec2EmotionDataset(mode=split)
        tta_on = args.tta
        acc, report, cm, _, _ = evaluate_ensemble(models, ds, device, use_tta=tta_on, weights=w)
        tag = f"{split.upper()}" + (" + TTA" if tta_on else "")
        print(f"\n{'='*60}")
        print(f"融合评估 [{tag}]  acc={acc:.2f}%  (模型数={len(models)})")
        print(f"{'='*60}")
        print(report)
        print("混淆矩阵:")
        print(cm)
        print()

    # ---- 单模型对比 ----
    print(f"\n{'='*60}")
    print("单模型 vs 融合对比:")
    print(f"{'='*60}")
    print(f"{'模型':<35} {'val_acc'}")
    print('-' * 45)
    for name, va in zip(names, val_accs):
        print(f"{name:<35} {va:.2f}%")
    print('-' * 45)
    tta_on = args.tta
    val_ds = Wav2Vec2EmotionDataset(mode='val')
    test_ds = Wav2Vec2EmotionDataset(mode='test')
    val_acc, _, _, _, _ = evaluate_ensemble(models, val_ds, device, use_tta=tta_on, weights=w)
    test_acc, _, _, _, _ = evaluate_ensemble(models, test_ds, device, use_tta=tta_on, weights=w)
    print(f"Ensemble{' + TTA' if tta_on else '':<22} val={val_acc:.2f}%  test={test_acc:.2f}%")
    print()


if __name__ == '__main__':
    main()
