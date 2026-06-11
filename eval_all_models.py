"""
综合评估脚本：test 集准确率、per-class F1、参数量统计、训练曲线
"""
import os, sys, json, re
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from models import (MultiBackboneModel, MultiBackboneMoEModel,
                    CTMWavLMModel, MULTIMODEL_CONFIGS)
from data_processor import Wav2Vec2EmotionDataset, wav2vec2_collate_fn

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['OMP_NUM_THREADS'] = '4'
DEVICE = config.DEVICE

# ---------- evaluate single model ----------
@torch.no_grad()
def evaluate_model(model, loader, desc=''):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        audio = batch['audio'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        logits = model(audio)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
    report = classification_report(all_labels, all_preds,
                                    target_names=config.EMOTIONS, digits=2, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    return acc, report, cm, all_preds, all_labels

# ---------- load test data ----------
print("Loading TEST dataset...")
test_ds = Wav2Vec2EmotionDataset(data_dir='./data', mode='test')
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False,
                          collate_fn=wav2vec2_collate_fn, num_workers=0, pin_memory=True)
print(f"  Test samples: {len(test_ds)}")

# ---------- define models to evaluate ----------
MODEL_LIST = [
    # (display_name, model_key, ckpt_name, use_ctm, use_moe, use_mean_pool,
    #  ctm_steps, supcon, center)
    ('WavLM Baseline (Attn)', 'wavlm-base', 'best_model_wavlm-base_best.pth',
     False, False, False, 4, 0, 0),
    ('WavLM Mean Pooling', 'wavlm-base', 'best_model_wavlm-base_meanpool_best.pth',
     False, False, True, 4, 0, 0),
    ('WavLM CTM K=1', 'wavlm-base', 'best_model_wavlm-base_ctm_k1_best.pth',
     True, False, False, 1, 0, 0),
    ('WavLM CTM K=2', 'wavlm-base', 'best_model_wavlm-base_ctm_k2_sup0.1_best.pth',
     True, False, False, 2, 0.1, 0),
    ('WavLM CTM K=4', 'wavlm-base', 'best_model_wavlm-base_ctm_k4_best.pth',
     True, False, False, 4, 0.1, 0),
    ('WavLM CTM K=4 + Center', 'wavlm-base', 'best_model_wavlm-base_ctm_k4_sup0.1_cen0.1_best.pth',
     True, False, False, 4, 0.1, 0.1),
    ('WavLM MoE', 'wavlm-base', 'best_model_wavlm-base_moe_best.pth',
     False, True, False, 4, 0, 0),
    ('HuBERT Baseline', 'hubert-base', 'best_model_hubert-base_best.pth',
     False, False, False, 4, 0, 0),
]

results = {}

for display, model_key, ckpt_name, use_ctm, use_moe, use_mean_pool, ctm_steps, supcon, center in MODEL_LIST:
    ckpt_path = f'./models/{ckpt_name}'
    if not os.path.exists(ckpt_path):
        print(f"\nSKIP {display}: checkpoint not found ({ckpt_name})")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating: {display}")
    print(f"  Checkpoint: {ckpt_name}")

    cfg = MULTIMODEL_CONFIGS[model_key]
    model_name = cfg['model_name']
    hidden_size = cfg['hidden_size']

    # Build model
    if use_ctm:
        model = CTMWavLMModel(model_name, config.NUM_CLASSES, dropout=0.3,
                               hidden_size=hidden_size, ctm_steps=ctm_steps)
    elif use_moe:
        model = MultiBackboneMoEModel(model_name, config.NUM_CLASSES, dropout=0.3,
                                       hidden_size=hidden_size)
    else:
        model = MultiBackboneModel(model_name, config.NUM_CLASSES, dropout=0.3,
                                    hidden_size=hidden_size)
        if use_mean_pool:
            model.attention_pool.init_as_mean_pool()

    model.to(DEVICE)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = total_params - backbone_params

    print(f"  Total params: {total_params/1e6:.2f}M (backbone: {backbone_params/1e6:.2f}M, head: {head_params/1e6:.2f}M)")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt

    # Handle missing keys (e.g., SupCon/criterion params)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded checkpoint. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")

    # Evaluate
    test_acc, report, cm, preds, labels = evaluate_model(model, test_loader)

    # Per-class metrics
    per_class = {}
    for emo in config.EMOTIONS:
        if emo in report:
            per_class[emo] = {
                'precision': round(report[emo]['precision'] * 100, 1),
                'recall': round(report[emo]['recall'] * 100, 1),
                'f1': round(report[emo]['f1-score'] * 100, 1),
            }

    print(f"  Test Acc: {test_acc:.2f}%")
    for emo in config.EMOTIONS:
        print(f"    {emo:8s}: P={per_class[emo]['precision']:5.1f}% R={per_class[emo]['recall']:5.1f}% F1={per_class[emo]['f1']:5.1f}%")

    results[display] = {
        'test_acc': round(test_acc, 2),
        'per_class': per_class,
        'total_params_M': round(total_params / 1e6, 2),
        'head_params_M': round(head_params / 1e6, 2),
        'ckpt_epoch': ckpt.get('epoch', '?'),
        'ckpt_best_val': ckpt.get('best_val_acc', 0),
    }

    # Save confusion matrix
    cm_path = f'./logs/cm_{display.replace(" ", "_").replace("(", "").replace(")", "")}.png'
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(config.EMOTIONS)))
    ax.set_yticks(range(len(config.EMOTIONS)))
    ax.set_xticklabels(config.EMOTIONS, rotation=45, fontsize=9)
    ax.set_yticklabels(config.EMOTIONS, fontsize=9)
    for i in range(len(config.EMOTIONS)):
        for j in range(len(config.EMOTIONS)):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    fontsize=8, color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_title(f'{display}\nTest Acc = {test_acc:.2f}%', fontsize=11)
    plt.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")

# Save results
with open('./logs/test_results.json', 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nTest results saved to ./logs/test_results.json")

# ---------- training curves ----------
print("\n\nGenerating training curves...")

# Parse all training logs
log_files = {
    'WavLM Baseline': 'logs/train_metrics_wavlm-base.txt',
    'Mean Pooling': 'logs/train_metrics_wavlm-base_meanpool.txt',
    'CTM K=1': 'logs/train_metrics_wavlm-base_ctm_k1.txt',
    'CTM K=2': 'logs/train_metrics_wavlm-base_ctm_k2_sup0.1.txt',
    'CTM K=4': 'logs/train_metrics_wavlm-base_ctm_k4.txt',
    'MoE': 'logs/train_metrics_wavlm-base_moe.txt',
    'HuBERT': 'logs/train_metrics_hubert-base.txt',
}

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# WavLM family
colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#95A5A6']
wavlm_logs = ['WavLM Baseline', 'Mean Pooling', 'CTM K=1', 'CTM K=2', 'CTM K=4', 'MoE']
for idx, name in enumerate(wavlm_logs):
    path = log_files.get(name, '')
    if not os.path.exists(path):
        continue
    epochs, val_accs = [], []
    with open(path) as f:
        for line in f:
            if not line.startswith('Epoch'):
                continue
            m = re.search(r'Epoch\s+(\d+)\s+\|.*?val=([\d.]+)%', line)
            if m:
                epochs.append(int(m.group(1)))
                val_accs.append(float(m.group(2)))
    if epochs:
        axes[0].plot(epochs, val_accs, 'o-', color=colors[idx], linewidth=1.5, markersize=3, label=name)

axes[0].set_title('WavLM Family: Val Accuracy Curves', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Validation Accuracy (%)')
axes[0].legend(fontsize=9, loc='lower right')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(10, 80)

# All models including HuBERT
all_logs = ['WavLM Baseline', 'CTM K=4', 'Mean Pooling', 'HuBERT']
for idx, name in enumerate(all_logs):
    path = log_files.get(name, '')
    if not os.path.exists(path):
        continue
    epochs, val_accs = [], []
    with open(path) as f:
        for line in f:
            m = re.search(r'Epoch\s+(\d+)\s+\|.*?val=([\d.]+)%', line)
            if m:
                epochs.append(int(m.group(1)))
                val_accs.append(float(m.group(2)))
    if epochs:
        axes[1].plot(epochs, val_accs, 'o-', color=colors[idx], linewidth=1.5, markersize=3, label=name)

axes[1].set_title('Cross-Backbone Comparison: Val Accuracy', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation Accuracy (%)')
axes[1].legend(fontsize=9, loc='lower right')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(10, 80)

plt.tight_layout()
fig.savefig('./logs/training_curves.png', dpi=150)
plt.close()
print("Training curves saved to ./logs/training_curves.png")

# Per-class F1 bar chart
print("\nGenerating per-class F1 comparison...")
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(config.EMOTIONS))
width = 0.12
key_models = ['WavLM Baseline (Attn)', 'WavLM Mean Pooling', 'WavLM CTM K=4',
              'WavLM CTM K=2', 'WavLM CTM K=1', 'WavLM MoE']
cmap = plt.cm.Set2

for i, name in enumerate(key_models):
    if name not in results:
        continue
    f1s = [results[name]['per_class'][emo]['f1'] for emo in config.EMOTIONS]
    bars = ax.bar(x + i * width - width * len(key_models) / 2 + width / 2, f1s,
                  width, label=name, color=cmap(i / len(key_models)), edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(config.EMOTIONS, fontsize=11)
ax.set_ylabel('F1 Score (%)', fontsize=11)
ax.set_title('Per-Class F1 Score Comparison (Test Set)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='lower left', ncol=2)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)
plt.tight_layout()
fig.savefig('./logs/per_class_f1.png', dpi=150)
plt.close()
print("Per-class F1 chart saved to ./logs/per_class_f1.png")

print("\nDone!")
