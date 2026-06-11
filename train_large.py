"""
Wav2Vec2-Large 平衡微调 v3

策略（对齐 base 74% 的成功经验，适度正则）:
  - 冻结 feature_extractor + 前 10 层，微调约一半 Transformer
  - lr=1e-5，分层 LR 与 base 一致（head ×10）
  - dropout/label_smooth/weight_decay 与 base 相同
  - 轻量数据增强（light），避免 v2 欠拟合
  - 早停 patience=8，验证集无提升则恢复最佳权重
"""
import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import Wav2Vec2EmotionModel
from data_processor import get_wav2vec2_dataloaders
from train import evaluate_wav2vec2_model


class Wav2Vec2LargeTrainer:
    def __init__(self, frozen_layers=None, lr=None, weight_decay=None,
                 label_smooth=None, dropout=None, early_stop=None):
        self.device = config.DEVICE
        config.WAV2VEC2_MODEL_NAME = config.WAV2VEC2_LARGE_MODEL_NAME

        self.frozen_layers = frozen_layers or config.WAV2VEC2_LARGE_FROZEN_LAYERS
        self.lr = lr or config.WAV2VEC2_LARGE_LR
        self.weight_decay = weight_decay or config.WAV2VEC2_LARGE_WEIGHT_DECAY
        self.label_smooth = label_smooth or config.WAV2VEC2_LARGE_LABEL_SMOOTH
        self.dropout = dropout or config.WAV2VEC2_LARGE_DROPOUT
        self.early_stop_patience = early_stop or config.WAV2VEC2_LARGE_EARLY_STOP

        print(f"加载 {config.WAV2VEC2_LARGE_MODEL_NAME} ...")
        self.model = Wav2Vec2EmotionModel(dropout=self.dropout).to(self.device)
        print("冻结底层参数:")
        self.model.freeze_for_finetuning(num_frozen_layers=self.frozen_layers)

        feat_params, enc_params, head_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name:
                head_params.append(param)
            elif 'feature_extractor' in name:
                feat_params.append(param)
            else:
                enc_params.append(param)

        param_groups = []
        if feat_params:
            param_groups.append({'params': feat_params, 'lr': self.lr * 0.5})
        if enc_params:
            param_groups.append({'params': enc_params, 'lr': self.lr})
        param_groups.append({'params': head_params, 'lr': self.lr * 10})

        self.optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=4, min_lr=1e-7
        )

        self.best_val_acc = 0.0
        self.best_state = None
        self.no_improve = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'lr': []
        }

        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.WAV2VEC2_LARGE_SAVE_PATH), exist_ok=True)

    def train_epoch(self, train_loader, epoch=None, log_interval=20):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        desc = f'Epoch {epoch} Train' if epoch else 'Training'
        pbar = tqdm(train_loader, desc=desc, leave=True, file=sys.stdout)
        for step, batch in enumerate(pbar, 1):
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(audio)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

            if step % log_interval == 0 or step == len(train_loader):
                pbar.set_postfix(
                    loss=f'{total_loss / step:.4f}',
                    acc=f'{100.0 * correct / total:.1f}%',
                    refresh=False,
                )

        return total_loss / len(train_loader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self, val_loader, epoch=None):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        desc = f'Epoch {epoch} Val' if epoch else 'Validating'
        for batch in tqdm(val_loader, desc=desc, leave=True, file=sys.stdout):
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(audio)
            total_loss += self.criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(val_loader), 100.0 * correct / total

    def _append_live_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        path = config.WAV2VEC2_LARGE_LIVE_LOG
        gap = train_acc - val_acc
        line = (
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
            f"gap={gap:.1f}% lr={lr:.2e} best={self.best_val_acc:.2f}%\n"
        )
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line)

    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_type': 'wav2vec2-large',
            'config': {
                'model_name': config.WAV2VEC2_LARGE_MODEL_NAME,
                'frozen_layers': self.frozen_layers,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'label_smooth': self.label_smooth,
                'dropout': self.dropout,
            }
        }, path)
        print(f"  模型已保存: {path}")

    def train(self, train_loader, val_loader, epochs=None, log_interval=20):
        epochs = epochs or config.WAV2VEC2_LARGE_EPOCHS
        start = time.time()

        print(f"\n{'='*60}")
        print("  Wav2Vec2-Large 平衡微调 v3")
        print(f"  设备: {self.device}")
        print(f"  epochs={epochs}  batch={config.WAV2VEC2_LARGE_BATCH_SIZE}")
        print(f"  lr={self.lr}  wd={self.weight_decay}  smooth={self.label_smooth}  drop={self.dropout}")
        print(f"  冻结前 {self.frozen_layers} 层  早停 patience={self.early_stop_patience}")
        print(f"  增强: {config.WAV2VEC2_AUGMENT_LEVEL}  保存: {config.WAV2VEC2_LARGE_SAVE_PATH}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            ep = epoch + 1
            print(f"\nEpoch [{ep}/{epochs}]", flush=True)
            print('-' * 40, flush=True)

            train_loss, train_acc = self.train_epoch(train_loader, epoch=ep, log_interval=log_interval)
            val_loss, val_acc = self.validate(val_loader, epoch=ep)
            self.scheduler.step(val_loss)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)

            gap = train_acc - val_acc
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%", flush=True)
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%  (gap={gap:.1f}%)", flush=True)
            print(f"LR: {current_lr:.2e}", flush=True)

            if gap > 25:
                print("  ⚠ 训练/验证差距较大，可能仍在过拟合", flush=True)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.no_improve = 0
                self.save_checkpoint(config.WAV2VEC2_LARGE_SAVE_PATH, ep)
                print(f"  ✓ 新的最佳模型! val_acc={val_acc:.2f}%", flush=True)
            else:
                self.no_improve += 1
                print(f"  未提升 ({self.no_improve}/{self.early_stop_patience})", flush=True)
                if self.no_improve >= self.early_stop_patience:
                    print(f"\n早停触发：验证集 {self.early_stop_patience} 轮无提升", flush=True)
                    break

            self._append_live_metrics(ep, train_loss, train_acc, val_loss, val_acc, current_lr)
            with open(config.WAV2VEC2_LARGE_HISTORY_PATH, 'w') as f:
                json.dump(self.history, f, indent=2)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            print(f"\n已恢复最佳权重 (val_acc={self.best_val_acc:.2f}%)")

        elapsed = (time.time() - start) / 60
        print(f"\n{'='*60}")
        print(f"训练完成! 用时 {elapsed:.1f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")

        with open(config.WAV2VEC2_LARGE_HISTORY_PATH, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"训练历史: {config.WAV2VEC2_LARGE_HISTORY_PATH}")

        return self.history


def main():
    parser = argparse.ArgumentParser(description='Wav2Vec2-Large 平衡微调 v3')
    parser.add_argument('--epochs', type=int, default=config.WAV2VEC2_LARGE_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.WAV2VEC2_LARGE_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.WAV2VEC2_LARGE_LR)
    parser.add_argument('--frozen-layers', type=int, default=config.WAV2VEC2_LARGE_FROZEN_LAYERS)
    parser.add_argument('--early-stop', type=int, default=config.WAV2VEC2_LARGE_EARLY_STOP)
    parser.add_argument('--log-interval', type=int, default=10, help='每 N 个 batch 刷新训练指标')
    parser.add_argument('--augment', choices=['light', 'full'], default=config.WAV2VEC2_AUGMENT_LEVEL)
    parser.add_argument('--no-eval', action='store_true', help='训练后跳过测试集评估')
    args = parser.parse_args()

    config.WAV2VEC2_LARGE_BATCH_SIZE = args.batch_size
    config.WAV2VEC2_AUGMENT_LEVEL = args.augment

    with open(config.WAV2VEC2_LARGE_LIVE_LOG, 'w', encoding='utf-8') as f:
        f.write('# v3 | epoch | train_loss train_acc | val_loss val_acc | gap lr best\n')

    print("加载数据...", flush=True)
    train_loader, val_loader, test_loader = get_wav2vec2_dataloaders(
        batch_size=args.batch_size,
        augment_level=args.augment,
    )

    trainer = Wav2Vec2LargeTrainer(
        frozen_layers=args.frozen_layers,
        lr=args.lr,
        early_stop=args.early_stop,
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs, log_interval=args.log_interval)

    if not args.no_eval:
        print("\n在测试集上评估最佳模型...")
        evaluate_wav2vec2_model(trainer.model, test_loader)


if __name__ == '__main__':
    main()
