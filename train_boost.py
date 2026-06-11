"""
Wav2Vec2-Base 提分微调：从 74% 最佳权重继续训练

策略:
  - 加载 best_model_wav2vec2_best.pth
  - 类别加权采样（neutral 样本少 + sad/disgust 弱类加强）
  - 较低学习率 2e-6，早停并恢复最佳权重
  - 轻量数据增强
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import Wav2Vec2EmotionModel
from data_processor import get_wav2vec2_dataloaders, compute_class_weights, count_samples_by_class
from train import evaluate_wav2vec2_model

# neutral=4, sad=5, disgust=1
WEAK_CLASS_BOOST = {4: 1.15, 5: 1.20, 1: 1.10}

DEFAULT_CKPT = './models/best_model_wav2vec2_best.pth'
SAVE_PATH = './models/best_model_wav2vec2_boost_best.pth'
HISTORY_PATH = './logs/wav2vec2_boost_history.json'
LIVE_LOG = './logs/train_metrics_boost.txt'


class BoostTrainer:
    def __init__(self, lr=2e-6, early_stop=6, label_smooth=0.05):
        self.device = config.DEVICE
        config.WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'
        self.lr = lr
        self.early_stop_patience = early_stop
        self.label_smooth = label_smooth

        self.model = Wav2Vec2EmotionModel(dropout=0.5).to(self.device)

        feat_params, enc_params, head_params = [], [], []
        for name, param in self.model.named_parameters():
            if 'classifier' in name:
                head_params.append(param)
            elif 'feature_extractor' in name:
                feat_params.append(param)
            else:
                enc_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': feat_params, 'lr': lr * 0.5},
            {'params': enc_params, 'lr': lr},
            {'params': head_params, 'lr': lr * 10},
        ], weight_decay=0.01)
        self.criterion = None  # set after class weights
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )

        self.best_val_acc = 0.0
        self.best_state = None
        self.no_improve = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}

    def set_class_weights(self, weights):
        weights = weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.label_smooth)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.best_val_acc = ckpt.get('best_val_acc', 0.0)
        if 'optimizer_state_dict' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                pass
        print(f"已加载 {path}  (val_acc={self.best_val_acc:.2f}%, epoch={ckpt.get('epoch', '?')})")
        self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

    def train_epoch(self, loader, log_interval=10):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(loader, desc='Train', leave=True, file=sys.stdout)
        for step, batch in enumerate(pbar, 1):
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(audio)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if step % log_interval == 0 or step == len(loader):
                pbar.set_postfix(loss=f'{total_loss/step:.4f}',
                                 acc=f'{100*correct/total:.1f}%', refresh=False)
        return total_loss / len(loader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(loader, desc='Val', leave=True, file=sys.stdout):
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(audio)
            total_loss += self.criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), 100.0 * correct / total

    def save(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_type': 'wav2vec2-base-boost',
        }, path)
        print(f"  模型已保存: {path}")

    def train(self, train_loader, val_loader, epochs=15, log_interval=10):
        counts = count_samples_by_class(mode='train')
        print(f"训练集各类样本: {dict(zip(config.EMOTIONS, counts))}")
        print(f"\n{'='*60}")
        print("  Wav2Vec2-Base 提分微调 (boost)")
        print(f"  lr={self.lr}  早停={self.early_stop_patience}  起点 val={self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")

        start = time.time()
        for epoch in range(epochs):
            ep = epoch + 1
            print(f"\nEpoch [{ep}/{epochs}]", flush=True)
            print('-' * 40, flush=True)
            train_loss, train_acc = self.train_epoch(train_loader, log_interval)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[1]['lr']
            gap = train_acc - val_acc

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(lr)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%", flush=True)
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%  (gap={gap:.1f}%)", flush=True)
            print(f"LR: {lr:.2e}", flush=True)

            with open(LIVE_LOG, 'a', encoding='utf-8') as f:
                f.write(
                    f"Epoch {ep:02d} | train={train_acc:.2f}% val={val_acc:.2f}% "
                    f"best={self.best_val_acc:.2f}% gap={gap:.1f}%\n"
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.no_improve = 0
                self.save(SAVE_PATH, ep)
                print(f"  ✓ 新的最佳! val_acc={val_acc:.2f}%", flush=True)
            else:
                self.no_improve += 1
                print(f"  未提升 ({self.no_improve}/{self.early_stop_patience})", flush=True)
                if self.no_improve >= self.early_stop_patience:
                    print("\n早停触发", flush=True)
                    break

            with open(HISTORY_PATH, 'w') as f:
                json.dump(self.history, f, indent=2)

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        print(f"\n训练完成 ({(time.time()-start)/60:.1f} min)  最佳 val={self.best_val_acc:.2f}%")
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Wav2Vec2-Base 提分微调')
    parser.add_argument('--checkpoint', default=DEFAULT_CKPT)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--early-stop', type=int, default=6)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--no-weighted-sampler', action='store_true')
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(LIVE_LOG, 'w') as f:
        f.write('# boost | epoch train val best gap\n')

    cw, _ = compute_class_weights(boost=WEAK_CLASS_BOOST)
    print("类别权重:", dict(zip(config.EMOTIONS, [f'{w:.3f}' for w in cw.tolist()])))

    train_loader, val_loader, test_loader = get_wav2vec2_dataloaders(
        batch_size=args.batch_size,
        augment_level='light',
        weighted_sampler=not args.no_weighted_sampler,
        class_boost=WEAK_CLASS_BOOST,
    )

    trainer = BoostTrainer(lr=args.lr, early_stop=args.early_stop)
    trainer.set_class_weights(cw)
    trainer.load_checkpoint(args.checkpoint)
    trainer.train(train_loader, val_loader, epochs=args.epochs, log_interval=args.log_interval)

    if not args.no_eval:
        print("\n测试集评估...")
        evaluate_wav2vec2_model(trainer.model, test_loader)


if __name__ == '__main__':
    main()
