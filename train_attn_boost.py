"""
从 74% base 权重热启动，仅微调 Attention Pooling + 轻量顶层 encoder。

策略:
  - 迁移 wav2vec2 + classifier 权重，attention 初始化为 mean pool 等价
  - Epoch 1-5: 冻结 encoder，只训 attention + 头
  - Epoch 6+:  解冻顶层 4 层，极低学习率
  - 类别加权 CE（非 Focal），轻量增强
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
from models import Wav2Vec2AttentionModel
from data_processor import get_wav2vec2_dataloaders, compute_class_weights
from train import evaluate_wav2vec2_model

WEAK_CLASS_BOOST = {4: 1.15, 5: 1.20, 1: 1.10}


def unfreeze_schedule(epoch, total_layers, head_epochs=5):
    if epoch <= head_epochs:
        return total_layers, True
    return max(total_layers - 4, 0), True


class AttnBoostTrainer:
    def __init__(self, lr_head=5e-5, lr_enc=3e-6, early_stop=5,
                 label_smooth=0.05, head_epochs=5):
        self.device = config.DEVICE
        config.WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'
        self.lr_head = lr_head
        self.lr_enc = lr_enc
        self.early_stop_patience = early_stop
        self.label_smooth = label_smooth
        self.head_epochs = head_epochs

        self.model = Wav2Vec2AttentionModel(dropout=0.5).to(self.device)
        self.total_layers = self.model.num_encoder_layers
        self.current_phase = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        self.best_val_acc = 0.0
        self.best_state = None
        self.no_improve = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'lr': [], 'phase': [],
        }

    def set_class_weights(self, weights):
        self.criterion = nn.CrossEntropyLoss(
            weight=weights.to(self.device),
            label_smoothing=self.label_smooth,
        )

    def load_warm_start(self, path):
        base_val = self.model.load_from_mean_pool_checkpoint(path, self.device)
        self.best_val_acc = base_val
        self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        self._apply_freeze(1)

    def _apply_freeze(self, epoch):
        frozen, freeze_fe = unfreeze_schedule(epoch, self.total_layers, self.head_epochs)
        self.model.freeze_for_finetuning(num_frozen_layers=frozen, freeze_feature_extractor=freeze_fe)
        phase = f"ep{epoch}:frozen={frozen},fe={freeze_fe}"
        if phase != self.current_phase:
            self.current_phase = phase
            print(f"\n>> 解冻阶段: {phase}")

        feat, enc, head = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if 'classifier' in name or 'attention_pool' in name:
                head.append(p)
            elif 'feature_extractor' in name:
                feat.append(p)
            else:
                enc.append(p)

        self.optimizer = optim.AdamW([
            {'params': feat, 'lr': self.lr_enc * 0.5},
            {'params': enc, 'lr': self.lr_enc},
            {'params': head, 'lr': self.lr_head},
        ], weight_decay=0.01)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )

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
            'model_type': 'wav2vec2-attn-boost',
        }, path)
        print(f"  模型已保存: {path}")

    def train(self, train_loader, val_loader, epochs=12, log_interval=10):
        live_log = config.WAV2VEC2_ATTN_BOOST_LIVE_LOG
        save_path = config.WAV2VEC2_ATTN_BOOST_SAVE_PATH
        history_path = config.WAV2VEC2_ATTN_BOOST_HISTORY_PATH

        print(f"\n{'='*60}")
        print("  Wav2Vec2 Attention 热启动微调 (attn-boost)")
        print(f"  起点 val={self.best_val_acc:.2f}%  head_epochs={self.head_epochs}")
        print(f"  lr_head={self.lr_head}  lr_enc={self.lr_enc}")
        print(f"{'='*60}\n")

        start = time.time()
        for epoch in range(epochs):
            ep = epoch + 1
            self._apply_freeze(ep)
            print(f"\nEpoch [{ep}/{epochs}]", flush=True)
            print('-' * 40, flush=True)

            train_loss, train_acc = self.train_epoch(train_loader, log_interval)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[-1]['lr']
            gap = train_acc - val_acc

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(lr)
            self.history['phase'].append(self.current_phase)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%", flush=True)
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%  (gap={gap:.1f}%)", flush=True)
            print(f"LR head: {lr:.2e}  | {self.current_phase}", flush=True)

            with open(live_log, 'a', encoding='utf-8') as f:
                f.write(
                    f"Epoch {ep:02d} | train={train_acc:.2f}% val={val_acc:.2f}% "
                    f"best={self.best_val_acc:.2f}% gap={gap:.1f}% | {self.current_phase}\n"
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.no_improve = 0
                self.save(save_path, ep)
                print(f"  ✓ 新的最佳! val_acc={val_acc:.2f}%", flush=True)
            else:
                self.no_improve += 1
                print(f"  未提升 ({self.no_improve}/{self.early_stop_patience})", flush=True)
                if self.no_improve >= self.early_stop_patience:
                    print("\n早停触发", flush=True)
                    break

            with open(history_path, 'w') as f:
                json.dump(self.history, f, indent=2)

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        elapsed = (time.time() - start) / 60
        print(f"\n{'='*60}")
        print(f"训练完成! 用时 {elapsed:.1f} 分钟  最佳 val={self.best_val_acc:.2f}%")
        print(f"{'='*60}")
        return self.history


def main():
    parser = argparse.ArgumentParser(description='74% 热启动 + Attention 微调')
    parser.add_argument('--checkpoint', default=config.WAV2VEC2_ATTN_BOOST_CKPT)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr-head', type=float, default=5e-5)
    parser.add_argument('--lr-enc', type=float, default=3e-6)
    parser.add_argument('--head-epochs', type=int, default=5)
    parser.add_argument('--early-stop', type=int, default=5)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    with open(config.WAV2VEC2_ATTN_BOOST_LIVE_LOG, 'w') as f:
        f.write('# attn-boost | epoch train val best gap phase\n')

    cw, _ = compute_class_weights(boost=WEAK_CLASS_BOOST)
    print("类别权重:", dict(zip(config.EMOTIONS, [f'{w:.3f}' for w in cw.tolist()])))

    train_loader, val_loader, test_loader = get_wav2vec2_dataloaders(
        batch_size=args.batch_size,
        augment_level='light',
        weighted_sampler=True,
        class_boost=WEAK_CLASS_BOOST,
    )

    trainer = AttnBoostTrainer(
        lr_head=args.lr_head,
        lr_enc=args.lr_enc,
        early_stop=args.early_stop,
        head_epochs=args.head_epochs,
    )
    trainer.set_class_weights(cw)
    trainer.load_warm_start(args.checkpoint)
    trainer.train(train_loader, val_loader, epochs=args.epochs, log_interval=args.log_interval)

    if not args.no_eval:
        print("\n在测试集上评估最佳模型...")
        evaluate_wav2vec2_model(trainer.model, test_loader)


if __name__ == '__main__':
    main()
