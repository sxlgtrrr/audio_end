"""
Wav2Vec2-Base 改进微调：Attention Pooling + 渐进解冻 + Focal Loss

渐进解冻（默认）:
  Epoch 1-3:  仅训练 attention + 分类头（encoder 全冻结）
  Epoch 4-7:  解冻 encoder 顶层 4 层
  Epoch 8+:   解冻全部 encoder（feature_extractor 仍冻结，最后阶段可选放开）
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from models import Wav2Vec2AttentionModel
from data_processor import get_wav2vec2_dataloaders
from train import evaluate_wav2vec2_model


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha[targets] * loss
        return loss.mean()


def unfreeze_schedule(epoch, total_layers, head_epochs=3, mid_epochs=7):
    """返回 (num_frozen_encoder_layers, freeze_feature_extractor)。"""
    if epoch <= head_epochs:
        return total_layers, True
    if epoch <= mid_epochs:
        return max(total_layers - 4, 0), True
    return 0, True


class ImprovedTrainer:
    def __init__(self, lr=None, weight_decay=0.01, focal_gamma=2.0,
                 early_stop=None, head_epochs=3, mid_epochs=7):
        self.device = config.DEVICE
        config.WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'
        self.lr = lr or config.WAV2VEC2_IMPROVED_LR
        self.weight_decay = weight_decay
        self.focal_gamma = focal_gamma
        self.early_stop_patience = early_stop or config.WAV2VEC2_IMPROVED_EARLY_STOP
        self.head_epochs = head_epochs
        self.mid_epochs = mid_epochs

        print(f"加载 {config.WAV2VEC2_MODEL_NAME} (Attention 版) ...")
        self.model = Wav2Vec2AttentionModel(dropout=0.5).to(self.device)
        self.total_layers = self.model.num_encoder_layers
        self.current_phase = None

        alpha = torch.ones(config.NUM_CLASSES, device=self.device)
        for idx, emo in enumerate(config.EMOTIONS):
            if emo in ('sad', 'disgust'):
                alpha[idx] = 1.15
            elif emo == 'neutral':
                alpha[idx] = 1.10
        self.criterion = FocalLoss(gamma=self.focal_gamma, alpha=alpha)

        self.optimizer = None
        self.scheduler = None
        self._apply_freeze(1)

        self.best_val_acc = 0.0
        self.best_state = None
        self.no_improve = 0
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [], 'lr': [], 'phase': [],
        }

        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.WAV2VEC2_IMPROVED_SAVE_PATH), exist_ok=True)

    def _param_groups(self):
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
        groups = []
        if feat:
            groups.append({'params': feat, 'lr': self.lr * 0.5})
        if enc:
            groups.append({'params': enc, 'lr': self.lr})
        if head:
            groups.append({'params': head, 'lr': self.lr * 10})
        return groups

    def _rebuild_optimizer(self):
        groups = self._param_groups()
        if not groups:
            raise RuntimeError('没有可训练参数')
        self.optimizer = optim.AdamW(groups, weight_decay=self.weight_decay)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )

    def _apply_freeze(self, epoch):
        frozen, freeze_fe = unfreeze_schedule(
            epoch, self.total_layers, self.head_epochs, self.mid_epochs
        )
        phase = f'ep{epoch}:frozen={frozen},fe={freeze_fe}'
        if phase == self.current_phase:
            return False
        self.current_phase = phase
        print(f"\n>> 渐进解冻: epoch {epoch} | 冻结 encoder 前 {frozen}/{self.total_layers} 层"
              f" | feature_extractor={'冻结' if freeze_fe else '可训'}", flush=True)
        self.model.freeze_for_finetuning(num_frozen_layers=frozen, freeze_feature_extractor=freeze_fe)
        self._rebuild_optimizer()
        return True

    def train_epoch(self, loader, epoch=None, log_interval=10):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        desc = f'Epoch {epoch} Train' if epoch else 'Train'
        pbar = tqdm(loader, desc=desc, leave=True, file=sys.stdout)
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
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            if step % log_interval == 0 or step == len(loader):
                pbar.set_postfix(loss=f'{total_loss/step:.4f}',
                                 acc=f'{100*correct/total:.1f}%', refresh=False)
        return total_loss / len(loader), 100.0 * correct / total

    @torch.no_grad()
    def validate(self, loader, epoch=None):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        desc = f'Epoch {epoch} Val' if epoch else 'Val'
        for batch in tqdm(loader, desc=desc, leave=True, file=sys.stdout):
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)
            outputs = self.model(audio)
            total_loss += self.criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), 100.0 * correct / total

    def save_checkpoint(self, path, epoch):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_type': 'wav2vec2-base-improved',
        }, path)
        print(f"  模型已保存: {path}", flush=True)

    def train(self, train_loader, val_loader, epochs=25, log_interval=10):
        start = time.time()
        print(f"\n{'='*60}")
        print("  Wav2Vec2-Base 改进微调")
        print("  Attention Pooling + 渐进解冻 + Focal Loss")
        print(f"  lr={self.lr}  focal_gamma={self.focal_gamma}  早停={self.early_stop_patience}")
        print(f"  解冻阶段: 1-{self.head_epochs} 仅头 | "
              f"{self.head_epochs+1}-{self.mid_epochs} 顶层4层 | "
              f"{self.mid_epochs+1}+ 全encoder")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            ep = epoch + 1
            self._apply_freeze(ep)
            print(f"\nEpoch [{ep}/{epochs}]", flush=True)
            print('-' * 40, flush=True)

            train_loss, train_acc = self.train_epoch(train_loader, ep, log_interval)
            val_loss, val_acc = self.validate(val_loader, ep)
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
            print(f"LR (head): {lr:.2e}  |  {self.current_phase}", flush=True)

            with open(config.WAV2VEC2_IMPROVED_LIVE_LOG, 'a', encoding='utf-8') as f:
                f.write(
                    f"Epoch {ep:02d} | train={train_acc:.2f}% val={val_acc:.2f}% "
                    f"best={self.best_val_acc:.2f}% gap={gap:.1f}% | {self.current_phase}\n"
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.no_improve = 0
                self.save_checkpoint(config.WAV2VEC2_IMPROVED_SAVE_PATH, ep)
                print(f"  ✓ 新的最佳! val_acc={val_acc:.2f}%", flush=True)
            else:
                self.no_improve += 1
                print(f"  未提升 ({self.no_improve}/{self.early_stop_patience})", flush=True)
                if self.no_improve >= self.early_stop_patience:
                    print("\n早停触发", flush=True)
                    break

            with open(config.WAV2VEC2_IMPROVED_HISTORY_PATH, 'w') as f:
                json.dump(self.history, f, indent=2)

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        elapsed = (time.time() - start) / 60
        print(f"\n{'='*60}")
        print(f"训练完成! 用时 {elapsed:.1f} 分钟  最佳 val={self.best_val_acc:.2f}%")
        print(f"{'='*60}")
        return self.history


def main():
    parser = argparse.ArgumentParser(description='Wav2Vec2 改进微调')
    parser.add_argument('--epochs', type=int, default=config.WAV2VEC2_IMPROVED_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.WAV2VEC2_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=config.WAV2VEC2_IMPROVED_LR)
    parser.add_argument('--focal-gamma', type=float, default=config.WAV2VEC2_IMPROVED_FOCAL_GAMMA)
    parser.add_argument('--early-stop', type=int, default=config.WAV2VEC2_IMPROVED_EARLY_STOP)
    parser.add_argument('--head-epochs', type=int, default=3, help='仅训分类头阶段')
    parser.add_argument('--mid-epochs', type=int, default=7, help='部分解冻阶段结束 epoch')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()

    config.WAV2VEC2_BATCH_SIZE = args.batch_size
    with open(config.WAV2VEC2_IMPROVED_LIVE_LOG, 'w', encoding='utf-8') as f:
        f.write('# improved | epoch train val best gap phase\n')

    print("加载数据...", flush=True)
    train_loader, val_loader, test_loader = get_wav2vec2_dataloaders(
        batch_size=args.batch_size, augment_level='light'
    )

    trainer = ImprovedTrainer(
        lr=args.lr,
        focal_gamma=args.focal_gamma,
        early_stop=args.early_stop,
        head_epochs=args.head_epochs,
        mid_epochs=args.mid_epochs,
    )
    trainer.train(train_loader, val_loader, epochs=args.epochs, log_interval=args.log_interval)

    if not args.no_eval:
        print("\n在测试集上评估最佳模型...")
        evaluate_wav2vec2_model(trainer.model, test_loader)


if __name__ == '__main__':
    main()
