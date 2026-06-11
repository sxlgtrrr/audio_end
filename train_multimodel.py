"""
通用多骨架微调：HuBERT / WavLM / wav2vec2-xlsr / wav2vec2-large

用法:
  python train_multimodel.py --model hubert-base              # HuBERT Base
  python train_multimodel.py --model wavlm-base                # WavLM Base+
  python train_multimodel.py --model wav2vec2-xlsr             # wav2vec2-XLSR
  python train_multimodel.py --model wavlm-large --strong-aug  # WavLM Large + 强增强

策略:
  - Attention Pooling + MLP 分类头
  - 渐进解冻: epoch 1-3 仅头 → epoch 4-7 顶层4层 → epoch 8+ 全部
  - Focal Loss (gamma=2) 处理类别不平衡
  - 可选 Mixup + Strong Augmentation
  - 早停 patience=8
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
from models import (MultiBackboneModel, MultiBackboneMoEModel,
                    CTMWavLMModel, CenterLoss, MULTIMODEL_CONFIGS)
from data_processor import (Wav2Vec2EmotionDataset, Wav2Vec2StrongDataset,
                            get_wav2vec2_dataloaders, wav2vec2_collate_fn,
                            mixup_data, mixup_criterion)
from torch.utils.data import DataLoader


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


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss: 同类靠近，异类远离。

    基于 Khosla et al. 2020 的简化版。
    不直接在 logits 上计算，而在 CTM 输出的 pooled 特征上做对比。"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] L2-normalized
            labels: [B] class indices
        """
        device = features.device
        B = features.size(0)

        # [B, B] cosine similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # 排除自身
        mask = torch.eye(B, device=device).bool()
        sim = sim.masked_fill(mask, -1e9)

        # 同类 mask
        labels = labels.contiguous().view(-1, 1)
        pos_mask = labels.eq(labels.T).float()
        pos_mask = pos_mask.masked_fill(mask, 0)

        # 分母：所有样本的 exp(sim) 之和（除自身）
        exp_sim = torch.exp(sim)
        denom = exp_sim.sum(dim=1, keepdim=True)

        # 分子：正样本的 exp(sim) 之和
        pos_sim = (exp_sim * pos_mask).sum(dim=1, keepdim=True)

        # 有同类样本才计算
        valid = pos_mask.sum(dim=1) > 0
        loss = -torch.log((pos_sim[valid] / (denom[valid] + 1e-8)) + 1e-8).mean()
        return loss


def unfreeze_schedule(epoch, total_layers, head_epochs=3, mid_epochs=7):
    if epoch <= head_epochs:
        return total_layers, True
    if epoch <= mid_epochs:
        return max(total_layers - 4, 0), True
    return 0, True


class MultiModelTrainer:
    def __init__(self, model_key='hubert-base', lr=2e-5, weight_decay=0.01,
                 focal_gamma=2.0, early_stop=8, head_epochs=3, mid_epochs=7,
                 dropout=0.5, augment_level='full', use_mixup=False,
                 mixup_alpha=0.2, label_smooth=0.0,
                 use_moe=False, use_ctm=False, ctm_steps=4,
                 supcon_weight=0.1, center_weight=0.1,
                 use_mean_pool=False):
        self.device = config.DEVICE
        self.model_key = model_key
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_moe = use_moe
        self.use_ctm = use_ctm
        self.ctm_steps = ctm_steps
        self.supcon_weight = supcon_weight
        self.supcon = None
        self.center_weight = center_weight
        self.use_mean_pool = use_mean_pool
        self.center_loss = None

        cfg = MULTIMODEL_CONFIGS[model_key]
        model_name = cfg['model_name']
        hidden_size = cfg['hidden_size']

        if use_ctm:
            tag = f'CTM (K={ctm_steps})'
        elif use_moe:
            tag = 'MoE (Conv Experts)'
        elif use_mean_pool:
            tag = 'Mean Pooling'
        else:
            tag = 'Attention Pooling'
        print(f"加载 {model_name} ({tag}) ...")

        if use_ctm:
            self.model = CTMWavLMModel(
                model_name, config.NUM_CLASSES, dropout=dropout,
                hidden_size=hidden_size, ctm_steps=ctm_steps
            ).to(self.device)
        elif use_moe:
            self.model = MultiBackboneMoEModel(
                model_name, config.NUM_CLASSES, dropout=dropout, hidden_size=hidden_size
            ).to(self.device)
        else:
            self.model = MultiBackboneModel(
                model_name, config.NUM_CLASSES, dropout=dropout, hidden_size=hidden_size
            ).to(self.device)
            if use_mean_pool:
                self.model.attention_pool.init_as_mean_pool()

        self.total_layers = self.model.num_encoder_layers
        print(f"  Encoder 层数: {self.total_layers}   hidden_size: {hidden_size}")

        self.current_phase = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop
        self.head_epochs = head_epochs
        self.mid_epochs = mid_epochs
        self.augment_level = augment_level

        # Focal Loss with slight class boost
        alpha = torch.ones(config.NUM_CLASSES, device=self.device)
        for idx, emo in enumerate(config.EMOTIONS):
            if emo in ('sad', 'disgust'):
                alpha[idx] = 1.15
            elif emo == 'neutral':
                alpha[idx] = 1.10

        # 如果用 mixup 或 label_smooth，用普通 CE（FocalLoss 不支持 soft target）
        if self.use_mixup:
            self.criterion = nn.CrossEntropyLoss(weight=alpha, reduction='none')
            self.val_criterion = nn.CrossEntropyLoss(weight=alpha)
        elif label_smooth > 0:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth,
                                                  weight=alpha)
            self.val_criterion = self.criterion
        else:
            self.criterion = FocalLoss(gamma=focal_gamma, alpha=alpha)
            self.val_criterion = self.criterion
        self.label_smooth = label_smooth

        # SupCon loss (仅 CTM 模型使用)
        if supcon_weight > 0:
            self.supcon = SupConLoss(temperature=0.07)
            self.supcon_weight = supcon_weight
            print(f"  + SupCon loss (weight={supcon_weight})")

        # Center loss (CTM + baseline 都可使用)
        if center_weight > 0:
            self.center_loss = CenterLoss(config.NUM_CLASSES, hidden_size)
            self.center_loss.to(self.device)
            self.center_weight = center_weight
            print(f"  + Center loss (weight={center_weight})")

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

        save_path = self._save_path()
        history_path = self._history_path()
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 清除 live log
        live_log = self._live_log_path()
        with open(live_log, 'w', encoding='utf-8') as f:
            f.write(f'# {model_key} | epoch train val best gap phase\n')

    def _exp_suffix(self):
        """根据实验配置生成不同后缀，防止覆盖。"""
        parts = []
        if self.use_ctm:
            parts.append(f'ctm_k{self.ctm_steps}')
        elif self.use_moe:
            parts.append('moe')
        elif self.use_mean_pool:
            parts.append('meanpool')
        if self.supcon_weight > 0:
            parts.append(f'sup{self.supcon_weight}')
        if self.center_weight > 0:
            parts.append(f'cen{self.center_weight}')
        return '_' + '_'.join(parts) if parts else ''

    def _save_path(self):
        return f'./models/best_model_{self.model_key}{self._exp_suffix()}_best.pth'

    def _history_path(self):
        return f'./logs/{self.model_key}{self._exp_suffix()}_history.json'

    def _live_log_path(self):
        return f'./logs/train_metrics_{self.model_key}{self._exp_suffix()}.txt'

    def _param_groups(self):
        feat, enc, head = [], [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if ('classifier' in name or 'attention_pool' in name
                    or 'moe' in name or 'ctm' in name):
                head.append(p)
            elif 'feature_extractor' in name or 'feature_projection' in name:
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
        # Center loss 参数也加入优化器
        if self.center_loss is not None:
            groups.append({'params': self.center_loss.parameters(), 'lr': self.lr * 10})
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
        print(f"\n>> 渐进解冻: epoch {epoch} | 冻结前 {frozen}/{self.total_layers} 层"
              f" | feature_extractor={'冻结' if freeze_fe else '可训'}", flush=True)
        self.model.freeze_for_finetuning(num_frozen_layers=frozen,
                                          freeze_feature_extractor=freeze_fe)
        self._rebuild_optimizer()
        return True

    def train_epoch(self, loader, epoch=None):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        desc = f'Epoch {epoch} Train' if epoch else 'Train'
        pbar = tqdm(loader, desc=desc, leave=True, file=sys.stdout)
        use_supcon = self.supcon is not None
        use_center = self.center_loss is not None
        need_features = use_supcon or use_center

        for step, batch in enumerate(pbar, 1):
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)

            if self.use_mixup and self.mixup_alpha > 0:
                audio, y_a, y_b, lam = mixup_data(audio, labels, self.mixup_alpha)

            self.optimizer.zero_grad()

            # 需要池化特征的 loss（SupCon / Center）
            if need_features:
                logits, features = self.model(audio, return_features=True)
                ce_loss = self.criterion(logits, labels)
                loss = ce_loss
                features_norm = F.normalize(features, dim=1)

                if use_supcon:
                    sup_loss = self.supcon(features_norm, labels)
                    loss = loss + self.supcon_weight * sup_loss
                if use_center:
                    center_loss = self.center_loss(features_norm, labels)
                    loss = loss + self.center_weight * center_loss

                if self.use_mixup:
                    loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                    if use_supcon:
                        loss = loss + self.supcon_weight * self.supcon(features_norm, labels)
                    if use_center:
                        loss = loss + self.center_weight * self.center_loss(features_norm, labels)
            else:
                outputs = self.model(audio)

                if self.use_mixup:
                    loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                else:
                    loss = self.criterion(outputs, labels)
                logits = outputs

            correct += (logits.argmax(1) == labels).sum().item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], max_norm=1.0
            )
            self.optimizer.step()

            total_loss += loss.item()
            total += labels.size(0)
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
            total_loss += self.val_criterion(outputs, labels).item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / len(loader), 100.0 * correct / total

    def save_checkpoint(self, path, epoch):
        data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_type': self.model_key +
                           ('-ctm' if self.use_ctm else
                            '-moe' if self.use_moe else ''),
        }
        if self.center_loss is not None:
            data['center_loss_state_dict'] = self.center_loss.state_dict()
        torch.save(data, path)
        print(f"  模型已保存: {path}", flush=True)

    def train(self, train_loader, val_loader, epochs=25, start_epoch=1):
        start = time.time()
        aug_tag = f'+Mixup(α={self.mixup_alpha})' if self.use_mixup else f'+aug={self.augment_level}'
        print(f"\n{'='*60}")
        print(f"  多骨架微调: {self.model_key}  {aug_tag}")
        print(f"  lr={self.lr}  wd={self.weight_decay}"
              f"  focal_gamma={getattr(self.criterion, 'gamma', 'N/A')}")
        print(f"  解冻: 1-{self.head_epochs} head | "
              f"{self.head_epochs+1}-{self.mid_epochs} top4 | {self.mid_epochs+1}+ all")
        print(f"{'='*60}\n")

        for epoch in range(start_epoch - 1, epochs):
            ep = epoch + 1
            self._apply_freeze(ep)
            print(f"\nEpoch [{ep}/{epochs}]", flush=True)
            print('-' * 40, flush=True)

            train_loss, train_acc = self.train_epoch(train_loader, ep)
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

            with open(self._live_log_path(), 'a', encoding='utf-8') as f:
                f.write(
                    f"Epoch {ep:02d} | train={train_acc:.2f}% val={val_acc:.2f}% "
                    f"best={self.best_val_acc:.2f}% gap={gap:.1f}% | {self.current_phase}\n"
                )

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.no_improve = 0
                self.save_checkpoint(self._save_path(), ep)
                print(f"  ✓ 新的最佳! val_acc={val_acc:.2f}%", flush=True)
            else:
                self.no_improve += 1
                print(f"  未提升 ({self.no_improve}/{self.early_stop_patience})", flush=True)
                if self.no_improve >= self.early_stop_patience:
                    print("\n早停触发", flush=True)
                    break

            with open(self._history_path(), 'w') as f:
                json.dump(self.history, f, indent=2)

        if self.best_state:
            self.model.load_state_dict(self.best_state)
        elapsed = (time.time() - start) / 60
        print(f"\n{'='*60}")
        print(f"训练完成! 用时 {elapsed:.1f} 分钟  最佳 val={self.best_val_acc:.2f}%")
        print(f"{'='*60}")
        return self.history


def get_dataloaders_with_strong(augment_level, batch_size=8):
    """获取数据加载器，支持 strong augmentation。"""
    import multiprocessing
    # 在某些环境下 num_workers>0 会导致死锁，保守设 0
    try:
        _nw = min(2, multiprocessing.cpu_count() // 4)
    except Exception:
        _nw = 0

    if augment_level in ('full', 'light', 'strong'):
        train_ds = Wav2Vec2StrongDataset(mode='train', augment_level=augment_level)
    else:
        train_ds = Wav2Vec2EmotionDataset(mode='train', augment_level=augment_level)

    val_ds = Wav2Vec2EmotionDataset(mode='val')
    test_ds = Wav2Vec2EmotionDataset(mode='test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=wav2vec2_collate_fn, num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           collate_fn=wav2vec2_collate_fn, num_workers=0,
                           pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=wav2vec2_collate_fn, num_workers=0,
                            pin_memory=True)
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='多骨架微调训练')
    parser.add_argument('--model', default='hubert-base',
                        choices=list(MULTIMODEL_CONFIGS.keys()),
                        help='预训练模型骨架')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--focal-gamma', type=float, default=2.0)
    parser.add_argument('--early-stop', type=int, default=8)
    parser.add_argument('--head-epochs', type=int, default=3)
    parser.add_argument('--mid-epochs', type=int, default=7)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--augment', default='full',
                        choices=['none', 'light', 'full', 'strong'])
    parser.add_argument('--mixup', action='store_true', help='启用 Mixup 增强')
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--label-smooth', type=float, default=0.0,
                        help='>0 则用 LabelSmoothing 替代 Focal Loss')
    parser.add_argument('--no-eval', action='store_true')
    parser.add_argument('--resume', default='', help='从 checkpoint 续训')
    parser.add_argument('--moe', action='store_true', help='使用 MoE 分类头 (Conv Experts)')
    parser.add_argument('--ctm', action='store_true', help='使用 CTM 时序推理块')
    parser.add_argument('--ctm-steps', type=int, default=4, help='CTM 内部推理步数 K')
    parser.add_argument('--supcon-weight', type=float, default=0.1,
                        help='SupCon loss 权重 (0=禁用, 推荐 0.1)')
    parser.add_argument('--center-weight', type=float, default=0.0,
                        help='Center loss 权重 (0=禁用, 推荐 0.1)')
    parser.add_argument('--mean-pool', action='store_true',
                        help='使用 Mean Pooling 替代 Attention Pooling')
    args = parser.parse_args()

    print("加载数据...", flush=True)
    train_loader, val_loader, test_loader = get_dataloaders_with_strong(
        args.augment, args.batch_size
    )

    trainer = MultiModelTrainer(
        model_key=args.model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        focal_gamma=args.focal_gamma,
        early_stop=args.early_stop,
        head_epochs=args.head_epochs,
        mid_epochs=args.mid_epochs,
        dropout=args.dropout,
        augment_level=args.augment,
        use_mixup=args.mixup,
        mixup_alpha=args.mixup_alpha,
        label_smooth=args.label_smooth,
        use_moe=args.moe,
        use_ctm=args.ctm,
        ctm_steps=args.ctm_steps,
        supcon_weight=args.supcon_weight,
        center_weight=args.center_weight,
        use_mean_pool=args.mean_pool,
    )

    start_epoch = 1
    if args.resume:
        ckpt_path = args.resume
        print(f"  续训 checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        if 'center_loss_state_dict' in ckpt and trainer.center_loss is not None:
            trainer.center_loss.load_state_dict(ckpt['center_loss_state_dict'])
        trainer.optimizer = optim.AdamW(
            trainer._param_groups(), weight_decay=trainer.weight_decay
        )
        if 'optimizer_state_dict' in ckpt:
            try:
                trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception:
                print("  警告: 无法恢复优化器状态，使用全新优化器")
        trainer.scheduler = ReduceLROnPlateau(
            trainer.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
        )
        trainer.best_val_acc = ckpt.get('best_val_acc', 0.0)
        start_epoch = ckpt.get('epoch', 0) + 1
        history_path = trainer._history_path()
        if os.path.isfile(history_path):
            with open(history_path) as f:
                trainer.history = json.load(f)
        print(f"  续训起始 epoch={start_epoch}, best_val={trainer.best_val_acc:.2f}%")

    trainer.train(train_loader, val_loader, epochs=args.epochs, start_epoch=start_epoch)

    if not args.no_eval:
        print("\n在测试集上评估最佳模型...")
        from train import evaluate_wav2vec2_model
        evaluate_wav2vec2_model(trainer.model, test_loader)


if __name__ == '__main__':
    main()
