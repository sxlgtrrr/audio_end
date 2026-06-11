import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from transformers import Wav2Vec2Model


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, num_lstm_layers=2):
        super(SpeechEmotionModel, self).__init__()

        self.mfcc_proj = nn.Conv2d(1, 32, 1)
        self.mel_proj = nn.Conv2d(1, 32, 1)
        self.chroma_proj = nn.Conv2d(1, 32, 1)

        self.stem = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        lstm_input = 256

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=256,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.4 if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, mfcc, mel_spec=None, chroma=None):
        batch_size = mfcc.size(0)

        x1 = F.interpolate(mfcc, size=(128, 94), mode='bilinear', align_corners=False)
        x1 = self.mfcc_proj(x1)

        if mel_spec is not None:
            x2 = F.interpolate(mel_spec, size=(128, 94), mode='bilinear', align_corners=False)
            x2 = self.mel_proj(x2)
        else:
            x2 = torch.zeros(batch_size, 32, 128, 94, device=mfcc.device)

        if chroma is not None:
            x3 = F.interpolate(chroma, size=(128, 94), mode='bilinear', align_corners=False)
            x3 = self.chroma_proj(x3)
        else:
            x3 = torch.zeros(batch_size, 32, 128, 94, device=mfcc.device)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        x = torch.cat([lstm_out[:, -1, :256], lstm_out[:, 0, 256:]], dim=1)

        out = self.classifier(x)
        return out


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(EmotionCNN, self).__init__()

        self.mfcc_proj = nn.Conv2d(1, 32, 1)
        self.mel_proj = nn.Conv2d(1, 32, 1)
        self.chroma_proj = nn.Conv2d(1, 32, 1)

        self.stem = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 1)),
            nn.Flatten(),
            nn.Linear(256 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, mfcc, mel_spec=None, chroma=None):
        b = mfcc.size(0)
        x1 = F.interpolate(mfcc, size=(128, 94), mode='bilinear', align_corners=False)
        x1 = self.mfcc_proj(x1)
        if mel_spec is not None:
            x2 = F.interpolate(mel_spec, size=(128, 94), mode='bilinear', align_corners=False)
            x2 = self.mel_proj(x2)
        else:
            x2 = torch.zeros(b, 32, 128, 94, device=mfcc.device)
        if chroma is not None:
            x3 = F.interpolate(chroma, size=(128, 94), mode='bilinear', align_corners=False)
            x3 = self.chroma_proj(x3)
        else:
            x3 = torch.zeros(b, 32, 128, 94, device=mfcc.device)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


class EmotionLSTM(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=248,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = torch.cat([lstm_out[:, -1, :256], lstm_out[:, 0, 256:]], dim=1)
        return self.classifier(x)


HybridCNNLSTM = SpeechEmotionModel


class AttentionPooling(nn.Module):
    """对时间维学习注意力权重，替代 mean pooling。"""

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def init_as_mean_pool(self):
        """零初始化最后一层，使 softmax 权重均匀 ≈ mean pooling。"""
        last = self.attn[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, hidden_states):
        scores = self.attn(hidden_states).squeeze(-1)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1)


class Wav2Vec2EmotionModel(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, dropout=None):
        super(Wav2Vec2EmotionModel, self).__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.WAV2VEC2_MODEL_NAME)

        hidden_size = self.wav2vec2.config.hidden_size
        drop = dropout if dropout is not None else 0.5

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop),

            nn.Linear(256, num_classes)
        )

        self._init_classifier()

    def freeze_for_finetuning(self, num_frozen_layers=None, freeze_feature_extractor=True):
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False

        encoder_layers = self.wav2vec2.encoder.layers
        total_layers = len(encoder_layers)
        if num_frozen_layers is None:
            num_frozen_layers = total_layers
        num_frozen_layers = min(num_frozen_layers, total_layers)

        for i, layer in enumerate(encoder_layers):
            requires_grad = i >= num_frozen_layers
            for param in layer.parameters():
                param.requires_grad = requires_grad

        for param in self.wav2vec2.encoder.layer_norm.parameters():
            param.requires_grad = num_frozen_layers < total_layers

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  可训练参数: {trainable/1e6:.1f}M / {total/1e6:.1f}M ({100*trainable/total:.1f}%)")
        return trainable, total

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, audio):
        outputs = self.wav2vec2(audio)
        x = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(x)


class Wav2Vec2AttentionModel(nn.Module):
    """Wav2Vec2 + Attention Pooling + MLP 分类头（改进版）。"""

    def __init__(self, num_classes=config.NUM_CLASSES, dropout=None):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.WAV2VEC2_MODEL_NAME)
        hidden_size = self.wav2vec2.config.hidden_size
        drop = dropout if dropout is not None else 0.5

        self.attention_pool = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in list(self.classifier) + list(self.attention_pool.attn):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @property
    def num_encoder_layers(self):
        return len(self.wav2vec2.encoder.layers)

    def freeze_for_finetuning(self, num_frozen_layers=None, freeze_feature_extractor=True):
        if freeze_feature_extractor:
            for p in self.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False

        total = self.num_encoder_layers
        num_frozen = total if num_frozen_layers is None else min(num_frozen_layers, total)

        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            trainable = i >= num_frozen
            for p in layer.parameters():
                p.requires_grad = trainable

        for p in self.wav2vec2.encoder.layer_norm.parameters():
            p.requires_grad = num_frozen < total

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.parameters())
        print(f"  可训练参数: {trainable/1e6:.1f}M / {total_p/1e6:.1f}M ({100*trainable/total_p:.1f}%)"
              f"  (encoder 冻结前 {num_frozen}/{self.num_encoder_layers} 层)")
        return trainable, total_p

    def forward(self, audio):
        hidden = self.wav2vec2(audio).last_hidden_state
        x = self.attention_pool(hidden)
        return self.classifier(x)

    def load_from_mean_pool_checkpoint(self, path, device=None):
        """从 Wav2Vec2EmotionModel(mean pool) 权重加载 backbone + 分类头。"""
        map_loc = device or 'cpu'
        ckpt = torch.load(path, map_location=map_loc)
        src = ckpt.get('model_state_dict', ckpt)
        dst = self.state_dict()
        loaded, skipped = [], []
        for k, v in src.items():
            if k in dst and dst[k].shape == v.shape:
                dst[k] = v
                loaded.append(k)
            else:
                skipped.append(k)
        self.load_state_dict(dst)
        self.attention_pool.init_as_mean_pool()
        base_val = ckpt.get('best_val_acc', 0.0)
        print(f"  已迁移 {len(loaded)} 个张量 (跳过 {len(skipped)})  起点 val≈{base_val:.2f}%")
        return base_val


# ============================================================
#  通用多骨架模型（HuBERT / WavLM / wav2vec2-xlsr 等）
# ============================================================
MULTIMODEL_CONFIGS = {
    'hubert-base': {
        'model_name': 'facebook/hubert-base-ls960',
        'hidden_size': 768,
    },
    'hubert-large': {
        'model_name': 'facebook/hubert-large-ll60k',
        'hidden_size': 1024,
    },
    'wavlm-base': {
        'model_name': 'microsoft/wavlm-base-plus',
        'hidden_size': 768,
    },
    'wavlm-large': {
        'model_name': 'microsoft/wavlm-large',
        'hidden_size': 1024,
    },
    'wav2vec2-xlsr': {
        'model_name': 'facebook/wav2vec2-xls-r-300m',
        'hidden_size': 1024,
    },
    'wav2vec2-large': {
        'model_name': 'facebook/wav2vec2-large',
        'hidden_size': 1024,
    },
    'wav2vec2-base': {
        'model_name': 'facebook/wav2vec2-base',
        'hidden_size': 768,
    },
}


class ConvExpert(nn.Module):
    """1D 卷积专家：用一维卷积提取时序局部特征，替代全连接层。

    输入 [B, T, H] → Conv1D 逐时间步提取局部模式 → 输出类别 logits。
    """

    def __init__(self, hidden_size, num_classes, expert_dim=256):

        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_size, expert_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(expert_dim),
            nn.GELU(),
            nn.Conv1d(expert_dim, expert_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(expert_dim // 2),
            nn.GELU(),
            nn.Conv1d(expert_dim // 2, expert_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(expert_dim // 4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(expert_dim // 4, num_classes)
        self._init()

    def _init(self):
        for m in self.conv_block:
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [B, T, H] → [B, H, T]
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.pool(x).squeeze(-1)          # [B, expert_dim//4]
        return self.head(x)                   # [B, num_classes]


class MoEClassifier(nn.Module):
    """MoE 分类头：专用专家（各情绪）+ 共享专家（正负强度）+ 路由门控。

    - specialized: 每个情绪类一个专用 1D Conv 专家
    - shared: 2~3 个共享专家，判别情绪极性 / 唤醒度
    - router: 学习如何加权组合各专家输出
    """

    def __init__(self, hidden_size, num_classes=6, num_specialized=6,
                 num_shared=2, expert_dim=256, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        num_experts = num_specialized + num_shared

        self.specialized = nn.ModuleList([
            ConvExpert(hidden_size, num_classes, expert_dim)
            for _ in range(num_specialized)
        ])
        self.shared = nn.ModuleList([
            ConvExpert(hidden_size, num_classes, expert_dim)
            for _ in range(num_shared)
        ])

        # 路由网络：从时序 mean-pooled 特征 → 专家权重
        self.router = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_experts),
        )
        self.temperature = nn.Parameter(torch.ones(1))

        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(num_classes, num_classes * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_classes * 2, num_classes),
        )

    @property
    def num_experts(self):
        return len(self.specialized) + len(self.shared)

    def forward(self, hidden_states, return_routing=False):
        # hidden_states: [B, T, H]
        B = hidden_states.size(0)

        # 路由
        pooled = hidden_states.mean(dim=1)
        route_logits = self.router(pooled)
        route_weights = F.softmax(route_logits * self.temperature, dim=-1)

        # 收集所有专家输出
        outputs = []
        for expert in self.specialized:
            outputs.append(expert(hidden_states))
        for expert in self.shared:
            outputs.append(expert(hidden_states))

        # [B, num_experts, num_classes]
        expert_stack = torch.stack(outputs, dim=1)

        # 加权融合: [B, num_classes]
        fused = (route_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
        logits = self.fusion(fused)

        if return_routing:
            return logits, route_weights
        return logits


class MultiBackboneMoEModel(nn.Module):
    """MultiBackbone + MoE 分类头：不同专家负责不同情绪，共享专家判别正负强度。"""

    def __init__(self, model_name, num_classes=config.NUM_CLASSES, dropout=0.5,
                 hidden_size=None, expert_dim=256, num_shared=2):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)

        if hidden_size is None:
            cfg = self.backbone.config
            if hasattr(cfg, 'hidden_size'):
                hidden_size = cfg.hidden_size
            elif hasattr(cfg, 'encoder_embed_dim'):
                hidden_size = cfg.encoder_embed_dim
            else:
                hidden_size = 768

        self.moe = MoEClassifier(
            hidden_size=hidden_size,
            num_classes=num_classes,
            num_specialized=num_classes,
            num_shared=num_shared,
            expert_dim=expert_dim,
            dropout=dropout,
        )

    @property
    def num_encoder_layers(self):
        cfg = self.backbone.config
        if hasattr(cfg, 'num_hidden_layers'):
            return cfg.num_hidden_layers
        if hasattr(cfg, 'encoder_layers'):
            return cfg.encoder_layers
        return 12

    def freeze_for_finetuning(self, num_frozen_layers=None, freeze_feature_extractor=True):
        if freeze_feature_extractor and hasattr(self.backbone, 'feature_extractor'):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False

        total = self.num_encoder_layers
        num_frozen = total if num_frozen_layers is None else min(num_frozen_layers, total)

        layers = None
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layers'):
            layers = self.backbone.encoder.layers
        elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer

        if layers is not None:
            for i, layer in enumerate(layers):
                trainable = i >= num_frozen
                for p in layer.parameters():
                    p.requires_grad = trainable

        if hasattr(self.backbone.encoder, 'layer_norm'):
            for p in self.backbone.encoder.layer_norm.parameters():
                p.requires_grad = num_frozen < total

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.parameters())
        print(f"  可训练参数: {trainable/1e6:.1f}M / {total_p/1e6:.1f}M ({100*trainable/total_p:.1f}%)"
              f"  (冻结前 {num_frozen}/{self.num_encoder_layers} 层)")
        return trainable, total_p

    def forward(self, audio, return_routing=False):
        outputs = self.backbone(audio)
        hidden = outputs.last_hidden_state          # [B, T, H]
        return self.moe(hidden, return_routing=return_routing)


class MultiBackboneModel(nn.Module):
    """通用 Wav2Vec2 类模型微调框架，支持 HuBERT / WavLM / wav2vec2-xlsr。

    所有模型均输出 last_hidden_state，API 统一，只需换 model_name。
    使用 Attention Pooling + MLP 分类头。
    """

    def __init__(self, model_name, num_classes=config.NUM_CLASSES, dropout=0.5,
                 hidden_size=None):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)

        # 自动检测 hidden_size
        if hidden_size is None:
            cfg = self.backbone.config
            if hasattr(cfg, 'hidden_size'):
                hidden_size = cfg.hidden_size
            elif hasattr(cfg, 'encoder_embed_dim'):
                hidden_size = cfg.encoder_embed_dim
            else:
                hidden_size = 768

        self.attention_pool = AttentionPooling(hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in list(self.classifier) + list(self.attention_pool.attn):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @property
    def num_encoder_layers(self):
        """获取 backbone encoder 层数（对不同模型灵活适配）。"""
        cfg = self.backbone.config
        if hasattr(cfg, 'num_hidden_layers'):
            return cfg.num_hidden_layers
        if hasattr(cfg, 'encoder_layers'):
            return cfg.encoder_layers
        return 12  # 默认

    def freeze_for_finetuning(self, num_frozen_layers=None, freeze_feature_extractor=True):
        """渐进解冻。"""
        # 冻结 feature extractor
        if freeze_feature_extractor and hasattr(self.backbone, 'feature_extractor'):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False

        total = self.num_encoder_layers
        num_frozen = total if num_frozen_layers is None else min(num_frozen_layers, total)

        # 获取 encoder layers
        layers = None
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layers'):
            layers = self.backbone.encoder.layers
        elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer

        if layers is not None:
            for i, layer in enumerate(layers):
                trainable = i >= num_frozen
                for p in layer.parameters():
                    p.requires_grad = trainable

        # 冻结 layer_norm
        if hasattr(self.backbone.encoder, 'layer_norm'):
            for p in self.backbone.encoder.layer_norm.parameters():
                p.requires_grad = num_frozen < total

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.parameters())
        print(f"  可训练参数: {trainable/1e6:.1f}M / {total_p/1e6:.1f}M ({100*trainable/total_p:.1f}%)"
              f"  (冻结前 {num_frozen}/{self.num_encoder_layers} 层)")
        return trainable, total_p

    def forward(self, audio, return_features=False):
        outputs = self.backbone(audio)
        hidden = outputs.last_hidden_state
        x = self.attention_pool(hidden)
        logits = self.classifier(x)
        if return_features:
            return logits, x
        return logits


# ============ Emotion CTM (Continuous Token-based Model) Temporal Reasoning ============

class EmotionCTMBlock(nn.Module):
    """轻量级 CTM-inspired 内部时序推理模块。

    核心思想：情绪不是某一帧决定的，而是由语速、停顿、能量、
    音高、颤音等随时间演化形成的。CTM Block 通过 K 步内部迭代，
    逐步从 WavLM 帧级特征中「读出」情绪状态。

    架构：
      H ∈ R^{T×D}  (WavLM 输出)
        ↓
      Adapter: D → d_attn
        ↓
      state_0 = 可学习 query (learned emotional prior)
        ↓
      For k in 1..K:
          state_k = state_{k-1} + CrossAttn(state_{k-1}, H)
          state_k = state_k + FFN(state_k)
          sync_k = cosine_sim(state_k, H)   # synchronization map
        ↓
      weighted_pool = Σ(softmax(sync_K) ⊙ H)   # frame-state synced pooling
        ↓
      → [B, D] utterance-level representation
    """

    def __init__(self, hidden_size, num_steps=4, embed_dim=256, num_heads=4,
                 dropout=0.3, ff_expand=4):
        super().__init__()
        self.num_steps = num_steps
        self.embed_dim = embed_dim

        # 维度压缩
        self.in_proj = nn.Linear(hidden_size, embed_dim)

        # 可学习的初始情绪 query
        self.emotion_query = nn.Parameter(torch.randn(1, embed_dim) * 0.02)

        # 内部迭代：cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ff_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ff_expand, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_ln = nn.LayerNorm(embed_dim)

        # 输出投影
        self.out_proj = nn.Linear(embed_dim, hidden_size)

    def _cross_attend(self, query, key_value):
        """query: [B, 1, d], key_value: [B, T, d]."""
        out, _ = self.cross_attn(query, key_value, key_value)
        return self.attn_ln(query + out)  # residual

    def _ffn_step(self, x):
        return self.ffn_ln(x + self.ffn(x))

    def forward(self, hidden_states, return_sync=False):
        """
        Args:
            hidden_states: [B, T, D]
        Returns:
            pooled: [B, D] utterance-level emotion representation
            (optional) sync_maps: list of [B, T] synchronization for each step
        """
        B, T, D = hidden_states.shape

        # 1. 维度压缩
        H = self.in_proj(hidden_states)          # [B, T, d]

        # 2. 初始 state
        state = self.emotion_query.expand(B, 1, -1)  # [B, 1, d]
        sync_maps = [] if return_sync else None

        for step in range(self.num_steps):
            # Cross-attend to all frames
            state = self._cross_attend(state, H)   # [B, 1, d]
            # FFN refine
            state = self._ffn_step(state)           # [B, 1, d]

            if return_sync:
                sync = F.cosine_similarity(
                    state, H, dim=-1           # [B, T]
                )
                sync_maps.append(sync)

        # 3. 最终 synchronization → attention pooling
        sync_final = F.cosine_similarity(
            state, H, dim=-1               # [B, T] (broadcast [B,1,d] × [B,T,d])
        )
        attn_weights = torch.softmax(sync_final, dim=-1)   # [B, T]
        pooled = (attn_weights.unsqueeze(-1) * hidden_states).sum(dim=1)  # [B, D]

        if return_sync:
            return pooled, sync_maps + [sync_final]
        return pooled


class CTMWavLMModel(nn.Module):
    """WavLM/HuBERT Backbone + Emotion CTM Temporal Reasoning + Classifier.

    pipeline:
      waveform → Backbone → [B,T,D] → CTM Block (K-step reasoning)
              → pooled [B,D] → classifier → [B, num_classes]
    """

    def __init__(self, model_name, num_classes=config.NUM_CLASSES, dropout=0.5,
                 hidden_size=None, ctm_steps=4, ctm_dim=256, ctm_heads=4):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name)

        if hidden_size is None:
            cfg = self.backbone.config
            hidden_size = cfg.hidden_size if hasattr(cfg, 'hidden_size') else 768

        # 中游：CTM temporal reasoning block
        self.ctm = EmotionCTMBlock(
            hidden_size=hidden_size,
            num_steps=ctm_steps,
            embed_dim=ctm_dim,
            num_heads=ctm_heads,
            dropout=dropout,
        )

        # 下游分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    @property
    def num_encoder_layers(self):
        cfg = self.backbone.config
        if hasattr(cfg, 'num_hidden_layers'):
            return cfg.num_hidden_layers
        if hasattr(cfg, 'encoder_layers'):
            return cfg.encoder_layers
        return 12

    def freeze_for_finetuning(self, num_frozen_layers=None, freeze_feature_extractor=True):
        if freeze_feature_extractor and hasattr(self.backbone, 'feature_extractor'):
            for p in self.backbone.feature_extractor.parameters():
                p.requires_grad = False

        total = self.num_encoder_layers
        num_frozen = total if num_frozen_layers is None else min(num_frozen_layers, total)

        layers = None
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layers'):
            layers = self.backbone.encoder.layers
        elif hasattr(self.backbone, 'encoder') and hasattr(self.backbone.encoder, 'layer'):
            layers = self.backbone.encoder.layer

        if layers is not None:
            for i, layer in enumerate(layers):
                trainable = i >= num_frozen
                for p in layer.parameters():
                    p.requires_grad = trainable

        if hasattr(self.backbone.encoder, 'layer_norm'):
            for p in self.backbone.encoder.layer_norm.parameters():
                p.requires_grad = num_frozen < total

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in self.parameters())
        print(f"  可训练参数: {trainable/1e6:.1f}M / {total_p/1e6:.1f}M ({100*trainable/total_p:.1f}%)"
              f"  (冻结前 {num_frozen}/{self.num_encoder_layers} 层)")
        return trainable, total_p

    def forward(self, audio, return_features=False):
        outputs = self.backbone(audio)
        hidden = outputs.last_hidden_state          # [B, T, D]
        pooled = self.ctm(hidden)                   # [B, D]
        logits = self.classifier(pooled)
        if return_features:
            return logits, pooled
        return logits


class CenterLoss(nn.Module):
    """Center Loss: 让同类样本的特征靠近其类中心，增强类内紧凑性。

    Wen et al. ECCV 2016。被 MLL 2025 在 SER 中验证有效。
    与 SupCon 互补：SupCon 拉大类间距离，Center Loss 缩紧类内距离。
    """

    def __init__(self, num_classes, feat_dim, init_scale=0.1):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim) * init_scale)

    def forward(self, features, labels):
        """features: [B, D] (L2 normalized), labels: [B]"""
        batch_size = features.size(0)
        centers_batch = self.centers[labels]       # [B, D]
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean() / 2.0
        return loss


class STDAttentionPooling(nn.Module):
    """STD + Attention 联合池化（WavLM SER 2024 验证有效）。

    同时利用两种信息：
    - Attention: 「哪些帧重要」
    - STD: 「帧间变化多大」（情绪动态 = 高方差）
    两种池化后的向量拼接，投射回原维度。
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, T, D]
        Returns:
            pooled: [B, D]
        """
        # Attention pooling
        scores = self.attn(hidden_states).squeeze(-1)     # [B, T]
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        attn_pooled = (hidden_states * weights).sum(dim=1)  # [B, D]

        # STD pooling
        std_pooled = hidden_states.std(dim=1)              # [B, D]

        # Concat fusion
        fused = torch.cat([attn_pooled, std_pooled], dim=-1)  # [B, 2D]
        return self.fusion(fused)                          # [B, D]


def get_model(model_type='hybrid', num_classes=config.NUM_CLASSES):
    models_map = {
        'cnn': EmotionCNN(num_classes=num_classes),
        'lstm': EmotionLSTM(num_classes=num_classes),
        'hybrid': SpeechEmotionModel(num_classes=num_classes),
        'wav2vec2': Wav2Vec2EmotionModel(num_classes=num_classes),
        'wav2vec2_attn': Wav2Vec2AttentionModel(num_classes=num_classes),
    }

    if model_type not in models_map:
        raise ValueError(f"Unknown model type: {model_type}")

    return models_map[model_type].to(config.DEVICE)


if __name__ == '__main__':
    model = get_model('hybrid')
    print(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
