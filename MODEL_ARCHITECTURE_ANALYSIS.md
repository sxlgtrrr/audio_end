# 语音情感识别 (SER) 模型架构分析

> 生成日期: 2026-06-11

---

## 1. 系统总览

本项目是一个基于深度学习的语音情感识别系统，从**手工声学特征 + 轻量神经网络**逐步演进到**大规模预训练语音模型微调**，并在池化策略和时序推理上做了多轮改进。系统支持 6 类情感识别：\ngry\, \disgust\, \ear\, \happy\, \
eutral\, \sad\。

### 整体数据流

\\\
原始音频 (16kHz, 3s)
  ↓
AudioProcessor: 加载 → 归一化 → 预加重 → 数据增强 (噪声/时移/音高/变速)
  ↓
FeatureExtractor: MFCC(含Delta) + Mel频谱 + Chroma + Spectral Contrast + Tonnetz + ZCR + RMS
  ↓
模型 (多种可选架构，见下文)
  ↓
6 类情感概率输出
\\\

---

## 2. 第一条路线：手工特征 + 轻量网络

核心思路：提取传统声学特征作为输入，模型从零训练，参数量小，推理快。

### 2.1 SpeechEmotionModel (HybridCNN-LSTM) — 主力混合模型

\\\
输入: MFCC [1,40,T] / Mel [1,128,T] / Chroma [1,12,T]
  │  各自双线性插值到 [128,94]，1×1 卷积升到 32 维
  │  三路拼接 → [96,128,94]
  ▼
Stem: Conv2d(96→64, 3×3) + BN + ReLU
  ▼
Res Block ×2 (64→64, 步长=1)            ── 空间: 128×94
  ▼
Res Block ×2 (64→128, 步长=2)           ── 空间: 64×47
  ▼
Res Block ×2 (128→256, 步长=2)          ── 空间: 32×24
  ▼
AdaptiveAvgPool2d((1, None))            ── 时间维保留，频率维压缩 [B,256,1,T']
  ▼
BiLSTM (2 层, 隐藏=256, dropout=0.4)    ── 建模时序依赖
  ▼  last-forward ⊕ first-backward 拼接 → [B,512]
  ▼
MLP: Linear(512→256) → BN → ReLU → Dropout(0.5) → Linear(256→6)
\\\

**设计思路**：
- CNN 部分类似 ResNet 的残差结构，在时频图上做局部特征提取，逐步降采样扩大感受野。
- BiLSTM 接在 CNN 之后，捕捉帧间的长期依赖关系（语调升降、节奏变化等）。
- 取前向最后一帧和反向第一帧拼接，同时保留句子起始和结束的上下文。
- 1×1 卷积各自投影三种特征到同一维度，允许不同特征在通道维融合。

### 2.2 EmotionCNN — 纯 CNN 版本

去掉 LSTM，直接用 \AdaptiveAvgPool2d((4,1))\ 拉平后接 MLP。速度快但缺乏长期时序建模能力。

### 2.3 EmotionLSTM — 纯 LSTM 版本

跳过 CNN 特征提取，直接将特征序列送入 2 层 BiLSTM。表达能力弱于混合模型，因为缺乏频域局部特征学习。

---

## 3. 第二条路线：预训练语音模型微调

核心思路：利用 Wav2Vec2 / HuBERT / WavLM 等在大规模无标注语音上预训练的 Transformer 模型做 backbone，仅微调分类头或部分层，效果通常优于手工特征路线。

### 3.1 Wav2Vec2EmotionModel — 基准微调模型

\\\
波形 (16kHz raw audio)
  ▼
Wav2Vec2Model (12 层 Transformer, 768 维)
  │  • 7 层 CNN 特征提取器 (feature encoder)
  │  • 12 层 Transformer encoder
  │  • 输出: last_hidden_state [B, T', 768]
  ▼
Mean Pooling (时间维求平均) → [B, 768]
  ▼
MLP: Linear(768→512) → LayerNorm → GELU → Dropout(0.5)
     Linear(512→256) → LayerNorm → GELU → Dropout(0.5)
     Linear(256→6)
\\\

**训练策略**：
- 提供 \reeze_for_finetuning()\ 方法，可冻结 feature extractor 和指定数量的 encoder 层。
- 通过逐步解冻实现渐进式微调，避免灾难性遗忘。

### 3.2 Wav2Vec2AttentionModel — Attention Pooling 改进

将 Mean Pooling 替换为**可学习的 AttentionPooling**：

\\\
class AttentionPooling:
  input: [B, T, 768]
    1. Linear(T, 384) → Tanh → Linear(384, 1) → scores [B, T, 1]
    2. Softmax 归一化得到注意力权重
    3. 加权求和 → [B, 768]
\\\

相比 Mean Pooling，Attention Pooling 允许模型自主学习哪些时间帧对情感判别更重要。

### 3.3 MultiBackboneModel — 统一多框架模型

通过 HuggingFace \AutoModel\ 抽象，统一支持以下 backbone：

| 模型名 | HuggingFace 标识 | 隐藏维度 |
|---|---|---|
| HuBERT-Base | \acebook/hubert-base-ls960\ | 768 |
| HuBERT-Large | \acebook/hubert-large-ll60k\ | 1024 |
| WavLM-Base | \microsoft/wavlm-base-plus\ | 768 |
| WavLM-Large | \microsoft/wavlm-large\ | 1024 |
| Wav2Vec2-XLSR | \acebook/wav2vec2-xls-r-300m\ | 1024 |
| Wav2Vec2-Large | \acebook/wav2vec2-large\ | 1024 |
| Wav2Vec2-Base | \acebook/wav2vec2-base\ | 768 |

所有模型使用相同的 **Attention Pooling + 2 层 MLP** 分类头，训练策略统一，方便横向对比和集成。

---

## 4. 进阶分类头设计

### 4.1 MultiBackboneMoEModel — Mixture of Experts 分类头

\\\
Backbone → [B, T, H]
  ▼
MoE 分类头:
  ├─ 6 个 Specialized Experts
  │   每个 Expert = 3×Conv1D → AdaptiveAvgPool1d → Linear(→6)
  │   每个专家擅长一种情感的模式识别
  ├─ 2 个 Shared Experts
  │   结构同上，判断情感正负强度等通用属性
  └─ Router 网络
       mean-pooled [B,H] → Linear(H→H/2) → GELU → Dropout → Linear(H/2→8)
       → softmax(× temperature) → 8 维权重
  ▼
加权融合: Σ(route_weight_i × expert_output_i) → [B,6]
  ▼
Fusion MLP: Linear(6→12) → GELU → Dropout → Linear(12→6)
\\\

**设计思路**：
- 不同的情感可能由不同的声学模式主导，MoE 允许每个"专家"专门学习一类模式。
- Shared Expert 捕捉通用特征（如能量高低、语速快慢），减少参数冗余。
- Router 动态决定每段语音应该信任哪些专家，temperature 参数控制权重的"软硬"程度。

### 4.2 ConvExpert — 专家网络结构

\\\
输入 [B, T, H]
  → transpose → [B, H, T]
  → Conv1D(H→256, k=3, p=1) → BN → GELU
  → Conv1D(256→128, k=3, p=1) → BN → GELU
  → Conv1D(128→64, k=3, p=1) → BN → GELU
  → AdaptiveAvgPool1d(1) → [B, 64]
  → Linear(64→num_classes) → [B, 6]
\\\

Conv1D 在时间维上滑动，提取局部时序模式，相比全连接层能更好地捕捉帧间局部变化。

---

## 5. 时序推理模块 (CTM)

### 5.1 CTMWavLMModel — 基于 Continuous Token Memory 的时序推理

\\\
Backbone (WavLM) → [B, T, D]
  ▼
EmotionCTMBlock:
  1. 维度压缩: Linear(D→256) → [B, T, 256]
  2. 初始化情感状态:
     emotion_query = nn.Parameter([1, 256])  (可学习的情感先验)
     state = emotion_query.expand(B, 1, 256)
  3. K 步内部迭代 (K=4):
     a) Cross-Attention(state, 帧特征, 帧特征)
        → MultiheadAttention(4 heads, d=256)
        → Residual + LayerNorm
     b) FFN Refine:
        Linear(256→1024) → GELU → Dropout → Linear(1024→256) → Residual + LayerNorm
     c) 同步图: cosine_similarity(state, 所有帧) → [B, T]
  4. 最终同步图 → softmax → 注意力加权池化 → [B, 256]
  5. 投影回原维度: Linear(256→D)
  ▼
MLP 分类头 (3 层):
  Linear(D→256) → LN → GELU → Dropout
  Linear(256→128) → LN → GELU → Dropout
  Linear(128→6)
\\\

**核心思想**：

> 情感不是某一帧决定的，而是由语速、停顿、能量、音高、韵律等随时间演化形成的。

CTM Block 通过 K 步**内部迭代**逐步从帧级特征中"读出"情感状态。每一轮迭代，状态通过 Cross-Attention 与所有帧交互、通过 FFN 精炼，逐步收敛到 utterance 级别的情感表征。这个过程有点像"慢慢品味一段语音"：第一遍听个大概，后面几遍注意到更多细节。

---

## 6. 池化策略对比

| 池化方法 | 说明 | 可学习参数 | 特点 |
|---|---|---|---|
| Mean Pooling | 时间维求平均 | 无 | 简单基准，丢失帧级重要性信息 |
| Attention Pooling | 学习帧级注意力权重后加权求和 | O(H²) | 自适应选择重要帧 |
| CTM 同步池化 | 多步迭代后 cosine 相似度作为注意力 | O(K·H²) | 建模帧与情感状态的动态对齐 |
| STD + Attention | 注意力池化与标准差池化拼接后融合 | O(H²) | 同时利用"重要性"和"动态性" |

---

## 7. 辅助损失：Center Loss

\\\python
class CenterLoss:
    centers: nn.Parameter([num_classes, feat_dim])  # 6 个可学习的类中心

    forward(features, labels):
        centers_batch = centers[labels]       # 取对应类中心
        loss = ||features - centers_batch||²   # 最小化类内距离
        return loss.mean() / 2
\\\

与交叉熵联合使用，让同类样本在特征空间中更紧凑，同时交叉熵保持类间可分。与 SupCon (监督对比学习) 互补：SupCon 拉大类间距离，Center Loss 缩小类内距离。

---

## 8. 训练配置

| 配置项 | Hybrid 路线 | Wav2Vec2 路线 | WavLM/Large 路线 |
|---|---|---|---|
| 批量大小 | 32 | 8 | 8 |
| 学习率 | 1e-3 | 2e-5 | 1e-5 |
| 训练轮数 | 50 | 20~25 | 30 |
| 早停 patience | — | 8 | 8 |
| 标签平滑 | — | — | 0.05 |
| 权重衰减 | — | — | 0.01 |
| Dropout | 0.5 | 0.5 | 0.5 |
| 冻结策略 | 无 | 冻结 feature encoder | 冻结前 10 层 encoder |

---

## 9. 模型演进脉络总结

\\\
手工特征 + 轻量网络
  ├── EmotionCNN (纯 CNN, ResNet 风格)
  ├── EmotionLSTM (纯 RNN)
  └── HybridCNN-LSTM (CNN + BiLSTM 混合) ← 手工特征路线的主力
        │
        ▼
预训练模型微调
  ├── Wav2Vec2-base + Mean Pooling ← 基线
  ├── Wav2Vec2-base + Attention Pooling ← 改进池化
  ├── MultiBackbone (HuBERT / WavLM / XLSR 统一框架)
  │     ├── + Attention Pooling
  │     ├── + MoE 分类头 (6 专属 + 2 共享专家)
  │     └── + CTM 时序推理 (K=4 步迭代)
  └── + Center Loss (增强类内紧凑性)
        │
        ▼       （可选）
    模型集成 / TTA (Test Time Augmentation)
\\\

整体演进方向：
1. **特征层面**：手工特征 → 预训练自监督特征
2. **池化策略**：简单平均 → 可学习注意力 → 同步图迭代
3. **分类头**：单路径 MLP → MoE 多专家融合
4. **时序建模**：BiLSTM → Cross-Attention 迭代推理
5. **训练范式**：全量训练 → 渐进式微调 → 联合损失优化
