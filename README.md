# 语音情感识别系统 (Speech Emotion Recognition)

基于深度学习的语音情感识别系统，支持多种模型架构（CNN+LSTM、Wav2Vec2、HuBERT、WavLM），融合 CTM 时序推理、MoE 专家混合、对比学习等前沿技术，并提供 Flask Web 应用实现录音→识别→回复的完整交互。

## 情感类别

`angry` `disgust` `fear` `happy` `neutral` `sad` — 共 6 类

## 项目结构

```
├── config.py              # 全局配置（采样率、超参数、模型路径）
├── data_processor.py      # 音频加载/归一化/增强、Dataset 定义、DataLoader 工厂
├── feature_extractor.py   # 手工特征提取（MFCC、Mel、Chroma、Spectral Contrast 等）
├── data_utils.py          # 数据目录初始化与数据集准备
├── models.py              # 全部模型定义（核心）
├── train.py               # 手工特征模型 + Wav2Vec2 基线训练
├── train_improved.py      # Wav2Vec2 + Attention Pooling + 渐进解冻 + Focal Loss
├── train_attn_boost.py    # 从基线热启动微调 Attention Pooling
├── train_large.py         # Wav2Vec2-large 微调
├── train_multimodel.py    # 多框架统一训练（HuBERT/WavLM/XLSR/MoE/CTM）
├── train_boost.py         # 增强训练
├── train_continue.py      # 断点续训
├── eval_all_models.py     # 全模型综合评估
├── eval_ensemble.py       # 多模型概率融合 + 网格权重搜索
├── eval_final.py          # 快速评估
├── eval_tta.py            # Test Time Augmentation
├── visualize.py           # 训练曲线、混淆矩阵、特征可视化
├── app.py                 # Flask Web 应用
├── demo.py                # 命令行 Demo（录音→Whisper ASR→情感识别→TTS 回复）
├── speaker_id.py          # 说话人识别（MFCC + GMM）
├── gen_ppt.py             # 课程汇报 PPT 生成
├── templates/index.html   # Web 前端 UI
├── models/                # 训练好的模型权重
└── logs/                  # 训练日志与指标
```

## 模型架构

### 路线 A：手工特征网络

`CNN (ResNet-style) + BiLSTM → 分类头`

- 输入：MFCC + Mel + Chroma 等多通道声学特征
- 可选：纯 CNN (`EmotionCNN`)、纯 BiLSTM (`EmotionLSTM`)、CNN+LSTM 混合 (`SpeechEmotionModel`)

### 路线 B：预训练模型微调

`Backbone → Pooling → 分类头`

| Backbone | Pooling 策略 | 说明 |
|---|---|---|
| Wav2Vec2-base | Mean / Attention | Facebook 预训练，~74% val acc |
| Wav2Vec2-large | Mean / Attention | 更大 backbone，10 层冻结 |
| HuBERT-base | Attention | 验证集 ~71.4% |
| WavLM-base | Mean / CTM / MoE | **最优 backbone** |

### Pooling 演进

| 方法 | 原理 | Test Acc |
|---|---|---|
| Mean Pooling | 所有帧等权平均 | 69.83% |
| Attention Pooling | 可学习加权 | — |
| **CTM K=4** | K 步迭代 Cross-Attention 时序推理 | **72.89%** |
| MoE | 6 专属专家 + 2 共享专家 + Router | 70.62% |

### CTM 时序推理（核心创新）

CTM (Cognition-like Temporal Module) 受思维链启发，通过 K 步迭代从帧序列中逐步"读出"情感状态：

```
H ∈ R^{T×D}  (Backbone 帧特征)
  ↓
state₀ = 可学习情感 Query
  ↓
For k = 1..K:
    state_k = state_{k-1} + CrossAttn(state_{k-1}, H)  # 阅读帧
    state_k = state_k + FFN(state_k)                     # 精炼
  ↓
sync_K = cosine_sim(state_K, H)  → softmax → 加权池化 → [B, D]
```

- K=1 → 68.09% | K=2 → 68.70% | **K=4 → 72.89%**
- CTM head 仅增 0.6M 参数 (0.6%)，参数效率极高
- 对最难类别 **sad 的 F1 提升最显著**（52.0% → 63.1%，+11pp）

### MoE 分类头

6 个专属专家各负责一种情感 + 2 个共享专家判断正负强度，Router 动态决定信任哪些专家。

## 训练策略

| 策略 | 说明 |
|---|---|
| **渐进解冻** | 阶段 1：训分类头 → 阶段 2：解冻顶层 4 层 → 阶段 3：解冻全部 encoder |
| **Focal Loss** (γ=2) | 压低易分类样本权重，让模型聚焦 sad、fear 等难类 |
| **SupCon Loss** | 监督对比学习，拉近同类、推远异类 |
| **Center Loss** | 学习类中心，增强类内紧凑性 |
| **Mixup** | 数据混合增强 |
| **Label Smoothing** | 标签平滑 α=0.05 |

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练

```bash
# 手工特征模型
python train.py

# Wav2Vec2 基线
python train.py --wav2vec2

# Wav2Vec2 + Attention + 渐进解冻
python train_improved.py

# Wav2Vec2-large
python train_large.py

# 多框架（WavLM + CTM K=4 + SupCon）
python train_multimodel.py --name wavlm-base --model wavlm-base --ctm --ctm-steps 4 --supcon 0.1
```

### 评估

```bash
# 全模型综合评估
python eval_all_models.py

# 模型集成
python eval_ensemble.py
```

### Web 应用

```bash
python app.py
```

浏览器打开 `http://localhost:5000`，支持：
- 录音 → Whisper 语音识别 → 情感识别 → 个性化回复
- 说话人识别与管理
- 情感雷达图可视化

### 命令行 Demo

```bash
python demo.py        # 录音并识别
python main.py        # 完整 pipeline（含特征可视化）
```

## 测试结果

| 模型 | Val Acc | Test Acc |
|---|---|---|
| WavLM + Mean Pooling | 73.0% | 69.83% |
| WavLM + CTM K=1 | 72.7% | 68.09% |
| WavLM + CTM K=2 | 73.6% | 68.70% |
| **WavLM + CTM K=4** | **74.6%** | **72.89%** |
| WavLM + CTM K=4 + Center | 74.3% | 71.14% |
| WavLM + MoE | 72.6% | 70.62% |
| HuBERT Baseline | 71.4% | 70.01% |
| 5 模型 Ensemble (含 CTM) | 75.5% | **75.85%** |

## 技术栈

- **深度学习**: PyTorch, Transformers (HuggingFace)
- **音频处理**: torchaudio, librosa, soundfile
- **ASR**: OpenAI Whisper
- **Web**: Flask + Chart.js
- **说话人识别**: 手写 MFCC + GMM
- **可视化**: matplotlib, seaborn
