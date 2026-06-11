# 语音情感识别 (SER) 项目全貌总结

> 生成日期: 2026-06-11
> 项目路径: G:\真语音信息处理课设\语音信息处理课设

---

## 1. 项目定位

本项目是一个完整的**语音情感识别系统**，从声学特征提取、模型训练到 Web 应用部署全覆盖。核心任务是从 3 秒语音片段中识别出 6 类情感：ngry, disgust, fear, happy, neutral, sad。

---

## 2. 原理：语音情感识别为什么可行？

### 情感在语音中的声学表现

不同情感状态下，人的发声生理机制会发生变化，反映在声学信号中：

| 声学属性 | happy | sad | angry | fear | neutral |
|---|---|---|---|---|---|
| **语速** | 加快 | 减慢 | 加快 | 加快 | 正常 |
| **基频均值** | 升高 | 降低 | 升高 | 升高 | 正常 |
| **基频范围** | 宽 | 窄 | 宽 | 宽 | 中等 |
| **能量** | 高 | 低 | 高 | 中低 | 中等 |
| **高频能量** | 多 | 少 | 多 | 多 | 中等 |
| **韵律变化** | 丰富 | 单调 | 剧烈 | 颤抖 | 平稳 |

本项目的特征提取层（MFCC、Mel、Chroma 等）精确捕捉以上声学属性，模型则学习从这些属性到情感的映射。

### 为什么用深度学习？

传统方法（SVM、GMM 等）需要手工设计统计特征（均值、方差等），丢失了时序动态信息。深度学习（CNN + LSTM / Transformer）可以直接从帧级特征序列中学习情感随时间演化的模式——这也是为什么系统的演进方向始终在强化**时序建模能力**。

---

## 3. 全部工作清单

### 3.1 数据层

| 文件 | 职责 |
|---|---|
| config.py | 全局配置：采样率 16kHz、帧长 3s、6 类情感、训练超参数 |
| data_utils.py | 目录初始化、RAVDESS/CREMA-D/SAVEE 数据准备脚本、示例数据生成 |
| data_processor.py | AudioProcessor（加载/归一化/增强）、EmotionDataset（手工特征 Dataset）、Wav2Vec2EmotionDataset（原始音频 Dataset）、get_dataloaders 工厂函数 |
| eature_extractor.py | FeatureExtractor：MFCC（含一阶二阶 Delta）、Mel 频谱、Chroma、Spectral Contrast、Tonnetz、ZCR、RMS |

**数据增强策略**：
- 加性高斯噪声（
oise_factor=0.005）
- 时移（shift_max=0.2）
- 音高偏移（
_steps=2）
- 时间拉伸

**训练/验证/测试划分**：按说话人划分（speaker-independent），70%/15%/15%。

---

### 3.2 模型层（核心）

| 文件 | 包含的模型类 | 说明 |
|---|---|---|
| models.py | SpeechEmotionModel | CNN + BiLSTM 混合模型（主力手工特征模型） |
| models.py | EmotionCNN | 纯 CNN 版本 |
| models.py | EmotionLSTM | 纯 BiLSTM 版本 |
| models.py | Wav2Vec2EmotionModel | Wav2Vec2-base + Mean Pooling（预训练基线） |
| models.py | Wav2Vec2AttentionModel | Wav2Vec2-base + Attention Pooling |
| models.py | MultiBackboneModel | 统一框架：HuBERT/WavLM/XLSR + Attention Pooling |
| models.py | MultiBackboneMoEModel | Backbone + MoE 分类头（6专属+2共享专家） |
| models.py | CTMWavLMModel | Backbone + CTM 时序推理块 |
| models.py | ConvExpert | Conv1D 专家网络（MoE 组件） |
| models.py | EmotionCTMBlock | K 步内部迭代推理块 |
| models.py | AttentionPooling | 可学习注意力池化 |
| models.py | STDAttentionPooling | 注意力 + 标准差联合池化 |
| models.py | CenterLoss | 类中心损失，增强类内紧凑性 |

**Pooling 演进原理**：
- Mean Pooling：所有帧等权，简单粗暴
- Attention Pooling：让模型自己学"哪些帧对情感更重要"
- CTM 同步池化：多步迭代后，用状态-帧的余弦相似度作为注意力权重，对齐情感状态与时间帧
- STD + Attention：不仅看"哪些帧重要"，还看"帧间变化多大"（情感动态）

---

### 3.3 训练层

| 文件 | 训练内容 | 损失函数 | 优化策略 |
|---|---|---|---|
| 	rain.py | 手工特征模型 + Wav2Vec2-base | CrossEntropy + label_smoothing=0.05 | AdamW + ReduceLROnPlateau + 分层LR |
| 	rain_improved.py | Wav2Vec2-base + Attention + 渐进解冻 | FocalLoss(gamma=2.0) + 类别加权 | 3 阶段解冻：头→顶4层→全encoder |
| 	rain_attn_boost.py | 从 base 74% 热启动 + Attention | 类别加权 CE | 冻结 encoder 只训 attention→解冻顶4层 |
| 	rain_large.py | Wav2Vec2-large v3 | label_smoothing=0.05 | 冻结前10层 + dropout=0.5 |
| 	rain_multimodel.py | HuBERT/WavLM/XLSR/MoE/CTM | FocalLoss + SupCon + CenterLoss | 渐进解冻 + Mixup |
| 	rain_boost.py | base 模型增强训练 | — | — |
| 	rain_continue.py | 断点续训 | — | — |

**渐进解冻原理**（	rain_improved.py）：
`
Epoch 1-3:    仅训练 attention + 分类头（encoder 全冻结）
Epoch 4-7:    解冻 encoder 顶层 4 层
Epoch 8+:     解冻全部 encoder（feature_extractor 保持冻结）
`
模仿人类学习：先学"怎么判断"（分类头），再学"听什么"（微调 encoder）。

**MoE 原理**：6 个专属专家各负责一类情感 + 2 个共享专家判断正负强度，Router 网络动态决定每段语音信任哪些专家。

**CTM 原理**：可学习情感 query 经过 K 步 Cross-Attention 迭代，逐步从帧特征中"读出"情感状态，最后用余弦相似度做同步注意力池化。

---

### 3.4 评估层

| 文件 | 功能 |
|---|---|
| eval_all_models.py | 综合评估：全部 8 个变体在 test 集上的 Accuracy + Per-class F1 + 混淆矩阵 + 训练曲线 |
| eval_final.py | Wav2Vec2-base 快速评估 |
| eval_ensemble.py | 多模型概率融合 + 网格权重搜索 + TTA |
| eval_tta.py | Test Time Augmentation |
| isualize.py | 训练曲线、混淆矩阵、波形图、MFCC/Mel 图、情感分布、特征对比、完整报告生成 |

---

### 3.5 应用层

| 文件 | 功能 |
|---|---|
| demo.py | 命令行 Demo：麦克风录音 → Whisper 语音识别 → Wav2Vec2 情感识别 → pyttsx3 语音回复 |
| pp.py | Flask Web 应用：录音 → ASR(Whisper) + 情感识别(Wav2Vec2) + 说话人识别(MFCC+GMM) + 个性化回复 |
| speaker_id.py | 说话人识别引擎：手写 MFCC 提取 + GMM 声纹建模 |
| 	emplates/index.html | Web 前端：深色风格 UI + Chart.js 概率雷达图 + 说话人管理 |
| gen_ppt.py | 课程设计 ppt 生成脚本 |
| check_labels.py | 数据标签检查脚本 |
| debug_model.py / debug_pipeline.py / 	est_pipeline.py | 调试和测试脚本 |

---

### 3.6 训练成果（模型检查点）

| 检查点 | 参数规模 | 验证集准确率 | 说明 |
|---|---|---|---|
| est_model_best.pth | ~67MB | — | 手工特征混合模型 |
| est_model_wav2vec2_best.pth | ~1.1GB | ~74% | Wav2Vec2-base 基线 |
| est_model_wav2vec2_improved_best.pth | ~1.1GB | — | + Attention + Focal Loss |
| est_model_wav2vec2_large_v3_best.pth | ~2.8GB | — | Wav2Vec2-large + 渐进解冻 |
| est_model_wavlm-base_best.pth | ~428MB | ~25% | WavLM 基线（Attention 池化，早停） |
| **est_model_wavlm-base_meanpool_best.pth** | ~1.1GB | **~73.0%** | WavLM + Mean Pooling |
| est_model_wavlm-base_ctm_k1_best.pth | ~1.1GB | ~72.7% | WavLM + CTM K=1 |
| est_model_wavlm-base_ctm_k2_sup0.1_best.pth | ~1.1GB | ~73.6% | WavLM + CTM K=2 + SupCon |
| **est_model_wavlm-base_ctm_k4_best.pth** | ~1.1GB | **~74.6%** | WavLM + CTM K=4 → **验证集最高** |
| est_model_wavlm-base_ctm_k4_sup0.1_cen0.1_best.pth | ~1.1GB | ~74.3% | CTM K=4 + SupCon + CenterLoss |
| est_model_wavlm-base_moe_best.pth | ~1.2GB | ~72.6% | WavLM + MoE |
| est_model_hubert-base_best.pth | ~1.1GB | ~71.4% | HuBERT 基线 |

---

### 3.7 测试集最终结果（test accuracy）

| 模型变体 | Test Acc | 最佳类别(F1) | 最弱类别(F1) |
|---|---|---|---|
| WavLM Baseline (Attention) | 24.59% | angry (37.3%) | neutral (5.5%) — 仅训练2轮早停 |
| **WavLM + Mean Pooling** | **69.83%** | neutral (79.2%) | sad (52.0%) |
| WavLM + CTM K=1 | 68.09% | neutral (75.7%) | sad (46.9%) |
| WavLM + CTM K=2 | 68.70% | happy (75.1%) | sad (55.9%) |
| **WavLM + CTM K=4** | **72.89%** | neutral (79.6%) | **sad (63.1%)** |
| WavLM + CTM K=4 + Center | 71.14% | neutral (78.6%) | sad (63.2%) |
| WavLM + MoE | 70.62% | angry (74.1%) | sad (54.4%) |
| HuBERT Baseline | 70.01% | neutral (79.8%) | sad (56.6%) |

**关键发现**：
- CTM K=4 在所有模型中取得最佳测试准确率（72.89%），且对最难类别 sad 的 F1 提升最显著（63.1%，比 Mean Pooling 的 52.0% 高出 11 个百分点）
- sad 和 ear 是普遍最难区分的类别，这与情感语音领域的先验一致（悲伤与恐惧在声学空间中有重叠）
- Mean Pooling 简单但有效，远超 Attention Pooling（早停导致的不公平对比）
- MoE 参数量最大（+5M 专家参数）但性能并非最高，说明专家路由的学习可能不够充分

---

## 4. 系统架构图

`
┌─────────────────────────────────────────────────────────────────────┐
│                     语音情感识别系统架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐    │
│  │ 原始音频  │───▶│ AudioProcessor │───▶│   FeatureExtractor      │    │
│  │ (16kHz)  │    │ 加载/归一化/  │    │  MFCC/Mel/Chroma/...   │    │
│  └──────────┘    │ 数据增强      │    └─────────┬───────────────┘    │
│                  └──────────────┘              │                    │
│                                                ▼                    │
│                           ┌─────────────────────────────┐          │
│                           │       模型层 (可切换)         │          │
│                           │                             │          │
│              路线A        │     路线B（预训练微调）       │          │
│         ┌─────────────┐  │  ┌──────────────────────┐   │          │
│         │手工特征网络   │  │  │Backbone: Wav2Vec2/   │   │          │
│         │CNN+LSTM/     │  │  │HuBERT/WavLM/XLSR     │   │          │
│         │CNN only/     │  │  └──────────┬───────────┘   │          │
│         │LSTM only     │  │             ▼               │          │
│         └──────┬───────┘  │  ┌──────────────────────┐   │          │
│                │          │  │Pooling: Mean/Attention│   │          │
│                │          │  │CTM/MoE/STD+Attn      │   │          │
│                │          │  └──────────┬───────────┘   │          │
│                │          │             ▼               │          │
│                ▼          │  ┌──────────────────────┐   │          │
│         ┌─────────────┐  │  │MLP 分类头 (2-3层)     │   │          │
│         │6类情感概率   │  │  └──────────────────────┘   │          │
│         └─────────────┘  └─────────────────────────────┘          │
│                                                │                    │
│                                                ▼                    │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                    应用层                                 │      │
│  │  ┌──────────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  │      │
│  │  │命令行Demo│  │Flask Web│  │说话人识别│  │模型集成  │  │      │
│  │  │(demo.py) │  │(app.py) │  │(speaker  │  │(ensemble│  │      │
│  │  │ Whisper  │  │ Whisp+  │  │ _id.py)  │  │ .py)    │  │      │
│  │  │ +TTS     │  │ +前端UI │  │ MFCC+GMM │  │ 加权融合│  │      │
│  │  └──────────┘  └─────────┘  └──────────┘  └─────────┘  │      │
│  └──────────────────────────────────────────────────────────┘      │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────┐      │
│  │                   可视化评估                              │      │
│  │  训练曲线 │ 混淆矩阵 │ MFCC/频谱图 │ 情感分布 │ Per-class F1  │      │
│  └──────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
`

---

## 5. 关键技术原理

### 5.1 Wav2Vec2 预训练原理

Wav2Vec2 是 Facebook AI 提出的自监督语音预训练模型：
1. **特征编码器**：7 层 CNN，将原始波形映射到帧级隐状态（24.9Hz 帧率，约每 40ms 一帧）
2. **量化器**：将隐状态离散化，作为自监督目标
3. **Transformer**：12 层（base）/ 24 层（large），对帧序列建模上下文
4. **预训练目标**：对比学习——区分被掩码帧的正确量化码本与负样本

本项目将预训练好的 Wav2Vec2 backbone 取出，接上情感分类头进行微调。backbone 已掌握丰富的语音语言学知识（音素、韵律、声调），微调将其中与情感相关的部分激活。

### 5.2 渐进解冻原理

`
Epoch 1-3:  梯度只更新分类头 ← 学习"如何判断"
Epoch 4-7:  梯度更新分类头 + 顶层4层 encoder ← 学习"有价值的上层特征"
Epoch 8+:   梯度更新除 feature_extractor 外的所有参数 ← 精细调整全部表示
`

预训练模型的特征提取器（feature_extractor）在情感任务中通常保持冻结，因为它是学习基础声学特征的，与情感无关的下游任务可以复用。

### 5.3 Focal Loss 原理

标准交叉熵对所有样本一视同仁。Focal Loss 引入 (1-pt)^γ 调制因子：
- 易分类样本（pt 接近 1）: 贡献被大幅压低
- 难分类样本（pt 小）: 贡献相对增大

γ=2 时，一个 pt=0.9 的样本权重仅为 0.01，而 pt=0.3 的样本权重为 0.49。这使得模型更关注 sad、ear 等难区分的类别。

### 5.4 SupCon + Center Loss 联合原理

- **SupCon Loss**（监督对比学习）：在特征空间中拉近同类样本、推远异类样本，增强类间可分性
- **Center Loss**：学习每类的中心向量，让同类样本向中心聚拢，增强类内紧凑性

两者互补：SupCon 管"边界"，Center Loss 管"内部"。

### 5.5 CTM 时序推理

CTM 的核心洞察：

> 情感不是某一帧决定的，而是随时间累积的——一段语音的开头可能还在平静，中间逐步激动，结尾归于平静。

CTM 通过 K 步 Cross-Attention 迭代，让一个可学习的 query 向量逐步"阅读"所有帧的信息，最终收敛到 utterance 级别的表征。K 越大，模型"读得越细"。实验证明 K=4 是最优的——太少读不全，太多会过拟合。

### 5.6 MoE 分类头

MoE 的核心洞察：
- ngry 和 happy 都高能量，但韵律模式不同 → 需要不同的专家
- 
eutral 能量适中，但特征稳定 → Shared Expert 即可

6 个专用专家各负责一种情感，2 个共享专家负责通用特征（能量高低、语速快慢），Router 网络学习如何调度。

---

## 6. 项目文件一览

| 类别 | 文件 | 行数估算 | 重要性 |
|---|---|---|---|
| 配置 | config.py | ~50 | ★★★★★ |
| 数据 | data_processor.py | ~350 | ★★★★★ |
| 数据 | eature_extractor.py | ~180 | ★★★★ |
| 数据 | data_utils.py | ~250 | ★★★ |
| 模型 | models.py | ~530 | ★★★★★ |
| 训练 | 	rain.py | ~280 | ★★★★★ |
| 训练 | 	rain_improved.py | ~240 | ★★★★★ |
| 训练 | 	rain_multimodel.py | ~400 | ★★★★★ |
| 训练 | 	rain_large.py | ~200 | ★★★★ |
| 训练 | 	rain_attn_boost.py | ~200 | ★★★★ |
| 训练 | 	rain_boost.py | — | ★★ |
| 训练 | 	rain_continue.py | — | ★★ |
| 评估 | eval_all_models.py | ~250 | ★★★★★ |
| 评估 | eval_ensemble.py | ~250 | ★★★★ |
| 评估 | eval_final.py | ~30 | ★★★ |
| 评估 | eval_tta.py | — | ★★★ |
| 评估 | isualize.py | ~300 | ★★★★ |
| 应用 | pp.py | ~220 | ★★★★★ |
| 应用 | demo.py | ~150 | ★★★★ |
| 应用 | speaker_id.py | ~240 | ★★★★ |
| 前端 | 	emplates/index.html | ~400 | ★★★★ |
| 辅助 | main.py | ~200 | ★★★★ |
| 辅助 | gen_ppt.py | — | ★★ |
| 辅助 | check_labels.py | — | ★★ |
| 辅助 | debug_*.py / 	est_*.py | — | ★★ |

**总计：约 4500+ 行代码，22 个 Python 脚本，1 个 HTML 前端，15 个训练好的模型检查点。**

---

## 7. 演进脉络总结

`
初始阶段（手工特征网络）
  EmotionCNN (~2M params)         ← 纯 CNN 基线
  EmotionLSTM                     ← 纯 RNN 基线
  SpeechEmotionModel (CNN+LSTM)   ← 手工特征路线主力
  Train with CE + label_smooth    ← 基础训练

↓ 引入预训练模型

Wav2Vec2-base + Mean Pooling     ← 预训练基线 (~74%)
Wav2Vec2-base + Attention Pooling ← 改进池化策略
Wav2Vec2-base + Focal Loss        ← 处理类别不平衡
Wav2Vec2-base + 渐进解冻           ← 防止灾难性遗忘
Wav2Vec2-large v3                 ← 更大 backbone

↓ 多框架统一

MultiBackbone (HuBERT / WavLM / XLSR)
  ├── + Attention Pooling
  ├── + CTM (K=1→2→4 逐步提升)
  ├── + SupCon + Center Loss
  └── + MoE 分类头
         └── 6 Specialist + 2 Shared + Router

↓ 集成与部署

模型集成 (Ensemble + 网格权重搜索)
Test Time Augmentation
Web 应用 (Flask + Whisper + Speaker ID)
`

整个项目的演进清晰地遵循着一条线索：**如何让模型更好地理解"情感在语音中是如何随时间变化的"**——从 CNN 的局部感受野到 LSTM 的序列建模，再到 Transformer 的自注意力、CTM 的多步迭代推理，每一步都在深化对时变情感信号的建模能力。
