import os
import torch
import numpy as np

if not os.environ.get('HF_ENDPOINT'):
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SAMPLE_RATE = 16000
DURATION = 3.0
N_MFCC = 40
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 1024

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
NUM_CLASSES = len(EMOTIONS)

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

WAV2VEC2_MODEL_NAME = 'facebook/wav2vec2-base'
WAV2VEC2_BATCH_SIZE = 8
WAV2VEC2_LR = 2e-5
WAV2VEC2_EPOCHS = 20

# wav2vec2-large 平衡微调 v3（介于 base 全量微调与 v2 强正则之间）
WAV2VEC2_LARGE_MODEL_NAME = 'facebook/wav2vec2-large'
WAV2VEC2_LARGE_BATCH_SIZE = 8
WAV2VEC2_LARGE_LR = 1e-5
WAV2VEC2_LARGE_EPOCHS = 30
WAV2VEC2_LARGE_FROZEN_LAYERS = 10
WAV2VEC2_LARGE_WEIGHT_DECAY = 0.01
WAV2VEC2_LARGE_LABEL_SMOOTH = 0.05
WAV2VEC2_LARGE_DROPOUT = 0.5
WAV2VEC2_LARGE_EARLY_STOP = 8
WAV2VEC2_LARGE_SAVE_PATH = './models/best_model_wav2vec2_large_v3_best.pth'
WAV2VEC2_LARGE_HISTORY_PATH = './logs/wav2vec2_large_v3_history.json'
WAV2VEC2_LARGE_LIVE_LOG = './logs/train_metrics_live_v3.txt'
WAV2VEC2_AUGMENT_LEVEL = 'light'  # light | full

# wav2vec2-base 改进版：Attention pooling + 渐进解冻 + Focal Loss
WAV2VEC2_IMPROVED_SAVE_PATH = './models/best_model_wav2vec2_improved_best.pth'
WAV2VEC2_IMPROVED_HISTORY_PATH = './logs/wav2vec2_improved_history.json'
WAV2VEC2_IMPROVED_LIVE_LOG = './logs/train_metrics_improved.txt'
WAV2VEC2_IMPROVED_EPOCHS = 25
WAV2VEC2_IMPROVED_LR = 2e-5
WAV2VEC2_IMPROVED_EARLY_STOP = 8
WAV2VEC2_IMPROVED_FOCAL_GAMMA = 2.0

# 从 74% base 热启动，仅微调 Attention pooling
WAV2VEC2_ATTN_BOOST_CKPT = './models/best_model_wav2vec2_best.pth'
WAV2VEC2_ATTN_BOOST_SAVE_PATH = './models/best_model_wav2vec2_attn_boost_best.pth'
WAV2VEC2_ATTN_BOOST_HISTORY_PATH = './logs/wav2vec2_attn_boost_history.json'
WAV2VEC2_ATTN_BOOST_LIVE_LOG = './logs/train_metrics_attn_boost.txt'

DATA_DIR = './data'
MODEL_SAVE_PATH = './models/best_model.pth'
LOG_DIR = './logs'

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)