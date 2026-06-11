import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import os
import config
import random
import scipy.io.wavfile as wav
from scipy import signal

class AudioProcessor:
    def __init__(self, sample_rate=config.SAMPLE_RATE, duration=config.DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

    def load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        except Exception:
            try:
                sr, data = wav.read(file_path)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                else:
                    data = data.astype(np.float32)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != self.sample_rate:
                    data = signal.resample(data, int(len(data) * self.sample_rate / sr))
                num_samples = int(self.sample_rate * self.duration)
                if len(data) < num_samples:
                    data = np.pad(data, (0, num_samples - len(data)), mode='constant')
                else:
                    data = data[:num_samples]
                audio = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return np.zeros(self.n_samples)

        if len(audio) < self.n_samples:
            padding = self.n_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            audio = audio[:self.n_samples]
        return audio

    def normalize_audio(self, audio):
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio

    def pre_emphasis(self, audio, coeff=0.97):
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio

    def time_shift(self, audio, shift_max=0.2):
        shift = int(random.uniform(-shift_max, shift_max) * len(audio))
        if shift > 0:
            audio = np.roll(audio, shift)
        elif shift < 0:
            audio = np.roll(audio, shift)
        return audio

    def pitch_shift(self, audio, n_steps=2):
        return librosa.effects.pitch_shift(y=audio, sr=self.sample_rate, n_steps=n_steps)

    def time_stretch(self, audio, rate=1.0):
        return librosa.effects.time_stretch(y=audio, rate=rate)


class EmotionDataset(Dataset):
    def __init__(self, data_dir=config.DATA_DIR, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.processor = AudioProcessor()
        self.audio_files = []
        self.labels = []

        self._load_dataset()

    @staticmethod
    def _is_valid_wav(file_path):
        if not os.path.isfile(file_path) or os.path.getsize(file_path) < 44:
            return False
        with open(file_path, 'rb') as f:
            return f.read(4) == b'RIFF'

    def _load_dataset(self):
        skipped = 0
        for idx, emotion in enumerate(config.EMOTIONS):
            emotion_dir = os.path.join(self.data_dir, self.mode, emotion)
            if os.path.exists(emotion_dir):
                for file_name in os.listdir(emotion_dir):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(emotion_dir, file_name)
                        if not self._is_valid_wav(file_path):
                            skipped += 1
                            continue
                        self.audio_files.append(file_path)
                        self.labels.append(idx)
        if skipped:
            print(f"  跳过 {skipped} 个无效 wav ({self.mode})")
        print(f"Loaded {len(self.audio_files)} samples for {self.mode} set")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        audio = self.processor.load_audio(audio_path)
        audio = self.processor.normalize_audio(audio)

        if self.transform and self.mode == 'train':
            if random.random() > 0.5:
                audio = self.processor.add_noise(audio)
            if random.random() > 0.4:
                audio = self.processor.time_shift(audio)
            if random.random() > 0.5:
                audio = self.processor.pitch_shift(audio, n_steps=random.uniform(-3, 3))

        mfcc = self._extract_mfcc(audio)
        mel_spec = self._extract_mel_spectrogram(audio)
        chroma = self._extract_chroma(audio)

        if self.transform and self.mode == 'train':
            mfcc = self._spec_augment(mfcc, freq_mask=10, time_mask=10)
            mel_spec = self._spec_augment(mel_spec, freq_mask=16, time_mask=10)

        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        features = {
            'mfcc': torch.FloatTensor(mfcc),
            'mel_spectrogram': torch.FloatTensor(mel_spec),
            'chroma': torch.FloatTensor(chroma),
            'label': torch.LongTensor([label])
        }
        return features

    def _extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.processor.sample_rate,
            n_mfcc=config.N_MFCC,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.concatenate([mfcc, delta, delta2], axis=0)
        return combined

    def _extract_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.processor.sample_rate,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def _extract_chroma(self, audio):
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.processor.sample_rate,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        return chroma

    def _spec_augment(self, spec, freq_mask=8, time_mask=10):
        if spec.ndim == 2:
            num_freq, num_time = spec.shape
        else:
            _, num_freq, num_time = spec.shape
        freq_masked = spec.copy()

        for _ in range(random.randint(1, 3)):
            f = random.randint(0, max(0, num_freq - freq_mask))
            f_len = random.randint(1, min(freq_mask, num_freq - f))
            freq_masked[f:f+f_len, :] = freq_masked.mean()

        for _ in range(random.randint(1, 3)):
            t = random.randint(0, max(0, num_time - time_mask))
            t_len = random.randint(1, min(time_mask, num_time - t))
            freq_masked[:, t:t+t_len] = freq_masked.mean()

        return freq_masked


def collate_fn(batch):
    mfccs = []
    mel_specs = []
    chromas = []
    labels = []

    for item in batch:
        mfccs.append(item['mfcc'])
        mel_specs.append(item['mel_spectrogram'])
        chromas.append(item['chroma'])
        labels.append(item['label'])

    mfccs = torch.stack(mfccs)
    mel_specs = torch.stack(mel_specs)
    chromas = torch.stack(chromas)
    labels = torch.cat(labels)

    return {
        'mfcc': mfccs.unsqueeze(1),
        'mel_spectrogram': mel_specs.unsqueeze(1),
        'chroma': chromas.unsqueeze(1),
        'label': labels
    }


def get_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    import multiprocessing
    nw = min(4, multiprocessing.cpu_count() // 2)

    train_dataset = EmotionDataset(data_dir=data_dir, mode='train', transform=True)
    val_dataset = EmotionDataset(data_dir=data_dir, mode='val', transform=False)
    test_dataset = EmotionDataset(data_dir=data_dir, mode='test', transform=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=nw,
                             pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=nw,
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=nw,
                            pin_memory=True)

    return train_loader, val_loader, test_loader


class Wav2Vec2EmotionDataset(Dataset):
    def __init__(self, data_dir=config.DATA_DIR, mode='train', augment_level='full'):
        self.data_dir = data_dir
        self.mode = mode
        self.augment_level = augment_level
        self.processor = AudioProcessor()
        self.audio_files = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        skipped = 0
        for idx, emotion in enumerate(config.EMOTIONS):
            emotion_dir = os.path.join(self.data_dir, self.mode, emotion)
            if os.path.exists(emotion_dir):
                for file_name in os.listdir(emotion_dir):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(emotion_dir, file_name)
                        if not EmotionDataset._is_valid_wav(file_path):
                            skipped += 1
                            continue
                        self.audio_files.append(file_path)
                        self.labels.append(idx)
        if skipped:
            print(f"  跳过 {skipped} 个无效 wav ({self.mode})")
        print(f"Loaded {len(self.audio_files)} samples for {self.mode} set")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        audio = self.processor.load_audio(audio_path)
        audio = self.processor.normalize_audio(audio)

        if self.mode == 'train':
            if self.augment_level == 'light':
                if random.random() < 0.3:
                    audio = self.processor.add_noise(audio, noise_factor=random.uniform(0.002, 0.006))
                if random.random() < 0.3:
                    audio = self.processor.time_shift(audio, shift_max=0.15)
                if random.random() < 0.2:
                    audio = self.processor.pitch_shift(audio, n_steps=random.uniform(-1.5, 1.5))
            else:
                if random.random() < 0.6:
                    audio = self.processor.add_noise(audio, noise_factor=random.uniform(0.002, 0.008))
                if random.random() < 0.5:
                    audio = self.processor.time_shift(audio, shift_max=0.2)
                if random.random() < 0.4:
                    audio = self.processor.pitch_shift(audio, n_steps=random.uniform(-2, 2))
                if random.random() < 0.35:
                    audio = self._time_mask(audio, max_mask_ratio=0.15)
                if random.random() < 0.25:
                    rate = random.uniform(0.9, 1.1)
                    audio = self.processor.time_stretch(audio, rate=rate)
                    if len(audio) < self.processor.n_samples:
                        audio = np.pad(audio, (0, self.processor.n_samples - len(audio)))
                    else:
                        audio = audio[:self.processor.n_samples]

        return {
            'audio': torch.FloatTensor(audio),
            'label': torch.LongTensor([label])
        }

    def _time_mask(self, audio, max_mask_ratio=0.15):
        mask_len = int(len(audio) * random.uniform(0.05, max_mask_ratio))
        if mask_len <= 0:
            return audio
        start = random.randint(0, max(0, len(audio) - mask_len))
        audio = audio.copy()
        audio[start:start + mask_len] = 0
        return audio


def wav2vec2_collate_fn(batch):
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    return {'audio': audios, 'label': labels}


# ============================================================
#  Mixup 数据增强
# ============================================================
def mixup_data(audio, labels, alpha=0.2):
    """对 batch 做 mixup 增强，返回混合后的音频和标签。

    Args:
        audio: (B, T) 音频张量
        labels: (B,) 标签张量
        alpha: Beta 分布参数（越小越接近原始）

    Returns:
        mixed_audio: (B, T)
        y_a, y_b: (B,) 原始标签
        lam: 混合系数
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = audio.size(0)
    index = torch.randperm(batch_size, device=audio.device)

    mixed_audio = lam * audio + (1 - lam) * audio[index]
    y_a, y_b = labels, labels[index]
    return mixed_audio, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup 损失: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)，返回标量。"""
    return (lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)).mean()


# ============================================================
#  增强版 Wav2Vec2 Dataset（支持 strong 增强级别）
# ============================================================
class Wav2Vec2StrongDataset(Dataset):
    """更强数据增强的 Wav2Vec2 数据集。

    augment_level:
      - 'none':   无增强（val/test 用）
      - 'light':  轻度增强（高斯噪声 + 时间偏移 + 音调）
      - 'full':   标准增强（加时间遮罩 + 时间拉伸）
      - 'strong': 强增强（加 band-stop 频率遮罩 + 更强噪声 + 多遮罩）
    """

    def __init__(self, data_dir=config.DATA_DIR, mode='train', augment_level='full'):
        self.data_dir = data_dir
        self.mode = mode
        self.augment_level = augment_level
        self.processor = AudioProcessor()
        self.audio_files = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        skipped = 0
        for idx, emotion in enumerate(config.EMOTIONS):
            emotion_dir = os.path.join(self.data_dir, self.mode, emotion)
            if os.path.exists(emotion_dir):
                for file_name in os.listdir(emotion_dir):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(emotion_dir, file_name)
                        if not EmotionDataset._is_valid_wav(file_path):
                            skipped += 1
                            continue
                        self.audio_files.append(file_path)
                        self.labels.append(idx)
        if skipped:
            print(f"  跳过 {skipped} 个无效 wav ({self.mode})")
        print(f"Loaded {len(self.audio_files)} samples for {self.mode} set")

    def __len__(self):
        return len(self.audio_files)

    def _band_stop_mask(self, audio, max_width_hz=800):
        """用带阻滤波器模拟频率遮罩 (SpecAugment freq-mask 在波形域的等价)。"""
        sr = self.processor.sample_rate
        nyq = sr / 2
        center = random.uniform(300, max(301, nyq - 200))
        width = random.uniform(200, max_width_hz)
        low = np.clip((center - width / 2) / nyq, 0.001, 0.998)
        high = np.clip((center + width / 2) / nyq, 0.002, 0.999)
        if low >= high:
            low, high = 0.001, 0.999
        from scipy.signal import butter, sosfilt
        sos = butter(4, [low, high], btype='bandstop', output='sos')
        return sosfilt(sos, audio)

    def _multi_time_mask(self, audio, max_mask_ratio=0.12, n_masks=2):
        """多次时间遮罩（比单次更强）。"""
        audio = audio.copy()
        for _ in range(random.randint(1, n_masks)):
            mask_len = int(len(audio) * random.uniform(0.03, max_mask_ratio))
            if mask_len <= 0:
                continue
            start = random.randint(0, max(0, len(audio) - mask_len))
            audio[start:start + mask_len] = 0
        return audio

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        audio = self.processor.load_audio(audio_path)
        audio = self.processor.normalize_audio(audio)

        if self.mode == 'train':
            lvl = self.augment_level

            # ---- 噪声 ----
            noise_prob = 0.3 if lvl == 'light' else (0.6 if lvl == 'full' else 0.75)
            noise_range = (0.002, 0.006) if lvl == 'light' else ((0.002, 0.008) if lvl == 'full' else (0.003, 0.012))
            if random.random() < noise_prob:
                audio = self.processor.add_noise(audio, noise_factor=random.uniform(*noise_range))

            # ---- 时间偏移 ----
            shift_prob = 0.3 if lvl == 'light' else (0.5 if lvl == 'full' else 0.6)
            shift_max = 0.15 if lvl == 'light' else (0.2 if lvl == 'full' else 0.25)
            if random.random() < shift_prob:
                audio = self.processor.time_shift(audio, shift_max=shift_max)

            # ---- 音调变换 ----
            pitch_prob = 0.2 if lvl == 'light' else (0.4 if lvl == 'full' else 0.5)
            pitch_range = (-1.5, 1.5) if lvl == 'light' else ((-2, 2) if lvl == 'full' else (-3, 3))
            if random.random() < pitch_prob:
                audio = self.processor.pitch_shift(audio, n_steps=random.uniform(*pitch_range))

            # ---- 时间遮罩 ----
            if lvl == 'full' and random.random() < 0.35:
                audio = self._multi_time_mask(audio, max_mask_ratio=0.15, n_masks=1)
            elif lvl == 'strong':
                if random.random() < 0.5:
                    audio = self._multi_time_mask(audio, max_mask_ratio=0.15, n_masks=3)
                # ---- 频率遮罩 (band-stop) ----
                if random.random() < 0.3:
                    audio = self._band_stop_mask(audio, max_width_hz=800)

            # ---- 时间拉伸 ----
            stretch_prob = 0.25 if lvl == 'full' else (0.35 if lvl == 'strong' else 0)
            if stretch_prob > 0 and random.random() < stretch_prob:
                rate = random.uniform(0.9, 1.1)
                audio = self.processor.time_stretch(audio, rate=rate)
                if len(audio) < self.processor.n_samples:
                    audio = np.pad(audio, (0, self.processor.n_samples - len(audio)))
                else:
                    audio = audio[:self.processor.n_samples]

        return {
            'audio': torch.FloatTensor(audio),
            'label': torch.LongTensor([label])
        }


def count_samples_by_class(data_dir=config.DATA_DIR, mode='train'):
    counts = []
    for emotion in config.EMOTIONS:
        emotion_dir = os.path.join(data_dir, mode, emotion)
        if not os.path.exists(emotion_dir):
            counts.append(0)
            continue
        n = sum(
            1 for f in os.listdir(emotion_dir)
            if f.endswith('.wav') and EmotionDataset._is_valid_wav(os.path.join(emotion_dir, f))
        )
        counts.append(n)
    return counts


def compute_class_weights(data_dir=config.DATA_DIR, mode='train', boost=None):
    """逆频率类别权重；boost 为 {class_idx: factor} 额外加权弱类。"""
    counts = count_samples_by_class(data_dir, mode)
    total = sum(counts)
    n_classes = len(counts)
    weights = [total / (n_classes * c) if c > 0 else 1.0 for c in counts]
    if boost:
        for idx, factor in boost.items():
            weights[idx] *= factor
    return torch.tensor(weights, dtype=torch.float32), counts


def get_wav2vec2_dataloaders(data_dir=config.DATA_DIR, batch_size=config.WAV2VEC2_BATCH_SIZE,
                             augment_level='full', weighted_sampler=False,
                             class_boost=None):
    import multiprocessing
    from torch.utils.data import WeightedRandomSampler
    nw = min(2, multiprocessing.cpu_count() // 4)

    train_dataset = Wav2Vec2EmotionDataset(data_dir=data_dir, mode='train',
                                           augment_level=augment_level)
    val_dataset = Wav2Vec2EmotionDataset(data_dir=data_dir, mode='val')
    test_dataset = Wav2Vec2EmotionDataset(data_dir=data_dir, mode='test')

    train_kwargs = dict(batch_size=batch_size, collate_fn=wav2vec2_collate_fn,
                        num_workers=nw, pin_memory=True)
    if weighted_sampler:
        cw, _ = compute_class_weights(data_dir, 'train', boost=class_boost)
        sample_weights = [cw[label].item() for label in train_dataset.labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, sampler=sampler, **train_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **train_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=wav2vec2_collate_fn, num_workers=nw,
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=wav2vec2_collate_fn, num_workers=nw,
                            pin_memory=True)

    return train_loader, val_loader, test_loader