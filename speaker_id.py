"""
说话人识别模块 — MFCC + GMM 声纹识别
手写算法实现：MFCC 提取 + 高斯混合模型注册/识别
"""
import numpy as np
import pickle
import os
from sklearn.mixture import GaussianMixture
from scipy.fft import dct

# ============================================
#  MFCC 提取（手写，和训练用 librosa 无关）
# ============================================
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def mel_filterbank(n_filters, n_fft, sample_rate):
    """Mel 滤波器组"""
    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(sample_rate / 2)
    mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_filters, n_fft // 2 + 1))
    for m in range(1, n_filters + 1):
        for k in range(bins[m-1], bins[m]):
            filters[m-1, k] = (k - bins[m-1]) / max(1, bins[m] - bins[m-1])
        for k in range(bins[m], min(bins[m+1], n_fft//2+1)):
            filters[m-1, k] = (bins[m+1] - k) / max(1, bins[m+1] - bins[m])
    return filters


def extract_mfcc(audio, sample_rate=16000, n_mfcc=20, n_fft=512,
                 hop_length=160, n_mels=40, preemphasis=0.97):
    """
    提取 MFCC 特征（手写实现）

    步骤:
    1. 预加重
    2. 分帧 + 汉明窗
    3. FFT → 功率谱
    4. Mel 滤波器组
    5. Log
    6. DCT → MFCC
    """
    if len(audio) < n_fft:
        return np.zeros((1, n_mfcc))

    # 1. 预加重
    audio = np.append(audio[0], audio[1:] - preemphasis * audio[:-1])

    # 2. 分帧
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    if n_frames <= 0:
        return np.zeros((1, n_mfcc))
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        frames[i] = audio[i*hop_length : i*hop_length+n_fft]

    # 3. 汉明窗
    window = np.hamming(n_fft)
    frames *= window

    # 4. FFT → 功率谱
    mag = np.abs(np.fft.rfft(frames, n_fft))
    power = (mag ** 2) / n_fft

    # 5. Mel 滤波器组
    filters = mel_filterbank(n_mels, n_fft, sample_rate)
    mel_energy = np.dot(power, filters.T)
    mel_energy = np.maximum(mel_energy, 1e-10)

    # 6. Log
    log_mel = np.log(mel_energy)

    # 7. DCT → MFCC
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]

    # 8. 添加 delta (一阶差分)
    delta = np.zeros_like(mfcc)
    delta[2:-2] = (mfcc[4:] - mfcc[:-4]) / 2
    mfcc_delta = np.concatenate([mfcc, delta], axis=1)

    return mfcc_delta  # (n_frames, 2*n_mfcc)


# ============================================
#  说话人模型（GMM）
# ============================================
class SpeakerModel:
    def __init__(self, name, n_components=16, n_mfcc=20):
        self.name = name
        self.gmm = GaussianMixture(
            n_components=n_components,
            covariance_type='diag',
            max_iter=200,
            n_init=3,
            random_state=42
        )
        self.is_trained = False
        self.n_mfcc = n_mfcc

    def enroll(self, audio_chunks, sample_rate=16000):
        """用多段音频注册说话人"""
        all_features = []
        for chunk in audio_chunks:
            if len(chunk) < 1600:  # 至少 0.1 秒
                continue
            mfcc = extract_mfcc(chunk, sample_rate, n_mfcc=self.n_mfcc)
            if mfcc.shape[0] > 0:
                all_features.append(mfcc)

        if not all_features:
            return False

        features = np.concatenate(all_features, axis=0)
        self.gmm.fit(features)
        self.is_trained = True
        return True

    def score(self, audio, sample_rate=16000):
        """对一段音频打分（越高越像）"""
        if not self.is_trained:
            return float('-inf')
        mfcc = extract_mfcc(audio, sample_rate, n_mfcc=self.n_mfcc)
        if mfcc.shape[0] < 3:
            return float('-inf')
        return self.gmm.score(mfcc)

    def to_dict(self):
        return {
            'name': self.name,
            'weights': self.gmm.weights_.tolist(),
            'means': self.gmm.means_.tolist(),
            'covariances': self.gmm.covariances_.tolist(),
            'n_mfcc': self.n_mfcc
        }

    @classmethod
    def from_dict(cls, data):
        sm = cls(data['name'], n_components=len(data['weights']),
                 n_mfcc=data.get('n_mfcc', 20))
        sm.gmm.weights_ = np.array(data['weights'])
        sm.gmm.means_ = np.array(data['means'])
        sm.gmm.covariances_ = np.array(data['covariances'])
        sm.is_trained = True
        # 补齐属性让 GMM 能用 score
        sm.gmm.precisions_cholesky_ = _compute_precision_cholesky(
            sm.gmm.covariances_, sm.gmm.covariance_type
        )
        return sm


def _compute_precision_cholesky(covariances, covariance_type):
    """计算 GMM 需要的精度矩阵"""
    from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
    return _compute_precision_cholesky(covariances, covariance_type)


# ============================================
#  说话人识别引擎
# ============================================
class SpeakerRecognitionEngine:
    def __init__(self, storage_path='speaker_models.pkl'):
        self.storage_path = storage_path
        self.speakers = {}  # name → SpeakerModel
        self.load()

    def register(self, name, audio_chunks, sample_rate=16000):
        """注册新说话人或向已有说话人追加语音"""
        if name not in self.speakers:
            self.speakers[name] = SpeakerModel(name)
        ok = self.speakers[name].enroll(audio_chunks, sample_rate)
        if ok:
            self.save()
        return ok

    def identify(self, audio, sample_rate=16000, threshold=-500):
        """识别说话人，返回 (name, score) 或 (None, None)"""
        if not self.speakers:
            return None, None

        scores = {}
        for name, model in self.speakers.items():
            s = model.score(audio, sample_rate)
            if s > float('-inf'):
                scores[name] = s

        if not scores:
            return None, None

        best = max(scores, key=scores.get)
        if scores[best] < threshold:
            return None, None

        return best, scores[best]

    def get_all_speakers(self):
        return list(self.speakers.keys())

    def delete_speaker(self, name):
        if name in self.speakers:
            del self.speakers[name]
            self.save()
            return True
        return False

    def save(self):
        data = {name: m.to_dict() for name, m in self.speakers.items()}
        with open(self.storage_path, 'wb') as f:
            pickle.dump(data, f)

    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                for name, d in data.items():
                    try:
                        self.speakers[name] = SpeakerModel.from_dict(d)
                    except Exception as e:
                        print(f"  ⚠ 加载说话人 {name} 失败: {e}")
                print(f"已加载 {len(self.speakers)} 个说话人模型")
            except Exception:
                pass


# ============================================
#  Demo
# ============================================
if __name__ == '__main__':
    engine = SpeakerRecognitionEngine()

    # 注册说话人 A
    audio_a = np.random.randn(48000).astype(np.float32) * 0.1
    engine.register("Alice", [audio_a, audio_a * 1.1])

    # 注册说话人 B
    audio_b = np.random.randn(48000).astype(np.float32) * 0.1
    engine.register("Bob", [audio_b, audio_b * 0.9])

    # 识别
    name, score = engine.identify(audio_a * 1.05)
    print(f"Identified: {name} (score={score:.1f})")

    engine.save()
    print("OK")
