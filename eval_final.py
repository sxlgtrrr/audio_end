"""测试集评估 (单线程，兼容 Windows)"""
import torch, sys, numpy as np
sys.path.insert(0, '.')
import config
from models import get_model
from data_processor import Wav2Vec2EmotionDataset, wav2vec2_collate_fn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

ds = Wav2Vec2EmotionDataset(data_dir='./data', mode='test')
test_loader = DataLoader(ds, 8, False, collate_fn=wav2vec2_collate_fn, num_workers=0, pin_memory=True)
model = get_model('wav2vec2')
ckpt = torch.load('models/best_model_wav2vec2_best.pth', map_location=config.DEVICE)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

all_preds, all_labels = [], []
for batch in test_loader:
    audio = batch['audio'].to(config.DEVICE)
    labels = batch['label'].to(config.DEVICE)
    with torch.no_grad():
        all_preds.extend(torch.softmax(model(audio),1).argmax(1).cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
print(f'Test Acc: {acc:.2f}%')
print(classification_report(all_labels, all_preds, target_names=config.EMOTIONS, digits=4))

from visualize import plot_confusion_matrix
plot_confusion_matrix(all_labels, all_preds, save_path='./logs/wav2vec2_test.png')
print('Confusion matrix saved.')
