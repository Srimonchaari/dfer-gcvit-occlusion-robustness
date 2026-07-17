# ============================================================
# FILE 7: extension/evaluate.py
# PURPOSE: Loads BOTH trained models (baseline + occluded) and
#          tests each one under 3 conditions:
#          1. Clean images (no occlusion)
#          2. Eye occluded (sunglasses simulation)
#          3. Mouth occluded (surgical mask simulation)
# ← This is the final comparison that produces our key results.
# ← Professor's evaluation only tested clean accuracy.
#   We added occlusion testing to measure real-world robustness.
# ============================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

from KMU_adapted import KMU
from models.gcvitt import VanillaSwinT1
from occlusion import apply_eye_occlusion, apply_mouth_occlusion   # ← our occlusion functions

NUM_FOLDS   = 10
NUM_CLASSES = 6
HDF5        = 'KMUtada/baseline.h5'   # ← always test on CLEAN baseline images
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,)),
    transforms.ToTensor(),
    transforms.Normalize((0.2274,), (0.2353,)),
])


class TestDataset(Dataset):
    # ← Simple wrapper: holds a list of images and labels
    #   Used to wrap occluded images into a DataLoader
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return transform_test(self.images[idx]), self.labels[idx]


def get_loader(kmu_ds, occ_fn=None):
    # ============================================================
    # ← THIS IS WHERE OCCLUSION IS APPLIED AT TEST TIME
    # ← kmu_ds.test_data = raw numpy image arrays from the dataset
    # ← If occ_fn is None: use clean images as-is
    # ← If occ_fn = apply_eye_occlusion: black out eyes on every image
    # ← If occ_fn = apply_mouth_occlusion: black out mouth on every image
    # ← The model never knows occlusion was applied — it just receives
    #   the modified pixel array and tries to predict the emotion
    # ============================================================
    images = [occ_fn(img) if occ_fn else img.copy() for img in kmu_ds.test_data]
    return DataLoader(TestDataset(images, list(kmu_ds.test_labels)), batch_size=16, shuffle=False)


def evaluate(model, loader):
    # ← Runs inference on all images in loader
    #   Returns accuracy % and macro F1 score %
    model.eval()
    preds, targets = [], []
    with torch.no_grad():    # ← no gradient needed — just measuring performance
        for imgs, lbls in loader:
            preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().tolist())   # ← highest score = prediction
            targets.extend(lbls.tolist())
    acc = 100. * sum(p == t for p, t in zip(preds, targets)) / len(targets)
    return round(acc, 2), round(f1_score(targets, preds, average='macro') * 100, 2)


def load_model(path):
    # ← Load a saved model checkpoint from disk
    #   weights_only=True is a security setting (prevents arbitrary code execution)
    m = VanillaSwinT1(n_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE)


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    rows = []

    for fold in range(1, NUM_FOLDS + 1):
        print(f'\nFold {fold}/{NUM_FOLDS}')

        # ← Load the test split for this fold (clean images)
        test_ds = KMU(hdf5_path=HDF5, split='Testing', fold=fold)

        # ← Load BOTH models for this fold
        model_a = load_model(f'results/baseline/best_fold_{fold}.pth')   # ← trained on clean images
        model_b = load_model(f'results/occluded/best_fold_{fold}.pth')   # ← trained on occluded images

        # ============================================================
        # MAIN COMPARISON LOOP
        # ← For each model (baseline and occluded):
        #   Test under all 3 conditions and record accuracy + F1
        # ← c_acc = clean accuracy (no occlusion)
        # ← e_acc = eye occluded accuracy
        # ← m_acc = mouth occluded accuracy
        # ← eye_drop  = how much accuracy fell when eyes were blocked
        # ← mouth_drop = how much accuracy fell when mouth was blocked
        #   SMALLER DROP = MORE ROBUST MODEL
        # ============================================================
        for model, name in [(model_a, 'baseline'), (model_b, 'occluded')]:
            c_acc, c_f1 = evaluate(model, get_loader(test_ds))                         # ← clean
            e_acc, e_f1 = evaluate(model, get_loader(test_ds, apply_eye_occlusion))    # ← eyes blocked
            m_acc, m_f1 = evaluate(model, get_loader(test_ds, apply_mouth_occlusion))  # ← mouth blocked

            rows.append({
                'fold':       fold,
                'model':      name,
                'clean_acc':  c_acc,
                'clean_f1':   c_f1,
                'eye_acc':    e_acc,
                'eye_f1':     e_f1,
                'mouth_acc':  m_acc,
                'mouth_f1':   m_f1,
                'eye_drop':   round(c_acc - e_acc, 2),   # ← accuracy lost due to eye occlusion
                'mouth_drop': round(c_acc - m_acc, 2),   # ← accuracy lost due to mouth occlusion
            })
            print(f'  {name:<10} clean={c_acc}%  eye={e_acc}% (drop={c_acc - e_acc:.1f}%)  mouth={m_acc}% (drop={c_acc - m_acc:.1f}%)')

        del model_a, model_b
        torch.cuda.empty_cache()   # ← free GPU memory between folds

    # ← Save all results to CSV — 10 folds × 2 models × 3 conditions = 60 rows
    fields = ['fold', 'model', 'clean_acc', 'clean_f1', 'eye_acc', 'eye_f1', 'mouth_acc', 'mouth_f1', 'eye_drop', 'mouth_drop']
    with open('results/evaluation.csv', 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
        csv.DictWriter(f, fieldnames=fields).writerows(rows)

    # ← Print final averaged results across all 10 folds
    print('\n' + '=' * 55)
    for name in ['baseline', 'occluded']:
        mr = [r for r in rows if r['model'] == name]
        n  = len(mr)
        print(f'\n{name.upper()} MODEL')
        print(f"  Clean Acc:     {sum(r['clean_acc'] for r in mr) / n:.2f}%  |  F1: {sum(r['clean_f1'] for r in mr) / n:.2f}%")
        print(f"  Eye Occ Acc:   {sum(r['eye_acc']   for r in mr) / n:.2f}%  |  F1: {sum(r['eye_f1']   for r in mr) / n:.2f}%")
        print(f"  Mouth Occ Acc: {sum(r['mouth_acc'] for r in mr) / n:.2f}%  |  F1: {sum(r['mouth_f1'] for r in mr) / n:.2f}%")
        print(f"  Eye Drop:      {sum(r['eye_drop']  for r in mr) / n:.2f}%")   # ← baseline=3.18%, occluded=2.73%
        print(f"  Mouth Drop:    {sum(r['mouth_drop']for r in mr) / n:.2f}%")   # ← baseline=60.36%, occluded=18.73% ← KEY RESULT

    print('\nSaved: results/evaluation.csv')
