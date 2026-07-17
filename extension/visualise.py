import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader

from KMU_adapted import KMU
from models.gcvitt import VanillaSwinT1
from occlusion import apply_eye_occlusion, apply_mouth_occlusion

CLASSES     = ['Anger','Disgust','Fear','Happy','Sadness','Surprise']
NUM_FOLDS   = 10
NUM_CLASSES = 6
HDF5        = 'KMUtada/baseline.h5'
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT         = 'visualisations'

transform_test = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224,)),
    transforms.ToTensor(), transforms.Normalize((0.2274,), (0.2353,)),
])


# ── helpers ────────────────────────────────────────────────────────────────────

class TestDS(Dataset):
    def __init__(self, images, labels):
        self.images = images; self.labels = labels
    def __len__(self): return len(self.images)
    def __getitem__(self, i): return transform_test(self.images[i]), self.labels[i]


def loader(kmu_ds, occ_fn=None):
    imgs = [occ_fn(x) if occ_fn else x.copy() for x in kmu_ds.test_data]
    return DataLoader(TestDS(imgs, list(kmu_ds.test_labels)), batch_size=16, shuffle=False)


def load_model(path):
    m = VanillaSwinT1(n_classes=NUM_CLASSES)
    m.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()


def get_preds(model, dl):
    preds, targets = [], []
    with torch.no_grad():
        for imgs, lbls in dl:
            preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().tolist())
            targets.extend(lbls.tolist())
    return np.array(targets), np.array(preds)


def occlusion_sensitivity(model, img_np, target_cls, patch=32):
    # Slide black patch across image, measure drop in target class probability
    heatmap = np.zeros((224, 224))
    with torch.no_grad():
        base_inp = transform_test(img_np).unsqueeze(0).to(DEVICE)
        base_prob = torch.softmax(model(base_inp), 1)[0, target_cls].item()
    for y in range(0, 224, patch):
        for x in range(0, 224, patch):
            occ = img_np.copy()
            occ[y:y+patch, x:x+patch] = 0
            with torch.no_grad():
                inp = transform_test(occ).unsqueeze(0).to(DEVICE)
                prob = torch.softmax(model(inp), 1)[0, target_cls].item()
            heatmap[y:y+patch, x:x+patch] = base_prob - prob
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


# ── 1. Accuracy comparison bar chart ──────────────────────────────────────────
def chart_accuracy():
    data = {
        'Baseline': {'Clean': 95.82, 'Eye Occluded': 92.64, 'Mouth Occluded': 35.45},
        'Occluded': {'Clean': 93.45, 'Eye Occluded': 90.73, 'Mouth Occluded': 74.73},
    }
    conditions = ['Clean', 'Eye Occluded', 'Mouth Occluded']
    x = np.arange(len(conditions)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, [data['Baseline'][c] for c in conditions], w, label='Baseline', color='steelblue')
    b2 = ax.bar(x + w/2, [data['Occluded'][c] for c in conditions], w, label='Occluded Model', color='coral')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy: Baseline vs Occlusion-Augmented Model', fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylim(0, 100); ax.legend(fontsize=11)
    ax.axhline(100, color='gray', linestyle='--', linewidth=0.5)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2.5,
                f'{bar.get_height():.2f}%', ha='center', va='top', fontsize=9,
                fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(f'{OUT}/1_accuracy_comparison.png', dpi=150)
    plt.close(); print('Saved: 1_accuracy_comparison.png')


# ── 2. Accuracy drop chart ─────────────────────────────────────────────────────
def chart_drop():
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ['Baseline', 'Occluded Model']
    eye   = [3.18,  2.73]
    mouth = [60.36, 18.73]
    x = np.arange(len(models)); w = 0.35
    b1 = ax.bar(x - w/2, eye,   w, label='Eye Drop',   color='dodgerblue')
    b2 = ax.bar(x + w/2, mouth, w, label='Mouth Drop', color='tomato')
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('Accuracy Drop Under Occlusion (Lower = More Robust)', fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10)
    ax.annotate('41.6% improvement\nin robustness', xy=(1.18, 18.73),
                xytext=(1.45, 35), fontsize=10, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    plt.tight_layout()
    plt.savefig(f'{OUT}/2_accuracy_drop.png', dpi=150)
    plt.close(); print('Saved: 2_accuracy_drop.png')


# ── 3. F1 comparison ───────────────────────────────────────────────────────────
def chart_f1():
    data = {
        'Baseline': [95.60, 92.20, 30.41],
        'Occluded': [93.40, 90.70, 73.95],
    }
    conditions = ['Clean', 'Eye Occluded', 'Mouth Occluded']
    x = np.arange(len(conditions)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, data['Baseline'], w, label='Baseline',       color='steelblue')
    b2 = ax.bar(x + w/2, data['Occluded'], w, label='Occluded Model', color='coral')
    ax.set_ylabel('Macro F1 Score (%)', fontsize=12)
    ax.set_title('Macro F1 Score: Baseline vs Occlusion-Augmented Model', fontsize=13)
    ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylim(0, 100); ax.legend(fontsize=11)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 2.5,
                f'{bar.get_height():.2f}%', ha='center', va='top', fontsize=9,
                fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(f'{OUT}/3_f1_comparison.png', dpi=150)
    plt.close(); print('Saved: 3_f1_comparison.png')


# ── 4. Per-fold accuracy line chart ────────────────────────────────────────────
def chart_per_fold():
    baseline = [100.0, 90.91, 99.09, 96.36, 90.91, 100.0, 93.64, 96.36, 92.73, 98.18]
    occluded = [100.0, 87.27, 100.0, 97.27,  80.0, 99.09,  90.0, 95.45, 87.27, 98.18]
    folds = list(range(1, 11))
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(folds, baseline, 'o-', color='steelblue', linewidth=2, markersize=8,
            label='Baseline', zorder=3)
    ax.plot(folds, occluded, 's--', color='coral',    linewidth=2, markersize=8,
            label='Occluded Model', zorder=3)
    ax.axhline(np.mean(baseline), color='steelblue', linestyle=':', alpha=0.6,
               label=f'Baseline avg  {np.mean(baseline):.2f}%')
    ax.axhline(np.mean(occluded), color='coral',     linestyle=':', alpha=0.6,
               label=f'Occluded avg  {np.mean(occluded):.2f}%')

    # Exact value labels on every data point
    # Y-axis capped at exactly 100 — accuracy cannot exceed 100%
    ax.set_ylim(74, 100)
    ax.set_yticks([75, 80, 85, 90, 95, 100])
    ax.axhline(100, color='grey', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Fold Cross-Validation Accuracy — Clean Images  (best held-out fold score, 10-fold CV)', fontsize=11)
    ax.set_xticks(folds)
    ax.legend(fontsize=10, loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{OUT}/4_per_fold_accuracy.png', dpi=150)
    plt.close(); print('Saved: 4_per_fold_accuracy.png')


# ── 5. Summary results table ────────────────────────────────────────────────────
def chart_table():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    headers = ['Model', 'Clean Acc', 'Clean F1', 'Eye Acc', 'Eye F1', 'Eye Drop', 'Mouth Acc', 'Mouth F1', 'Mouth Drop']
    rows = [
        ['Baseline',       '95.82%', '95.60%', '92.64%', '92.20%', '3.18%',  '35.45%', '30.41%', '60.36%'],
        ['Occluded Model', '93.45%', '93.40%', '90.73%', '90.70%', '2.73%',  '74.73%', '73.95%', '18.73%'],
    ]
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2)
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50'); table[0, j].set_text_props(color='white', fontweight='bold')
    table[1, 8].set_facecolor('#ffcccc'); table[2, 8].set_facecolor('#ccffcc')
    table[1, 6].set_facecolor('#ffcccc'); table[2, 6].set_facecolor('#ccffcc')
    ax.set_title('Complete Evaluation Results Summary', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{OUT}/5_results_table.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Saved: 5_results_table.png')


# ── 6. Confusion matrices (fold 1 sample) ─────────────────────────────────────
def chart_confusion():
    print('Building confusion matrices (fold 1)...')
    test_ds  = KMU(hdf5_path=HDF5, split='Testing', fold=1)
    model_a  = load_model('results/baseline/best_fold_1.pth')
    model_b  = load_model('results/occluded/best_fold_1.pth')

    conditions = [('Clean', None), ('Eye Occluded', apply_eye_occlusion), ('Mouth Occluded', apply_mouth_occlusion)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for col, (cond_name, occ_fn) in enumerate(conditions):
        for row, (model, mname) in enumerate([(model_a, 'Baseline'), (model_b, 'Occluded')]):
            y_true, y_pred = get_preds(model, loader(test_ds, occ_fn))
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=[c[:3] for c in CLASSES])
            disp.plot(ax=axes[row][col], colorbar=False, cmap='Blues')
            axes[row][col].set_title(f'{mname} — {cond_name}', fontsize=11, fontweight='bold')

    plt.suptitle('Confusion Matrices: Fold 1 Sample', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUT}/6_confusion_matrices.png', dpi=130, bbox_inches='tight')
    plt.close()
    del model_a, model_b; torch.cuda.empty_cache()
    print('Saved: 6_confusion_matrices.png')


# ── 7. Occlusion sensitivity heatmap ──────────────────────────────────────────
def chart_heatmap():
    print('Building occlusion sensitivity heatmaps (finer patches)...')
    import h5py
    model = load_model('results/baseline/best_fold_1.pth')
    with h5py.File(HDF5, 'r') as f:
        pixels = f['data_pixel'][:]
        labels = f['data_label'][:]

    cls_samples = {}
    for i, lbl in enumerate(labels):
        if int(lbl) not in cls_samples:
            cls_samples[int(lbl)] = pixels[i]
        if len(cls_samples) == 6:
            break

    fig, axes = plt.subplots(2, 6, figsize=(26, 9))
    for cls_id in range(6):
        img = cls_samples[cls_id]
        hm  = occlusion_sensitivity(model, img, cls_id, patch=16)  # finer grid
        ax0 = axes[0][cls_id]
        ax1 = axes[1][cls_id]
        ax0.imshow(img, cmap='gray')
        ax0.set_title(CLASSES[cls_id], fontsize=14, fontweight='bold', pad=6)
        ax0.axis('off')
        ax1.imshow(img, cmap='gray')
        im = ax1.imshow(hm, cmap='jet', alpha=0.6)
        ax1.axis('off')

    axes[0][0].set_ylabel('Original Image', fontsize=13, fontweight='bold')
    axes[1][0].set_ylabel('Attention Map\n(red = critical region)', fontsize=12, fontweight='bold')
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Importance')
    plt.suptitle('Baseline Model — Occlusion Sensitivity Map\nRed regions = where model relies most for expression recognition',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.savefig(f'{OUT}/7_attention_heatmap.png', dpi=140, bbox_inches='tight')
    plt.close()
    del model; torch.cuda.empty_cache()
    print('Saved: 7_attention_heatmap.png')


# ── 8. Sample test conditions ──────────────────────────────────────────────────
def chart_samples():
    import h5py
    with h5py.File(HDF5, 'r') as f:
        pixels = f['data_pixel'][:]
        labels = f['data_label'][:]

    fig, axes = plt.subplots(6, 3, figsize=(12, 22))
    shown = {}
    for i, lbl in enumerate(labels):
        if int(lbl) not in shown:
            shown[int(lbl)] = pixels[i]
        if len(shown) == 6: break

    col_titles = ['Original', 'Eye Occluded\n(simulates sunglasses)', 'Mouth Occluded\n(simulates surgical mask)']
    for row, cls_id in enumerate(range(6)):
        img = shown[cls_id]
        for col, (title, occ_fn) in enumerate([('Original', None),
                                                ('Eye Occluded\n(simulates sunglasses)', apply_eye_occlusion),
                                                ('Mouth Occluded\n(simulates surgical mask)', apply_mouth_occlusion)]):
            ax = axes[row][col]
            ax.imshow(occ_fn(img) if occ_fn else img, cmap='gray')
            ax.axis('off')
            if row == 0:
                ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
            if col == 0:
                ax.text(-0.12, 0.5, CLASSES[cls_id], fontsize=13, fontweight='bold',
                        transform=ax.transAxes, va='center', ha='right', rotation=90)

    plt.suptitle('KMU-FED Test Conditions: Original vs Eye vs Mouth Occlusion', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{OUT}/8_sample_conditions.png', dpi=140, bbox_inches='tight')
    plt.close()
    print('Saved: 8_sample_conditions.png')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    print(f'Device: {DEVICE}\nGenerating visualisations...\n')

    chart_accuracy()
    chart_drop()
    chart_f1()
    chart_per_fold()
    chart_table()
    chart_confusion()
    chart_heatmap()
    chart_samples()

    print(f'\nAll visualisations saved to {OUT}/')
