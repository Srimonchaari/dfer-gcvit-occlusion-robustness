# ============================================================
# FILE 6: extension/train_baseline.py
# PURPOSE: Trains GC-ViT on CLEAN (unoccluded) images across
#          all 10 folds in one run. This is our BASELINE model.
# ← We rewrote the professor's mainKMU.py with 3 key improvements:
#   1. Mixed precision (autocast + GradScaler) — faster GPU training
#   2. Cosine Annealing LR scheduler — better than fixed learning rate
#   3. Gradient accumulation — simulates larger batch size
# ← Results saved to results/baseline/
# ============================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import multiprocessing
multiprocessing.freeze_support()

import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler   # ← for mixed precision training
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from KMU_adapted import KMU
from models.gcvitt import VanillaSwinT1

# ============================================================
# CONFIGURATION
# ← HDF5_PATH: clean images (no occlusion applied)
# ← 60 epochs vs professor's 35 — more training time
# ← LR 0.0001 vs professor's 0.005 — smaller, more careful steps
# ← ACCUM_STEPS: gradient accumulation (explained below)
# ============================================================
HDF5_PATH   = 'KMUtada/baseline.h5'    # ← clean images dataset
RESULTS_DIR = 'results/baseline'        # ← where checkpoints and CSV are saved
NUM_FOLDS   = 10
EPOCHS      = 60                        # ← 60 epochs (professor used 35)
BATCH_SIZE  = 16
ACCUM_STEPS = 4                         # ← gradient accumulation steps (effective batch = 16×4 = 64)
LR          = 0.0001                    # ← lower learning rate than professor's 0.005
NUM_CLASSES = 6

torch.backends.cudnn.benchmark = True   # ← tells GPU to auto-optimise for our image size

# ← Same transforms as professor's mainKMU.py — kept identical for fair comparison
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,)),
    transforms.RandomHorizontalFlip(),                              # ← mirror face randomly
    transforms.RandomRotation(40),                                  # ← tilt up to 40 degrees
    transforms.RandomAffine(degrees=40, scale=(.3, 1.1), shear=0.15),  # ← stretch/shear
    transforms.ToTensor(),
    transforms.Normalize((0.2274,), (0.2325,)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,)),
    transforms.ToTensor(),
    transforms.Normalize((0.2274,), (0.2353,)),                    # ← no augmentation for test
])


def train_one_fold(model, train_loader, test_loader, fold, device, ckpt_path):
    # ← Trains the model for EPOCHS epochs and saves the best checkpoint
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # ============================================================
    # IMPROVEMENT 1: Mixed Precision Training
    # ← GradScaler works with autocast to use 16-bit floats
    # ← Normal training: 32-bit floats (more memory, slower)
    # ← Mixed precision: 16-bit for most ops (2x faster, half memory)
    # ← GradScaler prevents underflow — tiny gradients becoming 0 in 16-bit
    # ← Professor's code: no mixed precision at all
    # ============================================================
    scaler = GradScaler('cuda')

    # ============================================================
    # IMPROVEMENT 2: Cosine Annealing Learning Rate Scheduler
    # ← Professor used a FIXED learning rate of 0.005 for all 35 epochs
    # ← We use a DECREASING learning rate that follows a cosine curve:
    #   Starts at LR=0.0001 → gradually decreases → near 0 by epoch 60
    # ← Early epochs: bigger steps to learn quickly
    # ← Later epochs: tiny steps to fine-tune precisely
    # ← Result: better final accuracy
    # ============================================================
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_acc = 0.0
    best_f1  = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = correct = total = 0
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            # ← autocast automatically uses 16-bit where safe, 32-bit where needed
            with autocast('cuda'):
                outputs = model(images)                              # ← forward pass
                # ← divide loss by ACCUM_STEPS for gradient accumulation
                loss = criterion(outputs, labels) / ACCUM_STEPS

            scaler.scale(loss).backward()   # ← backprop with scaled gradients

            # ============================================================
            # IMPROVEMENT 3: Gradient Accumulation
            # ← GPU memory limits us to batch_size=16
            # ← Ideally we want batch_size=64 for stable training
            # ← Solution: run 4 batches of 16, ADD UP the gradients,
            #   then update weights once — mathematically same as batch 64
            # ← ACCUM_STEPS=4 means update weights every 4 batches
            # ← Professor had no gradient accumulation
            # ============================================================
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)    # ← update weights after 4 batches accumulated
                scaler.update()
                optimizer.zero_grad()    # ← clear accumulated gradients

            total_loss += loss.item() * ACCUM_STEPS
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss  = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc, f1 = evaluate(model, test_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()    # ← move the cosine curve one step forward
        print(f'Fold {fold} | Epoch {epoch}/{EPOCHS} | LR: {current_lr:.5f} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

        # ← Save checkpoint only when test accuracy improves — same logic as professor
        if test_acc > best_acc:
            best_acc = test_acc
            best_f1  = f1
            torch.save(model.state_dict(), ckpt_path)   # ← save best model weights

    return best_acc, best_f1


def evaluate(model, test_loader, device):
    # ← Returns accuracy AND macro F1 score on the test set
    # ← Macro F1 = average F1 across all 6 classes — more fair than accuracy
    #   when class sizes are unequal
    model.eval()
    all_preds   = []
    all_targets = []
    with torch.no_grad():    # ← no gradient calculation needed during evaluation
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            with autocast('cuda'):
                outputs = model(images)
            _, predicted = outputs.max(1)           # ← highest score = predicted emotion
            all_preds.extend(predicted.cpu().tolist())
            all_targets.extend(labels.tolist())

    acc = 100. * sum(p == t for p, t in zip(all_preds, all_targets)) / len(all_targets)
    f1  = f1_score(all_targets, all_preds, average='macro') * 100
    return acc, f1


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = []

    # ← Loop over ALL 10 folds in one run
    #   Professor's script needed to be run 10 separate times manually
    for fold in range(1, NUM_FOLDS + 1):
        print(f'\n{"=" * 52}')
        print(f'FOLD {fold}/{NUM_FOLDS}')
        print(f'{"=" * 52}')

        train_ds = KMU(hdf5_path=HDF5_PATH, split='Training', fold=fold, transform=transform_train)
        test_ds  = KMU(hdf5_path=HDF5_PATH, split='Testing',  fold=fold, transform=transform_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model     = VanillaSwinT1(n_classes=NUM_CLASSES).to(device)   # ← fresh GC-ViT for each fold
        ckpt_path = os.path.join(RESULTS_DIR, f'best_fold_{fold}.pth')

        best_acc, best_f1 = train_one_fold(model, train_loader, test_loader, fold, device, ckpt_path)
        print(f'Fold {fold} | Best Test Acc: {best_acc:.2f}% | Best F1: {best_f1:.2f}%')

        results.append({'fold': fold, 'test_acc': round(best_acc, 4), 'f1_score': round(best_f1, 4)})

    # ← Save all fold results to CSV for analysis
    csv_path = os.path.join(RESULTS_DIR, 'fold_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'test_acc', 'f1_score'])
        writer.writeheader()
        writer.writerows(results)

    avg_acc = sum(r['test_acc'] for r in results) / NUM_FOLDS
    avg_f1  = sum(r['f1_score'] for r in results) / NUM_FOLDS

    print(f'\n{"=" * 52}')
    print(f'Average Test Accuracy: {avg_acc:.2f}%')   # ← final baseline accuracy = 95.82%
    print(f'Average F1 Score:      {avg_f1:.2f}%')
    print(f'Results saved to {csv_path}')
