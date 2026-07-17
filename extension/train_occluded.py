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
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from KMU_adapted import KMU
from models.gcvitt import VanillaSwinT1

HDF5_PATH   = 'KMUtada/occluded.h5'
RESULTS_DIR = 'results/occluded'
NUM_FOLDS    = 10
EPOCHS      = 60
BATCH_SIZE  = 16
ACCUM_STEPS = 4
LR          = 0.0001
NUM_CLASSES = 6

torch.backends.cudnn.benchmark = True

# Exact transforms from professor's mainKMU.py
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.RandomAffine(degrees=40, scale=(.3, 1.1), shear=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.2274,), (0.2325,)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,)),
    transforms.ToTensor(),
    transforms.Normalize((0.2274,), (0.2353,)),
])


def train_one_fold(model, train_loader, test_loader, fold, device, ckpt_path):
    # Train for EPOCHS, evaluate each epoch, save best checkpoint by test accuracy
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    best_acc  = 0.0
    best_f1   = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = correct = total = 0
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            with autocast('cuda'):
                outputs = model(images)
                loss    = criterion(outputs, labels) / ACCUM_STEPS
            scaler.scale(loss).backward()
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item() * ACCUM_STEPS
            _, predicted = outputs.max(1)
            total   += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_loss  = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        test_acc, f1 = evaluate(model, test_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        print(f'Fold {fold} | Epoch {epoch}/{EPOCHS} | LR: {current_lr:.5f} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%')

        if test_acc > best_acc:
            best_acc = test_acc
            best_f1  = f1
            torch.save(model.state_dict(), ckpt_path)

    return best_acc, best_f1


def evaluate(model, test_loader, device):
    # Return accuracy and macro F1 on the test loader
    model.eval()
    all_preds   = []
    all_targets = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            with autocast('cuda'):
                outputs = model(images)
            _, predicted = outputs.max(1)
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

    for fold in range(1, NUM_FOLDS + 1):
        print(f'\n{"=" * 52}')
        print(f'FOLD {fold}/{NUM_FOLDS}')
        print(f'{"=" * 52}')

        train_ds = KMU(hdf5_path=HDF5_PATH,              split='Training', fold=fold, transform=transform_train)
        test_ds  = KMU(hdf5_path='KMUtada/baseline.h5', split='Testing',  fold=fold, transform=transform_test)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model     = VanillaSwinT1(n_classes=NUM_CLASSES).to(device)
        ckpt_path = os.path.join(RESULTS_DIR, f'best_fold_{fold}.pth')

        best_acc, best_f1 = train_one_fold(model, train_loader, test_loader, fold, device, ckpt_path)
        print(f'Fold {fold} | Best Test Acc: {best_acc:.2f}% | Best F1: {best_f1:.2f}%')

        results.append({'fold': fold, 'test_acc': round(best_acc, 4), 'f1_score': round(best_f1, 4)})

    csv_path = os.path.join(RESULTS_DIR, 'fold_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'test_acc', 'f1_score'])
        writer.writeheader()
        writer.writerows(results)

    avg_acc = sum(r['test_acc'] for r in results) / NUM_FOLDS
    avg_f1  = sum(r['f1_score'] for r in results) / NUM_FOLDS

    print(f'\n{"=" * 52}')
    print(f'Average Test Accuracy: {avg_acc:.2f}%')
    print(f'Average F1 Score:      {avg_f1:.2f}%')
    print(f'Results saved to {csv_path}')
