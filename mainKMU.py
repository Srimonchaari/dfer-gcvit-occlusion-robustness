# ============================================================
# FILE 3: mainKMU.py  (TRAINING SCRIPT)
# PURPOSE: Trains the model for ONE fold at a time.
#          Run this 10 times (fold 1 to 10) to complete training.
# HOW TO RUN: python mainKMU.py --model gcvit --fold 1 --bs 64 --lr 0.005
# ============================================================

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import argparse
import time
from KMU import KMU
from torch.autograd import Variable
from models import gcvitt      # ← GC-ViT model definition

# ============================================================
# COMMAND LINE ARGUMENTS
# ← These allow you to choose model, fold, batch size, learning rate
#   without editing the code every time
# Example: python mainKMU.py --model gcvit --fold 1 --bs 64 --lr 0.005
# ============================================================
parser = argparse.ArgumentParser(description='PyTorch KMU-FED Training')
parser.add_argument('--model',   type=str,   default='gcvit',    help='CNN architecture')
parser.add_argument('--dataset', type=str,   default='kmualign', help='dataset')
parser.add_argument('--fold',    default=1,  type=int,           help='k fold number')
parser.add_argument('--bs',      default=64, type=int,           help='batch_size')
parser.add_argument('--lr',      default=0.005, type=float,      help='learning rate')
parser.add_argument('--resume',  '-r', action='store_true',      help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = "cuda:1" if torch.cuda.is_available() else "cpu"   # ← use GPU if available

best_Test_acc       = 0   # ← tracks the best accuracy seen so far
best_Test_acc_epoch = 0   # ← tracks which epoch gave best accuracy
start_epoch         = 0   # ← start from 0 unless resuming a checkpoint

total_epoch = 35          # ← professor trains for 35 epochs per fold

path = os.path.join(opt.dataset + '_' + opt.model, str(opt.fold))   # ← folder to save checkpoints

# ============================================================
# DATA TRANSFORMS
# ← transforms_vaild: used for TEST images — no augmentation
#   just resize to 224x224 and normalize pixel values
# ← transforms_train: used for TRAINING images — adds augmentation
#   to prevent overfitting on the small ~1000 image dataset
# ============================================================
transforms_vaild = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,)),                              # ← resize to 224x224 (GC-ViT input size)
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.2274,), (0.2353,))             # ← normalize: subtract mean, divide by std
])

transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,)),                              # ← resize to 224x224
    torchvision.transforms.RandomHorizontalFlip(),                      # ← randomly mirror face left/right
    torchvision.transforms.RandomRotation(40),                          # ← randomly tilt face up to 40 degrees
    torchvision.transforms.RandomAffine(degrees=40, scale=(.3, 1.1), shear=0.15),  # ← stretch/shear face
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.2274,), (0.2325,))             # ← normalize pixel values
])
# ← WHY AUGMENTATION? Only ~1000 images. Without augmentation the model
#   memorizes exact training images. Augmentation creates random variations
#   so the model learns general features, not specific images.

# ← Load training and test sets for the selected fold
print('==> Preparing data..')
trainset    = KMU(split='Training', fold=opt.fold, transform=transforms_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=0)
testset     = KMU(split='Testing',  fold=opt.fold, transform=transforms_vaild)
testloader  = torch.utils.data.DataLoader(testset,  batch_size=16, shuffle=False, num_workers=0)

# ← GC-ViT Base with custom 6-class head (defined in models/gcvitt.py)
print('==> Building model..')
net = gcvitt.VanillaSwinT1(n_classes=6)   # ← loads pretrained GC-ViT + our custom head

if opt.resume:
    # ← Resume training from a previously saved checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    checkpoint      = torch.load(os.path.join(path, 'Test_model.t7'))
    net.load_state_dict(checkpoint['net'])
    best_Test_acc       = checkpoint['best_Test_acc']
    best_Test_acc_epoch = checkpoint['best_Test_acc_epoch']
    start_epoch         = best_Test_acc_epoch + 1

# ← CrossEntropyLoss: measures how wrong the model's predictions are
#   Adam optimizer: adjusts model weights to reduce the loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr)   # ← lr=0.005 fixed learning rate (professor's version)


def epoch_time(start_time, end_time):
    elapsed_time  = end_time - start_time
    elapsed_hours = int(elapsed_time // 3600)
    elapsed_time  = elapsed_time - elapsed_hours * 3600
    elapsed_mins  = int(elapsed_time // 60)
    elapsed_secs  = int(elapsed_time % 60)
    return elapsed_hours, elapsed_mins, elapsed_secs


# ============================================================
# TRAINING FUNCTION — runs once per epoch
# ← The core learning loop:
#   1. Feed batch of images through model (forward pass)
#   2. Calculate how wrong the predictions are (loss)
#   3. Propagate error backwards through the network (backprop)
#   4. Adjust weights to reduce the error (optimizer step)
# ============================================================
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.to(device)
    net.train()        # ← puts model in training mode (enables dropout, batch norm updates)
    train_loss = 0
    correct    = 0
    total      = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)   # ← move data to GPU
        optimizer.zero_grad()                                       # ← clear gradients from last batch
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)                                       # ← FORWARD PASS: model predicts emotion
        loss    = criterion(outputs, targets)                       # ← compare prediction vs true label
        loss.backward()                                             # ← BACKPROP: calculate how to fix weights
        optimizer.step()                                            # ← UPDATE WEIGHTS: take one step

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)                  # ← pick highest scoring emotion
        total   += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()   # ← count correct predictions

        print('Batch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (batch_idx, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    Train_acc = 100. * correct / total


# ============================================================
# TEST FUNCTION — runs after every epoch
# ← Just forward pass to measure accuracy on unseen test images
# ← Saves checkpoint ONLY if this epoch beats the best accuracy so far
# ============================================================
def test(epoch):
    global Test_acc
    global best_Test_acc
    global best_Test_acc_epoch
    net.to(device)
    net.eval()          # ← puts model in eval mode (disables dropout, freezes batch norm)
    PrivateTest_loss = 0
    correct = 0
    total   = 0

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        outputs  = net(inputs)                                      # ← FORWARD PASS only, no backprop
        loss     = criterion(outputs, targets)
        PrivateTest_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)                  # ← highest score = predicted emotion
        total   += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().item()

        print('Batch %d | Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (batch_idx, PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    Test_acc = 100. * correct / total

    # ← SAVE CHECKPOINT only if this is the best accuracy seen so far
    #   This means we always keep the BEST model, not just the last epoch
    if Test_acc > best_Test_acc:
        print('Saving best model — Acc: %0.3f%%' % Test_acc)
        state = {
            'net':                 net.state_dict(),
            'best_Test_acc':       Test_acc,
            'best_Test_acc_epoch': epoch,
        }
        if not os.path.isdir(opt.dataset + '_' + opt.model):
            os.mkdir(opt.dataset + '_' + opt.model)
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'Test_model.t7'))   # ← save model weights to disk
        best_Test_acc       = Test_acc
        best_Test_acc_epoch = epoch


# ============================================================
# MAIN TRAINING LOOP — runs train() then test() for 35 epochs
# ← After all 35 epochs, the saved checkpoint = best accuracy model
# ============================================================
total_start_time = time.monotonic()
for epoch in range(start_epoch, total_epoch):
    start_time = time.monotonic()
    train(epoch)     # ← train for one epoch
    test(epoch)      # ← evaluate and maybe save checkpoint
    end_time = time.monotonic()
    epoch_hours, epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_hours}h {epoch_mins}m {epoch_secs}s')

total_end_time = time.monotonic()
total_hours, total_mins, total_secs = epoch_time(total_start_time, total_end_time)
print(f'Total Time: {total_hours}h {total_mins}m {total_secs}s')
print("Best Test Accuracy: %0.3f%%" % best_Test_acc)
print("Best Epoch: %d" % best_Test_acc_epoch)
