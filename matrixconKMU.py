# ============================================================
# FILE: matrixconKMU.py  (PROFESSOR'S EVALUATION SCRIPT)
# PURPOSE: After training all 10 folds, this loads every saved
#          checkpoint and evaluates them together to produce:
#          1. Overall accuracy across all folds
#          2. Classification report (precision, recall, F1 per emotion)
#          3. Confusion matrix image
# ← This only tests CLEAN images — no occlusion (that is our extension)
# ← Run AFTER mainKMU.py has finished all 10 folds
# ============================================================

from __future__ import print_function

import torch
import torchvision
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from torch.autograd import Variable
from models import gcvitt     # ← GC-ViT model definition
from KMU import KMU

# ← Choose which saved model folder to load from
#   dataset name + model name must match what was used in mainKMU.py
parser_args_dataset = 'kmuFgcvit31m2'   # ← folder name used during training
parser_args_model   = 'gcvit'           # ← model name used during training

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ← Test transform: resize + normalize, NO augmentation
#   We test on real images exactly as they are
transforms_vaild = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((224,)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.2274,), (0.2353,))
])


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # ← Draws the confusion matrix as a colour grid
    # ← Rows = true emotion, Columns = predicted emotion
    # ← Diagonal = correct predictions (we want these to be high)
    # ← Off-diagonal = mistakes (model predicted wrong emotion)
    # ← normalize=True shows percentages instead of raw counts
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)       # ← actual emotion label
    plt.xlabel('Predicted label', fontsize=18)  # ← what the model predicted
    plt.tight_layout()


# ← The 6 emotion class names — used as axis labels on the confusion matrix
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sadness', 'Surprise']

# ← Build GC-ViT model (must match architecture used during training)
net = gcvitt.VanillaSwinT1(n_classes=6)   # ← GC-ViT Base with custom 6-class head

correct = 0
total   = 0

# ============================================================
# MAIN EVALUATION LOOP — loads each of the 10 fold checkpoints
# ← For each fold:
#   1. Load the saved best-accuracy checkpoint from that fold
#   2. Load the test images for that fold
#   3. Run inference — model predicts emotion for each test image
#   4. Collect all predictions and true labels
# ← After all 10 folds: combine predictions → confusion matrix
# ← This means EVERY image in the dataset gets tested exactly once
#   across the 10 folds — complete evaluation, nothing left out
# ============================================================
for i in range(10):
    print("%d fold" % (i + 1))

    # ← Build path to saved checkpoint: e.g. "kmuFgcvit31m2_gcvit/1/Test_model.t7"
    path       = os.path.join(parser_args_dataset + '_' + parser_args_model, '%d' % (i + 1))
    checkpoint = torch.load(os.path.join(path, 'Test_model.t7'))   # ← load saved weights

    net.load_state_dict(checkpoint['net'])   # ← restore model to its best-accuracy state
    net.to(device)
    net.eval()   # ← evaluation mode: no dropout, batch norm uses stored stats

    # ← Load test images for this specific fold
    testset    = KMU(split='Testing', fold=i + 1, transform=transforms_vaild)
    testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)

        outputs      = net(inputs)                         # ← forward pass: model predicts emotions
        _, predicted = torch.max(outputs.data, 1)         # ← highest score = predicted emotion class
        total       += targets.size(0)
        correct     += predicted.eq(targets.data).cpu().sum()   # ← count correct predictions

        # ← Accumulate predictions and true labels across all folds
        if batch_idx == 0 and i == 0:
            all_predicted = predicted
            all_targets   = targets
        else:
            all_predicted = torch.cat((all_predicted, predicted), 0)   # ← append to running list
            all_targets   = torch.cat((all_targets, targets), 0)

    acc = 100. * correct / total
    print("accuracy: %0.3f" % acc)   # ← prints running accuracy after each fold

# ============================================================
# FINAL RESULTS — after all 10 folds complete
# ← confusion_matrix: rows=true label, cols=predicted label
# ← classification_report: precision, recall, F1 per emotion
# ← These two together show WHERE the model makes mistakes
#   (e.g. confuses fear with disgust) not just overall accuracy
# ============================================================
matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
np.set_printoptions(precision=2)

# ← classification_report shows per-emotion breakdown:
#   precision = of all images predicted as "angry", how many were truly angry
#   recall    = of all truly angry images, how many did the model catch
#   f1-score  = balance of precision and recall
print('Classification Report:\n', classification_report(
    all_targets.data.cpu().numpy(),
    all_predicted.cpu().numpy(),
    target_names=class_names
))

# ← Plot and save the confusion matrix as an image
plt.figure(figsize=(10, 8))
plot_confusion_matrix(matrix, classes=class_names, normalize=False,
                      title='Confusion Matrix (Accuracy: %0.3f%%)' % acc)
plt.savefig(os.path.join(parser_args_dataset + '_' + parser_args_model, 'ConfusionMatrix.png'))   # ← save image
plt.close()
