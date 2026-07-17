# ============================================================
# FILE 2: KMU.py
# PURPOSE: PyTorch Dataset class — tells PyTorch HOW to load
#          images from the .h5 file and split them into
#          10 folds for cross-validation.
# KEY POINT: ALL images are used across all 10 folds.
#            Every image gets tested exactly once.
# ============================================================

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class KMU(data.Dataset):
    """`CK+ Dataset.

    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        there are 135,177,75,207,84,249,54 images in data
        we choose 123,159,66,186,75,225,48 images for training
        we choose 12,8,9,21,9,24,6 images for testing
        the split are in order according to the fold number
    """

    def __init__(self, split='Training', fold=1, transform=None):
        self.transform = transform
        self.split = split   # ← 'Training' or 'Testing' — which subset to load
        self.fold = fold     # ← which fold number (1 to 10) we are currently running

        # ← Open the .h5 file created by preprocess_kmu.py
        #   driver='core' loads the whole file into RAM — faster access
        self.data = h5py.File('KMUtada\mtcnnkmunew.h5', 'r', driver='core')

        number = len(self.data['data_label'])   # ← total number of images in dataset (~981)

        # ============================================================
        # 10-FOLD SPLIT CONFIGURATION
        # ← sum_number: where each emotion's images START and END in the file
        #   [0, 196, 316, 516, 725, 905, 1104]
        #   anger:   index 0   to 195  (196 images)
        #   disgust: index 196 to 315  (120 images)
        #   fear:    index 316 to 515  (200 images)
        #   happy:   index 516 to 724  (209 images)
        #   sadness: index 725 to 904  (180 images)
        #   surprise:index 905 to 1103 (199 images)
        # ← test_number: how many images per emotion go into each fold's test set
        #   roughly 10% of each emotion class
        # ============================================================
        sum_number  = [0, 196, 316, 516, 725, 905, 1104]  # ← start/end index of each emotion
        test_number = [19, 12, 20, 21, 18, 20]            # ← test images per emotion per fold

        test_index  = []   # ← will hold indices of test images for this fold
        train_index = []   # ← will hold indices of training images for this fold

        # ============================================================
        # FOLD LOGIC — Select which images are test for this fold
        # ← For fold 1: takes first  19 anger, first  12 disgust, etc.
        #   For fold 2: takes next   19 anger, next   12 disgust, etc.
        #   For fold 10: takes LAST images from each class (special case)
        # ← This ensures each fold has a BALANCED mix of all 6 emotions
        # ============================================================
        for j in range(len(test_number)):        # ← loop over each emotion (0 to 5)
            for k in range(test_number[j]):      # ← loop over how many test images for that emotion
                if self.fold != 10:              # ← folds 1-9: slide along from the start
                    test_index.append(sum_number[j] + (self.fold - 1) * test_number[j] + k)
                else:                            # ← fold 10: take from the END of each emotion block
                    test_index.append(sum_number[j + 1] - 1 - k)

        # ← Everything NOT in test_index goes into training
        #   This is why ALL images are used — train + test together = full dataset
        for i in range(number):
            if i not in test_index:
                train_index.append(i)

        print(len(train_index), len(test_index))
        print(f"Fold {self.fold}: Train samples: {len(train_index)}, Test samples: {len(test_index)}")

        # ← Load the actual pixel arrays and labels into memory for the selected split
        if self.split == 'Training':
            self.train_data   = []
            self.train_labels = []
            for ind in range(len(train_index)):
                self.train_data.append(self.data['data_pixel'][train_index[ind]])   # ← image array
                self.train_labels.append(self.data['data_label'][train_index[ind]]) # ← label 0-5

        elif self.split == 'Testing':
            self.test_data   = []
            self.test_labels = []
            for ind in range(len(test_index)):
                self.test_data.append(self.data['data_pixel'][test_index[ind]])
                self.test_labels.append(self.data['data_label'][test_index[ind]])

    def __getitem__(self, index):
        # ============================================================
        # __getitem__ — PyTorch calls this automatically for every image in a batch
        # ← Returns ONE image and its label
        #   If transform is set, applies augmentation before returning
        # ============================================================
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'Testing':
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)   # ← apply resize, flip, rotation etc. here

        return img, target              # ← returns (image tensor, label 0-5)

    def __len__(self):
        # ← PyTorch needs this to know how many images are in the dataset
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'Testing':
            return len(self.test_data)
