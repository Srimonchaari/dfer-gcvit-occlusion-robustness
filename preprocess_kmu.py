# ============================================================
# FILE 1: preprocess_kmu.py
# PURPOSE: Runs ONCE offline — reads face images from folders
#          and saves them into a single .h5 database file.
# NOTE: MTCNN face detection was already run BEFORE this file.
#       The folder name "KMUMTCN" proves it — faces are already
#       cropped and aligned. This file just packages them.
# ============================================================

import csv
import os
import numpy as np
import h5py
import skimage.io
import torch

# ← STEP 1: Path to the folder containing MTCNN-cropped face images
#   "KMUMTCN" in the name means faces were already detected and cropped by MTCNN
ck_path = 'datasets/KMUMTCN'

# ← STEP 2: Each emotion has its own sub-folder
#   The model never sees folder names — only the label numbers 0-5
anger_path   = os.path.join(ck_path, 'anger')
disgust_path = os.path.join(ck_path, 'disgust')
fear_path    = os.path.join(ck_path, 'fear')
happy_path   = os.path.join(ck_path, 'happy')
sadness_path = os.path.join(ck_path, 'sadness')
surprise_path = os.path.join(ck_path, 'surprise')

# ← STEP 3: Two empty lists — one for pixel data, one for labels
data_x = []   # will hold all image pixel arrays
data_y = []   # will hold all labels (0,1,2,3,4,5)

# ← STEP 4: Output path — the .h5 file we are creating
#   h5 is like a zip file for numpy arrays — much faster to load than reading 1000 images
datapath = os.path.join('KMUtada','mtcnnkmunew.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# ============================================================
# STEP 5: Read every image and assign labels
# ← files.sort() is CRITICAL — alphabetical order ensures
#   the same image always ends up in the same fold every run.
#   Without sort(), order is random and the train/test split
#   would be different every time — unreliable experiments.
# ============================================================

files = os.listdir(anger_path)
files.sort()                                          # ← sort alphabetically — deterministic order
for filename in files:
    I = skimage.io.imread(os.path.join(anger_path, filename))  # ← read image as pixel array
    data_x.append(I.tolist())                         # ← add pixel values to list
    data_y.append(0)                                  # ← anger = label 0

files = os.listdir(disgust_path)
files.sort()                                          # ← same sort for every emotion
for filename in files:
    I = skimage.io.imread(os.path.join(disgust_path, filename))
    data_x.append(I.tolist())
    data_y.append(1)                                  # ← disgust = label 1

files = os.listdir(fear_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(fear_path, filename))
    data_x.append(I.tolist())
    data_y.append(2)                                  # ← fear = label 2

files = os.listdir(happy_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(happy_path, filename))
    data_x.append(I.tolist())
    data_y.append(3)                                  # ← happy = label 3

files = os.listdir(sadness_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(sadness_path, filename))
    data_x.append(I.tolist())
    data_y.append(4)                                  # ← sadness = label 4

files = os.listdir(surprise_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(surprise_path, filename))
    data_x.append(I.tolist())
    data_y.append(5)                                  # ← surprise = label 5

print(np.shape(data_x))   # ← prints total images count and image dimensions
print(np.shape(data_y))   # ← prints total labels count

# ============================================================
# STEP 6: Save everything into one .h5 file
# ← This is the final output used by KMU.py for training
#   "data_pixel" stores all image arrays
#   "data_label" stores all labels (0-5)
#   After this runs once, we never need the image folders again
# ============================================================
datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype='uint8', data=data_x)   # ← all pixel arrays saved
datafile.create_dataset("data_label", dtype='int64', data=data_y)   # ← all labels saved
datafile.close()

print("Save data finish!!!")
