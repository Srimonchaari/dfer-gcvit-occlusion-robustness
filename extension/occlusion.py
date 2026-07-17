# ============================================================
# FILE 5: extension/occlusion.py
# PURPOSE: Simulates real-world face occlusion by painting
#          black rectangles over specific face regions.
# THIS IS THE CORE IDEA OF OUR EXTENSION.
# ← The professor's code has ZERO occlusion anywhere.
#   We invented this research question:
#   "Can the model still recognize emotions when part of the
#    face is hidden — like sunglasses or a surgical mask?"
# ← These two functions are the foundation of every experiment,
#   every chart, and every result in our extension.
# ============================================================


def apply_eye_occlusion(image):
    # ← Simulates a person wearing SUNGLASSES
    # ← image is a numpy array of shape (224, 224) — pixel values 0-255
    # ← image[rows, cols] = 0 sets those pixels to BLACK
    # ← Rows 60-100 = eye region in the 224x224 cropped face
    # ← Cols 30-195 = horizontal span across both eyes
    # ← We chose these coordinates by visually inspecting
    #   where eyes appear in the KMU-FED face images
    occluded = image.copy()          # ← copy so we don't destroy the original image
    occluded[60:100, 30:195] = 0    # ← paint black rectangle over eye region
    return occluded


def apply_mouth_occlusion(image):
    # ← Simulates a person wearing a SURGICAL MASK
    # ← Rows 150-185 = lower portion of face where mouth sits
    # ← Cols 40-185 = horizontal span across the mouth area
    # ← Relevant especially post-COVID — masks are common in real life
    #   A model that breaks when mouth is hidden is not deployable
    occluded = image.copy()           # ← copy so we don't destroy the original image
    occluded[150:185, 40:185] = 0    # ← paint black rectangle over mouth region
    return occluded
