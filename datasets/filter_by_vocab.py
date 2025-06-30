# Filter based on the vacab ; recognition models doesn't need chinese, russian and other languages..

import json
import os
import shutil
import string
import sys

# Import vocabulary from vocabs.py
from doctane.datasets.vocabs import VOCABS

# Convert VOCABS to a set for fast lookup (if it's not already)
doctr_vocab = set(VOCABS['french'])

# Set up paths
base_dir = "./dataset"
image_dir = os.path.join(base_dir, "word_crops")
labels_file = os.path.join(base_dir, "recognition_labels.json")

# Output paths
valid_image_dir = os.path.join(base_dir, "filtered_recog_images")
valid_labels_file = os.path.join(base_dir, "filtered_recog_labels.json")

# Create output image directory
os.makedirs(valid_image_dir, exist_ok=True)

# Load original labels
with open(labels_file, "r") as f:
    labels_data = json.load(f)

# Store only valid labels and copy corresponding images
valid_labels_data = {}
for image, label in labels_data.items():
    if all(char in doctr_vocab for char in label):
        valid_labels_data[image] = label
        image_path = os.path.join(image_dir, image)
        dest_path = os.path.join(valid_image_dir, image)
        if os.path.exists(image_path):
            shutil.copy(image_path, dest_path)
    else:
        print(f"Excluded '{label}' from {image} due to unsupported characters.")

# Save valid labels
with open(valid_labels_file, "w") as f:
    json.dump(valid_labels_data, f, indent=4)

print("Filtering complete: Valid images and labels saved.")
