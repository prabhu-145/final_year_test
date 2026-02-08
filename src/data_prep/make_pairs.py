import os
import shutil
import random

# Paths
SKETCH_DIR = "data/aligned/sketches"
PHOTO_DIR  = "data/aligned/photos"

TRAIN_A = "data/paired/trainA"
TRAIN_B = "data/paired/trainB"
VAL_A   = "data/paired/valA"
VAL_B   = "data/paired/valB"

# Create output folders
for path in [TRAIN_A, TRAIN_B, VAL_A, VAL_B]:
    os.makedirs(path, exist_ok=True)

# Get common filenames (same sketch & photo)
sketch_files = set(os.listdir(SKETCH_DIR))
photo_files  = set(os.listdir(PHOTO_DIR))

paired_files = list(sketch_files.intersection(photo_files))
paired_files.sort()

print(f"Total paired images found: {len(paired_files)}")

# Shuffle and split (80% train, 20% val)
random.shuffle(paired_files)
split_idx = int(0.8 * len(paired_files))

train_files = paired_files[:split_idx]
val_files   = paired_files[split_idx:]

def copy_files(file_list, src_sketch, src_photo, dstA, dstB):
    for fname in file_list:
        shutil.copy(os.path.join(src_sketch, fname), os.path.join(dstA, fname))
        shutil.copy(os.path.join(src_photo, fname),  os.path.join(dstB, fname))

# Copy training data
copy_files(train_files, SKETCH_DIR, PHOTO_DIR, TRAIN_A, TRAIN_B)

# Copy validation data
copy_files(val_files, SKETCH_DIR, PHOTO_DIR, VAL_A, VAL_B)

print("✅ Paired dataset created successfully")
print(f"Training pairs: {len(train_files)}")
print(f"Validation pairs: {len(val_files)}")
