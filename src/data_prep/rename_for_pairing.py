import os

SKETCH_DIR = "data/aligned/sketches"
PHOTO_DIR  = "data/aligned/photos"

# Get file lists
sketch_files = sorted(os.listdir(SKETCH_DIR))
photo_files  = sorted(os.listdir(PHOTO_DIR))

# Use minimum count to avoid mismatch
count = min(len(sketch_files), len(photo_files))

print(f"Renaming {count} sketch-photo pairs")

for i in range(count):
    new_name = f"{i+1:03}.jpg"

    # Rename sketch
    os.rename(
        os.path.join(SKETCH_DIR, sketch_files[i]),
        os.path.join(SKETCH_DIR, new_name)
    )

    # Rename photo
    os.rename(
        os.path.join(PHOTO_DIR, photo_files[i]),
        os.path.join(PHOTO_DIR, new_name)
    )

print("✅ Renaming completed successfully")
