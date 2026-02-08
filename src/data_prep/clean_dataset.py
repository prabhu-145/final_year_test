import os
from PIL import Image

def clean_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        try:
            img = Image.open(os.path.join(input_dir, file))
            img.verify()
            img = Image.open(os.path.join(input_dir, file))
            img = img.convert("RGB")
            img.save(os.path.join(output_dir, file))
        except:
            print("Removed:", file)

# Run cleaning
clean_folder("data/raw/sketches", "data/cleaned/sketches")
clean_folder("data/raw/photos", "data/cleaned/photos")
