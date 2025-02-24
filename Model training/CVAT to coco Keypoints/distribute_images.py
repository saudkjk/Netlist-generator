# import os
# import shutil
# import random

# # Define paths
# base_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(base_dir)
# source_images_dir = os.path.join(base_dir, 'images')
# destination_base = os.path.join(parent_dir, 'data', 'images')
# train_dir = os.path.join(destination_base, 'train')
# val_dir = os.path.join(destination_base, 'val')

# # Ensure destination directories exist
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)

# # Get list of image files
# image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# # Shuffle images for randomness
# random.shuffle(image_files)

# # Split data (80% train, 20% val)
# split_index = int(0.8 * len(image_files))
# train_files = image_files[:split_index]
# val_files = image_files[split_index:]

# # Copy files to respective folders
# for file in train_files:
#     shutil.copy(os.path.join(source_images_dir, file), os.path.join(train_dir, file))

# for file in val_files:
#     shutil.copy(os.path.join(source_images_dir, file), os.path.join(val_dir, file))

# print(f"Distributed {len(train_files)} images to train/ and {len(val_files)} images to val/")

import os
import shutil
import random

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
source_images_dir = os.path.join(base_dir, 'images')
destination_base = os.path.join(parent_dir, 'data', 'images')
train_dir = os.path.join(destination_base, 'train')
val_dir = os.path.join(destination_base, 'val')

# Ensure destination directories exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get list of image files
image_files = [f for f in os.listdir(source_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Sort images to maintain order before grouping
image_files.sort()

# Split into groups of 10 images each
group_size = 10
image_groups = [image_files[i:i + group_size] for i in range(0, len(image_files), group_size)]

train_files = []
val_files = []

# Process each group
for group in image_groups:
    random.shuffle(group)  # Shuffle within the group
    split_index = int(0.7 * len(group))  # 70% to train, 30% to val
    
    train_files.extend(group[:split_index])
    val_files.extend(group[split_index:])

# Copy files to respective folders
for file in train_files:
    shutil.copy(os.path.join(source_images_dir, file), os.path.join(train_dir, file))

for file in val_files:
    shutil.copy(os.path.join(source_images_dir, file), os.path.join(val_dir, file))

print(f"Distributed {len(train_files)} images to train/ and {len(val_files)} images to val/")
