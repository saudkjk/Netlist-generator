import os
import shutil

# Define directory paths
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
# cvat_to_coco_dir = os.path.join(base_dir, 'CVAT to coco Keypoints')
data_dir = os.path.join(parent_dir, 'data')
images_dir = os.path.join(data_dir, 'images')
labels_dir = os.path.join(data_dir, 'labels')
out_dir = os.path.join(base_dir, 'out')

# Get train and val directories for images and labels
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')

# Get the list of image filenames without extensions in train and val
train_files = {os.path.splitext(f)[0] for f in os.listdir(train_images_dir) if f.endswith('.jpg')}
val_files = {os.path.splitext(f)[0] for f in os.listdir(val_images_dir) if f.endswith('.jpg')}

# Iterate over .txt files in the out directory
for txt_file in os.listdir(out_dir):
    if txt_file.endswith('.txt'):
        file_name = os.path.splitext(txt_file)[0]  # Get filename without extension
        txt_path = os.path.join(out_dir, txt_file)
        
        if file_name in train_files:
            shutil.copy(txt_path, os.path.join(train_labels_dir, txt_file))
        elif file_name in val_files:
            shutil.copy(txt_path, os.path.join(val_labels_dir, txt_file))

print("Files copied successfully!")
