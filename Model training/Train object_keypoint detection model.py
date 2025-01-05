import os
import shutil
from ultralytics import YOLO
# from config import PROJECT_PATH 

current_dir = os.getcwd()  # Get the current working directory
PROJECT_PATH = os.path.dirname(current_dir)  # Get the parent directory (project root directory)

# Define source and destination paths dynamically
source_path = os.path.join(PROJECT_PATH, 'runs')
destination_path = os.path.join(PROJECT_PATH, 'Current trained model')

from ultralytics import YOLO

def main():
    # Load and train the YOLO model
    model = YOLO('yolov8m-pose.pt')
    model.train(
        data=os.path.join(PROJECT_PATH, 'Model training/config.yaml'),
        epochs=150,
        imgsz=640
    )
    
    # Copy the 'runs' directory (YOLO training output) to the specified destination
    if os.path.exists(source_path):
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        print(f"Training results have been copied to {destination_path}")
    else:
        print(f"Source path '{source_path}' does not exist. Make sure the training ran successfully.")

if __name__ == '__main__':
    main()