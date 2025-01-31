import os
import shutil
from ultralytics import YOLO
from prepare_config import prepare_config_file, prepare_paths

def main():
    # Step 1: Prepare configuration file and paths
    config_path, project_path = prepare_config_file()
    source_path, destination_path = prepare_paths(project_path)

    # Step 2: Load and train the YOLO model
    model = YOLO('yolov8m-pose.pt')
    model.train(
        data=config_path,  # Use the prepared config file
        epochs=100,
        imgsz=640
    )
    
    # Step 3: Copy the 'runs' directory (YOLO training output) to the specified destination
    if os.path.exists(source_path):
        shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        print(f"Training results have been copied to {destination_path}")
    else:
        print(f"Source path '{source_path}' does not exist. Make sure the training ran successfully.")

if __name__ == '__main__':
    main()
