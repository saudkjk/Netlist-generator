from ultralytics import YOLO
import os
from prepare_config import prepare_config_file

def validate_model():
    current_dir = os.getcwd()
    PROJECT_PATH = os.path.dirname(current_dir)  # Project path is one level up

    # Find the latest training folder dynamically
    pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
    train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

    # Check if there are train folders
    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the pose directory.")

    # Determine the latest train folder
    def extract_suffix(folder_name):
        return int(folder_name[5:]) if folder_name != "train" else 0

    latest_train_folder = max(train_folders, key=extract_suffix)

    # latest_model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')
    latest_model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'best.pt')
    
    config_path, project_path = prepare_config_file()

    print(f"Using model from: {latest_model_path}")
    print(f"Using config from: {config_path}")

    # Load the trained YOLOv8 model
    model = YOLO(latest_model_path)
    
    # Run validation
    results = model.val(data=config_path)

    # Extract per-class mAP scores properly
    try:
        per_class_map = results.box.maps  # Correct way to get per-class mAP
    except AttributeError:
        print("⚠️ Could not extract per-class mAP scores. Using overall mAP instead.")
        per_class_map = [results.box.map50] * len(results.names)  # Fallback: Use overall mAP

    # Display results
    print("\n📌 Per-Class mAP@50 Scores:")
    for idx, mAP in enumerate(per_class_map):
        class_name = results.names[idx]  # Get class name
        print(f"Class {idx} ({class_name}): {mAP:.3f}")

    # Identify underperforming and well-performing classes
    underperforming_classes = [idx for idx, mAP in enumerate(per_class_map) if mAP < 0.70]
    well_performing_classes = [idx for idx, mAP in enumerate(per_class_map) if mAP >= 0.85]

    # Display findings
    print("\n⚠️ Underperforming Classes (mAP < 70%):", [results.names[i] for i in underperforming_classes])
    print("✅ Well-Performing Classes (mAP > 85%):", [results.names[i] for i in well_performing_classes])

    # Save results to a file
    with open("yolo_validation_results.txt", "w") as f:
        f.write("Per-Class mAP@50 Scores:\n")
        for idx, mAP in enumerate(per_class_map):
            f.write(f"Class {idx} ({results.names[idx]}): {mAP:.3f}\n")
        f.write("\nUnderperforming Classes (mAP < 70%): " + str([results.names[i] for i in underperforming_classes]))
        f.write("\nWell-Performing Classes (mAP > 85%): " + str([results.names[i] for i in well_performing_classes]))

    print("\n✅ Validation complete. Results saved to 'yolo_validation_results.txt'.")

# ✅ Fix for Windows multiprocessing
if __name__ == "__main__":
    validate_model()
