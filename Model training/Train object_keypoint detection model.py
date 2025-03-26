import os
import shutil
from ultralytics import YOLO
from prepare_config import prepare_config_file, prepare_paths

def main():
    # Step 1: Prepare configuration file and paths
    config_path, project_path = prepare_config_file()
    source_path, destination_path = prepare_paths(project_path)

    # Step 2: Load and train the YOLO model
    model = YOLO('yolo11m-pose.pt')
    model.train(
        data=config_path,  # Use the prepared config file
        epochs=150,
        batch=-1,
        # warmup_epochs=5,
        device=0,
        patience=20,  # Stop training if no improvement for 20 epochs
        imgsz=640,
    )
    
    # Step 3: Copy the 'runs' directory (YOLO training output) to the specified destination
    # if os.path.exists(source_path):
    #     shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
    #     print(f"Training results have been copied to {destination_path}")
    # else:
    #     print(f"Source path '{source_path}' does not exist. Make sure the training ran successfully.")

if __name__ == '__main__':
    main()

# import os
# import shutil
# from ultralytics import YOLO
# from prepare_config import prepare_config_file, prepare_paths

# def main():
#     # Step 1: Prepare configuration file and paths
#     config_path, project_path = prepare_config_file()
#     source_path, destination_path = prepare_paths(project_path)

#     # Step 2: Load and train the YOLO model
#     model = YOLO('yolo11m-pose.pt')
#     model.train(
#         data=config_path,  # Use the prepared config file
#         epochs=100,
#         batch=-1,
#         warmup_epochs=5,
#         device=0,
#         patience=15,  # Stop training if no improvement for 10 epochs
#         imgsz=640,
#     )
    
#     # Step 3: Copy the 'runs' directory (YOLO training output) to the specified destination
#     if os.path.exists(source_path):
#         shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
#         print(f"Training results have been copied to {destination_path}")
#     else:
#         print(f"Source path '{source_path}' does not exist. Make sure the training ran successfully.")

# if __name__ == '__main__':
#     main()


# import os
# import shutil
# from ultralytics import YOLO
# from prepare_config import prepare_config_file, prepare_paths

# def main():
#     # Step 1: Prepare configuration file and paths
#     config_path, project_path = prepare_config_file()
#     source_path, destination_path = prepare_paths(project_path)

#     # Step 2: Load and train the YOLO model
#     model = YOLO('yolo11m-pose.pt')

#     results = model.train(
#         data=config_path,  # Use the prepared config file
#         epochs=200,  # Reduce if model converges early
#         batch=32,  # Increase batch size for faster training
#         imgsz=512,  # Lower resolution for speed optimization
#         device=[0,1],  # Multi-GPU training (modify as needed)
#         optimizer="AdamW",
#         patience=10,  # Stop training if no improvement
#         auto_weight=True,  # Balance class losses dynamically

#         # Augmentation improvements
#         mosaic=0.5,  # Reduce mosaic augmentation
#         mixup=0.2,  # Lower mixup for stability
#         copy_paste=0.2,  # Augment rare-class objects

#         # Loss function adjustments for better learning
#         cls=2.0,  # Higher weight on classification loss
#         kobj=4.0,  # Objectness loss increased
#         box=10.0,  # Emphasize bounding box accuracy
#         dfl=2.0,  # Improve keypoint distribution focal loss

#         # Other efficiency improvements
#         warmup_epochs=2,  # Faster early training stabilization
#         cos_lr=False,  # Disable cosine learning rate decay (speed boost)
#         multi_scale=False,  # Disable multi-scale (improves speed)
#         save=True,
#         save_period=20,  # Save model less frequently
#         pretrained=True  # Use pretrained weights
#     )

#     # Step 3: Evaluate model on validation data
#     metrics = model.val()
#     print(f"Validation Performance:\n"
#           f"mAP50: {metrics.metrics.mAP50:.4f}\n"
#           f"mAP50-95: {metrics.metrics.mAP50_95:.4f}\n"
#           f"Precision: {metrics.metrics.precision:.4f}\n"
#           f"Recall: {metrics.metrics.recall:.4f}")

#     # Step 4: Copy training results if training was successful
#     if os.path.exists(source_path):
#         shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
#         print(f"Training results have been copied to {destination_path}")
#     else:
#         print(f"Source path '{source_path}' does not exist. Make sure the training ran successfully.")

# if __name__ == '__main__':
#     main()



# import os
# import shutil
# from ultralytics import YOLO
# from prepare_config import prepare_config_file, prepare_paths

# def main():
#     # Step 1: Prepare configuration file and paths
#     config_path, project_path = prepare_config_file()
#     source_path, destination_path = prepare_paths(project_path)

#     # Step 2: Load and train the YOLO model
#     model = YOLO('yolo11m-pose.pt')
#     model.train(
#         data=config_path,  # Use the prepared config file
#         epochs=100,
#         batch=-1,
#         warmup_epochs=5,
#         device=0,
#         multi_scale=True,  # Improves generalization
#         patience=10,  # Stop training if no improvement for 10 epochs
#         imgsz=640,
#     )
    
#     # Step 3: Copy the 'runs' directory (YOLO training output) to the specified destination
#     if os.path.exists(source_path):
#         shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
#         print(f"Training results have been copied to {destination_path}")
#     else:
#         print(f"Source path '{source_path}' does not exist. Make sure the training ran successfully.")

# if __name__ == '__main__':
#     main()


#    flipud=0.5,  # Occasional upside-down flips is bad since wehn wew have a compoentns upside down with keypoint it will assume
#    that the point upside down is the same as the one that is not upside down making the two points to be the same. so better object detection but bad keypoint detection


# import os
# import shutil
# from ultralytics import YOLO
# from prepare_config import prepare_config_file, prepare_paths

# def main():
#     # Step 1: Prepare configuration file and paths
#     config_path, project_path = prepare_config_file()
#     source_path, destination_path = prepare_paths(project_path)

#     # Step 2: Load and train the YOLO model
#     model = YOLO('yolo11m-pose.pt')

#     # Train the model with optimal settings
#     results = model.train(
#     data=config_path,  # Use the prepared config file
#     epochs=300,  # Train for high accuracy
#     batch=-1,  # Auto-adjust batch size based on GPU memory
#     imgsz=640,  # High resolution for detailed detection
#     device=0,  # Use GPU (change to device=[0,1] for multi-GPU)
#     optimizer="AdamW",  # Optimized for generalization
#     lr0=0.001,  # Lower starting learning rate
#     lrf=0.1,  # Final learning rate (cosine decay)
#     momentum=0.937,  # Standard momentum
#     weight_decay=0.0001,  # Prevent overfitting
#     warmup_epochs=5,  # Stable early training
#     warmup_momentum=0.9,  # Higher warmup momentum
#     cos_lr=True,  # Use cosine learning rate decay
#     patience=10,  # Stop training if no improvement for 10 epochs
#     rect=False,  # Avoid rectangular training
#     multi_scale=True,  # Improves generalization
#     close_mosaic=10,  # Disable mosaic in last 10 epochs for stability
#     pose=12.0,  # Prioritize keypoint accuracy
#     kobj=3.0,  # Increase keypoint objectness loss
#     cls=1.0,  # Standard classification loss
#     box=7.5,  # Balanced bounding box loss
#     dfl=1.5,  # Distribution focal loss for keypoints
#     augment=True,  # Enable strong augmentations
#     hsv_h=0.05,  # Hue shift
#     hsv_s=0.5,  # Saturation changes
#     hsv_v=0.3,  # Brightness variability
#     degrees=15,  # Small rotations
#     translate=0.2,  # Symbol position shifts
#     scale=0.6,  # Scale variations
#     shear=5.0,  # Minor shearing
#     perspective=0.0005,  # Minimal perspective warp
#     flipud=0.1,  # Occasional upside-down flips
#     # fliplr=0.5,  # Frequent left/right flips
#     mosaic=1.0,  # Strong data augmentation
#     mixup=0.3,  # Mix component images
#     copy_paste=0.3,  # Paste components into different circuits
#     save=True,  # Save best models
#     # save_conf=True,  # Save confidence scores
#     # show_conf=True,  # Display confidence scores
#     save_period=10,  # Save model every 10 epochs
#     pretrained=True  # Start with pretrained weights
#     )
#     # Print training summary
#     print("Training Complete. Best model saved in:", results.save_dir)
    
#     # Step 3: Copy the 'runs' directory (YOLO training output) to the specified destination
#     if os.path.exists(source_path):
#         shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
#         print(f"Training results have been copied to {destination_path}")
#     else:
#         print(f"Source path '{source_path}' does not exist. Make sure the training ran successfully.")

# if __name__ == '__main__':
#     main()

