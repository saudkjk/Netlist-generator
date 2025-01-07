import os
import cv2
from tkinter import Tk, filedialog
from ultralytics import YOLO

# Get the current working directory and define the project root
current_dir = os.getcwd()
PROJECT_PATH = os.path.dirname(os.path.dirname(current_dir))  # Project path is two levels up

# Function for processing all images
def process_all_images():
    # Find the latest training folder dynamically
    pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
    train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

    # Check if there are train folders
    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the pose directory.")

    # Determine the latest train folder
    def extract_suffix(folder_name):
        if folder_name == "train":
            return 0
        else:
            return int(folder_name[5:])

    latest_train_folder = max(train_folders, key=extract_suffix)
    latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

    # Define the image folder path dynamically
    image_folder = os.path.join(PROJECT_PATH, 'Program test/Model test/Test images')

    # Define the output folder for processed images
    output_folder = os.path.join(PROJECT_PATH, 'Program test/Model test/Model test results')
    os.makedirs(output_folder, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from: {latest_train_path}")
    model = YOLO(latest_train_path)

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Process and save each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)

        # Load image
        img = cv2.imread(image_path)

        # Run inference
        results = model(image_path)[0]

        # Process each result
        for result in results:
            for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
                                             result.keypoints.xy.cpu().numpy(),
                                             result.boxes.xyxy.cpu().numpy()):
                class_idx = int(cls)
                object_name = results.names[class_idx]

                # Extract bounding box coordinates
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Draw bounding box and keypoints
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(img, object_name, (x_min, y_min),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1)

                for keypoint in keypoints:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(img, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

        # Save the processed image to the output folder
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, img)
        print(f"Processed image saved to: {output_image_path}")

# Function for processing a single image with file selection
def process_single_image():
    # Use tkinter to select the image file
    print("Please select an image from the Test Images folder.")
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window
    root.lift()  # Bring the dialog to the front
    root.attributes("-topmost", True)  # Ensure the dialog appears on top

    # Define the image folder path dynamically
    image_folder = os.path.join(PROJECT_PATH, 'Program test/Model test/Test images')

    # Let the user select a file
    selected_file = filedialog.askopenfilename(initialdir=image_folder, title="Select an image",
                                               filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))

    root.destroy()  # Destroy the root window after selection

    # If the user cancels the selection, exit the function
    if not selected_file:
        print("No file selected. Exiting...")
        return

    # Load YOLO model
    pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
    train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

    # Check if there are train folders
    if not train_folders:
        raise FileNotFoundError("No 'train' folders found in the pose directory.")

    # Determine the latest train folder
    def extract_suffix(folder_name):
        if folder_name == "train":
            return 0
        else:
            return int(folder_name[5:])

    latest_train_folder = max(train_folders, key=extract_suffix)
    latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

    print(f"Loading model from: {latest_train_path}")
    model = YOLO(latest_train_path)

    # Load the selected image
    img = cv2.imread(selected_file)

    # Run inference
    results = model(selected_file)[0]

    # Process each result
    for result in results:
        keypoints = result.keypoints.xy.cpu().numpy()[0]
        class_idx = int(result.boxes.cls.cpu().numpy()[0])
        object_name = results.names[class_idx]

        # Draw bounding box
        bbox = result.boxes.xyxy.cpu().numpy()[0]
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(img, object_name, (x_min, y_min),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 255), 1)

        # Draw keypoints
        for keypoint in keypoints:
            x, y = int(keypoint[0]), int(keypoint[1])
            cv2.circle(img, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

    # Display the processed image with the window appearing on top
    print("Displaying processed image...")
    cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)  # Create a named window
    cv2.setWindowProperty("Processed Image", cv2.WND_PROP_TOPMOST, 1)  # Set it to always be on top
    cv2.imshow("Processed Image", img)
    cv2.waitKey(0)  # Wait for a key press to close the image
    cv2.destroyAllWindows()

# Main function with user selection
def main():
    print("Select an option:")
    print("1. Process multiple images")
    print("2. Process a single image")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        process_all_images()
    elif choice == '2':
        process_single_image()
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == '__main__':
    main()
