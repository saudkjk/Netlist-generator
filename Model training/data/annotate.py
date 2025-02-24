import os
import cv2
import sys

# Constants for directories
IMAGE_DIR = os.path.join("images", "val")
LABEL_DIR = os.path.join("labels", "val")
OUTPUT_DIR = "."  # Save output in the data directory

def parse_label_file(label_path):
    """Reads the label file and extracts bounding boxes and keypoints."""
    with open(label_path, "r") as file:
        lines = file.readlines()

    annotations = []
    for line in lines:
        values = list(map(float, line.strip().split()))
        if len(values) < 5:
            continue  # Skip invalid labels

        class_id = int(values[0])
        bbox = values[1:5]  # Bounding box (center_x, center_y, width, height)
        keypoints = values[5:]  # Remaining values are keypoints (x, y, visibility)

        annotations.append((class_id, bbox, keypoints))

    return annotations

def draw_annotations(image, annotations):
    """Draw bounding boxes and keypoints on the image."""
    height, width, _ = image.shape

    for class_id, bbox, keypoints in annotations:
        if len(bbox) < 4:
            continue  # Skip if bbox is incomplete

        # Convert normalized bbox to pixel values
        center_x, center_y, box_w, box_h = bbox
        x1 = int((center_x - box_w / 2) * width)
        y1 = int((center_y - box_h / 2) * height)
        x2 = int((center_x + box_w / 2) * width)
        y2 = int((center_y + box_h / 2) * height)

        # Draw bounding box
        color = (0, 255, 0)  # Green for boxes
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, str(class_id), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw keypoints
        for i in range(0, len(keypoints), 3):  # Keypoints stored as (x, y, visibility)
            kp_x, kp_y, visibility = keypoints[i:i+3]
            if visibility > 0:  # Only draw visible keypoints
                px = int(kp_x * width)
                py = int(kp_y * height)
                cv2.circle(image, (px, py), 3, (0, 0, 255), -1)  # Red dots for keypoints

    return image

def annotate_image(index):
    """Annotates the image with bounding boxes and keypoints and saves it."""
    # Get the list of image files
    image_files = sorted(os.listdir(IMAGE_DIR))
    
    if index < 0 or index >= len(image_files):
        print("Index out of range!")
        return

    # Select image and corresponding label file
    image_name = image_files[index]
    label_name = os.path.splitext(image_name)[0] + ".txt"

    image_path = os.path.join(IMAGE_DIR, image_name)
    label_path = os.path.join(LABEL_DIR, label_name)

    if not os.path.exists(image_path):
        print(f"Image file {image_path} not found!")
        return
    if not os.path.exists(label_path):
        print(f"Label file {label_path} not found!")
        return

    # Load image and annotations
    image = cv2.imread(image_path)
    annotations = parse_label_file(label_path)

    # Draw annotations
    annotated_image = draw_annotations(image, annotations)

    # Save the output
    output_path = os.path.join(OUTPUT_DIR, f"annotated_{image_name}")
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved as {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python annotate.py <index>")
        sys.exit(1)

    try:
        index = int(sys.argv[1])
        annotate_image(index)
    except ValueError:
        print("Index must be an integer.")
