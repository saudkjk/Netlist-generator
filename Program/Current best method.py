import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import label as connected_label
from scipy.spatial import distance

# Define helper functions
def detect_edges(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    connected_edges = cv2.dilate(edges, kernel, iterations=2)

    return connected_edges

def mask_components(connected_edges, components):
    masked_edges = connected_edges.copy()
    for component in components:
        bbox = component["bounding_box"]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)

    return masked_edges

def find_nearest_edge(masked_edges, px, py):
    edge_points = np.argwhere(masked_edges > 0)  # Get all edge points
    if edge_points.size == 0:
        return None  # No edges in the mask
    distances = distance.cdist([(py, px)], edge_points)  # Compute distances from the point to all edges
    nearest_index = np.argmin(distances)  # Index of the nearest edge

    return edge_points[nearest_index]  # Coordinates of the nearest edge (y, x)

def is_point_connected_or_nearest(masked_edges, px, py):

    # Find the nearest edge
    nearest_edge = find_nearest_edge(masked_edges, px, py)
    if nearest_edge is not None:
        return True, (nearest_edge[1], nearest_edge[0])  # Return x, y of the nearest edge

    print(f"No connection found for point ({px}, {py})")  # Debug message if something went wrong
    return False, None  # No connection and no nearest edge

def overlay_and_find_nodes_with_connected_regions(masked_edges, components, results_path, image_file):
    labeled_edges, num_regions = connected_label(masked_edges)
    region_to_node = {}
    current_node_id = 1
    gnd_regions = set()

    for component in components:
        for point in component["connection_points"]:
            px, py = point
            if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
                continue

            is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
            if is_connected:
                connected_px, connected_py = connection_point
                region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
                if region > 0:
                    if region not in region_to_node:
                        region_to_node[region] = current_node_id
                        current_node_id += 1
                    if component["label"].upper() == "GND":
                        gnd_regions.add(region)

    # Rearrange node IDs based on top-left-most pixel
    region_top_left = {}
    for region in range(1, num_regions + 1):
        pixels = np.argwhere(labeled_edges == region)  # Get all pixels in the region
        if len(pixels) > 0:
            top_left_pixel = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))][0]  # Sort by (y, then x) and pick the first
            region_top_left[region] = top_left_pixel

    # Sort regions by top-left-most pixel
    sorted_regions = sorted(region_top_left.items(), key=lambda x: (x[1][0], x[1][1]))  # Sort by (y, x)

    # Reassign node IDs
    new_region_to_node = {}
    new_node_id = 1
    for region, _ in sorted_regions:
        if region in region_to_node:
            new_region_to_node[region] = new_node_id
            new_node_id += 1
    
    # Adjust for gnd_regions
    gnd_nodes = [new_region_to_node[region] for region in gnd_regions if region in new_region_to_node]
    if gnd_nodes:  # If there are any ground nodes
        # Find the smallest node in gnd_nodes
        smallest_gnd_node = min(gnd_nodes)
        
        # Update new_region_to_node so that all gnd_regions point to the smallest_gnd_node
        for region in gnd_regions:
            if region in new_region_to_node:
                new_region_to_node[region] = smallest_gnd_node

    # Create a new text file for node positions
    results_file = os.path.join(results_path, os.path.splitext(image_file)[0] + '.txt')
    with open(results_file, 'w') as results:
        # Dictionary to keep track of label counts
        label_counts = {}

        for component in components:
            # Skip GND components
            if component["label"].upper() == "GND":
                continue

            connected_nodes = []
            for point in component["connection_points"]:
                px, py = point
                if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
                    continue

                is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
                if is_connected:
                    connected_px, connected_py = connection_point
                    region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
                    if region > 0 and region in new_region_to_node:
                        node_id = new_region_to_node[region]
                        connected_nodes.append(node_id)

            # Ensure we only write components that have at least one connected node
            connected_nodes = list(set(connected_nodes))
            if connected_nodes:  # Check if there are any connected nodes
                unique_label = component["label"]

                # Increment the count for the current label
                if unique_label not in label_counts:
                    label_counts[unique_label] = 0
                label_counts[unique_label] += 1

                # Create a new label with numbering
                numbered_label = f"{unique_label}_{label_counts[unique_label]}"

                # Write to the results file
                results.write(f"{numbered_label} {' '.join(map(str, connected_nodes))}\n")

def process_all_images(images_folder, model_path, results_path):
    os.makedirs(results_path, exist_ok=True)

    model = YOLO(model_path)

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)

        results = model(image_path)[0]
        circuit_info = []

        for result in results:
            for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
                                             result.keypoints.xy.cpu().numpy(),
                                             result.boxes.xyxy.cpu().numpy()):
                class_idx = int(cls)
                object_name = results.names[class_idx]

                x_min, y_min, x_max, y_max = map(int, bbox)
                bounding_box = [x_min, y_min, x_max, y_max]

                connection_points = [
                    [int(point[0]), int(point[1])] for point in keypoints if not (point[0] == 0 and point[1] == 0)
                ]

                circuit_info.append({
                    "label": object_name,
                    "bounding_box": bounding_box,
                    "connection_points": connection_points
                })

        connected_edges = detect_edges(image_path)
        masked_edges = mask_components(connected_edges, circuit_info)
        overlay_and_find_nodes_with_connected_regions(
             masked_edges, circuit_info, results_path, image_file)

if __name__ == '__main__':
    current_path = os.getcwd()
    pose_folder = os.path.join(os.path.dirname(os.getcwd()), 'Current trained model/pose')
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

    images_folder = os.path.join(current_path, 'Images/')
    results_path = os.path.join(current_path, 'Results/')

    process_all_images(images_folder, latest_train_path, results_path)