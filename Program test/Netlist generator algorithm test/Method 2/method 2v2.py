

import os
import cv2
import numpy as np
from ultralytics import YOLO
import json
import random
from collections import defaultdict

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    


def identify_circuit_nodes(merged_labels, components, output_folder):
    """
    Identify circuit nodes by segmenting the merged label using component bounding boxes.
    Filter to find real nodes where two or more circuit components are connected.
    Identify connection points between components and nodes, and write to a text file.
    
    Args:
        merged_labels: Binary image where all circuit traces are marked as 1
        components: List of component dictionaries with bounding box information
        output_folder: Path to the output folder for component connections
    
    Returns:
        node_labels: Label matrix where each circuit node has a unique label ID
        node_info: Dictionary with information about each node
        node_vis: Visualization of circuit nodes
        node_vis_with_boxes: Visualization with component bounding boxes
        circuit_cut: Binary image showing where the circuit was cut
        overlap_debug: Debug visualization showing overlap detection
        real_nodes: Set of node IDs that are real nodes (connected to 2+ components)
        real_node_info: Dictionary with information about real nodes
        real_node_vis: Visualization of real nodes
        real_node_vis_with_boxes: Visualization of real nodes with component bounding boxes
        component_connections: Dictionary mapping component IDs to lists of connection points
    """
    h, w = merged_labels.shape
    
    # Create a copy of the merged labels
    circuit_image = merged_labels.copy()
    
    # Create a mask for all component bounding boxes (slightly reduced)
    component_mask = np.zeros_like(circuit_image)
    
    # Track the reduced bounding boxes for each component
    component_reduced_boxes = []
    
    # The amount to shrink bounding boxes by (in pixels)
    shrink_amount = 3
    
    for component in components:
        x_min, y_min, x_max, y_max = component["bbox"]
        
        # Reduce the bounding box slightly to avoid cutting at component boundaries
        x_min_reduced = x_min + shrink_amount
        y_min_reduced = y_min + shrink_amount
        x_max_reduced = x_max - shrink_amount
        y_max_reduced = y_max - shrink_amount
        
        # Make sure the reduced box is valid
        if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
            # Record the reduced bounding box
            component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
            # Fill the mask with 1s for this component's reduced bounding box
            component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
    # Cut the circuit at component bounding boxes
    # (Set circuit pixels to 0 where component_mask is 1)
    circuit_cut = circuit_image.copy()
    circuit_cut[component_mask == 1] = 0
    
    # Now perform connected components analysis on the cut circuit
    num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
    # Create a debug visualization to check overlap detection
    overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Set up colors for debug visualization
    # Circuit traces - green
    overlap_debug[circuit_image > 0] = [0, 255, 0]
    
    # Check which nodes connect to component bounding boxes
    # (These are the "real" nodes that we want to keep)
    valid_nodes = set()
    node_to_components = defaultdict(set)
    component_to_nodes = defaultdict(set)
    
    # Check every pixel in the image (exhaustive but guaranteed to catch all overlaps)
    for i, component in enumerate(components):
        x_min, y_min, x_max, y_max = component["bbox"]
        comp_id = component["id"]
        
        # Draw component bounding box on debug image - blue
        cv2.rectangle(overlap_debug, (x_min, y_min), (x_max, y_max), [255, 0, 0], 1)
        
        # Create a mask for just this component's bounding box perimeter
        # This ensures we only check the perimeter, not the interior
        comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        
        # Top and bottom edges
        if y_min >= 0 and y_min < h:
            comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
        if y_max >= 0 and y_max < h:
            comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
            
        # Left and right edges
        if x_min >= 0 and x_min < w:
            comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
        if x_max >= 0 and x_max < w:
            comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
        
        # Get node labels that intersect with this component's perimeter
        for label_id in range(1, num_labels):  # Skip background (0)
            # Check if this node intersects with the component perimeter
            node_mask = (node_labels == label_id)
            intersection = np.logical_and(comp_perimeter, node_mask)
            
            if np.any(intersection):
                # This node intersects with the component's perimeter
                valid_nodes.add(label_id)
                node_to_components[int(label_id)].add(comp_id)
                component_to_nodes[comp_id].add(int(label_id))
                
                # For debug visualization - red for detected intersections
                overlap_debug[intersection] = [0, 0, 255]
    
    # Create a visualization of the circuit nodes
    node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Generate a color for each valid node
    node_colors = {}
    for node_id in valid_nodes:
        node_colors[node_id] = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255)
        )
    
    # Color each node
    for node_id in valid_nodes:
        node_vis[node_labels == node_id] = node_colors[node_id]
    
    # Overlay the original component bounding boxes
    node_vis_with_boxes = node_vis.copy()
    for component in components:
        x_min, y_min, x_max, y_max = component["bbox"]
        cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
    # Collect node information
    node_info = {
        "valid_nodes": list(valid_nodes),
        "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
        "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
        "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
    }
    
    # Find real nodes (nodes where two or more circuit components are connected)
    real_nodes = set()
    for node_id, components_list in node_to_components.items():
        if len(components_list) >= 2:
            real_nodes.add(node_id)
    
    # Create a filtered node_info dictionary for real nodes
    real_node_info = {
        "valid_nodes": list(real_nodes),
        "node_to_components": {int(k): list(v) for k, v in node_to_components.items() if k in real_nodes},
        "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items() if k in real_nodes}
    }
    
    # Update component_to_nodes mapping to include only real nodes
    component_to_real_nodes = {}
    for comp_id, nodes in component_to_nodes.items():
        real_comp_nodes = [node_id for node_id in nodes if node_id in real_nodes]
        if real_comp_nodes:
            component_to_real_nodes[comp_id] = real_comp_nodes
    
    real_node_info["component_to_nodes"] = component_to_real_nodes
    
    # Create a visualization of real nodes
    real_node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color each real node using its original color
    for node_id in real_nodes:
        if node_id in node_colors:
            real_node_vis[node_labels == node_id] = node_colors[node_id]
            
    # Create a visualization of real nodes with component bounding boxes
    real_node_vis_with_boxes = real_node_vis.copy()
    for component in components:
        x_min, y_min, x_max, y_max = component["bbox"]
        cv2.rectangle(real_node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
    # Find connection points between components and real nodes
    component_connections = {}
    
    # For each component, find its connections to real nodes
    for component in components:
        comp_id = component["id"]
        x_min, y_min, x_max, y_max = component["bbox"]
        
        # Skip if component is not connected to any real nodes
        if comp_id not in real_node_info["component_to_nodes"]:
            continue
        
        # Get the real nodes this component connects to
        connected_real_nodes = real_node_info["component_to_nodes"][comp_id]
        
        # Find connection points for each real node
        connections = []
        for node_id in connected_real_nodes:
            # Create a mask for this node
            node_mask = (node_labels == node_id)
            
            # Create a mask for component perimeter
            comp_perimeter = np.zeros_like(node_mask)
            
            # Top and bottom edges
            if y_min >= 0 and y_min < h:
                comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
            if y_max >= 0 and y_max < h:
                comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
                
            # Left and right edges
            if x_min >= 0 and x_min < w:
                comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
            if x_max >= 0 and x_max < w:
                comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
            
            # Find intersection points
            intersection = np.logical_and(node_mask, comp_perimeter)
            intersection_points = np.where(intersection)
            
            if len(intersection_points[0]) > 0:
                # Calculate the middle connection point
                y_points = intersection_points[0]
                x_points = intersection_points[1]
                
                # Get the middle point
                middle_idx = len(x_points) // 2
                middle_y = y_points[middle_idx]
                middle_x = x_points[middle_idx]
                
                connections.append((middle_x, middle_y))
        
        # Store the connections for this component
        component_connections[comp_id] = connections
    
    # Write component connections to a text file
    import os
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create the file path by joining the folder path with a filename
    output_file = os.path.join(output_folder, "component_connections.txt")
    
    try:
        with open(output_file, 'w') as f:
            for comp_id, points in component_connections.items():
                connection_str = f"{comp_id} "
                for i, (x, y) in enumerate(points):
                    if i > 0:
                        connection_str += ", "
                    connection_str += f"({x}, {y})"
                f.write(connection_str + "\n")
    except Exception as e:
        print(f"Warning: Could not write to {output_file}: {e}")


def process_circuit_image(image_path, model_path, output_folder, first):
    """Process a circuit image to identify components and their connections."""
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load model and image
    model = YOLO(model_path)
    original_image = cv2.imread(image_path)
    
    if original_image is None:
        print(f"Error: Could not read image at {image_path}")
        return None, None
    
    # Create copies for visualization
    detection_image = original_image.copy()
    
    # Detect components
    results = model(image_path)[0]
    # for result in results:
    #     for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
    #                                      result.keypoints.xy.cpu().numpy(),
    #                                      result.boxes.xyxy.cpu().numpy()):
    #         class_idx = int(cls)
    #         object_name = results.names[class_idx]
    #         print(object_name)
    #         print("bbox:")
    #         print(bbox)
    #         first = False
    # Get component information
    components = []
    for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
                                       results.boxes.xyxy.cpu().numpy())):
        class_idx = int(cls)  # Convert to Python int
        class_name = results.names[class_idx]
        x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
        # Assign a unique ID to each component
        component_id = f"{class_name}_{i}"
        
        # Choose color based on class for visualization
        color = (random.randint(100, 255), 
                 random.randint(100, 255), 
                 random.randint(100, 255))
        
        # Draw bounding box
        cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(detection_image, f"{class_name}", 
                    (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, color, 2)
        
        components.append({
            "id": component_id,
            "type": class_name,
            "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
            "color": [int(c) for c in color]  # Explicit conversion
        })
    
    # Preprocess image for connected component analysis
    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to separate components and wires from background
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Save the original binary image for comparison
    cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
    # Thicken lines to ensure connections
    kernel = np.ones((3, 3), np.uint8)
    thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
    # Save the thickened binary image for comparison
    cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
    # Create a merged label image where all foreground pixels are set to 1
    merged_labels = np.zeros_like(thickened_binary)
    merged_labels[thickened_binary > 0] = 1
    
    # Save the merged label image
    merged_label_vis = np.zeros_like(original_image)
    merged_label_vis[merged_labels > 0] = (0, 255, 0)  # Green for all circuit traces
    cv2.imwrite(os.path.join(output_folder, "merged_labels.png"), merged_label_vis)
    
    # Identify circuit nodes
    # node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug = identify_circuit_nodes(merged_labels, components)
    # node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug, real_nodes, real_node_info, real_node_vis = identify_circuit_nodes(merged_labels, components)
    identify_circuit_nodes(merged_labels, components, output_folder)

def process_all_images(images_folder, model_path, output_folder):
    """Process all circuit images in a folder."""
    os.makedirs(output_folder, exist_ok=True)
    
    image_files = [f for f in os.listdir(images_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    first = True
    for image_file in image_files:
        print(f"Processing {image_file}...")
        image_path = os.path.join(images_folder, image_file)
        
        # Create output folder for this image
        image_output_folder = os.path.join(output_folder, 
                                          os.path.splitext(image_file)[0])
        os.makedirs(image_output_folder, exist_ok=True)
        
        # Process the image
        try:
            process_circuit_image(image_path, model_path, image_output_folder, first)
            first = False
        except Exception as e:
            print(f"  Error processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    parent_dir = os.path.dirname(os.getcwd()) # Parent directory
    PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
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

    parent_dir = os.path.dirname(os.getcwd()) # Parent directory
    test_images_folder = os.path.join(parent_dir, 'Test images/')
    output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
    test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

    process_all_images(test_images_folder, latest_train_path, output_files_path)




