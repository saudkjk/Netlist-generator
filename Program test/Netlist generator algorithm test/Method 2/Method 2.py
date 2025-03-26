
# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from scipy.ndimage import label as connected_label
# from scipy.spatial import distance
# import json

# # Define helper functions
# def detect_edges(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grayscale, 50, 150)
#     kernel = np.ones((5, 5), np.uint8)
#     connected_edges = cv2.dilate(edges, kernel, iterations=2)

#     return connected_edges

# def mask_components(connected_edges, components):
#     masked_edges = connected_edges.copy()
#     for component in components:
#         bbox = component["bounding_box"]
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)

#     return masked_edges

# def find_nearest_edge(masked_edges, px, py):
#     edge_points = np.argwhere(masked_edges > 0)  # Get all edge points
#     if edge_points.size == 0:
#         return None  # No edges in the mask
#     distances = distance.cdist([(py, px)], edge_points)  # Compute distances from the point to all edges
#     nearest_index = np.argmin(distances)  # Index of the nearest edge

#     return edge_points[nearest_index]  # Coordinates of the nearest edge (y, x)

# def is_point_connected_or_nearest(masked_edges, px, py):

#     # Find the nearest edge
#     nearest_edge = find_nearest_edge(masked_edges, px, py)
#     if nearest_edge is not None:
#         return True, (nearest_edge[1], nearest_edge[0])  # Return x, y of the nearest edge

#     print(f"No connection found for point ({px}, {py})")  # Debug message if something went wrong
#     return False, None  # No connection and no nearest edge

# def overlay_and_find_nodes_with_connected_regions(connected_edges, masked_edges, components, test_results_path, image_file, output_files_path):
#     labeled_edges, num_regions = connected_label(masked_edges)
#     region_to_node = {}
#     current_node_id = 1
#     gnd_regions = set()

#     for component in components:
#         for point in component["connection_points"]:
#             px, py = point
#             if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                 continue

#             is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#             if is_connected:
#                 connected_px, connected_py = connection_point
#                 region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                 if region > 0:
#                     if region not in region_to_node:
#                         region_to_node[region] = current_node_id
#                         current_node_id += 1
#                     if component["label"].upper() == "GND":
#                         gnd_regions.add(region)

#     # Rearrange node IDs based on top-left-most pixel
#     region_top_left = {}
#     for region in range(1, num_regions + 1):
#         pixels = np.argwhere(labeled_edges == region)  # Get all pixels in the region
#         if len(pixels) > 0:
#             top_left_pixel = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))][0]  # Sort by (y, then x) and pick the first
#             region_top_left[region] = top_left_pixel

#     # Sort regions by top-left-most pixel
#     sorted_regions = sorted(region_top_left.items(), key=lambda x: (x[1][0], x[1][1]))  # Sort by (y, x)
#     # Reassign node IDs
#     new_region_to_node = {}
#     new_node_id = 1
#     for region, _ in sorted_regions:
#         if region in region_to_node:
#             new_region_to_node[region] = new_node_id
#             new_node_id += 1

#     # Adjust for gnd_regions
#     gnd_nodes = [new_region_to_node[region] for region in gnd_regions if region in new_region_to_node]
#     if gnd_nodes:  # If there are any ground nodes
#         # Find the smallest node in gnd_nodes
#         smallest_gnd_node = min(gnd_nodes)
        
#         # Update new_region_to_node so that all gnd_regions point to the smallest_gnd_node
#         for region in gnd_regions:
#             if region in new_region_to_node:
#                 new_region_to_node[region] = smallest_gnd_node

#     # Create a new text file for node positions
#     results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0] + '.txt')
#     with open(results_file, 'w') as results:
#         # Dictionary to keep track of label counts
#         label_counts = {}

#         for component in components:
#             # Skip GND components
#             if component["label"].upper() == "GND":
#                 continue
#             connected_nodes = []
#             for point in component["connection_points"]:
#                 px, py = point
#                 if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                     continue
#                 is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#                 if is_connected:
#                     connected_px, connected_py = connection_point
#                     region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                     if region > 0 and region in new_region_to_node:
#                         node_id = new_region_to_node[region]
#                         connected_nodes.append(node_id)
#             # Ensure we only write components that have at least one connected node
#             connected_nodes = list(set(connected_nodes))
#             if connected_nodes:  # Check if there are any connected nodes
#                 unique_label = component["label"]
#                 # Increment the count for the current label
#                 if unique_label not in label_counts:
#                     label_counts[unique_label] = 0
#                 label_counts[unique_label] += 1
#                 # Create a new label with numbering
#                 numbered_label = f"{unique_label}_{label_counts[unique_label]}"
#                 # Write to the results file
#                 results.write(f"{numbered_label} {' '.join(map(str, connected_nodes))}\n")

# def process_all_images(images_folder, model_path, results_path, test_results_path):
#     os.makedirs(results_path, exist_ok=True)

#     model = YOLO(model_path)

#     image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     for image_file in image_files:
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_files_path, os.path.splitext(image_file)[0])
#         json_path = os.path.join(image_output_folder, 'circuit_info.json')

#         results = model(image_path)[0]
#         circuit_info = []

#         for result in results:
#             for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
#                                              result.keypoints.xy.cpu().numpy(),
#                                              result.boxes.xyxy.cpu().numpy()):
#                 class_idx = int(cls)
#                 object_name = results.names[class_idx]

#                 x_min, y_min, x_max, y_max = map(int, bbox)
#                 bounding_box = [x_min, y_min, x_max, y_max]

#                 connection_points = [
#                     [int(point[0]), int(point[1])] for point in keypoints if not (point[0] == 0 and point[1] == 0)
#                 ]

#                 circuit_info.append({
#                     "label": object_name,
#                     "bounding_box": bounding_box,
#                     "connection_points": connection_points
#                 })

#         # connected_edges = detect_edges(image_path)
#         # masked_edges = mask_components(connected_edges, circuit_info)
#         # overlay_and_find_nodes_with_connected_regions(
#         #      masked_edges, circuit_info, results_path, image_file)
#         with open(json_path, 'w') as json_file:
#             json.dump(circuit_info, json_file, indent=4)
            
#         with open(json_path, "r") as json_file:
#             components = json.load(json_file)

#         connected_edges = detect_edges(image_path)
#         masked_edges = mask_components(connected_edges, components)
#         overlay_and_find_nodes_with_connected_regions(
#             connected_edges, masked_edges, components, test_results_path, image_file, output_files_path)

# if __name__ == '__main__':
#     # current_path = os.getcwd()
#     # pose_folder = os.path.join(os.path.dirname(os.getcwd()), 'Current trained model/pose')
#     # train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     # images_folder = os.path.join(current_path, 'Images/')
#     # results_path = os.path.join(current_path, 'Results/')

#     # process_all_images(images_folder, latest_train_path, results_path)

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path, test_results_path)










# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from scipy.ndimage import label as connected_label
# from scipy.spatial import distance
# import json

# # Define helper functions
# def detect_edges(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grayscale, 50, 150)
#     kernel = np.ones((5, 5), np.uint8)
#     connected_edges = cv2.dilate(edges, kernel, iterations=2)

#     return connected_edges

# def mask_components(connected_edges, components):
#     masked_edges = connected_edges.copy()
#     for component in components:
#         bbox = component["bounding_box"]
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)

#     return masked_edges

# def find_nearest_edge(masked_edges, px, py):
#     edge_points = np.argwhere(masked_edges > 0)  # Get all edge points
#     if edge_points.size == 0:
#         return None  # No edges in the mask
#     distances = distance.cdist([(py, px)], edge_points)  # Compute distances from the point to all edges
#     nearest_index = np.argmin(distances)  # Index of the nearest edge

#     return edge_points[nearest_index]  # Coordinates of the nearest edge (y, x)

# def is_point_connected_or_nearest(masked_edges, px, py):

#     # Find the nearest edge
#     nearest_edge = find_nearest_edge(masked_edges, px, py)
#     if nearest_edge is not None:
#         return True, (nearest_edge[1], nearest_edge[0])  # Return x, y of the nearest edge

#     print(f"No connection found for point ({px}, {py})")  # Debug message if something went wrong
#     return False, None  # No connection and no nearest edge

# def overlay_and_find_nodes_with_connected_regions(connected_edges, masked_edges, components, test_results_path, image_file, output_files_path):
#     labeled_edges, num_regions = connected_label(masked_edges)
#     region_to_node = {}
#     current_node_id = 1
#     gnd_regions = set()

#     for component in components:
#         for point in component["connection_points"]:
#             px, py = point
#             if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                 continue

#             is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#             if is_connected:
#                 connected_px, connected_py = connection_point
#                 region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                 if region > 0:
#                     if region not in region_to_node:
#                         region_to_node[region] = current_node_id
#                         current_node_id += 1
#                     if component["label"].upper() == "GND":
#                         gnd_regions.add(region)

#     # Rearrange node IDs based on top-left-most pixel
#     region_top_left = {}
#     for region in range(1, num_regions + 1):
#         pixels = np.argwhere(labeled_edges == region)  # Get all pixels in the region
#         if len(pixels) > 0:
#             top_left_pixel = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))][0]  # Sort by (y, then x) and pick the first
#             region_top_left[region] = top_left_pixel

#     # Sort regions by top-left-most pixel
#     sorted_regions = sorted(region_top_left.items(), key=lambda x: (x[1][0], x[1][1]))  # Sort by (y, x)
#     # Reassign node IDs
#     new_region_to_node = {}
#     new_node_id = 1
#     for region, _ in sorted_regions:
#         if region in region_to_node:
#             new_region_to_node[region] = new_node_id
#             new_node_id += 1

#     # Adjust for gnd_regions
#     gnd_nodes = [new_region_to_node[region] for region in gnd_regions if region in new_region_to_node]
#     if gnd_nodes:  # If there are any ground nodes
#         # Find the smallest node in gnd_nodes
#         smallest_gnd_node = min(gnd_nodes)
        
#         # Update new_region_to_node so that all gnd_regions point to the smallest_gnd_node
#         for region in gnd_regions:
#             if region in new_region_to_node:
#                 new_region_to_node[region] = smallest_gnd_node

#     # Create a new text file for node positions
#     results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0] + '.txt')
#     with open(results_file, 'w') as results:
#         # Dictionary to keep track of label counts
#         label_counts = {}

#         for component in components:
#             # Skip GND components
#             if component["label"].upper() == "GND":
#                 continue
#             connected_nodes = []
#             for point in component["connection_points"]:
#                 px, py = point
#                 if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                     continue
#                 is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#                 if is_connected:
#                     connected_px, connected_py = connection_point
#                     region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                     if region > 0 and region in new_region_to_node:
#                         node_id = new_region_to_node[region]
#                         connected_nodes.append(node_id)
#             # Ensure we only write components that have at least one connected node
#             connected_nodes = list(set(connected_nodes))
#             if connected_nodes:  # Check if there are any connected nodes
#                 unique_label = component["label"]
#                 # Increment the count for the current label
#                 if unique_label not in label_counts:
#                     label_counts[unique_label] = 0
#                 label_counts[unique_label] += 1
#                 # Create a new label with numbering
#                 numbered_label = f"{unique_label}_{label_counts[unique_label]}"
#                 # Write to the results file
#                 results.write(f"{numbered_label} {' '.join(map(str, connected_nodes))}\n")

# def process_all_images(images_folder, model_path, results_path, test_results_path):
#     os.makedirs(results_path, exist_ok=True)

#     model = YOLO(model_path)

#     image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     for image_file in image_files:
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_files_path, os.path.splitext(image_file)[0])
#         json_path = os.path.join(image_output_folder, 'circuit_info.json')

#         results = model(image_path)[0]
#         circuit_info = []

#         for result in results:
#             for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
#                                              result.keypoints.xy.cpu().numpy(),
#                                              result.boxes.xyxy.cpu().numpy()):
#                 class_idx = int(cls)
#                 object_name = results.names[class_idx]

#                 x_min, y_min, x_max, y_max = map(int, bbox)
#                 bounding_box = [x_min, y_min, x_max, y_max]

#                 # json: replace connection_points with image processing to identify each connection point.
#                 connection_points = [
#                     [int(point[0]), int(point[1])] for point in keypoints if not (point[0] == 0 and point[1] == 0)
#                 ]
#                 print("connecttion points are: ********************")
#                 print(connection_points)
#                 print("done with connection points ********************")

#                 circuit_info.append({
#                     "label": object_name,
#                     "bounding_box": bounding_box,
#                     "connection_points": connection_points
#                 })

#         with open(json_path, 'w') as json_file:
#             json.dump(circuit_info, json_file, indent=4)
            
#         with open(json_path, "r") as json_file:
#             components = json.load(json_file)

#         connected_edges = detect_edges(image_path)
#         masked_edges = mask_components(connected_edges, components)
#         overlay_and_find_nodes_with_connected_regions(
#             connected_edges, masked_edges, components, test_results_path, image_file, output_files_path)

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path, test_results_path)































# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from scipy.ndimage import label as connected_label
# from scipy.spatial import distance
# import json

# # Define helper functions
# def detect_edges(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grayscale, 50, 150)
#     kernel = np.ones((5, 5), np.uint8)
#     connected_edges = cv2.dilate(edges, kernel, iterations=2)

#     return connected_edges

# def mask_components(connected_edges, components):
#     masked_edges = connected_edges.copy()
#     for component in components:
#         bbox = component["bounding_box"]
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)

#     return masked_edges

# def find_nearest_edge(masked_edges, px, py):
#     edge_points = np.argwhere(masked_edges > 0)  # Get all edge points
#     if edge_points.size == 0:
#         return None  # No edges in the mask
#     distances = distance.cdist([(py, px)], edge_points)  # Compute distances from the point to all edges
#     nearest_index = np.argmin(distances)  # Index of the nearest edge

#     return edge_points[nearest_index]  # Coordinates of the nearest edge (y, x)

# def is_point_connected_or_nearest(masked_edges, px, py):

#     # Find the nearest edge
#     nearest_edge = find_nearest_edge(masked_edges, px, py)
#     if nearest_edge is not None:
#         return True, (nearest_edge[1], nearest_edge[0])  # Return x, y of the nearest edge

#     print(f"No connection found for point ({px}, {py})")  # Debug message if something went wrong
#     return False, None  # No connection and no nearest edge

# def overlay_and_find_nodes_with_connected_regions(connected_edges, masked_edges, components, test_results_path, image_file, output_files_path):
#     labeled_edges, num_regions = connected_label(masked_edges)
#     region_to_node = {}
#     current_node_id = 1
#     gnd_regions = set()

#     for component in components:
#         for point in component["connection_points"]:
#             px, py = point
#             if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                 continue

#             is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#             if is_connected:
#                 connected_px, connected_py = connection_point
#                 region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                 if region > 0:
#                     if region not in region_to_node:
#                         region_to_node[region] = current_node_id
#                         current_node_id += 1
#                     if component["label"].upper() == "GND":
#                         gnd_regions.add(region)

#     # Rearrange node IDs based on top-left-most pixel
#     region_top_left = {}
#     for region in range(1, num_regions + 1):
#         pixels = np.argwhere(labeled_edges == region)  # Get all pixels in the region
#         if len(pixels) > 0:
#             top_left_pixel = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))][0]  # Sort by (y, then x) and pick the first
#             region_top_left[region] = top_left_pixel

#     # Sort regions by top-left-most pixel
#     sorted_regions = sorted(region_top_left.items(), key=lambda x: (x[1][0], x[1][1]))  # Sort by (y, x)
#     # Reassign node IDs
#     new_region_to_node = {}
#     new_node_id = 1
#     for region, _ in sorted_regions:
#         if region in region_to_node:
#             new_region_to_node[region] = new_node_id
#             new_node_id += 1

#     # Adjust for gnd_regions
#     gnd_nodes = [new_region_to_node[region] for region in gnd_regions if region in new_region_to_node]
#     if gnd_nodes:  # If there are any ground nodes
#         # Find the smallest node in gnd_nodes
#         smallest_gnd_node = min(gnd_nodes)
        
#         # Update new_region_to_node so that all gnd_regions point to the smallest_gnd_node
#         for region in gnd_regions:
#             if region in new_region_to_node:
#                 new_region_to_node[region] = smallest_gnd_node

#     # Create a new text file for node positions
#     results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0] + '.txt')
#     with open(results_file, 'w') as results:
#         # Dictionary to keep track of label counts
#         label_counts = {}

#         for component in components:
#             # Skip GND components
#             if component["label"].upper() == "GND":
#                 continue
#             connected_nodes = []
#             for point in component["connection_points"]:
#                 px, py = point
#                 if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                     continue
#                 is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#                 if is_connected:
#                     connected_px, connected_py = connection_point
#                     region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                     if region > 0 and region in new_region_to_node:
#                         node_id = new_region_to_node[region]
#                         connected_nodes.append(node_id)
#             # Ensure we only write components that have at least one connected node
#             connected_nodes = list(set(connected_nodes))
#             if connected_nodes:  # Check if there are any connected nodes
#                 unique_label = component["label"]
#                 # Increment the count for the current label
#                 if unique_label not in label_counts:
#                     label_counts[unique_label] = 0
#                 label_counts[unique_label] += 1
#                 # Create a new label with numbering
#                 numbered_label = f"{unique_label}_{label_counts[unique_label]}"
#                 # Write to the results file
#                 results.write(f"{numbered_label} {' '.join(map(str, connected_nodes))}\n")

# # def process_all_images(images_folder, model_path, results_path, test_results_path):
# #     os.makedirs(results_path, exist_ok=True)

# #     model = YOLO(model_path)

# #     image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# #     for image_file in image_files:
# #         image_path = os.path.join(images_folder, image_file)
# #         image_output_folder = os.path.join(output_files_path, os.path.splitext(image_file)[0])

# #         results = model(image_path)[0]
# #         circuit_info = []

# #         for result in results:
# #             for cls, bbox in zip(result.boxes.cls.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
# #                 class_idx = int(cls)
# #                 object_name = results.names[class_idx]

# #                 x_min, y_min, x_max, y_max = map(int, bbox)
# #                 bounding_box = [x_min, y_min, x_max, y_max]

# #                 # TODO: draw the bouunding box over the image and save the image in image_output_folder

# #                 # TODO: color the contents inside the bounding box with a unique color for each object and save the image in image_output_folder
                
# #                 # TODO: figure out the intersection between thte bouning box and the colored object and the intersetion would be the connection point draw them on the image and save the image in image_output_folder

# #                 # TODO: fill the connection_points with the intersection points
# #                 connection_points = ...

# #                 print(connection_points)

# #                 circuit_info.append({
# #                     "label": object_name,
# #                     "bounding_box": bounding_box,
# #                     "connection_points": connection_points
# #                 })



# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# def process_all_images(images_folder, model_path, results_path, test_results_path):
#     import os
#     import cv2
#     import numpy as np
#     from ultralytics import YOLO
#     import random
#     import json
    
#     # Custom JSON encoder to handle NumPy types
#     class NumpyEncoder(json.JSONEncoder):
#         def default(self, obj):
#             if isinstance(obj, np.integer):
#                 return int(obj)
#             if isinstance(obj, np.floating):
#                 return float(obj)
#             if isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             return super(NumpyEncoder, self).default(obj)
    
#     os.makedirs(results_path, exist_ok=True)
#     model = YOLO(model_path)
    
#     # Define color dictionary to use consistent colors for each object type
#     color_dict = {}
    
#     image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     for image_file in image_files:
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(results_path, os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Load the image
#         original_image = cv2.imread(image_path)
#         segmentation_image = original_image.copy()
#         connection_image = original_image.copy()
        
#         results = model(image_path)[0]
#         circuit_info = []
        
#         for result in results:
#             for cls, bbox in zip(result.boxes.cls.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
#                 class_idx = int(cls)
#                 object_name = results.names[class_idx]
#                 x_min, y_min, x_max, y_max = map(int, bbox)
#                 bounding_box = [int(x_min), int(y_min), int(x_max), int(y_max)]  # Explicitly convert to Python int
                
#                 # Generate a unique color for each object type if not already defined
#                 if object_name not in color_dict:
#                     color_dict[object_name] = (
#                         random.randint(0, 255),
#                         random.randint(0, 255),
#                         random.randint(0, 255)
#                     )
#                 color = color_dict[object_name]
                
#                 # Draw bounding box on the original image
#                 cv2.rectangle(original_image, (x_min, y_min), (x_max, y_max), color, 2)
#                 cv2.putText(original_image, object_name, (x_min, y_min - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
#                 # Create a mask for the component within the bounding box
#                 component_mask = np.zeros(segmentation_image.shape[:2], dtype=np.uint8)
#                 roi = segmentation_image[y_min:y_max, x_min:x_max]
                
#                 # Convert ROI to grayscale
#                 roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
#                 # Apply threshold to identify the component
#                 _, component_binary = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)
                
#                 # Find contours in the binary image
#                 contours, _ = cv2.findContours(component_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
#                 # Fill the component mask
#                 for contour in contours:
#                     # Adjust contour coordinates to the original image
#                     adjusted_contour = contour.copy()
#                     adjusted_contour[:, :, 0] += x_min
#                     adjusted_contour[:, :, 1] += y_min
#                     cv2.drawContours(component_mask, [adjusted_contour], -1, 255, -1)
                
#                 # Apply the mask to color the component
#                 colored_component = np.zeros_like(segmentation_image)
#                 colored_component[component_mask == 255] = color
                
#                 # Blend the colored component with the original image
#                 alpha = 0.6  # Transparency factor
#                 mask_3d = cv2.cvtColor(component_mask, cv2.COLOR_GRAY2BGR)
#                 segmentation_image = np.where(mask_3d > 0, 
#                                             cv2.addWeighted(segmentation_image, 1-alpha, colored_component, alpha, 0),
#                                             segmentation_image)
                
#                 # Identify boundary pixels of the component
#                 kernel = np.ones((3, 3), np.uint8)
#                 dilated = cv2.dilate(component_mask, kernel, iterations=1)
#                 boundary = dilated - component_mask
                
#                 # Define rectangular contour for the bounding box
#                 rect_points = np.array([
#                     [x_min, y_min],
#                     [x_max, y_min],
#                     [x_max, y_max],
#                     [x_min, y_max]
#                 ])
                
#                 # Create a mask for the bounding box
#                 bbox_mask = np.zeros_like(component_mask)
#                 cv2.drawContours(bbox_mask, [rect_points], -1, 255, 2)
                
#                 # Find intersection between component boundary and bounding box
#                 intersection = np.logical_and(boundary, bbox_mask).astype(np.uint8) * 255
                
#                 # Extract connection points
#                 connection_points_indices = np.where(intersection > 0)
#                 connection_points = []
                
#                 # Limiting to a reasonable number of connection points
#                 step = max(1, len(connection_points_indices[0]) // 10)  # Take at most ~10 points
#                 for i in range(0, len(connection_points_indices[0]), step):
#                     y, x = connection_points_indices[0][i], connection_points_indices[1][i]
#                     # Convert numpy int64 to Python int for JSON serialization
#                     connection_points.append((int(x), int(y)))
                    
#                     # Draw connection points on the connection image
#                     cv2.circle(connection_image, (int(x), int(y)), 3, (0, 0, 255), -1)
                
#                 circuit_info.append({
#                     "label": object_name,
#                     "bounding_box": bounding_box,
#                     "connection_points": connection_points
#                 })
        
#         # Save all generated images
#         cv2.imwrite(os.path.join(image_output_folder, "bbox_" + image_file), original_image)
#         cv2.imwrite(os.path.join(image_output_folder, "segmentation_" + image_file), segmentation_image)
#         cv2.imwrite(os.path.join(image_output_folder, "connections_" + image_file), connection_image)
        
#         # Save circuit info to a JSON file using the custom encoder
#         with open(os.path.join(image_output_folder, "circuit_info.json"), "w") as f:
#             json.dump(circuit_info, f, indent=2, cls=NumpyEncoder)
        
#         # Display performance metrics in test results
#         with open(os.path.join(test_results_path, f"test_results_{image_file}.txt"), "w") as f:
#             f.write(f"Processed {image_file}\n")
#             f.write(f"Detected {len(circuit_info)} components\n")
#             for item in circuit_info:
#                 f.write(f"Component: {item['label']}, Connection points: {len(item['connection_points'])}\n")



# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path, test_results_path)









# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# # Custom JSON encoder to handle NumPy types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)
    


# def identify_main_circuit_label(labels, components):
#     """
#     Identify the main circuit label that connects most components.
    
#     Args:
#         labels: The connected components label matrix
#         components: List of component dictionaries
        
#     Returns:
#         main_label: The label ID that likely represents the main circuit
#         connection_counts: Dictionary with counts of how many components each label connects
#     """
#     # Count how many bounding boxes each label intersects with
#     label_to_components = defaultdict(set)
    
#     # For each component
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Get the labels within this bounding box
#         box_labels = labels[y_min:y_max, x_min:x_max]
#         unique_labels = np.unique(box_labels)
        
#         # Skip background (0)
#         unique_labels = unique_labels[unique_labels > 0]
        
#         # Add component to each label
#         for label in unique_labels:
#             label_to_components[int(label)].add(component["id"])
    
#     # Count connections for each label
#     connection_counts = {label: len(components_set) 
#                         for label, components_set in label_to_components.items()}
    
#     # Find the label with the most connections
#     main_label = max(connection_counts.items(), key=lambda x: x[1])[0] if connection_counts else None
    
#     return main_label, connection_counts

# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
#     connection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Perform connected component analysis on thickened image
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thickened_binary, connectivity=8)
    
#     # Identify the main circuit label
#     main_circuit_label, label_connection_counts = identify_main_circuit_label(labels, components)
    
#     # Print information about the main circuit label
#     print(f"Identified main circuit label: {main_circuit_label}")
#     print(f"Label connection counts: {label_connection_counts}")
    
#     # Highlight the main circuit in a separate visualization
#     main_circuit_image = np.zeros_like(original_image)
#     if main_circuit_label is not None:
#         # Assign a bright color to the main circuit
#         main_circuit_image[labels == main_circuit_label] = (0, 255, 0)  # Bright green
    
#     # Save the main circuit visualization
#     cv2.imwrite(os.path.join(output_folder, "main_circuit.png"), main_circuit_image)
    
#     # Create a colored label image for visualization with special color for main circuit
#     label_colors = np.zeros((num_labels, 3), dtype=np.uint8)
#     # Reserve color 0 for background (black)
#     for i in range(1, num_labels):
#         if i == main_circuit_label:
#             label_colors[i] = (0, 255, 0)  # Bright green for main circuit
#         else:
#             label_colors[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

#     # Create label visualization image
#     label_image = np.zeros_like(original_image)
#     for i in range(1, num_labels):
#         label_image[labels == i] = label_colors[i]

    
#     # Map components to connected regions
#     component_to_regions = defaultdict(set)
#     region_to_components = defaultdict(set)
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Get label masks for the component's bounding box
#         component_labels = labels[y_min:y_max, x_min:x_max]
        
#         # Find all region labels that intersect with this component
#         unique_labels = np.unique(component_labels)
        
#         # Filter out background (label 0)
#         unique_labels = unique_labels[unique_labels != 0]
        
#         # Add these regions to the component
#         for label in unique_labels:
#             component_to_regions[component["id"]].add(int(label))  # Convert to Python int
#             region_to_components[int(label)].add(component["id"])  # Convert to Python int
    
#     # Determine connections between components
#     connections = []
#     processed_pairs = set()
    
#     for region, component_ids in region_to_components.items():
#         # Skip regions that only connect to one component
#         if len(component_ids) < 2:
#             continue
            
#         # Create connections between all components sharing this region
#         for comp_id1 in component_ids:
#             for comp_id2 in component_ids:
#                 if comp_id1 != comp_id2:
#                     # Create an ordered pair to avoid duplicates
#                     pair = tuple(sorted([comp_id1, comp_id2]))
                    
#                     if pair not in processed_pairs:
#                         # Find the components by ID
#                         comp1 = next(c for c in components if c["id"] == comp_id1)
#                         comp2 = next(c for c in components if c["id"] == comp_id2)
                        
#                         # Calculate connection points (use centroids of the shared region)
#                         mask = (labels == region)
#                         y_indices, x_indices = np.where(mask)
                        
#                         # Skip if no pixels found (shouldn't happen, but just in case)
#                         if len(x_indices) == 0:
#                             continue
                            
#                         # Find connection points
#                         bbox1 = comp1["bbox"]
#                         bbox2 = comp2["bbox"]
                        
#                         # Find region pixels that intersect with component bounding boxes
#                         comp1_points = []
#                         comp2_points = []
                        
#                         for x, y in zip(x_indices, y_indices):
#                             # Check if point is on the boundary of component 1
#                             if (bbox1[0] <= x <= bbox1[2] and 
#                                 (y == bbox1[1] or y == bbox1[3])) or \
#                                (bbox1[1] <= y <= bbox1[3] and 
#                                 (x == bbox1[0] or x == bbox1[2])):
#                                 comp1_points.append((int(x), int(y)))  # Convert to Python int
                                
#                             # Check if point is on the boundary of component 2
#                             if (bbox2[0] <= x <= bbox2[2] and 
#                                 (y == bbox2[1] or y == bbox2[3])) or \
#                                (bbox2[1] <= y <= bbox2[3] and 
#                                 (x == bbox2[0] or x == bbox2[2])):
#                                 comp2_points.append((int(x), int(y)))  # Convert to Python int
                        
#                         # If no intersection points found, use nearest points
#                         if not comp1_points or not comp2_points:
#                             # Use centroids or corners as fallback
#                             center1 = (int((bbox1[0] + bbox1[2]) / 2), 
#                                      int((bbox1[1] + bbox1[3]) / 2))
#                             center2 = (int((bbox2[0] + bbox2[2]) / 2), 
#                                      int((bbox2[1] + bbox2[3]) / 2))
                            
#                             # Find nearest mask point to each center
#                             if not comp1_points:
#                                 distances = [(x - center1[0])**2 + (y - center1[1])**2 
#                                            for x, y in zip(x_indices, y_indices)]
#                                 min_idx = np.argmin(distances)
#                                 comp1_points = [(int(x_indices[min_idx]), int(y_indices[min_idx]))]  # Convert to Python int
                                
#                             if not comp2_points:
#                                 distances = [(x - center2[0])**2 + (y - center2[1])**2 
#                                            for x, y in zip(x_indices, y_indices)]
#                                 min_idx = np.argmin(distances)
#                                 comp2_points = [(int(x_indices[min_idx]), int(y_indices[min_idx]))]  # Convert to Python int
                        
#                         # Take median points for stability
#                         comp1_point = comp1_points[len(comp1_points) // 2]
#                         comp2_point = comp2_points[len(comp2_points) // 2]
                        
#                         # Draw connection on visualization
#                         cv2.circle(connection_image, comp1_point, 4, (0, 0, 255), -1)
#                         cv2.circle(connection_image, comp2_point, 4, (0, 0, 255), -1)
#                         cv2.line(connection_image, comp1_point, comp2_point, (255, 0, 0), 2)
                        
#                         # Record connection
#                         connections.append({
#                             "component1": comp_id1,
#                             "component2": comp_id2,
#                             "connect_point1": [int(comp1_point[0]), int(comp1_point[1])],  # Convert to Python int
#                             "connect_point2": [int(comp2_point[0]), int(comp2_point[1])],  # Convert to Python int
#                             "region_id": int(region)  # Convert to Python int
#                         })
                        
#                         processed_pairs.add(pair)
    
#     # Generate netlist data
#     netlist_components = []
#     for comp in components:
#         netlist_components.append({
#             "id": comp["id"],
#             "type": comp["type"],
#             "position": {
#                 "x": int((comp["bbox"][0] + comp["bbox"][2]) / 2),  # Convert to Python int
#                 "y": int((comp["bbox"][1] + comp["bbox"][3]) / 2)   # Convert to Python int
#             },
#             "connections": [c for c in connections 
#                             if c["component1"] == comp["id"] or c["component2"] == comp["id"]]
#         })
    
#     # Create netlist data structure
#     netlist_data = {
#         "components": netlist_components,
#         "connections": connections
#     }
    
#     # Generate SPICE-like netlist
#     nodes = {}
#     node_counter = 1  # Start from 1 (0 is usually ground)
    
#     # Identify GND node if present
#     gnd_components = [c for c in components if "gnd" in c["type"].lower()]
#     if gnd_components:
#         # Find all components connected to ground
#         gnd_connections = []
#         for conn in connections:
#             if any(c["id"] == conn["component1"] or c["id"] == conn["component2"] 
#                   for c in gnd_components):
#                 other_comp = conn["component1"] if conn["component1"] not in [c["id"] for c in gnd_components] else conn["component2"]
#                 gnd_connections.append((other_comp, 0))  # Node 0 is ground
#                 nodes[other_comp] = 0
    
#     # Assign node numbers to all other connections
#     for conn in connections:
#         comp1, comp2 = conn["component1"], conn["component2"]
        
#         # Skip if both are already assigned to the same node
#         if comp1 in nodes and comp2 in nodes and nodes[comp1] == nodes[comp2]:
#             continue
            
#         # If one has a node, assign the other to the same node
#         if comp1 in nodes:
#             nodes[comp2] = nodes[comp1]
#         elif comp2 in nodes:
#             nodes[comp1] = nodes[comp2]
#         # Otherwise, create a new node
#         else:
#             nodes[comp1] = node_counter
#             nodes[comp2] = node_counter
#             node_counter += 1
    
#     # Generate SPICE-like netlist text
#     spice_netlist = "* Circuit netlist generated from image\n"
#     spice_netlist += f"* Source: {os.path.basename(image_path)}\n\n"
    
#     # Add component declarations
#     for comp in components:
#         comp_id = comp["id"]
#         comp_type = comp["type"]
        
#         # Find connections for this component
#         comp_connections = []
#         for conn in connections:
#             if conn["component1"] == comp_id:
#                 comp_connections.append((conn["component2"], nodes.get(conn["component2"], 999)))
#             elif conn["component2"] == comp_id:
#                 comp_connections.append((conn["component1"], nodes.get(conn["component1"], 999)))
        
#         # Format based on component type
#         if "resistor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"R{comp_id.split('_')[1]} {node_str} 1k\n"
#         elif "capacitor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"C{comp_id.split('_')[1]} {node_str} 1u\n"
#         elif "transistor" in comp_type.lower() or "bjt" in comp_type.lower():
#             # Assume 3 connections for transistors (collector, base, emitter)
#             if len(comp_connections) >= 3:
#                 nodes_str = " ".join(str(n[1]) for n in comp_connections[:3])
#                 spice_netlist += f"Q{comp_id.split('_')[1]} {nodes_str} 2N3904\n"
#         # Add more component types as needed
    
#     # Save all outputs
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Save visualization images
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_detection.png"), detection_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_labels.png"), label_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_connections.png"), connection_image)
    
#     # Save netlist data as JSON
#     with open(os.path.join(output_folder, f"{base_name}_netlist.json"), 'w') as f:
#         json.dump(netlist_data, f, indent=2, cls=NumpyEncoder)  # Use NumpyEncoder
    
#     # Save SPICE netlist
#     with open(os.path.join(output_folder, f"{base_name}.cir"), 'w') as f:
#         f.write(spice_netlist)
    
#     return netlist_data, spice_netlist

# def process_all_images(images_folder, model_path, output_folder):
#     """Process all circuit images in a folder."""
#     os.makedirs(output_folder, exist_ok=True)
    
#     image_files = [f for f in os.listdir(images_folder) 
#                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
        
#         # Create output folder for this image
#         image_output_folder = os.path.join(output_folder, 
#                                           os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Process the image
#         try:
#             netlist_data, spice_netlist = process_circuit_image(
#                 image_path, model_path, image_output_folder)
            
#             if netlist_data:
#                 print(f"  Generated netlist for {image_file}")
#                 print(f"  Output saved to {image_output_folder}")
#             else:
#                 print(f"  Failed to process {image_file}")
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path)







# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# # Custom JSON encoder to handle NumPy types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)
    
# # def identify_main_circuit_labels(labels, components):
# #     """
# #     Identify the label(s) that represent the main circuit.
    
# #     This function looks for labels that connect to component bounding boxes.
# #     It handles three possibilities:
# #     1. One label connects to all components - that's the main circuit
# #     2. Multiple labels connect to different components - all are considered part of the main circuit
# #     3. One or more components aren't connected to any label - indicates a design problem
    
# #     Args:
# #         labels: The connected components label matrix
# #         components: List of component dictionaries
        
# #     Returns:
# #         main_labels: Set of label IDs that represent the main circuit
# #         component_to_labels: Dictionary mapping component IDs to the labels they connect with
# #         disconnected_components: List of components not connected to any label
# #     """
# #     # Map labels to the components they connect with
# #     label_to_components = defaultdict(set)
    
# #     # Map components to the labels they connect with
# #     component_to_labels = defaultdict(set)
    
# #     # For each component
# #     for component in components:
# #         comp_id = component["id"]
# #         x_min, y_min, x_max, y_max = component["bbox"]
        
# #         # Get the labels within this bounding box
# #         box_labels = labels[y_min:y_max, x_min:x_max]
# #         unique_labels = np.unique(box_labels)
        
# #         # Skip background (0)
# #         unique_labels = unique_labels[unique_labels != 0]
        
# #         # Map component to labels and vice versa
# #         for label in unique_labels:
# #             label_int = int(label)
# #             label_to_components[label_int].add(comp_id)
# #             component_to_labels[comp_id].add(label_int)
    
# #     # Check for disconnected components (not connected to any label)
# #     all_components = {comp["id"] for comp in components}
# #     connected_components = set(component_to_labels.keys())
# #     disconnected_components = all_components - connected_components
    
# #     if disconnected_components:
# #         print(f"Warning: The following components are not connected to any circuit trace: {disconnected_components}")
    
# #     # Get all labels that connect to any component
# #     main_labels = set()
# #     for labels_set in component_to_labels.values():
# #         main_labels.update(labels_set)
    
# #     # Check if there's a single label connecting to all components
# #     for label, connected_comps in label_to_components.items():
# #         if connected_comps == connected_components:  # This label connects to all connected components
# #             print(f"Found a single label ({label}) that connects all components")
# #             main_labels = {label}
# #             break
    
# #     return main_labels, component_to_labels, list(disconnected_components)


# def identify_main_circuit_labels(labels, components):
#     """
#     Identify the label(s) that represent the main circuit.
    
#     This function looks for labels that connect to component bounding boxes.
#     It also performs graph traversal to find all connected labels that form
#     the complete circuit, even if they don't directly intersect with components.
    
#     Args:
#         labels: The connected components label matrix
#         components: List of component dictionaries
        
#     Returns:
#         main_labels: Set of label IDs that represent the main circuit
#         component_to_labels: Dictionary mapping component IDs to the labels they connect with
#         disconnected_components: List of components not connected to any label
#     """
#     # Map labels to the components they connect with
#     label_to_components = defaultdict(set)
    
#     # Map components to the labels they connect with
#     component_to_labels = defaultdict(set)
    
#     # For each component
#     for component in components:
#         comp_id = component["id"]
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Get the labels within this bounding box
#         box_labels = labels[y_min:y_max, x_min:x_max]
#         unique_labels = np.unique(box_labels)
        
#         # Skip background (0)
#         unique_labels = unique_labels[unique_labels != 0]
        
#         # Map component to labels and vice versa
#         for label in unique_labels:
#             label_int = int(label)
#             label_to_components[label_int].add(comp_id)
#             component_to_labels[comp_id].add(label_int)
    
#     # Check for disconnected components (not connected to any label)
#     all_components = {comp["id"] for comp in components}
#     connected_components = set(component_to_labels.keys())
#     disconnected_components = all_components - connected_components
    
#     if disconnected_components:
#         print(f"Warning: The following components are not connected to any circuit trace: {disconnected_components}")
    
#     # Get all labels that connect to any component as initial main labels
#     main_labels = set()
#     for labels_set in component_to_labels.values():
#         main_labels.update(labels_set)
    
#     # Check if there's a single label connecting to all components
#     single_connecting_label = None
#     for label, connected_comps in label_to_components.items():
#         if connected_comps == connected_components:  # This label connects to all connected components
#             print(f"Found a single label ({label}) that connects all components")
#             single_connecting_label = label
#             break
    
#     # If a single connecting label was found, we could use just that
#     # but we'll proceed with graph traversal anyway for completeness
    
#     # Perform graph traversal to find all labels connected to initial main labels
    
#     # Find all boundaries between different labels (where they touch)
#     h, w = labels.shape
#     label_boundaries = set()
    
#     # Check horizontal adjacencies
#     for y in range(h):
#         for x in range(w-1):
#             label1 = labels[y, x]
#             label2 = labels[y, x+1]
#             if label1 != 0 and label2 != 0 and label1 != label2:
#                 label_boundaries.add((int(label1), int(label2)))
    
#     # Check vertical adjacencies
#     for y in range(h-1):
#         for x in range(w):
#             label1 = labels[y, x]
#             label2 = labels[y+1, x]
#             if label1 != 0 and label2 != 0 and label1 != label2:
#                 label_boundaries.add((int(label1), int(label2)))
    
#     # Expand main_labels to include all connected labels through graph traversal
#     expanded_main_labels = set(main_labels)
#     all_processed = False
    
#     while not all_processed:
#         all_processed = True
#         new_labels = set()
        
#         for label1, label2 in label_boundaries:
#             if label1 in expanded_main_labels and label2 not in expanded_main_labels:
#                 new_labels.add(label2)
#                 all_processed = False
#             elif label2 in expanded_main_labels and label1 not in expanded_main_labels:
#                 new_labels.add(label1)
#                 all_processed = False
        
#         expanded_main_labels.update(new_labels)
    
#     print(f"Initial main labels from component intersections: {main_labels}")
#     print(f"Expanded main labels after graph traversal: {expanded_main_labels}")
#     print(f"Added {len(expanded_main_labels) - len(main_labels)} additional labels through connectivity")
    
#     return expanded_main_labels, component_to_labels, list(disconnected_components)

# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
#     connection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Save the thickened binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
#     # Perform connected component analysis on thickened image
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thickened_binary, connectivity=8)
    
#     # Identify the main circuit labels
#     main_circuit_labels, component_to_labels, disconnected_components = identify_main_circuit_labels(labels, components)
    
#     # Print information about the circuit structure
#     print(f"Identified main circuit labels: {main_circuit_labels}")
#     print(f"Components to labels mapping: {component_to_labels}")
    
#     if disconnected_components:
#         print(f"WARNING: {os.path.basename(image_path)} has a bad design - disconnected components: {disconnected_components}")
    
#     # Highlight the main circuit in a separate visualization
#     main_circuit_image = np.zeros_like(original_image)
#     for label in main_circuit_labels:
#         # Assign a bright color to the main circuit
#         main_circuit_image[labels == label] = (0, 255, 0)  # Bright green
    
#     # Save the main circuit visualization
#     cv2.imwrite(os.path.join(output_folder, "main_circuit.png"), main_circuit_image)
    
#     # Create a colored label image for visualization with special color for main circuit labels
#     label_colors = np.zeros((num_labels, 3), dtype=np.uint8)
#     # Reserve color 0 for background (black)
#     for i in range(1, num_labels):
#         if i in main_circuit_labels:
#             label_colors[i] = (0, 255, 0)  # Bright green for main circuit
#         else:
#             label_colors[i] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
#     # Create label visualization image
#     label_image = np.zeros_like(original_image)
#     for i in range(1, num_labels):
#         label_image[labels == i] = label_colors[i]
    
#     # Map components to connected regions
#     component_to_regions = defaultdict(set)
#     region_to_components = defaultdict(set)
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Get label masks for the component's bounding box
#         component_labels = labels[y_min:y_max, x_min:x_max]
        
#         # Find all region labels that intersect with this component
#         unique_labels = np.unique(component_labels)
        
#         # Filter out background (label 0)
#         unique_labels = unique_labels[unique_labels != 0]
        
#         # Add these regions to the component
#         for label in unique_labels:
#             component_to_regions[component["id"]].add(int(label))  # Convert to Python int
#             region_to_components[int(label)].add(component["id"])  # Convert to Python int
    
#     # Determine connections between components
#     connections = []
#     processed_pairs = set()
    
#     for region, component_ids in region_to_components.items():
#         # Skip regions that only connect to one component
#         if len(component_ids) < 2:
#             continue
            
#         # Create connections between all components sharing this region
#         for comp_id1 in component_ids:
#             for comp_id2 in component_ids:
#                 if comp_id1 != comp_id2:
#                     # Create an ordered pair to avoid duplicates
#                     pair = tuple(sorted([comp_id1, comp_id2]))
                    
#                     if pair not in processed_pairs:
#                         # Find the components by ID
#                         comp1 = next(c for c in components if c["id"] == comp_id1)
#                         comp2 = next(c for c in components if c["id"] == comp_id2)
                        
#                         # Calculate connection points (use centroids of the shared region)
#                         mask = (labels == region)
#                         y_indices, x_indices = np.where(mask)
                        
#                         # Skip if no pixels found (shouldn't happen, but just in case)
#                         if len(x_indices) == 0:
#                             continue
                            
#                         # Find connection points
#                         bbox1 = comp1["bbox"]
#                         bbox2 = comp2["bbox"]
                        
#                         # Find region pixels that intersect with component bounding boxes
#                         comp1_points = []
#                         comp2_points = []
                        
#                         for x, y in zip(x_indices, y_indices):
#                             # Check if point is on the boundary of component 1
#                             if (bbox1[0] <= x <= bbox1[2] and 
#                                 (y == bbox1[1] or y == bbox1[3])) or \
#                                (bbox1[1] <= y <= bbox1[3] and 
#                                 (x == bbox1[0] or x == bbox1[2])):
#                                 comp1_points.append((int(x), int(y)))  # Convert to Python int
                                
#                             # Check if point is on the boundary of component 2
#                             if (bbox2[0] <= x <= bbox2[2] and 
#                                 (y == bbox2[1] or y == bbox2[3])) or \
#                                (bbox2[1] <= y <= bbox2[3] and 
#                                 (x == bbox2[0] or x == bbox2[2])):
#                                 comp2_points.append((int(x), int(y)))  # Convert to Python int
                        
#                         # If no intersection points found, use nearest points
#                         if not comp1_points or not comp2_points:
#                             # Use centroids or corners as fallback
#                             center1 = (int((bbox1[0] + bbox1[2]) / 2), 
#                                      int((bbox1[1] + bbox1[3]) / 2))
#                             center2 = (int((bbox2[0] + bbox2[2]) / 2), 
#                                      int((bbox2[1] + bbox2[3]) / 2))
                            
#                             # Find nearest mask point to each center
#                             if not comp1_points:
#                                 distances = [(x - center1[0])**2 + (y - center1[1])**2 
#                                            for x, y in zip(x_indices, y_indices)]
#                                 min_idx = np.argmin(distances)
#                                 comp1_points = [(int(x_indices[min_idx]), int(y_indices[min_idx]))]  # Convert to Python int
                                
#                             if not comp2_points:
#                                 distances = [(x - center2[0])**2 + (y - center2[1])**2 
#                                            for x, y in zip(x_indices, y_indices)]
#                                 min_idx = np.argmin(distances)
#                                 comp2_points = [(int(x_indices[min_idx]), int(y_indices[min_idx]))]  # Convert to Python int
                        
#                         # Take median points for stability
#                         comp1_point = comp1_points[len(comp1_points) // 2]
#                         comp2_point = comp2_points[len(comp2_points) // 2]
                        
#                         # Draw connection on visualization
#                         cv2.circle(connection_image, comp1_point, 4, (0, 0, 255), -1)
#                         cv2.circle(connection_image, comp2_point, 4, (0, 0, 255), -1)
#                         cv2.line(connection_image, comp1_point, comp2_point, (255, 0, 0), 2)
                        
#                         # Record connection
#                         connections.append({
#                             "component1": comp_id1,
#                             "component2": comp_id2,
#                             "connect_point1": [int(comp1_point[0]), int(comp1_point[1])],  # Convert to Python int
#                             "connect_point2": [int(comp2_point[0]), int(comp2_point[1])],  # Convert to Python int
#                             "region_id": int(region)  # Convert to Python int
#                         })
                        
#                         processed_pairs.add(pair)
    
#     # Generate netlist data
#     netlist_components = []
#     for comp in components:
#         netlist_components.append({
#             "id": comp["id"],
#             "type": comp["type"],
#             "position": {
#                 "x": int((comp["bbox"][0] + comp["bbox"][2]) / 2),  # Convert to Python int
#                 "y": int((comp["bbox"][1] + comp["bbox"][3]) / 2)   # Convert to Python int
#             },
#             "connections": [c for c in connections 
#                             if c["component1"] == comp["id"] or c["component2"] == comp["id"]]
#         })
    
#     # Create netlist data structure
#     netlist_data = {
#         "components": netlist_components,
#         "connections": connections
#     }
    
#     # Generate SPICE-like netlist
#     nodes = {}
#     node_counter = 1  # Start from 1 (0 is usually ground)
    
#     # Identify GND node if present
#     gnd_components = [c for c in components if "gnd" in c["type"].lower()]
#     if gnd_components:
#         # Find all components connected to ground
#         gnd_connections = []
#         for conn in connections:
#             if any(c["id"] == conn["component1"] or c["id"] == conn["component2"] 
#                   for c in gnd_components):
#                 other_comp = conn["component1"] if conn["component1"] not in [c["id"] for c in gnd_components] else conn["component2"]
#                 gnd_connections.append((other_comp, 0))  # Node 0 is ground
#                 nodes[other_comp] = 0
    
#     # Assign node numbers to all other connections
#     for conn in connections:
#         comp1, comp2 = conn["component1"], conn["component2"]
        
#         # Skip if both are already assigned to the same node
#         if comp1 in nodes and comp2 in nodes and nodes[comp1] == nodes[comp2]:
#             continue
            
#         # If one has a node, assign the other to the same node
#         if comp1 in nodes:
#             nodes[comp2] = nodes[comp1]
#         elif comp2 in nodes:
#             nodes[comp1] = nodes[comp2]
#         # Otherwise, create a new node
#         else:
#             nodes[comp1] = node_counter
#             nodes[comp2] = node_counter
#             node_counter += 1
    
#     # Generate SPICE-like netlist text
#     spice_netlist = "* Circuit netlist generated from image\n"
#     spice_netlist += f"* Source: {os.path.basename(image_path)}\n\n"
    
#     # Add component declarations
#     for comp in components:
#         comp_id = comp["id"]
#         comp_type = comp["type"]
        
#         # Find connections for this component
#         comp_connections = []
#         for conn in connections:
#             if conn["component1"] == comp_id:
#                 comp_connections.append((conn["component2"], nodes.get(conn["component2"], 999)))
#             elif conn["component2"] == comp_id:
#                 comp_connections.append((conn["component1"], nodes.get(conn["component1"], 999)))
        
#         # Format based on component type
#         if "resistor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"R{comp_id.split('_')[1]} {node_str} 1k\n"
#         elif "capacitor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"C{comp_id.split('_')[1]} {node_str} 1u\n"
#         elif "transistor" in comp_type.lower() or "bjt" in comp_type.lower():
#             # Assume 3 connections for transistors (collector, base, emitter)
#             if len(comp_connections) >= 3:
#                 nodes_str = " ".join(str(n[1]) for n in comp_connections[:3])
#                 spice_netlist += f"Q{comp_id.split('_')[1]} {nodes_str} 2N3904\n"
#         # Add more component types as needed
    
#     # Save all outputs
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Save visualization images
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_detection.png"), detection_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_labels.png"), label_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_connections.png"), connection_image)
    
#     # Save netlist data as JSON
#     with open(os.path.join(output_folder, f"{base_name}_netlist.json"), 'w') as f:
#         json.dump(netlist_data, f, indent=2, cls=NumpyEncoder)  # Use NumpyEncoder
    
#     # Save SPICE netlist
#     with open(os.path.join(output_folder, f"{base_name}.cir"), 'w') as f:
#         f.write(spice_netlist)
    
#     return netlist_data, spice_netlist

# def process_all_images(images_folder, model_path, output_folder):
#     """Process all circuit images in a folder."""
#     os.makedirs(output_folder, exist_ok=True)
    
#     image_files = [f for f in os.listdir(images_folder) 
#                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
        
#         # Create output folder for this image
#         image_output_folder = os.path.join(output_folder, 
#                                           os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Process the image
#         try:
#             netlist_data, spice_netlist = process_circuit_image(
#                 image_path, model_path, image_output_folder)
            
#             if netlist_data:
#                 print(f"  Generated netlist for {image_file}")
#                 print(f"  Output saved to {image_output_folder}")
#             else:
#                 print(f"  Failed to process {image_file}")
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path)





# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# # Custom JSON encoder to handle NumPy types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)
    
# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
#     connection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Save the thickened binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
#     # Create a merged label image where all foreground pixels are set to 1
#     merged_labels = np.zeros_like(thickened_binary)
#     merged_labels[thickened_binary > 0] = 1
    
#     # Save the merged label image
#     merged_label_vis = np.zeros_like(original_image)
#     merged_label_vis[merged_labels > 0] = (0, 255, 0)  # Green for all circuit traces
#     cv2.imwrite(os.path.join(output_folder, "merged_labels.png"), merged_label_vis)
    
#     # Map components to connected regions (in this case, just the single merged label)
#     component_to_regions = defaultdict(set)
#     region_to_components = defaultdict(set)
    
#     # Create a single region ID for the entire circuit
#     circuit_region_id = 1
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Check if this component intersects with the merged label
#         component_region = merged_labels[y_min:y_max, x_min:x_max]
#         if np.any(component_region > 0):
#             component_to_regions[component["id"]].add(circuit_region_id)
#             region_to_components[circuit_region_id].add(component["id"])
    
#     # Identify disconnected components
#     all_components = {comp["id"] for comp in components}
#     connected_components = set(component_to_regions.keys())
#     disconnected_components = all_components - connected_components
    
#     if disconnected_components:
#         print(f"WARNING: {os.path.basename(image_path)} has a bad design - disconnected components: {disconnected_components}")
    
#     # Determine connections between components
#     connections = []
#     processed_pairs = set()
    
#     # All components connected to the merged label are connected to each other
#     component_ids = list(region_to_components[circuit_region_id])
    
#     for i in range(len(component_ids)):
#         for j in range(i+1, len(component_ids)):
#             comp_id1 = component_ids[i]
#             comp_id2 = component_ids[j]
            
#             # Create an ordered pair to avoid duplicates
#             pair = tuple(sorted([comp_id1, comp_id2]))
            
#             if pair not in processed_pairs:
#                 # Find the components by ID
#                 comp1 = next(c for c in components if c["id"] == comp_id1)
#                 comp2 = next(c for c in components if c["id"] == comp_id2)
                
#                 # Get bounding boxes
#                 bbox1 = comp1["bbox"]
#                 bbox2 = comp2["bbox"]
                
#                 # Find connection points (intersection with merged label)
#                 # For each component, find where its bounding box intersects with the circuit trace
                
#                 # Component 1 connection points
#                 comp1_mask = np.zeros_like(merged_labels)
#                 comp1_mask[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]] = 1
                
#                 # Get the boundary of the bounding box
#                 comp1_boundary = np.zeros_like(comp1_mask)
#                 comp1_boundary[bbox1[1], bbox1[0]:bbox1[2]] = 1  # Top edge
#                 comp1_boundary[bbox1[3]-1, bbox1[0]:bbox1[2]] = 1  # Bottom edge
#                 comp1_boundary[bbox1[1]:bbox1[3], bbox1[0]] = 1  # Left edge
#                 comp1_boundary[bbox1[1]:bbox1[3], bbox1[2]-1] = 1  # Right edge
                
#                 # Find intersection of bounding box boundary with circuit trace
#                 comp1_intersect = np.logical_and(comp1_boundary, merged_labels)
#                 comp1_y, comp1_x = np.where(comp1_intersect)
                
#                 # Component 2 connection points
#                 comp2_mask = np.zeros_like(merged_labels)
#                 comp2_mask[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]] = 1
                
#                 # Get the boundary of the bounding box
#                 comp2_boundary = np.zeros_like(comp2_mask)
#                 comp2_boundary[bbox2[1], bbox2[0]:bbox2[2]] = 1  # Top edge
#                 comp2_boundary[bbox2[3]-1, bbox2[0]:bbox2[2]] = 1  # Bottom edge
#                 comp2_boundary[bbox2[1]:bbox2[3], bbox2[0]] = 1  # Left edge
#                 comp2_boundary[bbox2[1]:bbox2[3], bbox2[2]-1] = 1  # Right edge
                
#                 # Find intersection of bounding box boundary with circuit trace
#                 comp2_intersect = np.logical_and(comp2_boundary, merged_labels)
#                 comp2_y, comp2_x = np.where(comp2_intersect)
                
#                 # If no intersection points found, use centers as fallback
#                 if len(comp1_x) == 0:
#                     center1_x = (bbox1[0] + bbox1[2]) // 2
#                     center1_y = (bbox1[1] + bbox1[3]) // 2
#                     comp1_point = (int(center1_x), int(center1_y))
#                 else:
#                     # Use the median point for stability
#                     idx = len(comp1_x) // 2
#                     comp1_point = (int(comp1_x[idx]), int(comp1_y[idx]))
                
#                 if len(comp2_x) == 0:
#                     center2_x = (bbox2[0] + bbox2[2]) // 2
#                     center2_y = (bbox2[1] + bbox2[3]) // 2
#                     comp2_point = (int(center2_x), int(center2_y))
#                 else:
#                     # Use the median point for stability
#                     idx = len(comp2_x) // 2
#                     comp2_point = (int(comp2_x[idx]), int(comp2_y[idx]))
                
#                 # Draw connection on visualization
#                 cv2.circle(connection_image, comp1_point, 4, (0, 0, 255), -1)
#                 cv2.circle(connection_image, comp2_point, 4, (0, 0, 255), -1)
#                 cv2.line(connection_image, comp1_point, comp2_point, (255, 0, 0), 2)
                
#                 # Record connection
#                 connections.append({
#                     "component1": comp_id1,
#                     "component2": comp_id2,
#                     "connect_point1": [int(comp1_point[0]), int(comp1_point[1])],
#                     "connect_point2": [int(comp2_point[0]), int(comp2_point[1])],
#                     "region_id": circuit_region_id
#                 })
                
#                 processed_pairs.add(pair)
    
#     # Generate netlist data
#     netlist_components = []
#     for comp in components:
#         netlist_components.append({
#             "id": comp["id"],
#             "type": comp["type"],
#             "position": {
#                 "x": int((comp["bbox"][0] + comp["bbox"][2]) / 2),
#                 "y": int((comp["bbox"][1] + comp["bbox"][3]) / 2)
#             },
#             "connections": [c for c in connections 
#                             if c["component1"] == comp["id"] or c["component2"] == comp["id"]]
#         })
    
#     # Create netlist data structure
#     netlist_data = {
#         "components": netlist_components,
#         "connections": connections
#     }
    
#     # Generate SPICE-like netlist
#     nodes = {}
#     node_counter = 1  # Start from 1 (0 is usually ground)
    
#     # Identify GND node if present
#     gnd_components = [c for c in components if "gnd" in c["type"].lower()]
#     if gnd_components:
#         # Find all components connected to ground
#         gnd_connections = []
#         for conn in connections:
#             if any(c["id"] == conn["component1"] or c["id"] == conn["component2"] 
#                   for c in gnd_components):
#                 other_comp = conn["component1"] if conn["component1"] not in [c["id"] for c in gnd_components] else conn["component2"]
#                 gnd_connections.append((other_comp, 0))  # Node 0 is ground
#                 nodes[other_comp] = 0
    
#     # Assign node numbers to all other connections
#     for conn in connections:
#         comp1, comp2 = conn["component1"], conn["component2"]
        
#         # Skip if both are already assigned to the same node
#         if comp1 in nodes and comp2 in nodes and nodes[comp1] == nodes[comp2]:
#             continue
            
#         # If one has a node, assign the other to the same node
#         if comp1 in nodes:
#             nodes[comp2] = nodes[comp1]
#         elif comp2 in nodes:
#             nodes[comp1] = nodes[comp2]
#         # Otherwise, create a new node
#         else:
#             nodes[comp1] = node_counter
#             nodes[comp2] = node_counter
#             node_counter += 1
    
#     # Generate SPICE-like netlist text
#     spice_netlist = "* Circuit netlist generated from image\n"
#     spice_netlist += f"* Source: {os.path.basename(image_path)}\n\n"
    
#     # Add component declarations
#     for comp in components:
#         comp_id = comp["id"]
#         comp_type = comp["type"]
        
#         # Skip disconnected components
#         if comp_id in disconnected_components:
#             continue
        
#         # Find connections for this component
#         comp_connections = []
#         for conn in connections:
#             if conn["component1"] == comp_id:
#                 comp_connections.append((conn["component2"], nodes.get(conn["component2"], 999)))
#             elif conn["component2"] == comp_id:
#                 comp_connections.append((conn["component1"], nodes.get(conn["component1"], 999)))
        
#         # Format based on component type
#         if "resistor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"R{comp_id.split('_')[1]} {node_str} 1k\n"
#         elif "capacitor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"C{comp_id.split('_')[1]} {node_str} 1u\n"
#         elif "transistor" in comp_type.lower() or "bjt" in comp_type.lower():
#             # Assume 3 connections for transistors (collector, base, emitter)
#             if len(comp_connections) >= 3:
#                 nodes_str = " ".join(str(n[1]) for n in comp_connections[:3])
#                 spice_netlist += f"Q{comp_id.split('_')[1]} {nodes_str} 2N3904\n"
#         # Add more component types as needed
    
#     # Save all outputs
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Save visualization images
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_detection.png"), detection_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_connections.png"), connection_image)
    
#     # Save netlist data as JSON
#     with open(os.path.join(output_folder, f"{base_name}_netlist.json"), 'w') as f:
#         json.dump(netlist_data, f, indent=2, cls=NumpyEncoder)  # Use NumpyEncoder
    
#     # Save SPICE netlist
#     with open(os.path.join(output_folder, f"{base_name}.cir"), 'w') as f:
#         f.write(spice_netlist)
    
#     return netlist_data, spice_netlist

# def process_all_images(images_folder, model_path, output_folder):
#     """Process all circuit images in a folder."""
#     os.makedirs(output_folder, exist_ok=True)
    
#     image_files = [f for f in os.listdir(images_folder) 
#                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
        
#         # Create output folder for this image
#         image_output_folder = os.path.join(output_folder, 
#                                           os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Process the image
#         try:
#             netlist_data, spice_netlist = process_circuit_image(
#                 image_path, model_path, image_output_folder)
            
#             if netlist_data:
#                 print(f"  Generated netlist for {image_file}")
#                 print(f"  Output saved to {image_output_folder}")
#             else:
#                 print(f"  Failed to process {image_file}")
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path)










# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# # Custom JSON encoder to handle NumPy types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)

# def identify_circuit_nodes(merged_labels, components):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#     """
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Get a margin around the bounding box to check for adjacent nodes
#         margin = 2  # pixels to check around bounding box
#         x_min_check = max(0, x_min - margin)
#         y_min_check = max(0, y_min - margin)
#         x_max_check = min(node_labels.shape[1] - 1, x_max + margin)
#         y_max_check = min(node_labels.shape[0] - 1, y_max + margin)
        
#         # Check the perimeter of the bounding box (with margin)
#         # Top and bottom edges
#         for x in range(x_min_check, x_max_check):
#             for y in [y_min_check, y_max_check - 1]:
#                 node_id = node_labels[y, x]
#                 if node_id > 0:  # Ignore background (0)
#                     valid_nodes.add(node_id)
#                     node_to_components[int(node_id)].add(comp_id)
#                     component_to_nodes[comp_id].add(int(node_id))
        
#         # Left and right edges
#         for y in range(y_min_check, y_max_check):
#             for x in [x_min_check, x_max_check - 1]:
#                 node_id = node_labels[y, x]
#                 if node_id > 0:  # Ignore background (0)
#                     valid_nodes.add(node_id)
#                     node_to_components[int(node_id)].add(comp_id)
#                     component_to_nodes[comp_id].add(int(node_id))
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((node_labels.shape[0], node_labels.shape[1], 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, 0

# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Save the thickened binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
#     # Create a merged label image where all foreground pixels are set to 1
#     merged_labels = np.zeros_like(thickened_binary)
#     merged_labels[thickened_binary > 0] = 1
    
#     # Save the merged label image
#     merged_label_vis = np.zeros_like(original_image)
#     merged_label_vis[merged_labels > 0] = (0, 255, 0)  # Green for all circuit traces
#     cv2.imwrite(os.path.join(output_folder, "merged_labels.png"), merged_label_vis)
    
#     # Identify circuit nodes
#     node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug = identify_circuit_nodes(merged_labels, components)
    
#     # Save the node visualizations
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes.png"), node_vis)
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes_with_boxes.png"), node_vis_with_boxes)
#     cv2.imwrite(os.path.join(output_folder, "circuit_cut.png"), circuit_cut * 255)  # Multiply by 255 to make visible
#     # cv2.imwrite(os.path.join(output_folder, "overlap_debug.png"), overlap_debug)
    
#     # Save node information as JSON
#     with open(os.path.join(output_folder, "node_info.json"), 'w') as f:
#         json.dump(node_info, f, indent=2, cls=NumpyEncoder)
    
#     # Extract node-to-components mapping from node_info
#     node_to_components = {int(k): set(v) for k, v in node_info["node_to_components"].items()}
    
#     # Print debug information about node-to-component connections
#     print("\n========== NODE CONNECTIONS DEBUG ==========")
#     print(f"Image: {os.path.basename(image_path)}")
#     print(f"Total nodes detected: {len(node_to_components)}")
#     print(f"Total components: {len(components)}")

#     # Print components
#     print("\nComponents detected:")
#     for comp in components:
#         print(f"  {comp['id']}: {comp['type']} at {comp['bbox']}")

#     # Print each node and its connected components
#     print("\nNodes and their connected components:")
#     for node_id, connected_comps in sorted(node_to_components.items()):
#         # Get node color if available
#         node_color = node_info["node_colors"].get(str(node_id), [0, 0, 0])
#         print(f"Node {node_id} (Color: RGB{node_color}):")
        
#         # List all components connected to this node
#         for comp_id in connected_comps:
#             # Find component details
#             comp = next((c for c in components if c["id"] == comp_id), None)
#             if comp:
#                 print(f"  - {comp_id}: {comp['type']} at {comp['bbox']}")
        
#         # Calculate node size (number of pixels)
#         node_mask = (node_labels == node_id)
#         node_size = np.sum(node_mask)
        
#         # Check if node touches image edge
#         h, w = node_labels.shape
#         touches_edge = (
#             np.any(node_mask[0, :]) or  # Top edge
#             np.any(node_mask[h-1, :]) or  # Bottom edge
#             np.any(node_mask[:, 0]) or  # Left edge
#             np.any(node_mask[:, w-1])  # Right edge
#         )
        
#         print(f"  Node size: {node_size} pixels, Touches edge: {touches_edge}")
#         print()

#     print("=========================================\n")

#     # Save the debug information to a text file
#     with open(os.path.join(output_folder, "node_connections_debug.txt"), 'w') as f:
#         f.write(f"========== NODE CONNECTIONS DEBUG ==========\n")
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total nodes detected: {len(node_to_components)}\n")
#         f.write(f"Total components: {len(components)}\n\n")
        
#         f.write("Components detected:\n")
#         for comp in components:
#             f.write(f"  {comp['id']}: {comp['type']} at {comp['bbox']}\n")
        
#         f.write("\nNodes and their connected components:\n")
#         for node_id, connected_comps in sorted(node_to_components.items()):
#             node_color = node_info["node_colors"].get(str(node_id), [0, 0, 0])
#             f.write(f"Node {node_id} (Color: RGB{node_color}):\n")
            
#             for comp_id in connected_comps:
#                 comp = next((c for c in components if c["id"] == comp_id), None)
#                 if comp:
#                     f.write(f"  - {comp_id}: {comp['type']} at {comp['bbox']}\n")
            
#             node_mask = (node_labels == node_id)
#             node_size = np.sum(node_mask)
            
#             h, w = node_labels.shape
#             touches_edge = (
#                 np.any(node_mask[0, :]) or  # Top edge
#                 np.any(node_mask[h-1, :]) or  # Bottom edge
#                 np.any(node_mask[:, 0]) or  # Left edge
#                 np.any(node_mask[:, w-1])  # Right edge
#             )
            
#             f.write(f"  Node size: {node_size} pixels, Touches edge: {touches_edge}\n\n")
        
#         f.write("=========================================\n")
    
#     # Create connections based on overlap detection results
#     connections = []
#     processed_pairs = set()
    
#     # For each node that connects to multiple components, create connections between those components
#     for node_id, connected_comps in node_to_components.items():
#         connected_comps_list = list(connected_comps)
        
#         # If this node connects multiple components, create connections between them
#         if len(connected_comps_list) >= 2:
#             for i in range(len(connected_comps_list)):
#                 for j in range(i+1, len(connected_comps_list)):
#                     comp_id1 = connected_comps_list[i]
#                     comp_id2 = connected_comps_list[j]
                    
#                     # Create an ordered pair to avoid duplicates
#                     pair = tuple(sorted([comp_id1, comp_id2]))
                    
#                     if pair not in processed_pairs:
#                         # Find the components by ID
#                         comp1 = next(c for c in components if c["id"] == comp_id1)
#                         comp2 = next(c for c in components if c["id"] == comp_id2)
                        
#                         # Get bounding boxes
#                         bbox1 = comp1["bbox"]
#                         bbox2 = comp2["bbox"]
                        
#                         # Find the node's mask
#                         node_mask = (node_labels == node_id)
#                         node_y, node_x = np.where(node_mask)
                        
#                         # For each component, find where its boundary intersects with this node
#                         # Component 1
#                         comp1_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox1
                        
#                         # Create a perimeter mask for the component
#                         comp1_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp1_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp1_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp1_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp1_intersect = np.logical_and(comp1_boundary, node_mask)
#                         comp1_y, comp1_x = np.where(comp1_intersect)
                        
#                         # Component 2
#                         comp2_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox2
                        
#                         # Create a perimeter mask for the component
#                         comp2_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp2_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp2_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp2_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp2_intersect = np.logical_and(comp2_boundary, node_mask)
#                         comp2_y, comp2_x = np.where(comp2_intersect)
                        
#                         # Get connection points
#                         if len(comp1_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp1_x) // 2
#                             comp1_point = (int(comp1_x[idx]), int(comp1_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp1_point = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
                        
#                         if len(comp2_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp2_x) // 2
#                             comp2_point = (int(comp2_x[idx]), int(comp2_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp2_point = (int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2))
                        
#                         # Record the connection
#                         connections.append({
#                             "component1": comp_id1,
#                             "component2": comp_id2,
#                             "connect_point1": [int(comp1_point[0]), int(comp1_point[1])],
#                             "connect_point2": [int(comp2_point[0]), int(comp2_point[1])],
#                             "node_id": int(node_id)
#                         })
                        
#                         processed_pairs.add(pair)
    
#     # Create a connection visualization
#     connection_image = original_image.copy()
    
#     # Draw all connections
#     for conn in connections:
#         comp1_point = tuple(conn["connect_point1"])
#         comp2_point = tuple(conn["connect_point2"])
        
#         # Draw connection line
#         cv2.line(connection_image, comp1_point, comp2_point, (255, 0, 0), 2)
        
#         # Draw connection points
#         cv2.circle(connection_image, comp1_point, 4, (0, 0, 255), -1)
#         cv2.circle(connection_image, comp2_point, 4, (0, 0, 255), -1)
    
#     # Identify disconnected components
#     all_components = {comp["id"] for comp in components}
#     connected_components = set()
#     for conn in connections:
#         connected_components.add(conn["component1"])
#         connected_components.add(conn["component2"])
#     disconnected_components = all_components - connected_components
    
#     if disconnected_components:
#         print(f"WARNING: {os.path.basename(image_path)} has a bad design - disconnected components: {disconnected_components}")
    
#     # Generate netlist data
#     netlist_components = []
#     for comp in components:
#         netlist_components.append({
#             "id": comp["id"],
#             "type": comp["type"],
#             "position": {
#                 "x": int((comp["bbox"][0] + comp["bbox"][2]) / 2),
#                 "y": int((comp["bbox"][1] + comp["bbox"][3]) / 2)
#             },
#             "connections": [c for c in connections 
#                             if c["component1"] == comp["id"] or c["component2"] == comp["id"]]
#         })
    
#     # Create netlist data structure
#     netlist_data = {
#         "components": netlist_components,
#         "connections": connections,
#         "nodes": node_info
#     }
    
#     # Generate SPICE-like netlist
#     nodes = {}
#     node_counter = 1  # Start from 1 (0 is usually ground)
    
#     # Identify GND node if present
#     gnd_components = [c for c in components if "gnd" in c["type"].lower()]
#     if gnd_components:
#         # Find all components connected to ground
#         gnd_connections = []
#         for conn in connections:
#             if any(c["id"] == conn["component1"] or c["id"] == conn["component2"] 
#                   for c in gnd_components):
#                 other_comp = conn["component1"] if conn["component1"] not in [c["id"] for c in gnd_components] else conn["component2"]
#                 gnd_connections.append((other_comp, 0))  # Node 0 is ground
#                 nodes[other_comp] = 0
    
#     # Assign node numbers to all other connections
#     for conn in connections:
#         comp1, comp2 = conn["component1"], conn["component2"]
        
#         # Skip if both are already assigned to the same node
#         if comp1 in nodes and comp2 in nodes and nodes[comp1] == nodes[comp2]:
#             continue
            
#         # If one has a node, assign the other to the same node
#         if comp1 in nodes:
#             nodes[comp2] = nodes[comp1]
#         elif comp2 in nodes:
#             nodes[comp1] = nodes[comp2]
#         # Otherwise, create a new node
#         else:
#             nodes[comp1] = node_counter
#             nodes[comp2] = node_counter
#             node_counter += 1
    
#     # Generate SPICE-like netlist text
#     spice_netlist = "* Circuit netlist generated from image\n"
#     spice_netlist += f"* Source: {os.path.basename(image_path)}\n\n"
    
#     # Add component declarations
#     for comp in components:
#         comp_id = comp["id"]
#         comp_type = comp["type"]
        
#         # Skip disconnected components
#         if comp_id in disconnected_components:
#             continue
        
#         # Find connections for this component
#         comp_connections = []
#         for conn in connections:
#             if conn["component1"] == comp_id:
#                 comp_connections.append((conn["component2"], nodes.get(conn["component2"], 999)))
#             elif conn["component2"] == comp_id:
#                 comp_connections.append((conn["component1"], nodes.get(conn["component1"], 999)))
        
#         # Format based on component type
#         if "resistor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"R{comp_id.split('_')[1]} {node_str} 1k\n"
#         elif "capacitor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"C{comp_id.split('_')[1]} {node_str} 1u\n"
#         elif "transistor" in comp_type.lower() or "bjt" in comp_type.lower():
#             # Assume 3 connections for transistors (collector, base, emitter)
#             if len(comp_connections) >= 3:
#                 nodes_str = " ".join(str(n[1]) for n in comp_connections[:3])
#                 spice_netlist += f"Q{comp_id.split('_')[1]} {nodes_str} 2N3904\n"
#         # Add more component types as needed
    
#     # Save all outputs
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Save visualization images
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_detection.png"), detection_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_connections.png"), connection_image)
    
#     # Save netlist data as JSON
#     with open(os.path.join(output_folder, f"{base_name}_netlist.json"), 'w') as f:
#         json.dump(netlist_data, f, indent=2, cls=NumpyEncoder)  # Use NumpyEncoder
    
#     # Save SPICE netlist
#     with open(os.path.join(output_folder, f"{base_name}.cir"), 'w') as f:
#         f.write(spice_netlist)
    
#     return netlist_data, spice_netlist

# def process_all_images(images_folder, model_path, output_folder):
#     """Process all circuit images in a folder."""
#     os.makedirs(output_folder, exist_ok=True)
    
#     image_files = [f for f in os.listdir(images_folder) 
#                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
        
#         # Create output folder for this image
#         image_output_folder = os.path.join(output_folder, 
#                                           os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Process the image
#         try:
#             netlist_data, spice_netlist = process_circuit_image(
#                 image_path, model_path, image_output_folder)
            
#             if netlist_data:
#                 print(f"  Generated netlist for {image_file}")
#                 print(f"  Output saved to {image_output_folder}")
#             else:
#                 print(f"  Failed to process {image_file}")
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path)























# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# # Custom JSON encoder to handle NumPy types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)

# def identify_circuit_nodes(merged_labels, components):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#     """
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Get a margin around the bounding box to check for adjacent nodes
#         margin = 2  # pixels to check around bounding box
#         x_min_check = max(0, x_min - margin)
#         y_min_check = max(0, y_min - margin)
#         x_max_check = min(node_labels.shape[1] - 1, x_max + margin)
#         y_max_check = min(node_labels.shape[0] - 1, y_max + margin)
        
#         # Check the perimeter of the bounding box (with margin)
#         # Top and bottom edges
#         for x in range(x_min_check, x_max_check):
#             for y in [y_min_check, y_max_check - 1]:
#                 node_id = node_labels[y, x]
#                 if node_id > 0:  # Ignore background (0)
#                     valid_nodes.add(node_id)
#                     node_to_components[int(node_id)].add(comp_id)
#                     component_to_nodes[comp_id].add(int(node_id))
        
#         # Left and right edges
#         for y in range(y_min_check, y_max_check):
#             for x in [x_min_check, x_max_check - 1]:
#                 node_id = node_labels[y, x]
#                 if node_id > 0:  # Ignore background (0)
#                     valid_nodes.add(node_id)
#                     node_to_components[int(node_id)].add(comp_id)
#                     component_to_nodes[comp_id].add(int(node_id))
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((node_labels.shape[0], node_labels.shape[1], 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut











# def identify_circuit_nodes(merged_labels, components):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#     """
#     h, w = merged_labels.shape
    
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Create a debug visualization to check overlap detection
#     overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Set up colors for debug visualization
#     # Circuit traces - green
#     overlap_debug[circuit_image > 0] = [0, 255, 0]
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     # Check every pixel in the image (exhaustive but guaranteed to catch all overlaps)
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Draw component bounding box on debug image - blue
#         cv2.rectangle(overlap_debug, (x_min, y_min), (x_max, y_max), [255, 0, 0], 1)
        
#         # Create a mask for just this component's bounding box perimeter
#         # This ensures we only check the perimeter, not the interior
#         comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        
#         # Top and bottom edges
#         if y_min >= 0 and y_min < h:
#             comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#         if y_max >= 0 and y_max < h:
#             comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
            
#         # Left and right edges
#         if x_min >= 0 and x_min < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#         if x_max >= 0 and x_max < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
        
#         # Get node labels that intersect with this component's perimeter
#         for label_id in range(1, num_labels):  # Skip background (0)
#             # Check if this node intersects with the component perimeter
#             node_mask = (node_labels == label_id)
#             intersection = np.logical_and(comp_perimeter, node_mask)
            
#             if np.any(intersection):
#                 # This node intersects with the component's perimeter
#                 valid_nodes.add(label_id)
#                 node_to_components[int(label_id)].add(comp_id)
#                 component_to_nodes[comp_id].add(int(label_id))
                
#                 # For debug visualization - red for detected intersections
#                 overlap_debug[intersection] = [0, 0, 255]
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug


# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Save the thickened binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
#     # Create a merged label image where all foreground pixels are set to 1
#     merged_labels = np.zeros_like(thickened_binary)
#     merged_labels[thickened_binary > 0] = 1
    
#     # Save the merged label image
#     merged_label_vis = np.zeros_like(original_image)
#     merged_label_vis[merged_labels > 0] = (0, 255, 0)  # Green for all circuit traces
#     cv2.imwrite(os.path.join(output_folder, "merged_labels.png"), merged_label_vis)
    
#     # Identify circuit nodes
#     node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug = identify_circuit_nodes(merged_labels, components)
    
#     # Save the node visualizations
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes.png"), node_vis)
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes_with_boxes.png"), node_vis_with_boxes)
#     cv2.imwrite(os.path.join(output_folder, "circuit_cut.png"), circuit_cut * 255)  # Multiply by 255 to make visible
#     cv2.imwrite(os.path.join(output_folder, "overlap_debug.png"), overlap_debug)
    
#     # Save node information as JSON
#     with open(os.path.join(output_folder, "node_info.json"), 'w') as f:
#         json.dump(node_info, f, indent=2, cls=NumpyEncoder)
    
#     # Extract node-to-components mapping from node_info
#     node_to_components = {int(k): set(v) for k, v in node_info["node_to_components"].items()}
    
#     # Create connections based on overlap detection results
#     connections = []
#     processed_pairs = set()
    
#     # For each node that connects to multiple components, create connections between those components
#     for node_id, connected_comps in node_to_components.items():
#         connected_comps_list = list(connected_comps)
        
#         # If this node connects multiple components, create connections between them
#         if len(connected_comps_list) >= 2:
#             for i in range(len(connected_comps_list)):
#                 for j in range(i+1, len(connected_comps_list)):
#                     comp_id1 = connected_comps_list[i]
#                     comp_id2 = connected_comps_list[j]
                    
#                     # Create an ordered pair to avoid duplicates
#                     pair = tuple(sorted([comp_id1, comp_id2]))
                    
#                     if pair not in processed_pairs:
#                         # Find the components by ID
#                         comp1 = next(c for c in components if c["id"] == comp_id1)
#                         comp2 = next(c for c in components if c["id"] == comp_id2)
                        
#                         # Get bounding boxes
#                         bbox1 = comp1["bbox"]
#                         bbox2 = comp2["bbox"]
                        
#                         # Find the node's mask
#                         node_mask = (node_labels == node_id)
#                         node_y, node_x = np.where(node_mask)
                        
#                         # For each component, find where its boundary intersects with this node
#                         # Component 1
#                         comp1_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox1
                        
#                         # Create a perimeter mask for the component
#                         comp1_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp1_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp1_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp1_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp1_intersect = np.logical_and(comp1_boundary, node_mask)
#                         comp1_y, comp1_x = np.where(comp1_intersect)
                        
#                         # Component 2
#                         comp2_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox2
                        
#                         # Create a perimeter mask for the component
#                         comp2_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp2_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp2_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp2_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp2_intersect = np.logical_and(comp2_boundary, node_mask)
#                         comp2_y, comp2_x = np.where(comp2_intersect)
                        
#                         # Get connection points
#                         if len(comp1_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp1_x) // 2
#                             comp1_point = (int(comp1_x[idx]), int(comp1_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp1_point = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
                        
#                         if len(comp2_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp2_x) // 2
#                             comp2_point = (int(comp2_x[idx]), int(comp2_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp2_point = (int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2))
                        
#                         # Record the connection
#                         connections.append({
#                             "component1": comp_id1,
#                             "component2": comp_id2,
#                             "connect_point1": [int(comp1_point[0]), int(comp1_point[1])],
#                             "connect_point2": [int(comp2_point[0]), int(comp2_point[1])],
#                             "node_id": int(node_id)
#                         })
                        
#                         processed_pairs.add(pair)
    
#     # Create a connection visualization
#     connection_image = original_image.copy()
    
#     # Draw all connections
#     for conn in connections:
#         comp1_point = tuple(conn["connect_point1"])
#         comp2_point = tuple(conn["connect_point2"])
        
#         # Draw connection line
#         cv2.line(connection_image, comp1_point, comp2_point, (255, 0, 0), 2)
        
#         # Draw connection points
#         cv2.circle(connection_image, comp1_point, 4, (0, 0, 255), -1)
#         cv2.circle(connection_image, comp2_point, 4, (0, 0, 255), -1)
    
#     # Identify disconnected components
#     all_components = {comp["id"] for comp in components}
#     connected_components = set()
#     for conn in connections:
#         connected_components.add(conn["component1"])
#         connected_components.add(conn["component2"])
#     disconnected_components = all_components - connected_components
    
#     if disconnected_components:
#         print(f"WARNING: {os.path.basename(image_path)} has a bad design - disconnected components: {disconnected_components}")
    
#     # Generate netlist data
#     netlist_components = []
#     for comp in components:
#         netlist_components.append({
#             "id": comp["id"],
#             "type": comp["type"],
#             "position": {
#                 "x": int((comp["bbox"][0] + comp["bbox"][2]) / 2),
#                 "y": int((comp["bbox"][1] + comp["bbox"][3]) / 2)
#             },
#             "connections": [c for c in connections 
#                             if c["component1"] == comp["id"] or c["component2"] == comp["id"]]
#         })
    
#     # Create netlist data structure
#     netlist_data = {
#         "components": netlist_components,
#         "connections": connections,
#         "nodes": node_info
#     }
    
#     # Generate SPICE-like netlist
#     nodes = {}
#     node_counter = 1  # Start from 1 (0 is usually ground)
    
#     # Identify GND node if present
#     gnd_components = [c for c in components if "gnd" in c["type"].lower()]
#     if gnd_components:
#         # Find all components connected to ground
#         gnd_connections = []
#         for conn in connections:
#             if any(c["id"] == conn["component1"] or c["id"] == conn["component2"] 
#                   for c in gnd_components):
#                 other_comp = conn["component1"] if conn["component1"] not in [c["id"] for c in gnd_components] else conn["component2"]
#                 gnd_connections.append((other_comp, 0))  # Node 0 is ground
#                 nodes[other_comp] = 0
    
#     # Assign node numbers to all other connections
#     for conn in connections:
#         comp1, comp2 = conn["component1"], conn["component2"]
        
#         # Skip if both are already assigned to the same node
#         if comp1 in nodes and comp2 in nodes and nodes[comp1] == nodes[comp2]:
#             continue
            
#         # If one has a node, assign the other to the same node
#         if comp1 in nodes:
#             nodes[comp2] = nodes[comp1]
#         elif comp2 in nodes:
#             nodes[comp1] = nodes[comp2]
#         # Otherwise, create a new node
#         else:
#             nodes[comp1] = node_counter
#             nodes[comp2] = node_counter
#             node_counter += 1
    
#     # Generate SPICE-like netlist text
#     spice_netlist = "* Circuit netlist generated from image\n"
#     spice_netlist += f"* Source: {os.path.basename(image_path)}\n\n"
    
#     # Add component declarations
#     for comp in components:
#         comp_id = comp["id"]
#         comp_type = comp["type"]
        
#         # Skip disconnected components
#         if comp_id in disconnected_components:
#             continue
        
#         # Find connections for this component
#         comp_connections = []
#         for conn in connections:
#             if conn["component1"] == comp_id:
#                 comp_connections.append((conn["component2"], nodes.get(conn["component2"], 999)))
#             elif conn["component2"] == comp_id:
#                 comp_connections.append((conn["component1"], nodes.get(conn["component1"], 999)))
        
#         # Format based on component type
#         if "resistor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"R{comp_id.split('_')[1]} {node_str} 1k\n"
#         elif "capacitor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"C{comp_id.split('_')[1]} {node_str} 1u\n"
#         elif "transistor" in comp_type.lower() or "bjt" in comp_type.lower():
#             # Assume 3 connections for transistors (collector, base, emitter)
#             if len(comp_connections) >= 3:
#                 nodes_str = " ".join(str(n[1]) for n in comp_connections[:3])
#                 spice_netlist += f"Q{comp_id.split('_')[1]} {nodes_str} 2N3904\n"
#         # Add more component types as needed
    
#     # Save all outputs
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Save visualization images
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_detection.png"), detection_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_connections.png"), connection_image)
    
#     # Save netlist data as JSON
#     with open(os.path.join(output_folder, f"{base_name}_netlist.json"), 'w') as f:
#         json.dump(netlist_data, f, indent=2, cls=NumpyEncoder)  # Use NumpyEncoder
    
#     # Save SPICE netlist
#     with open(os.path.join(output_folder, f"{base_name}.cir"), 'w') as f:
#         f.write(spice_netlist)
    
#     return netlist_data, spice_netlist

# def process_all_images(images_folder, model_path, output_folder):
#     """Process all circuit images in a folder."""
#     os.makedirs(output_folder, exist_ok=True)
    
#     image_files = [f for f in os.listdir(images_folder) 
#                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
        
#         # Create output folder for this image
#         image_output_folder = os.path.join(output_folder, 
#                                           os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Process the image
#         try:
#             netlist_data, spice_netlist = process_circuit_image(
#                 image_path, model_path, image_output_folder)
            
#             if netlist_data:
#                 print(f"  Generated netlist for {image_file}")
#                 print(f"  Output saved to {image_output_folder}")
#             else:
#                 print(f"  Failed to process {image_file}")
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path)




































































































# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# # Custom JSON encoder to handle NumPy types
# class NumpyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)

# def identify_circuit_nodes(merged_labels, components):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#     """
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Get a margin around the bounding box to check for adjacent nodes
#         margin = 2  # pixels to check around bounding box
#         x_min_check = max(0, x_min - margin)
#         y_min_check = max(0, y_min - margin)
#         x_max_check = min(node_labels.shape[1] - 1, x_max + margin)
#         y_max_check = min(node_labels.shape[0] - 1, y_max + margin)
        
#         # Check the perimeter of the bounding box (with margin)
#         # Top and bottom edges
#         for x in range(x_min_check, x_max_check):
#             for y in [y_min_check, y_max_check - 1]:
#                 node_id = node_labels[y, x]
#                 if node_id > 0:  # Ignore background (0)
#                     valid_nodes.add(node_id)
#                     node_to_components[int(node_id)].add(comp_id)
#                     component_to_nodes[comp_id].add(int(node_id))
        
#         # Left and right edges
#         for y in range(y_min_check, y_max_check):
#             for x in [x_min_check, x_max_check - 1]:
#                 node_id = node_labels[y, x]
#                 if node_id > 0:  # Ignore background (0)
#                     valid_nodes.add(node_id)
#                     node_to_components[int(node_id)].add(comp_id)
#                     component_to_nodes[comp_id].add(int(node_id))
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((node_labels.shape[0], node_labels.shape[1], 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, 0

# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Save the thickened binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
#     # Create a merged label image where all foreground pixels are set to 1
#     merged_labels = np.zeros_like(thickened_binary)
#     merged_labels[thickened_binary > 0] = 1
    
#     # Save the merged label image
#     merged_label_vis = np.zeros_like(original_image)
#     merged_label_vis[merged_labels > 0] = (0, 255, 0)  # Green for all circuit traces
#     cv2.imwrite(os.path.join(output_folder, "merged_labels.png"), merged_label_vis)
    
#     # Identify circuit nodes
#     node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug = identify_circuit_nodes(merged_labels, components)
    
#     # Save the node visualizations
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes.png"), node_vis)
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes_with_boxes.png"), node_vis_with_boxes)
#     cv2.imwrite(os.path.join(output_folder, "circuit_cut.png"), circuit_cut * 255)  # Multiply by 255 to make visible
#     # cv2.imwrite(os.path.join(output_folder, "overlap_debug.png"), overlap_debug)
    
#     # Save node information as JSON
#     with open(os.path.join(output_folder, "node_info.json"), 'w') as f:
#         json.dump(node_info, f, indent=2, cls=NumpyEncoder)
    
#     # Extract node-to-components mapping from node_info
#     node_to_components = {int(k): set(v) for k, v in node_info["node_to_components"].items()}
    
#     # Print debug information about node-to-component connections
#     print("\n========== NODE CONNECTIONS DEBUG ==========")
#     print(f"Image: {os.path.basename(image_path)}")
#     print(f"Total nodes detected: {len(node_to_components)}")
#     print(f"Total components: {len(components)}")

#     # Print components
#     print("\nComponents detected:")
#     for comp in components:
#         print(f"  {comp['id']}: {comp['type']} at {comp['bbox']}")

#     # Print each node and its connected components
#     print("\nNodes and their connected components:")
#     for node_id, connected_comps in sorted(node_to_components.items()):
#         # Get node color if available
#         node_color = node_info["node_colors"].get(str(node_id), [0, 0, 0])
#         print(f"Node {node_id} (Color: RGB{node_color}):")
        
#         # List all components connected to this node
#         for comp_id in connected_comps:
#             # Find component details
#             comp = next((c for c in components if c["id"] == comp_id), None)
#             if comp:
#                 print(f"  - {comp_id}: {comp['type']} at {comp['bbox']}")
        
#         # Calculate node size (number of pixels)
#         node_mask = (node_labels == node_id)
#         node_size = np.sum(node_mask)
        
#         # Check if node touches image edge
#         h, w = node_labels.shape
#         touches_edge = (
#             np.any(node_mask[0, :]) or  # Top edge
#             np.any(node_mask[h-1, :]) or  # Bottom edge
#             np.any(node_mask[:, 0]) or  # Left edge
#             np.any(node_mask[:, w-1])  # Right edge
#         )
        
#         print(f"  Node size: {node_size} pixels, Touches edge: {touches_edge}")
#         print()

#     print("=========================================\n")

#     # Save the debug information to a text file
#     with open(os.path.join(output_folder, "node_connections_debug.txt"), 'w') as f:
#         f.write(f"========== NODE CONNECTIONS DEBUG ==========\n")
#         f.write(f"Image: {os.path.basename(image_path)}\n")
#         f.write(f"Total nodes detected: {len(node_to_components)}\n")
#         f.write(f"Total components: {len(components)}\n\n")
        
#         f.write("Components detected:\n")
#         for comp in components:
#             f.write(f"  {comp['id']}: {comp['type']} at {comp['bbox']}\n")
        
#         f.write("\nNodes and their connected components:\n")
#         for node_id, connected_comps in sorted(node_to_components.items()):
#             node_color = node_info["node_colors"].get(str(node_id), [0, 0, 0])
#             f.write(f"Node {node_id} (Color: RGB{node_color}):\n")
            
#             for comp_id in connected_comps:
#                 comp = next((c for c in components if c["id"] == comp_id), None)
#                 if comp:
#                     f.write(f"  - {comp_id}: {comp['type']} at {comp['bbox']}\n")
            
#             node_mask = (node_labels == node_id)
#             node_size = np.sum(node_mask)
            
#             h, w = node_labels.shape
#             touches_edge = (
#                 np.any(node_mask[0, :]) or  # Top edge
#                 np.any(node_mask[h-1, :]) or  # Bottom edge
#                 np.any(node_mask[:, 0]) or  # Left edge
#                 np.any(node_mask[:, w-1])  # Right edge
#             )
            
#             f.write(f"  Node size: {node_size} pixels, Touches edge: {touches_edge}\n\n")
        
#         f.write("=========================================\n")
    
#     # Create connections based on overlap detection results
#     connections = []
#     processed_pairs = set()
    
#     # For each node that connects to multiple components, create connections between those components
#     for node_id, connected_comps in node_to_components.items():
#         connected_comps_list = list(connected_comps)
        
#         # If this node connects multiple components, create connections between them
#         if len(connected_comps_list) >= 2:
#             for i in range(len(connected_comps_list)):
#                 for j in range(i+1, len(connected_comps_list)):
#                     comp_id1 = connected_comps_list[i]
#                     comp_id2 = connected_comps_list[j]
                    
#                     # Create an ordered pair to avoid duplicates
#                     pair = tuple(sorted([comp_id1, comp_id2]))
                    
#                     if pair not in processed_pairs:
#                         # Find the components by ID
#                         comp1 = next(c for c in components if c["id"] == comp_id1)
#                         comp2 = next(c for c in components if c["id"] == comp_id2)
                        
#                         # Get bounding boxes
#                         bbox1 = comp1["bbox"]
#                         bbox2 = comp2["bbox"]
                        
#                         # Find the node's mask
#                         node_mask = (node_labels == node_id)
#                         node_y, node_x = np.where(node_mask)
                        
#                         # For each component, find where its boundary intersects with this node
#                         # Component 1
#                         comp1_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox1
                        
#                         # Create a perimeter mask for the component
#                         comp1_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp1_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp1_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp1_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp1_intersect = np.logical_and(comp1_boundary, node_mask)
#                         comp1_y, comp1_x = np.where(comp1_intersect)
                        
#                         # Component 2
#                         comp2_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox2
                        
#                         # Create a perimeter mask for the component
#                         comp2_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp2_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp2_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp2_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp2_intersect = np.logical_and(comp2_boundary, node_mask)
#                         comp2_y, comp2_x = np.where(comp2_intersect)
                        
#                         # Get connection points
#                         if len(comp1_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp1_x) // 2
#                             comp1_point = (int(comp1_x[idx]), int(comp1_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp1_point = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
                        
#                         if len(comp2_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp2_x) // 2
#                             comp2_point = (int(comp2_x[idx]), int(comp2_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp2_point = (int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2))
                        
#                         # Record the connection
#                         connections.append({
#                             "component1": comp_id1,
#                             "component2": comp_id2,
#                             "connect_point1": [int(comp1_point[0]), int(comp1_point[1])],
#                             "connect_point2": [int(comp2_point[0]), int(comp2_point[1])],
#                             "node_id": int(node_id)
#                         })
                        
#                         processed_pairs.add(pair)
    
#     # Create a connection visualization
#     connection_image = original_image.copy()
    
#     # Draw all connections
#     for conn in connections:
#         comp1_point = tuple(conn["connect_point1"])
#         comp2_point = tuple(conn["connect_point2"])
        
#         # Draw connection line
#         cv2.line(connection_image, comp1_point, comp2_point, (255, 0, 0), 2)
        
#         # Draw connection points
#         cv2.circle(connection_image, comp1_point, 4, (0, 0, 255), -1)
#         cv2.circle(connection_image, comp2_point, 4, (0, 0, 255), -1)
    
#     # Identify disconnected components
#     all_components = {comp["id"] for comp in components}
#     connected_components = set()
#     for conn in connections:
#         connected_components.add(conn["component1"])
#         connected_components.add(conn["component2"])
#     disconnected_components = all_components - connected_components
    
#     if disconnected_components:
#         print(f"WARNING: {os.path.basename(image_path)} has a bad design - disconnected components: {disconnected_components}")
    
#     # Generate netlist data
#     netlist_components = []
#     for comp in components:
#         netlist_components.append({
#             "id": comp["id"],
#             "type": comp["type"],
#             "position": {
#                 "x": int((comp["bbox"][0] + comp["bbox"][2]) / 2),
#                 "y": int((comp["bbox"][1] + comp["bbox"][3]) / 2)
#             },
#             "connections": [c for c in connections 
#                             if c["component1"] == comp["id"] or c["component2"] == comp["id"]]
#         })
    
#     # Create netlist data structure
#     netlist_data = {
#         "components": netlist_components,
#         "connections": connections,
#         "nodes": node_info
#     }
    
#     # Generate SPICE-like netlist
#     nodes = {}
#     node_counter = 1  # Start from 1 (0 is usually ground)
    
#     # Identify GND node if present
#     gnd_components = [c for c in components if "gnd" in c["type"].lower()]
#     if gnd_components:
#         # Find all components connected to ground
#         gnd_connections = []
#         for conn in connections:
#             if any(c["id"] == conn["component1"] or c["id"] == conn["component2"] 
#                   for c in gnd_components):
#                 other_comp = conn["component1"] if conn["component1"] not in [c["id"] for c in gnd_components] else conn["component2"]
#                 gnd_connections.append((other_comp, 0))  # Node 0 is ground
#                 nodes[other_comp] = 0
    
#     # Assign node numbers to all other connections
#     for conn in connections:
#         comp1, comp2 = conn["component1"], conn["component2"]
        
#         # Skip if both are already assigned to the same node
#         if comp1 in nodes and comp2 in nodes and nodes[comp1] == nodes[comp2]:
#             continue
            
#         # If one has a node, assign the other to the same node
#         if comp1 in nodes:
#             nodes[comp2] = nodes[comp1]
#         elif comp2 in nodes:
#             nodes[comp1] = nodes[comp2]
#         # Otherwise, create a new node
#         else:
#             nodes[comp1] = node_counter
#             nodes[comp2] = node_counter
#             node_counter += 1
    
#     # Generate SPICE-like netlist text
#     spice_netlist = "* Circuit netlist generated from image\n"
#     spice_netlist += f"* Source: {os.path.basename(image_path)}\n\n"
    
#     # Add component declarations
#     for comp in components:
#         comp_id = comp["id"]
#         comp_type = comp["type"]
        
#         # Skip disconnected components
#         if comp_id in disconnected_components:
#             continue
        
#         # Find connections for this component
#         comp_connections = []
#         for conn in connections:
#             if conn["component1"] == comp_id:
#                 comp_connections.append((conn["component2"], nodes.get(conn["component2"], 999)))
#             elif conn["component2"] == comp_id:
#                 comp_connections.append((conn["component1"], nodes.get(conn["component1"], 999)))
        
#         # Format based on component type
#         if "resistor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"R{comp_id.split('_')[1]} {node_str} 1k\n"
#         elif "capacitor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"C{comp_id.split('_')[1]} {node_str} 1u\n"
#         elif "transistor" in comp_type.lower() or "bjt" in comp_type.lower():
#             # Assume 3 connections for transistors (collector, base, emitter)
#             if len(comp_connections) >= 3:
#                 nodes_str = " ".join(str(n[1]) for n in comp_connections[:3])
#                 spice_netlist += f"Q{comp_id.split('_')[1]} {nodes_str} 2N3904\n"
#         # Add more component types as needed
    
#     # Save all outputs
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Save visualization images
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_detection.png"), detection_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_connections.png"), connection_image)
    
#     # Save netlist data as JSON
#     with open(os.path.join(output_folder, f"{base_name}_netlist.json"), 'w') as f:
#         json.dump(netlist_data, f, indent=2, cls=NumpyEncoder)  # Use NumpyEncoder
    
#     # Save SPICE netlist
#     with open(os.path.join(output_folder, f"{base_name}.cir"), 'w') as f:
#         f.write(spice_netlist)
    
#     return netlist_data, spice_netlist

# def process_all_images(images_folder, model_path, output_folder):
#     """Process all circuit images in a folder."""
#     os.makedirs(output_folder, exist_ok=True)
    
#     image_files = [f for f in os.listdir(images_folder) 
#                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
        
#         # Create output folder for this image
#         image_output_folder = os.path.join(output_folder, 
#                                           os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Process the image
#         try:
#             netlist_data, spice_netlist = process_circuit_image(
#                 image_path, model_path, image_output_folder)
            
#             if netlist_data:
#                 print(f"  Generated netlist for {image_file}")
#                 print(f"  Output saved to {image_output_folder}")
#             else:
#                 print(f"  Failed to process {image_file}")
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path)
























# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# class NumpyEncoder(json.JSONEncoder):
#     """
#     Custom JSON encoder for numpy types.
#     """
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)


# def identify_circuit_nodes(merged_labels, components):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#         overlap_debug: Debug visualization showing overlap detection
#     """
#     h, w = merged_labels.shape
    
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Create a debug visualization to check overlap detection
#     overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Set up colors for debug visualization
#     # Circuit traces - green
#     overlap_debug[circuit_image > 0] = [0, 255, 0]
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     # Check every pixel in the image (exhaustive but guaranteed to catch all overlaps)
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Draw component bounding box on debug image - blue
#         cv2.rectangle(overlap_debug, (x_min, y_min), (x_max, y_max), [255, 0, 0], 1)
        
#         # Create a mask for just this component's bounding box perimeter
#         # This ensures we only check the perimeter, not the interior
#         comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        
#         # Top and bottom edges
#         if y_min >= 0 and y_min < h:
#             comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#         if y_max >= 0 and y_max < h:
#             comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
            
#         # Left and right edges
#         if x_min >= 0 and x_min < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#         if x_max >= 0 and x_max < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
        
#         # Get node labels that intersect with this component's perimeter
#         for label_id in range(1, num_labels):  # Skip background (0)
#             # Check if this node intersects with the component perimeter
#             node_mask = (node_labels == label_id)
#             intersection = np.logical_and(comp_perimeter, node_mask)
            
#             if np.any(intersection):
#                 # This node intersects with the component's perimeter
#                 valid_nodes.add(label_id)
#                 node_to_components[int(label_id)].add(comp_id)
#                 component_to_nodes[comp_id].add(int(label_id))
                
#                 # For debug visualization - red for detected intersections
#                 overlap_debug[intersection] = [0, 0, 255]
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug

# def identify_circuit_nodes(merged_labels, components):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
#     Filter to find real nodes where two or more circuit components are connected.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#         overlap_debug: Debug visualization showing overlap detection
#         real_nodes: Set of node IDs that are real nodes (connected to 2+ components)
#         real_node_info: Dictionary with information about real nodes
#         real_node_vis: Visualization of real nodes
#     """
#     h, w = merged_labels.shape
    
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Create a debug visualization to check overlap detection
#     overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Set up colors for debug visualization
#     # Circuit traces - green
#     overlap_debug[circuit_image > 0] = [0, 255, 0]
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     # Check every pixel in the image (exhaustive but guaranteed to catch all overlaps)
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Draw component bounding box on debug image - blue
#         cv2.rectangle(overlap_debug, (x_min, y_min), (x_max, y_max), [255, 0, 0], 1)
        
#         # Create a mask for just this component's bounding box perimeter
#         # This ensures we only check the perimeter, not the interior
#         comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        
#         # Top and bottom edges
#         if y_min >= 0 and y_min < h:
#             comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#         if y_max >= 0 and y_max < h:
#             comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
            
#         # Left and right edges
#         if x_min >= 0 and x_min < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#         if x_max >= 0 and x_max < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
        
#         # Get node labels that intersect with this component's perimeter
#         for label_id in range(1, num_labels):  # Skip background (0)
#             # Check if this node intersects with the component perimeter
#             node_mask = (node_labels == label_id)
#             intersection = np.logical_and(comp_perimeter, node_mask)
            
#             if np.any(intersection):
#                 # This node intersects with the component's perimeter
#                 valid_nodes.add(label_id)
#                 node_to_components[int(label_id)].add(comp_id)
#                 component_to_nodes[comp_id].add(int(label_id))
                
#                 # For debug visualization - red for detected intersections
#                 overlap_debug[intersection] = [0, 0, 255]
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     # Find real nodes (nodes where two or more circuit components are connected)
#     real_nodes = set()
#     for node_id, components_list in node_to_components.items():
#         if len(components_list) >= 2:
#             real_nodes.add(node_id)
    
#     # Create a filtered node_info dictionary for real nodes
#     real_node_info = {
#         "valid_nodes": list(real_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items() if k in real_nodes},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items() if k in real_nodes}
#     }
    
#     # Update component_to_nodes mapping to include only real nodes
#     component_to_real_nodes = {}
#     for comp_id, nodes in component_to_nodes.items():
#         real_comp_nodes = [node_id for node_id in nodes if node_id in real_nodes]
#         if real_comp_nodes:
#             component_to_real_nodes[comp_id] = real_comp_nodes
    
#     real_node_info["component_to_nodes"] = component_to_real_nodes
    
#     # Create a visualization of real nodes
#     real_node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Color each real node using its original color
#     for node_id in real_nodes:
#         if node_id in node_colors:
#             real_node_vis[node_labels == node_id] = node_colors[node_id]
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug, real_nodes, real_node_info, real_node_vis

# def identify_circuit_nodes(merged_labels, components):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
#     Filter to find real nodes where two or more circuit components are connected.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#         overlap_debug: Debug visualization showing overlap detection
#         real_nodes: Set of node IDs that are real nodes (connected to 2+ components)
#         real_node_info: Dictionary with information about real nodes
#         real_node_vis: Visualization of real nodes
#         real_node_vis_with_boxes: Visualization of real nodes with component bounding boxes
#     """
#     h, w = merged_labels.shape
    
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Create a debug visualization to check overlap detection
#     overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Set up colors for debug visualization
#     # Circuit traces - green
#     overlap_debug[circuit_image > 0] = [0, 255, 0]
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     # Check every pixel in the image (exhaustive but guaranteed to catch all overlaps)
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Draw component bounding box on debug image - blue
#         cv2.rectangle(overlap_debug, (x_min, y_min), (x_max, y_max), [255, 0, 0], 1)
        
#         # Create a mask for just this component's bounding box perimeter
#         # This ensures we only check the perimeter, not the interior
#         comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        
#         # Top and bottom edges
#         if y_min >= 0 and y_min < h:
#             comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#         if y_max >= 0 and y_max < h:
#             comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
            
#         # Left and right edges
#         if x_min >= 0 and x_min < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#         if x_max >= 0 and x_max < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
        
#         # Get node labels that intersect with this component's perimeter
#         for label_id in range(1, num_labels):  # Skip background (0)
#             # Check if this node intersects with the component perimeter
#             node_mask = (node_labels == label_id)
#             intersection = np.logical_and(comp_perimeter, node_mask)
            
#             if np.any(intersection):
#                 # This node intersects with the component's perimeter
#                 valid_nodes.add(label_id)
#                 node_to_components[int(label_id)].add(comp_id)
#                 component_to_nodes[comp_id].add(int(label_id))
                
#                 # For debug visualization - red for detected intersections
#                 overlap_debug[intersection] = [0, 0, 255]
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     # Find real nodes (nodes where two or more circuit components are connected)
#     real_nodes = set()
#     for node_id, components_list in node_to_components.items():
#         if len(components_list) >= 2:
#             real_nodes.add(node_id)
    
#     # Create a filtered node_info dictionary for real nodes
#     real_node_info = {
#         "valid_nodes": list(real_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items() if k in real_nodes},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items() if k in real_nodes}
#     }
    
#     # Update component_to_nodes mapping to include only real nodes
#     component_to_real_nodes = {}
#     for comp_id, nodes in component_to_nodes.items():
#         real_comp_nodes = [node_id for node_id in nodes if node_id in real_nodes]
#         if real_comp_nodes:
#             component_to_real_nodes[comp_id] = real_comp_nodes
    
#     real_node_info["component_to_nodes"] = component_to_real_nodes
    
#     # Create a visualization of real nodes
#     real_node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Color each real node using its original color
#     for node_id in real_nodes:
#         if node_id in node_colors:
#             real_node_vis[node_labels == node_id] = node_colors[node_id]
            
#     # Create a visualization of real nodes with component bounding boxes
#     real_node_vis_with_boxes = real_node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(real_node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug, real_nodes, real_node_info, real_node_vis, real_node_vis_with_boxes

# def identify_circuit_nodes(merged_labels, components, output_folder):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
#     Filter to find real nodes where two or more circuit components are connected.
#     Identify connection points between components and nodes, and write to a text file.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
#         output_folder: Path to the output folder for component connections
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#         overlap_debug: Debug visualization showing overlap detection
#         real_nodes: Set of node IDs that are real nodes (connected to 2+ components)
#         real_node_info: Dictionary with information about real nodes
#         real_node_vis: Visualization of real nodes
#         real_node_vis_with_boxes: Visualization of real nodes with component bounding boxes
#         component_connections: Dictionary mapping component IDs to lists of connection points
#     """
#     h, w = merged_labels.shape
    
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Create a debug visualization to check overlap detection
#     overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Set up colors for debug visualization
#     # Circuit traces - green
#     overlap_debug[circuit_image > 0] = [0, 255, 0]
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     # Check every pixel in the image (exhaustive but guaranteed to catch all overlaps)
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Draw component bounding box on debug image - blue
#         cv2.rectangle(overlap_debug, (x_min, y_min), (x_max, y_max), [255, 0, 0], 1)
        
#         # Create a mask for just this component's bounding box perimeter
#         # This ensures we only check the perimeter, not the interior
#         comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        
#         # Top and bottom edges
#         if y_min >= 0 and y_min < h:
#             comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#         if y_max >= 0 and y_max < h:
#             comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
            
#         # Left and right edges
#         if x_min >= 0 and x_min < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#         if x_max >= 0 and x_max < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
        
#         # Get node labels that intersect with this component's perimeter
#         for label_id in range(1, num_labels):  # Skip background (0)
#             # Check if this node intersects with the component perimeter
#             node_mask = (node_labels == label_id)
#             intersection = np.logical_and(comp_perimeter, node_mask)
            
#             if np.any(intersection):
#                 # This node intersects with the component's perimeter
#                 valid_nodes.add(label_id)
#                 node_to_components[int(label_id)].add(comp_id)
#                 component_to_nodes[comp_id].add(int(label_id))
                
#                 # For debug visualization - red for detected intersections
#                 overlap_debug[intersection] = [0, 0, 255]
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     # Find real nodes (nodes where two or more circuit components are connected)
#     real_nodes = set()
#     for node_id, components_list in node_to_components.items():
#         if len(components_list) >= 2:
#             real_nodes.add(node_id)
    
#     # Create a filtered node_info dictionary for real nodes
#     real_node_info = {
#         "valid_nodes": list(real_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items() if k in real_nodes},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items() if k in real_nodes}
#     }
    
#     # Update component_to_nodes mapping to include only real nodes
#     component_to_real_nodes = {}
#     for comp_id, nodes in component_to_nodes.items():
#         real_comp_nodes = [node_id for node_id in nodes if node_id in real_nodes]
#         if real_comp_nodes:
#             component_to_real_nodes[comp_id] = real_comp_nodes
    
#     real_node_info["component_to_nodes"] = component_to_real_nodes
    
#     # Create a visualization of real nodes
#     real_node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Color each real node using its original color
#     for node_id in real_nodes:
#         if node_id in node_colors:
#             real_node_vis[node_labels == node_id] = node_colors[node_id]
            
#     # Create a visualization of real nodes with component bounding boxes
#     real_node_vis_with_boxes = real_node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(real_node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Find connection points between components and real nodes
#     component_connections = {}
    
#     # For each component, find its connections to real nodes
#     for component in components:
#         comp_id = component["id"]
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Skip if component is not connected to any real nodes
#         if comp_id not in real_node_info["component_to_nodes"]:
#             continue
        
#         # Get the real nodes this component connects to
#         connected_real_nodes = real_node_info["component_to_nodes"][comp_id]
        
#         # Find connection points for each real node
#         connections = []
#         for node_id in connected_real_nodes:
#             # Create a mask for this node
#             node_mask = (node_labels == node_id)
            
#             # Create a mask for component perimeter
#             comp_perimeter = np.zeros_like(node_mask)
            
#             # Top and bottom edges
#             if y_min >= 0 and y_min < h:
#                 comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#             if y_max >= 0 and y_max < h:
#                 comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
                
#             # Left and right edges
#             if x_min >= 0 and x_min < w:
#                 comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#             if x_max >= 0 and x_max < w:
#                 comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
            
#             # Find intersection points
#             intersection = np.logical_and(node_mask, comp_perimeter)
#             intersection_points = np.where(intersection)
            
#             if len(intersection_points[0]) > 0:
#                 # Calculate the middle connection point
#                 y_points = intersection_points[0]
#                 x_points = intersection_points[1]
                
#                 # Get the middle point
#                 middle_idx = len(x_points) // 2
#                 middle_y = y_points[middle_idx]
#                 middle_x = x_points[middle_idx]
                
#                 connections.append((middle_x, middle_y))
        
#         # Store the connections for this component
#         component_connections[comp_id] = connections
    
#     # Write component connections to a text file
#     import os
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Create the file path by joining the folder path with a filename
#     output_file = os.path.join(output_folder, "component_connections.txt")
    
#     try:
#         with open(output_file, 'w') as f:
#             for comp_id, points in component_connections.items():
#                 connection_str = f"{comp_id} "
#                 for i, (x, y) in enumerate(points):
#                     if i > 0:
#                         connection_str += ", "
#                     connection_str += f"({x}, {y})"
#                 f.write(connection_str + "\n")
#     except Exception as e:
#         print(f"Warning: Could not write to {output_file}: {e}")
    
#     return node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug, real_nodes, real_node_info, real_node_vis, real_node_vis_with_boxes, component_connections


# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Save the thickened binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
#     # Create a merged label image where all foreground pixels are set to 1
#     merged_labels = np.zeros_like(thickened_binary)
#     merged_labels[thickened_binary > 0] = 1
    
#     # Save the merged label image
#     merged_label_vis = np.zeros_like(original_image)
#     merged_label_vis[merged_labels > 0] = (0, 255, 0)  # Green for all circuit traces
#     cv2.imwrite(os.path.join(output_folder, "merged_labels.png"), merged_label_vis)
    
#     # Identify circuit nodes
#     # node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug = identify_circuit_nodes(merged_labels, components)
#     # node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug, real_nodes, real_node_info, real_node_vis = identify_circuit_nodes(merged_labels, components)
#     (node_labels, node_info, node_vis, node_vis_with_boxes, 
#     circuit_cut, overlap_debug, real_nodes, real_node_info, 
#     real_node_vis, real_node_vis_with_boxes) = identify_circuit_nodes(merged_labels, components, output_folder)

#     # Save the node visualizations
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes.png"), node_vis)
#     cv2.imwrite(os.path.join(output_folder, "real_circuit_nodes_with_boxes.png"), real_node_vis_with_boxes)
#     cv2.imwrite(os.path.join(output_folder, "real_circuit_nodes.png"), real_node_vis)
#     cv2.imwrite(os.path.join(output_folder, "circuit_nodes_with_boxes.png"), node_vis_with_boxes)
#     cv2.imwrite(os.path.join(output_folder, "circuit_cut.png"), circuit_cut * 255)  # Multiply by 255 to make visible
#     cv2.imwrite(os.path.join(output_folder, "overlap_debug.png"), overlap_debug)
    
#     # Save node information as JSON
#     with open(os.path.join(output_folder, "node_info.json"), 'w') as f:
#         json.dump(node_info, f, indent=2, cls=NumpyEncoder)
    
#     # Extract node-to-components mapping from node_info
#     node_to_components = {int(k): set(v) for k, v in node_info["node_to_components"].items()}
    
#     # Create connections based on overlap detection results
#     connections = []
#     processed_pairs = set()
    
#     # For each node that connects to multiple components, create connections between those components
#     for node_id, connected_comps in node_to_components.items():
#         connected_comps_list = list(connected_comps)
        
#         # If this node connects multiple components, create connections between them
#         if len(connected_comps_list) >= 2:
#             for i in range(len(connected_comps_list)):
#                 for j in range(i+1, len(connected_comps_list)):
#                     comp_id1 = connected_comps_list[i]
#                     comp_id2 = connected_comps_list[j]
                    
#                     # Create an ordered pair to avoid duplicates
#                     pair = tuple(sorted([comp_id1, comp_id2]))
                    
#                     if pair not in processed_pairs:
#                         # Find the components by ID
#                         comp1 = next(c for c in components if c["id"] == comp_id1)
#                         comp2 = next(c for c in components if c["id"] == comp_id2)
                        
#                         # Get bounding boxes
#                         bbox1 = comp1["bbox"]
#                         bbox2 = comp2["bbox"]
                        
#                         # Find the node's mask
#                         node_mask = (node_labels == node_id)
#                         node_y, node_x = np.where(node_mask)
                        
#                         # For each component, find where its boundary intersects with this node
#                         # Component 1
#                         comp1_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox1
                        
#                         # Create a perimeter mask for the component
#                         comp1_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp1_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp1_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp1_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp1_intersect = np.logical_and(comp1_boundary, node_mask)
#                         comp1_y, comp1_x = np.where(comp1_intersect)
                        
#                         # Component 2
#                         comp2_boundary = np.zeros_like(node_labels)
#                         x_min, y_min, x_max, y_max = bbox2
                        
#                         # Create a perimeter mask for the component
#                         comp2_boundary[y_min, x_min:x_max+1] = 1  # Top edge
#                         comp2_boundary[y_max, x_min:x_max+1] = 1  # Bottom edge
#                         comp2_boundary[y_min:y_max+1, x_min] = 1  # Left edge
#                         comp2_boundary[y_min:y_max+1, x_max] = 1  # Right edge
                        
#                         # Find intersection of component boundary with node
#                         comp2_intersect = np.logical_and(comp2_boundary, node_mask)
#                         comp2_y, comp2_x = np.where(comp2_intersect)
                        
#                         # Get connection points
#                         if len(comp1_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp1_x) // 2
#                             comp1_point = (int(comp1_x[idx]), int(comp1_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp1_point = (int((bbox1[0] + bbox1[2]) / 2), int((bbox1[1] + bbox1[3]) / 2))
                        
#                         if len(comp2_x) > 0:
#                             # Use median point for stability
#                             idx = len(comp2_x) // 2
#                             comp2_point = (int(comp2_x[idx]), int(comp2_y[idx]))
#                         else:
#                             # Fallback to component center
#                             comp2_point = (int((bbox2[0] + bbox2[2]) / 2), int((bbox2[1] + bbox2[3]) / 2))
                        
#                         # Record the connection
#                         connections.append({
#                             "component1": comp_id1,
#                             "component2": comp_id2,
#                             "connect_point1": [int(comp1_point[0]), int(comp1_point[1])],
#                             "connect_point2": [int(comp2_point[0]), int(comp2_point[1])],
#                             "node_id": int(node_id)
#                         })
                        
#                         processed_pairs.add(pair)
    
#     # Create a connection visualization
#     connection_image = original_image.copy()
    
#     # Draw all connections
#     for conn in connections:
#         comp1_point = tuple(conn["connect_point1"])
#         comp2_point = tuple(conn["connect_point2"])
        
#         # Draw connection line
#         cv2.line(connection_image, comp1_point, comp2_point, (255, 0, 0), 2)
        
#         # Draw connection points
#         cv2.circle(connection_image, comp1_point, 4, (0, 0, 255), -1)
#         cv2.circle(connection_image, comp2_point, 4, (0, 0, 255), -1)
    
#     # Group connection points by component
#     component_connection_points = defaultdict(list)

#     # Gather all connection points for each component
#     for conn in connections:
#         comp1_id = conn["component1"]
#         comp2_id = conn["component2"]
#         comp1_point = conn["connect_point1"]
#         comp2_point = conn["connect_point2"]
        
#         component_connection_points[comp1_id].append(tuple(comp1_point))
#         component_connection_points[comp2_id].append(tuple(comp2_point))

#     # Print connection points for each component
#     print(f"\nConnection points for {os.path.basename(image_path)}:")
#     for comp_id, points in component_connection_points.items():
#         # Remove duplicates while preserving order
#         unique_points = []
#         for point in points:
#             if point not in unique_points:
#                 unique_points.append(point)
        
#         points_str = ", ".join([f"({x}, {y})" for x, y in unique_points])
#         print(f"{comp_id}: {points_str}")
    
#     # Identify disconnected components
#     all_components = {comp["id"] for comp in components}
#     connected_components = set()
#     for conn in connections:
#         connected_components.add(conn["component1"])
#         connected_components.add(conn["component2"])
#     disconnected_components = all_components - connected_components
    
#     if disconnected_components:
#         print(f"WARNING: {os.path.basename(image_path)} has a bad design - disconnected components: {disconnected_components}")
    
#     # Generate netlist data
#     netlist_components = []
#     for comp in components:
#         netlist_components.append({
#             "id": comp["id"],
#             "type": comp["type"],
#             "position": {
#                 "x": int((comp["bbox"][0] + comp["bbox"][2]) / 2),
#                 "y": int((comp["bbox"][1] + comp["bbox"][3]) / 2)
#             },
#             "connections": [c for c in connections 
#                             if c["component1"] == comp["id"] or c["component2"] == comp["id"]]
#         })
    
#     # Create netlist data structure
#     netlist_data = {
#         "components": netlist_components,
#         "connections": connections,
#         "nodes": node_info
#     }
    
#     # Generate SPICE-like netlist
#     nodes = {}
#     node_counter = 1  # Start from 1 (0 is usually ground)
    
#     # Identify GND node if present
#     gnd_components = [c for c in components if "gnd" in c["type"].lower()]
#     if gnd_components:
#         # Find all components connected to ground
#         gnd_connections = []
#         for conn in connections:
#             if any(c["id"] == conn["component1"] or c["id"] == conn["component2"] 
#                   for c in gnd_components):
#                 other_comp = conn["component1"] if conn["component1"] not in [c["id"] for c in gnd_components] else conn["component2"]
#                 gnd_connections.append((other_comp, 0))  # Node 0 is ground
#                 nodes[other_comp] = 0
    
#     # Assign node numbers to all other connections
#     for conn in connections:
#         comp1, comp2 = conn["component1"], conn["component2"]
        
#         # Skip if both are already assigned to the same node
#         if comp1 in nodes and comp2 in nodes and nodes[comp1] == nodes[comp2]:
#             continue
            
#         # If one has a node, assign the other to the same node
#         if comp1 in nodes:
#             nodes[comp2] = nodes[comp1]
#         elif comp2 in nodes:
#             nodes[comp1] = nodes[comp2]
#         # Otherwise, create a new node
#         else:
#             nodes[comp1] = node_counter
#             nodes[comp2] = node_counter
#             node_counter += 1
    
#     # Generate SPICE-like netlist text
#     spice_netlist = "* Circuit netlist generated from image\n"
#     spice_netlist += f"* Source: {os.path.basename(image_path)}\n\n"
    
#     # Add component declarations
#     for comp in components:
#         comp_id = comp["id"]
#         comp_type = comp["type"]
        
#         # Skip disconnected components
#         if comp_id in disconnected_components:
#             continue
        
#         # Find connections for this component
#         comp_connections = []
#         for conn in connections:
#             if conn["component1"] == comp_id:
#                 comp_connections.append((conn["component2"], nodes.get(conn["component2"], 999)))
#             elif conn["component2"] == comp_id:
#                 comp_connections.append((conn["component1"], nodes.get(conn["component1"], 999)))
        
#         # Format based on component type
#         if "resistor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"R{comp_id.split('_')[1]} {node_str} 1k\n"
#         elif "capacitor" in comp_type.lower():
#             node_str = " ".join(str(n[1]) for n in comp_connections[:2])
#             spice_netlist += f"C{comp_id.split('_')[1]} {node_str} 1u\n"
#         elif "transistor" in comp_type.lower() or "bjt" in comp_type.lower():
#             # Assume 3 connections for transistors (collector, base, emitter)
#             if len(comp_connections) >= 3:
#                 nodes_str = " ".join(str(n[1]) for n in comp_connections[:3])
#                 spice_netlist += f"Q{comp_id.split('_')[1]} {nodes_str} 2N3904\n"
#         # Add more component types as needed
    
#     # Save all outputs
#     base_name = os.path.splitext(os.path.basename(image_path))[0]
    
#     # Save visualization images
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_detection.png"), detection_image)
#     cv2.imwrite(os.path.join(output_folder, f"{base_name}_connections.png"), connection_image)
    
#     # Save netlist data as JSON
#     with open(os.path.join(output_folder, f"{base_name}_netlist.json"), 'w') as f:
#         json.dump(netlist_data, f, indent=2, cls=NumpyEncoder)  # Use NumpyEncoder
    
#     # Save SPICE netlist
#     with open(os.path.join(output_folder, f"{base_name}.cir"), 'w') as f:
#         f.write(spice_netlist)
    
#     return netlist_data, spice_netlist









# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import json
# import random
# from collections import defaultdict

# class NumpyEncoder(json.JSONEncoder):
#     """
#     Custom JSON encoder for numpy types.
#     """
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NumpyEncoder, self).default(obj)
    


# def identify_circuit_nodes(merged_labels, components, output_folder):
#     """
#     Identify circuit nodes by segmenting the merged label using component bounding boxes.
#     Filter to find real nodes where two or more circuit components are connected.
#     Identify connection points between components and nodes, and write to a text file.
    
#     Args:
#         merged_labels: Binary image where all circuit traces are marked as 1
#         components: List of component dictionaries with bounding box information
#         output_folder: Path to the output folder for component connections
    
#     Returns:
#         node_labels: Label matrix where each circuit node has a unique label ID
#         node_info: Dictionary with information about each node
#         node_vis: Visualization of circuit nodes
#         node_vis_with_boxes: Visualization with component bounding boxes
#         circuit_cut: Binary image showing where the circuit was cut
#         overlap_debug: Debug visualization showing overlap detection
#         real_nodes: Set of node IDs that are real nodes (connected to 2+ components)
#         real_node_info: Dictionary with information about real nodes
#         real_node_vis: Visualization of real nodes
#         real_node_vis_with_boxes: Visualization of real nodes with component bounding boxes
#         component_connections: Dictionary mapping component IDs to lists of connection points
#     """
#     h, w = merged_labels.shape
    
#     # Create a copy of the merged labels
#     circuit_image = merged_labels.copy()
    
#     # Create a mask for all component bounding boxes (slightly reduced)
#     component_mask = np.zeros_like(circuit_image)
    
#     # Track the reduced bounding boxes for each component
#     component_reduced_boxes = []
    
#     # The amount to shrink bounding boxes by (in pixels)
#     shrink_amount = 3
    
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Reduce the bounding box slightly to avoid cutting at component boundaries
#         x_min_reduced = x_min + shrink_amount
#         y_min_reduced = y_min + shrink_amount
#         x_max_reduced = x_max - shrink_amount
#         y_max_reduced = y_max - shrink_amount
        
#         # Make sure the reduced box is valid
#         if x_min_reduced < x_max_reduced and y_min_reduced < y_max_reduced:
#             # Record the reduced bounding box
#             component_reduced_boxes.append((x_min_reduced, y_min_reduced, x_max_reduced, y_max_reduced))
            
#             # Fill the mask with 1s for this component's reduced bounding box
#             component_mask[y_min_reduced:y_max_reduced, x_min_reduced:x_max_reduced] = 1
    
#     # Cut the circuit at component bounding boxes
#     # (Set circuit pixels to 0 where component_mask is 1)
#     circuit_cut = circuit_image.copy()
#     circuit_cut[component_mask == 1] = 0
    
#     # Now perform connected components analysis on the cut circuit
#     num_labels, node_labels, stats, centroids = cv2.connectedComponentsWithStats(circuit_cut.astype(np.uint8), connectivity=8)
    
#     # Create a debug visualization to check overlap detection
#     overlap_debug = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Set up colors for debug visualization
#     # Circuit traces - green
#     overlap_debug[circuit_image > 0] = [0, 255, 0]
    
#     # Check which nodes connect to component bounding boxes
#     # (These are the "real" nodes that we want to keep)
#     valid_nodes = set()
#     node_to_components = defaultdict(set)
#     component_to_nodes = defaultdict(set)
    
#     # Check every pixel in the image (exhaustive but guaranteed to catch all overlaps)
#     for i, component in enumerate(components):
#         x_min, y_min, x_max, y_max = component["bbox"]
#         comp_id = component["id"]
        
#         # Draw component bounding box on debug image - blue
#         cv2.rectangle(overlap_debug, (x_min, y_min), (x_max, y_max), [255, 0, 0], 1)
        
#         # Create a mask for just this component's bounding box perimeter
#         # This ensures we only check the perimeter, not the interior
#         comp_perimeter = np.zeros((h, w), dtype=np.uint8)
        
#         # Top and bottom edges
#         if y_min >= 0 and y_min < h:
#             comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#         if y_max >= 0 and y_max < h:
#             comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
            
#         # Left and right edges
#         if x_min >= 0 and x_min < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#         if x_max >= 0 and x_max < w:
#             comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
        
#         # Get node labels that intersect with this component's perimeter
#         for label_id in range(1, num_labels):  # Skip background (0)
#             # Check if this node intersects with the component perimeter
#             node_mask = (node_labels == label_id)
#             intersection = np.logical_and(comp_perimeter, node_mask)
            
#             if np.any(intersection):
#                 # This node intersects with the component's perimeter
#                 valid_nodes.add(label_id)
#                 node_to_components[int(label_id)].add(comp_id)
#                 component_to_nodes[comp_id].add(int(label_id))
                
#                 # For debug visualization - red for detected intersections
#                 overlap_debug[intersection] = [0, 0, 255]
    
#     # Create a visualization of the circuit nodes
#     node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Generate a color for each valid node
#     node_colors = {}
#     for node_id in valid_nodes:
#         node_colors[node_id] = (
#             random.randint(50, 255),
#             random.randint(50, 255),
#             random.randint(50, 255)
#         )
    
#     # Color each node
#     for node_id in valid_nodes:
#         node_vis[node_labels == node_id] = node_colors[node_id]
    
#     # Overlay the original component bounding boxes
#     node_vis_with_boxes = node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Collect node information
#     node_info = {
#         "valid_nodes": list(valid_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items()},
#         "component_to_nodes": {k: list(v) for k, v in component_to_nodes.items()},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items()}
#     }
    
#     # Find real nodes (nodes where two or more circuit components are connected)
#     real_nodes = set()
#     for node_id, components_list in node_to_components.items():
#         if len(components_list) >= 2:
#             real_nodes.add(node_id)
    
#     # Create a filtered node_info dictionary for real nodes
#     real_node_info = {
#         "valid_nodes": list(real_nodes),
#         "node_to_components": {int(k): list(v) for k, v in node_to_components.items() if k in real_nodes},
#         "node_colors": {int(k): [int(c) for c in v] for k, v in node_colors.items() if k in real_nodes}
#     }
    
#     # Update component_to_nodes mapping to include only real nodes
#     component_to_real_nodes = {}
#     for comp_id, nodes in component_to_nodes.items():
#         real_comp_nodes = [node_id for node_id in nodes if node_id in real_nodes]
#         if real_comp_nodes:
#             component_to_real_nodes[comp_id] = real_comp_nodes
    
#     real_node_info["component_to_nodes"] = component_to_real_nodes
    
#     # Create a visualization of real nodes
#     real_node_vis = np.zeros((h, w, 3), dtype=np.uint8)
    
#     # Color each real node using its original color
#     for node_id in real_nodes:
#         if node_id in node_colors:
#             real_node_vis[node_labels == node_id] = node_colors[node_id]
            
#     # Create a visualization of real nodes with component bounding boxes
#     real_node_vis_with_boxes = real_node_vis.copy()
#     for component in components:
#         x_min, y_min, x_max, y_max = component["bbox"]
#         cv2.rectangle(real_node_vis_with_boxes, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)
    
#     # Find connection points between components and real nodes
#     component_connections = {}
    
#     # For each component, find its connections to real nodes
#     for component in components:
#         comp_id = component["id"]
#         x_min, y_min, x_max, y_max = component["bbox"]
        
#         # Skip if component is not connected to any real nodes
#         if comp_id not in real_node_info["component_to_nodes"]:
#             continue
        
#         # Get the real nodes this component connects to
#         connected_real_nodes = real_node_info["component_to_nodes"][comp_id]
        
#         # Find connection points for each real node
#         connections = []
#         for node_id in connected_real_nodes:
#             # Create a mask for this node
#             node_mask = (node_labels == node_id)
            
#             # Create a mask for component perimeter
#             comp_perimeter = np.zeros_like(node_mask)
            
#             # Top and bottom edges
#             if y_min >= 0 and y_min < h:
#                 comp_perimeter[y_min, max(0, x_min):min(w, x_max+1)] = 1
#             if y_max >= 0 and y_max < h:
#                 comp_perimeter[y_max, max(0, x_min):min(w, x_max+1)] = 1
                
#             # Left and right edges
#             if x_min >= 0 and x_min < w:
#                 comp_perimeter[max(0, y_min):min(h, y_max+1), x_min] = 1
#             if x_max >= 0 and x_max < w:
#                 comp_perimeter[max(0, y_min):min(h, y_max+1), x_max] = 1
            
#             # Find intersection points
#             intersection = np.logical_and(node_mask, comp_perimeter)
#             intersection_points = np.where(intersection)
            
#             if len(intersection_points[0]) > 0:
#                 # Calculate the middle connection point
#                 y_points = intersection_points[0]
#                 x_points = intersection_points[1]
                
#                 # Get the middle point
#                 middle_idx = len(x_points) // 2
#                 middle_y = y_points[middle_idx]
#                 middle_x = x_points[middle_idx]
                
#                 connections.append((middle_x, middle_y))
        
#         # Store the connections for this component
#         component_connections[comp_id] = connections
    
#     # Write component connections to a text file
#     import os
#     # Create the output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Create the file path by joining the folder path with a filename
#     output_file = os.path.join(output_folder, "component_connections.txt")
    
#     try:
#         with open(output_file, 'w') as f:
#             for comp_id, points in component_connections.items():
#                 connection_str = f"{comp_id} "
#                 for i, (x, y) in enumerate(points):
#                     if i > 0:
#                         connection_str += ", "
#                     connection_str += f"({x}, {y})"
#                 f.write(connection_str + "\n")
#     except Exception as e:
#         print(f"Warning: Could not write to {output_file}: {e}")


# def process_circuit_image(image_path, model_path, output_folder):
#     """Process a circuit image to identify components and their connections."""
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Load model and image
#     model = YOLO(model_path)
#     original_image = cv2.imread(image_path)
    
#     if original_image is None:
#         print(f"Error: Could not read image at {image_path}")
#         return None, None
    
#     # Create copies for visualization
#     detection_image = original_image.copy()
    
#     # Detect components
#     results = model(image_path)[0]
    
#     # Get component information
#     components = []
#     for i, (cls, bbox) in enumerate(zip(results.boxes.cls.cpu().numpy(), 
#                                        results.boxes.xyxy.cpu().numpy())):
#         class_idx = int(cls)  # Convert to Python int
#         class_name = results.names[class_idx]
#         x_min, y_min, x_max, y_max = map(int, bbox)  # Convert to Python int
        
#         # Assign a unique ID to each component
#         component_id = f"{class_name}_{i}"
        
#         # Choose color based on class for visualization
#         color = (random.randint(100, 255), 
#                  random.randint(100, 255), 
#                  random.randint(100, 255))
        
#         # Draw bounding box
#         cv2.rectangle(detection_image, (x_min, y_min), (x_max, y_max), color, 2)
#         cv2.putText(detection_image, f"{class_name}", 
#                     (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 
#                     0.5, color, 2)
        
#         components.append({
#             "id": component_id,
#             "type": class_name,
#             "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],  # Explicit conversion
#             "color": [int(c) for c in color]  # Explicit conversion
#         })
    
#     # Preprocess image for connected component analysis
#     # Convert to grayscale
#     gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
#     # Apply threshold to separate components and wires from background
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # Save the original binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "original_binary.png"), binary)
    
#     # Thicken lines to ensure connections
#     kernel = np.ones((3, 3), np.uint8)
#     thickened_binary = cv2.dilate(binary, kernel, iterations=1)
    
#     # Save the thickened binary image for comparison
#     cv2.imwrite(os.path.join(output_folder, "thickened_binary.png"), thickened_binary)
    
#     # Create a merged label image where all foreground pixels are set to 1
#     merged_labels = np.zeros_like(thickened_binary)
#     merged_labels[thickened_binary > 0] = 1
    
#     # Save the merged label image
#     merged_label_vis = np.zeros_like(original_image)
#     merged_label_vis[merged_labels > 0] = (0, 255, 0)  # Green for all circuit traces
#     cv2.imwrite(os.path.join(output_folder, "merged_labels.png"), merged_label_vis)
    
#     # Identify circuit nodes
#     # node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug = identify_circuit_nodes(merged_labels, components)
#     # node_labels, node_info, node_vis, node_vis_with_boxes, circuit_cut, overlap_debug, real_nodes, real_node_info, real_node_vis = identify_circuit_nodes(merged_labels, components)
#     identify_circuit_nodes(merged_labels, components, output_folder)

# def process_all_images(images_folder, model_path, output_folder):
#     """Process all circuit images in a folder."""
#     os.makedirs(output_folder, exist_ok=True)
    
#     image_files = [f for f in os.listdir(images_folder) 
#                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
#     for image_file in image_files:
#         print(f"Processing {image_file}...")
#         image_path = os.path.join(images_folder, image_file)
        
#         # Create output folder for this image
#         image_output_folder = os.path.join(output_folder, 
#                                           os.path.splitext(image_file)[0])
#         os.makedirs(image_output_folder, exist_ok=True)
        
#         # Process the image
#         try:
#             process_circuit_image(image_path, model_path, image_output_folder)
#         except Exception as e:
#             print(f"  Error processing {image_file}: {str(e)}")
#             import traceback
#             traceback.print_exc()

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path)























# import os
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from scipy.ndimage import label as connected_label
# from scipy.spatial import distance

# # Define helper functions
# def detect_edges(image_path):
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#     grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(grayscale, 50, 150)
#     kernel = np.ones((5, 5), np.uint8)
#     connected_edges = cv2.dilate(edges, kernel, iterations=2)

#     return connected_edges

# def mask_components(connected_edges, components):
#     masked_edges = connected_edges.copy()
#     for component in components:
#         bbox = component["bounding_box"]
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(masked_edges, (x1, y1), (x2, y2), 0, -1)

#     return masked_edges

# def find_nearest_edge(masked_edges, px, py):
#     edge_points = np.argwhere(masked_edges > 0)  # Get all edge points
#     if edge_points.size == 0:
#         return None  # No edges in the mask
#     distances = distance.cdist([(py, px)], edge_points)  # Compute distances from the point to all edges
#     nearest_index = np.argmin(distances)  # Index of the nearest edge

#     return edge_points[nearest_index]  # Coordinates of the nearest edge (y, x)

# def is_point_connected_or_nearest(masked_edges, px, py):

#     # Find the nearest edge
#     nearest_edge = find_nearest_edge(masked_edges, px, py)
#     if nearest_edge is not None:
#         return True, (nearest_edge[1], nearest_edge[0])  # Return x, y of the nearest edge

#     print(f"No connection found for point ({px}, {py})")  # Debug message if something went wrong
#     return False, None  # No connection and no nearest edge

# def overlay_and_find_nodes_with_connected_regions(masked_edges, components, results_path, image_file):
#     labeled_edges, num_regions = connected_label(masked_edges)
#     region_to_node = {}
#     current_node_id = 1
#     gnd_regions = set()

#     for component in components:
#         for point in component["connection_points"]:
#             px, py = point
#             if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                 continue

#             is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#             if is_connected:
#                 connected_px, connected_py = connection_point
#                 region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                 if region > 0:
#                     if region not in region_to_node:
#                         region_to_node[region] = current_node_id
#                         current_node_id += 1
#                     if component["label"].upper() == "GND":
#                         gnd_regions.add(region)

#     # Rearrange node IDs based on top-left-most pixel
#     region_top_left = {}
#     for region in range(1, num_regions + 1):
#         pixels = np.argwhere(labeled_edges == region)  # Get all pixels in the region
#         if len(pixels) > 0:
#             top_left_pixel = pixels[np.lexsort((pixels[:, 1], pixels[:, 0]))][0]  # Sort by (y, then x) and pick the first
#             region_top_left[region] = top_left_pixel

#     # Sort regions by top-left-most pixel
#     sorted_regions = sorted(region_top_left.items(), key=lambda x: (x[1][0], x[1][1]))  # Sort by (y, x)
#     # Reassign node IDs
#     new_region_to_node = {}
#     new_node_id = 1
#     for region, _ in sorted_regions:
#         if region in region_to_node:
#             new_region_to_node[region] = new_node_id
#             new_node_id += 1

#     # Adjust for gnd_regions
#     gnd_nodes = [new_region_to_node[region] for region in gnd_regions if region in new_region_to_node]
#     if gnd_nodes:  # If there are any ground nodes
#         # Find the smallest node in gnd_nodes
#         smallest_gnd_node = min(gnd_nodes)
        
#         # Update new_region_to_node so that all gnd_regions point to the smallest_gnd_node
#         for region in gnd_regions:
#             if region in new_region_to_node:
#                 new_region_to_node[region] = smallest_gnd_node

#     # Create a new text file for node positions
#     results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0] + '.txt')
#     with open(results_file, 'w') as results:
#         # Dictionary to keep track of label counts
#         label_counts = {}

#         for component in components:
#             # Skip GND components
#             if component["label"].upper() == "GND":
#                 continue
#             connected_nodes = []
#             for point in component["connection_points"]:
#                 px, py = point
#                 if py >= labeled_edges.shape[0] or px >= labeled_edges.shape[1]:
#                     continue
#                 is_connected, connection_point = is_point_connected_or_nearest(masked_edges, px, py)
#                 if is_connected:
#                     connected_px, connected_py = connection_point
#                     region = labeled_edges[connected_py, connected_px]  # Use the connected or nearest edge point
#                     if region > 0 and region in new_region_to_node:
#                         node_id = new_region_to_node[region]
#                         connected_nodes.append(node_id)
#             # Ensure we only write components that have at least one connected node
#             connected_nodes = list(set(connected_nodes))
#             if connected_nodes:  # Check if there are any connected nodes
#                 unique_label = component["label"]
#                 # Increment the count for the current label
#                 if unique_label not in label_counts:
#                     label_counts[unique_label] = 0
#                 label_counts[unique_label] += 1
#                 # Create a new label with numbering
#                 numbered_label = f"{unique_label}_{label_counts[unique_label]}"
#                 # Write to the results file
#                 results.write(f"{numbered_label} {' '.join(map(str, connected_nodes))}\n")

# def parse_connections_file(file_path):
#     connections_dict = {}
    
#     with open(file_path, 'r') as f:
#         for line in f:
#             # Extract component name and connection points
#             parts = line.strip().split(' ', 1)
#             if len(parts) < 2:
#                 continue
                
#             component_name = parts[0]
#             # Extract the connection points using regex
#             import re
#             points = re.findall(r'\((\d+), (\d+)\)', parts[1])
            
#             # Convert to the format we need
#             connection_points = [[int(x), int(y)] for x, y in points]
#             connections_dict[component_name] = connection_points
    
#     return connections_dict

# def process_all_images(images_folder, model_path, results_path, test_results_path):
#     os.makedirs(results_path, exist_ok=True)

#     model = YOLO(model_path)
#     first = True

#     image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

#     for image_file in image_files:
#         image_path = os.path.join(images_folder, image_file)
#         image_output_folder = os.path.join(output_files_path, os.path.splitext(image_file)[0])
#         json_path = os.path.join(image_output_folder, 'circuit_info.json')

#         results = model(image_path)[0]
#         circuit_info = []
#         for result in results:
#             for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
#                                              result.keypoints.xy.cpu().numpy(),
#                                              result.boxes.xyxy.cpu().numpy()):
#                 class_idx = int(cls)
#                 object_name = results.names[class_idx]
#                 print(object_name)
#                 print("bbox:")
#                 print(bbox)
#                 first = False

#         # for result in results:
#         #     for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
#         #                                      result.keypoints.xy.cpu().numpy(),
#         #                                      result.boxes.xyxy.cpu().numpy()):
#         #         class_idx = int(cls)
#         #         object_name = results.names[class_idx]

#         #         x_min, y_min, x_max, y_max = map(int, bbox)
#         #         bounding_box = [x_min, y_min, x_max, y_max]
                

#         #         connection_points = [
#         #             [int(point[0]), int(point[1])] for point in keypoints if not (point[0] == 0 and point[1] == 0)
#         #         ]

#         #         print("connection_points:")
#         #         print(connection_points)

#         #         circuit_info.append({
#         #             "label": object_name,
#         #             "bounding_box": bounding_box,
#         #             "connection_points": connection_points
#         #         })


#         connections_file_path = results_path + "component_connections.txt"
#         print(connections_file_path)
#         component_connections = parse_connections_file(connections_file_path)

#         for result in results:
#             for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
#                                             result.keypoints.xy.cpu().numpy(),
#                                             result.boxes.xyxy.cpu().numpy()):
#                 class_idx = int(cls)
#                 object_name = results.names[class_idx]

#                 # Extract component ID if it exists in the name
#                 # Check if this is a detection that has a corresponding entry in our text file
#                 matching_key = None
#                 for key in component_connections.keys():
#                     # Extract the base component type without the ID
#                     if key.split('_')[0] == object_name and object_name + "_" + key.split('_')[1] == key:
#                         matching_key = key
#                         break
                    
#                 x_min, y_min, x_max, y_max = map(int, bbox)
#                 bounding_box = [x_min, y_min, x_max, y_max]

#                 # Use connection points from the text file if available
#                 if matching_key and matching_key in component_connections:
#                     connection_points = component_connections[matching_key]
#                 else:
#                     # Fallback to the original keypoints if no match is found
#                     connection_points = [
#                         [int(point[0]), int(point[1])] for point in keypoints if not (point[0] == 0 and point[1] == 0)
#                     ]

#                 print("connection_points:")
#                 print(connection_points)

#                 circuit_info.append({
#                     "label": object_name,
#                     "bounding_box": bounding_box,
#                     "connection_points": connection_points
#                 })

#         connected_edges = detect_edges(image_path)
#         masked_edges = mask_components(connected_edges, circuit_info)
#         overlay_and_find_nodes_with_connected_regions(
#              masked_edges, circuit_info, results_path, image_file)

# if __name__ == '__main__':
#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     PROJECT_PATH = os.path.dirname(os.path.dirname(parent_dir))  # Project path is three levels up
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         if folder_name == "train":
#             return 0
#         else:
#             return int(folder_name[5:])

#     latest_train_folder = max(train_folders, key=extract_suffix)
#     latest_train_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')

#     parent_dir = os.path.dirname(os.getcwd()) # Parent directory
#     test_images_folder = os.path.join(parent_dir, 'Test images/')
#     output_files_path = os.path.join(parent_dir, 'Method 2/Test outputs for debugging/')
#     test_results_path = os.path.join(parent_dir, 'Method 2/Test results/')

#     process_all_images(test_images_folder, latest_train_path, output_files_path, test_results_path)





















import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import label as connected_label
from scipy.spatial import distance
import json

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

def overlay_and_find_nodes_with_connected_regions(connected_edges, masked_edges, components, test_results_path, image_file, output_files_path):
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
    results_file = os.path.join(test_results_path, os.path.splitext(image_file)[0] + '.txt')
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


def parse_connections_file(file_path):
    connections_list = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # Extract component name and connection points
            parts = line.strip().split(' ', 1)
            if len(parts) < 2:
                continue
                
            # Extract the connection points using regex
            import re
            points = re.findall(r'\((\d+), (\d+)\)', parts[1])
            
            # Convert to the format we need
            connection_points = [[int(x), int(y)] for x, y in points]
            connections_list.append(connection_points)
    
    return connections_list

def process_all_images(images_folder, model_path, results_path, test_results_path):
    os.makedirs(results_path, exist_ok=True)

    model = YOLO(model_path)

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(images_folder, image_file)
        image_output_folder = os.path.join(output_files_path, os.path.splitext(image_file)[0])
        json_path = os.path.join(image_output_folder, 'circuit_info.json')

        results = model(image_path)[0]
        circuit_info = []
        connections_file_path = image_output_folder + "/component_connections.txt"
        component_connections = parse_connections_file(connections_file_path)
        index = 0 
        for result in results:
            for cls, keypoints, bbox in zip(result.boxes.cls.cpu().numpy(),
                                           result.keypoints.xy.cpu().numpy(),
                                           result.boxes.xyxy.cpu().numpy()):
                class_idx = int(cls)
                object_name = results.names[class_idx]

                x_min, y_min, x_max, y_max = map(int, bbox)
                bounding_box = [x_min, y_min, x_max, y_max]

                # Make sure we don't go out of bounds
                if index < len(component_connections):
                    connection_points = component_connections[index]
                    index += 1
                else:
                    # Fallback to original keypoints if we run out of connections
                    connection_points = [
                        [int(point[0]), int(point[1])] for point in keypoints if not (point[0] == 0 and point[1] == 0)
                    ]
                    print(f"Warning: Using fallback keypoints for index {index}")

                circuit_info.append({
                    "label": object_name,
                    "bounding_box": bounding_box,
                    "connection_points": connection_points
                })

        with open(json_path, 'w') as json_file:
            json.dump(circuit_info, json_file, indent=4)
            
        with open(json_path, "r") as json_file:
            components = json.load(json_file)

        connected_edges = detect_edges(image_path)
        masked_edges = mask_components(connected_edges, components)
        overlay_and_find_nodes_with_connected_regions(
            connected_edges, masked_edges, components, test_results_path, image_file, output_files_path)

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

    process_all_images(test_images_folder, latest_train_path, output_files_path, test_results_path)