import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from collections import defaultdict

def test_pose_model(model_path, test_images_dir, annotations_dir=None, output_dir='test_results'):
    """
    Test a trained YOLO pose model and plot predictions vs ground truth
    
    Args:
        model_path (str): Path to the trained model
        test_images_dir (str): Directory containing test images
        annotations_dir (str): Directory containing annotation files
        output_dir (str): Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Get list of test images
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not test_images:
        print(f"No images found in {test_images_dir}")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Initialize metrics storage
    metrics = {
        'per_class': defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'keypoint_distance': [],
            'keypoint_visibility_match': [],
            'confidence': []
        }),
        'overall': {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'keypoint_distance': [],
            'keypoint_visibility_match': [],
            'confidence': []
        }
    }
    
    # For mAP calculation
    all_detections = []
    
    # Track class counts for weighted mAP
    class_counts = defaultdict(int)
    
    # Process each image
    for img_path in test_images:
        img_filename = os.path.basename(img_path)
        print(f"Processing {img_filename}...")
        
        # Run inference
        results = model(img_path)[0]
        
        # Load image for plotting
        img = plt.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        
        # Extract ground truth annotations if available
        ground_truth = []
        if annotations_dir:
            ann_file = os.path.join(annotations_dir, os.path.splitext(img_filename)[0] + '.txt')
            
            if os.path.exists(ann_file):
                with open(ann_file, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) < 5:  # Need at least class and bbox
                            continue
                        
                        # Parse class and bbox
                        class_id = int(parts[0])
                        cx, cy, w, h = [float(x) for x in parts[1:5]]
                        
                        # Increment class count for weighted mAP
                        class_counts[class_id] += 1
                        
                        # Convert normalized coordinates to pixel coordinates
                        x1 = (cx - w/2) * img_width
                        y1 = (cy - h/2) * img_height
                        x2 = (cx + w/2) * img_width
                        y2 = (cy + h/2) * img_height
                        
                        # Parse keypoints if available
                        keypoints = []
                        if len(parts) > 5:
                            kpts_data = [float(x) for x in parts[5:]]
                            num_keypoints = len(kpts_data) // 3
                            
                            for i in range(num_keypoints):
                                x = kpts_data[i*3] * img_width
                                y = kpts_data[i*3 + 1] * img_height
                                v = kpts_data[i*3 + 2]
                                keypoints.append((x, y, v))
                        
                        # Add to ground truth list
                        ground_truth.append({
                            'class_id': class_id,
                            'bbox': [x1, y1, x2, y2],
                            'keypoints': keypoints,
                            'matched': False  # Flag to track if this ground truth is matched to a prediction
                        })
                        
                        # Draw ground truth bbox
                        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            linewidth=2, edgecolor='blue', facecolor='none')
                        ax.add_patch(rect)
                        
                        # Plot visible keypoints
                        x_coords = [kp[0] for kp in keypoints if kp[2] > 0]
                        y_coords = [kp[1] for kp in keypoints if kp[2] > 0]
                        
                        if x_coords:  # Only plot if there are visible keypoints
                            ax.scatter(x_coords, y_coords, c='blue', s=25, marker='x', 
                                      label='Ground Truth' if 'Ground Truth' not in plt.gca().get_legend_handles_labels()[1] else "")               
        
        # Plot model predictions and calculate metrics
        if results.keypoints is not None:
            keypoints = results.keypoints.data.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
            classes = results.boxes.cls.cpu().numpy() if hasattr(results.boxes, 'cls') else []
            confidences = results.boxes.conf.cpu().numpy() if hasattr(results.boxes, 'conf') else []
            
            # Track predictions that match ground truth
            predictions_matched = [False] * len(boxes)
            
            # Plot each detection
            for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                # Get class ID
                class_id = int(classes[i]) if i < len(classes) else 0
                confidence = float(confidences[i]) if i < len(confidences) else 1.0
                
                # Class name for display
                class_name = model.names[class_id] if class_id in model.names else f"Class {class_id}"
                
                # Plot bounding box
                x1, y1, x2, y2 = box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # Add class name and confidence above the box
                ax.text(x1, y1-5, f"{class_name} {confidence:.2f}", fontsize=10, color='red', 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                
                # Plot keypoints
                x, y, conf = kpts[:, 0], kpts[:, 1], kpts[:, 2]
                visible = conf > 0.25
                ax.scatter(x[visible], y[visible], c='red', s=25, marker='o', 
                          label='Prediction' if i == 0 else "")
                
                # Create detection entry for mAP calculation
                detection_entry = {
                    'class_id': class_id,
                    'bbox': box.tolist(),
                    'confidence': confidence,
                    'matched': False,
                    'image_file': img_filename
                }
                
                # Match this prediction to ground truth if available
                if ground_truth:
                    # Get IoU with each ground truth box
                    best_match_idx = -1
                    best_match_iou = 0.3  # Minimum IoU threshold
                    
                    for gt_idx, gt in enumerate(ground_truth):
                        if gt['matched'] or gt['class_id'] != class_id:
                            continue  # Skip already matched items or different classes
                        
                        # Calculate IoU
                        iou = calculate_iou(box, gt['bbox'])
                        if iou > best_match_iou:
                            best_match_iou = iou
                            best_match_idx = gt_idx
                    
                    # If match found, calculate metrics
                    if best_match_idx >= 0:
                        gt = ground_truth[best_match_idx]
                        gt['matched'] = True
                        predictions_matched[i] = True
                        detection_entry['matched'] = True
                        
                        # Record true positive for this class
                        metrics['per_class'][class_id]['true_positives'] += 1
                        metrics['overall']['true_positives'] += 1
                        
                        # Add confidence score
                        metrics['per_class'][class_id]['confidence'].append(confidence)
                        metrics['overall']['confidence'].append(confidence)
                        
                        # Calculate keypoint metrics if keypoints exist in ground truth
                        if gt['keypoints'] and len(kpts) > 0:
                            kpt_distances = []
                            visibility_matches = []
                            
                            # Match the minimum number of keypoints between ground truth and prediction
                            min_kpts = min(len(gt['keypoints']), len(kpts))
                            
                            for k in range(min_kpts):
                                gt_kpt = gt['keypoints'][k]
                                pred_kpt = kpts[k]
                                
                                # Calculate Euclidean distance for visible keypoints
                                gt_visible = gt_kpt[2] > 0
                                pred_visible = pred_kpt[2] > 0.25
                                
                                # Check if visibility prediction matches
                                visibility_match = 1.0 if gt_visible == pred_visible else 0.0
                                visibility_matches.append(visibility_match)
                                
                                # Calculate distance if both keypoints are visible
                                if gt_visible and pred_visible:
                                    dist = np.sqrt((gt_kpt[0] - pred_kpt[0])**2 + (gt_kpt[1] - pred_kpt[1])**2)
                                    # Normalize by image size
                                    norm_dist = dist / np.sqrt(img_width**2 + img_height**2)
                                    kpt_distances.append(norm_dist)
                            
                            if kpt_distances:
                                metrics['per_class'][class_id]['keypoint_distance'].extend(kpt_distances)
                                metrics['overall']['keypoint_distance'].extend(kpt_distances)
                            
                            if visibility_matches:
                                metrics['per_class'][class_id]['keypoint_visibility_match'].extend(visibility_matches)
                                metrics['overall']['keypoint_visibility_match'].extend(visibility_matches)
                        
                    else:
                        # No match - false positive
                        metrics['per_class'][class_id]['false_positives'] += 1
                        metrics['overall']['false_positives'] += 1
                
                # Add to all detections for mAP calculation
                all_detections.append(detection_entry)
            
            # Count unmatched ground truth as false negatives
            for gt in ground_truth:
                if not gt['matched']:
                    metrics['per_class'][gt['class_id']]['false_negatives'] += 1
                    metrics['overall']['false_negatives'] += 1
        
        # Add title and legend
        plt.title(f"Pose Estimation: {img_filename}")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f"result_{img_filename}"))
        plt.close()
    
    # Calculate mAP and weighted mAP
    mAP, per_class_AP = calculate_map(all_detections, model.names)
    weighted_mAP = calculate_weighted_map(per_class_AP, class_counts)
    
    # Add mAP metrics to metrics dictionary
    metrics['mAP'] = mAP
    metrics['weighted_mAP'] = weighted_mAP
    metrics['per_class_AP'] = per_class_AP
    metrics['class_counts'] = dict(class_counts)  # Convert to regular dict for easier serialization
    
    # Generate performance report
    generate_performance_report(metrics, model.names, output_dir)
    
    print(f"Results saved to {output_dir}")
    return metrics

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate areas of each box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    if union > 0:
        return intersection / union
    else:
        return 0.0

def calculate_map(all_detections, class_names, iou_threshold=0.5):
    """
    Calculate Mean Average Precision
    
    Args:
        all_detections: List of all detections across all images
        class_names: Dictionary mapping class IDs to names
        iou_threshold: IoU threshold for considering a detection correct
    
    Returns:
        tuple: (mAP value, dictionary of per-class AP values)
    """
    # Group detections by class
    class_detections = defaultdict(list)
    for detection in all_detections:
        class_detections[detection['class_id']].append(detection)
    
    # Group ground truth by class and image
    gt_by_class_and_image = defaultdict(lambda: defaultdict(list))
    for detection in all_detections:
        if detection['matched']:  # This is a matched detection
            class_id = detection['class_id']
            image_file = detection['image_file']
            gt_by_class_and_image[class_id][image_file].append(detection)
    
    # Calculate AP for each class
    ap_values = {}
    for class_id, detections in class_detections.items():
        # Skip classes with no detections
        if not detections:
            continue
        
        # Count total ground truth for this class
        total_gt = sum(len(gt_list) for img, gt_list in gt_by_class_and_image[class_id].items())
        
        if total_gt == 0:
            # No ground truth for this class
            ap_values[class_id] = 0.0
            continue
        
        # Sort detections by confidence
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Initialize arrays for precision/recall curve
        tp = np.zeros(len(sorted_detections))
        fp = np.zeros(len(sorted_detections))
        
        # Create dictionary to count TPs per image
        gt_used = {img: [False] * len(gt_list) for img, gt_list in gt_by_class_and_image[class_id].items()}
        
        # Process each detection
        for i, detection in enumerate(sorted_detections):
            img_file = detection['image_file']
            
            # If the image has no ground truth for this class
            if img_file not in gt_by_class_and_image[class_id]:
                fp[i] = 1  # False positive
                continue
            
            # Get ground truth for this image
            gt_in_img = gt_by_class_and_image[class_id][img_file]
            
            # Find best matching ground truth
            best_iou = -1
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(gt_in_img):
                # Skip already matched ground truth
                if gt_used[img_file][gt_idx]:
                    continue
                
                # Calculate IoU
                iou = calculate_iou(detection['bbox'], gt['bbox'])
                
                # Update best match
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            # Check if we found a good match
            if best_iou >= iou_threshold:
                # Mark ground truth as used
                gt_used[img_file][best_gt_idx] = True
                tp[i] = 1  # True positive
            else:
                fp[i] = 1  # False positive
        
        # Compute cumulative sums
        cumsum_tp = np.cumsum(tp)
        cumsum_fp = np.cumsum(fp)
        
        # Compute precision and recall
        precision = cumsum_tp / (cumsum_tp + cumsum_fp + 1e-10)
        recall = cumsum_tp / total_gt
        
        # Compute AP using 11-point interpolation (Pascal VOC 2007)
        ap = 0
        recall_thresholds = np.linspace(0, 1, 11)
        
        for r in recall_thresholds:
            # Find precision at points where recall >= r
            prec_at_rec = precision[recall >= r]
            if len(prec_at_rec) > 0:
                ap += np.max(prec_at_rec) / 11
        
        ap_values[class_id] = ap
    
    # Calculate mAP
    if ap_values:
        mAP = sum(ap_values.values()) / len(ap_values)
    else:
        mAP = 0.0
    
    return mAP, ap_values

def calculate_weighted_map(per_class_AP, class_counts):
    """
    Calculate weighted Mean Average Precision based on class frequency
    
    Args:
        per_class_AP: Dictionary mapping class IDs to AP values
        class_counts: Dictionary mapping class IDs to number of ground truth instances
    
    Returns:
        float: Weighted mAP value
    """
    total_instances = sum(class_counts.values())
    
    if total_instances == 0:
        return 0.0
    
    weighted_sum = 0
    for class_id, ap in per_class_AP.items():
        weight = class_counts.get(class_id, 0) / total_instances
        weighted_sum += ap * weight
    
    return weighted_sum

def generate_performance_report(metrics, class_names, output_dir):
    """Generate a performance report text file"""
    report_path = os.path.join(output_dir, 'performance_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write("         POSE MODEL PERFORMANCE REPORT\n")
        f.write("="*50 + "\n\n")
        
        # Per-class metrics
        f.write("1. PER-CLASS PERFORMANCE\n")
        f.write("-"*50 + "\n")
        
        for class_id, class_metrics in sorted(metrics['per_class'].items()):
            # Get class name
            class_name = class_names.get(class_id, f"Class {class_id}")
            
            # Get instance count for this class
            instance_count = metrics['class_counts'].get(class_id, 0)
            
            # Calculate precision, recall, F1 score
            tp = class_metrics['true_positives']
            fp = class_metrics['false_positives']
            fn = class_metrics['false_negatives']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Get AP for this class
            ap = metrics['per_class_AP'].get(class_id, 0)
            
            # Calculate average metrics
            avg_confidence = np.mean(class_metrics['confidence']) if class_metrics['confidence'] else 0
            avg_keypoint_distance = np.mean(class_metrics['keypoint_distance']) if class_metrics['keypoint_distance'] else 0
            keypoint_visibility_accuracy = np.mean(class_metrics['keypoint_visibility_match']) if class_metrics['keypoint_visibility_match'] else 0
            
            # Calculate class weight for weighted mAP
            total_instances = sum(metrics['class_counts'].values())
            class_weight = instance_count / total_instances if total_instances > 0 else 0
            
            # Write class metrics
            f.write(f"\nClass: {class_name} (ID: {class_id})\n")
            f.write(f"  Class Statistics:\n")
            f.write(f"    - Instances in Test Set: {instance_count}\n")
            f.write(f"    - Weight in Weighted mAP: {class_weight:.4f} ({class_weight*100:.1f}%)\n")
            
            f.write(f"  Detection Metrics:\n")
            f.write(f"    - True Positives: {tp}\n")
            f.write(f"    - False Positives: {fp}\n")
            f.write(f"    - False Negatives: {fn}\n")
            f.write(f"    - Precision: {precision:.4f}\n")
            f.write(f"    - Recall: {recall:.4f}\n")
            f.write(f"    - F1 Score: {f1:.4f}\n")
            f.write(f"    - Average Precision (AP@0.5): {ap:.4f}\n")
            f.write(f"    - Average Confidence: {avg_confidence:.4f}\n")
            
            f.write(f"  Keypoint Metrics:\n")
            if class_metrics['keypoint_distance']:
                f.write(f"    - Average Normalized Keypoint Distance: {avg_keypoint_distance:.4f}\n")
                f.write(f"    - Keypoint Visibility Accuracy: {keypoint_visibility_accuracy:.4f}\n")
                f.write(f"    - Number of Evaluated Keypoints: {len(class_metrics['keypoint_distance'])}\n")
            else:
                f.write(f"    - No keypoint data available for this class\n")
            
            # Recommendation based on metrics
            f.write(f"  Recommendation:\n")
            if tp < 10:
                f.write(f"    - CRITICAL: Very few true positives ({tp}). Add significantly more training data!\n")
            elif recall < 0.5:
                f.write(f"    - WARNING: Low recall ({recall:.2f}). Model is missing many instances. Add more varied training examples.\n")
            elif precision < 0.5:
                f.write(f"    - WARNING: Low precision ({precision:.2f}). Model is making many false detections. Improve training data quality.\n")
            elif ap < 0.5:
                f.write(f"    - WARNING: Low AP ({ap:.2f}). Model struggles with high-confidence correct detections. Review training process.\n")
            elif avg_keypoint_distance > 0.1 and class_metrics['keypoint_distance']:
                f.write(f"    - IMPROVE: Keypoint accuracy is suboptimal. Add more examples with clearly visible keypoints.\n")
            elif f1 < 0.7:
                f.write(f"    - IMPROVE: Overall performance could be better (F1={f1:.2f}). Add more training data.\n")
            else:
                f.write(f"    - GOOD: Model performs well on this class. No immediate action needed.\n")
        
        # Overall metrics
        f.write("\n\n" + "="*50 + "\n")
        f.write("2. OVERALL MODEL PERFORMANCE\n")
        f.write("-"*50 + "\n")
        
        # Calculate overall detection metrics
        overall = metrics['overall']
        tp = overall['true_positives']
        fp = overall['false_positives']
        fn = overall['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_confidence = np.mean(overall['confidence']) if overall['confidence'] else 0
        
        # Keypoint metrics
        avg_keypoint_distance = np.mean(overall['keypoint_distance']) if overall['keypoint_distance'] else 0
        keypoint_visibility_accuracy = np.mean(overall['keypoint_visibility_match']) if overall['keypoint_visibility_match'] else 0
        
        # Write overall metrics
        f.write(f"\nObject Detection Metrics:\n")
        f.write(f"  - Total Predictions: {tp + fp}\n")
        f.write(f"  - Total Ground Truth: {tp + fn}\n")
        f.write(f"  - True Positives: {tp}\n")
        f.write(f"  - False Positives: {fp}\n")
        f.write(f"  - False Negatives: {fn}\n")
        f.write(f"  - Precision: {precision:.4f}\n")
        f.write(f"  - Recall: {recall:.4f}\n")
        f.write(f"  - F1 Score: {f1:.4f}\n")
        f.write(f"  - Mean Average Precision (mAP@0.5): {metrics['mAP']:.4f}\n")
        f.write(f"  - Weighted mAP@0.5 (by class frequency): {metrics['weighted_mAP']:.4f}\n")
        f.write(f"  - Average Confidence: {avg_confidence:.4f}\n")
        
        f.write(f"\nKeypoint Metrics:\n")
        if overall['keypoint_distance']:
            f.write(f"  - Average Normalized Keypoint Distance: {avg_keypoint_distance:.4f}\n")
            f.write(f"  - Keypoint Visibility Accuracy: {keypoint_visibility_accuracy:.4f}\n")
            f.write(f"  - Number of Evaluated Keypoints: {len(overall['keypoint_distance'])}\n")
        else:
            f.write(f"  - No keypoint data available\n")
        
        # Class distribution information
        f.write(f"\nClass Distribution:\n")
        total_instances = sum(metrics['class_counts'].values())
        f.write(f"  - Total instances: {total_instances}\n")
        
        for class_id, count in sorted(metrics['class_counts'].items(), key=lambda x: x[1], reverse=True):
            class_name = class_names.get(class_id, f"Class {class_id}")
            percentage = (count / total_instances) * 100 if total_instances > 0 else 0
            f.write(f"  - {class_name} (ID: {class_id}): {count} instances ({percentage:.1f}%)\n")
        
        # Overall recommendation
        f.write(f"\nOverall Recommendation:\n")
        if metrics['weighted_mAP'] < 0.5:
            f.write(f"  - CRITICAL: Model performance is poor (weighted mAP={metrics['weighted_mAP']:.2f}). Significant improvements needed.\n")
            f.write(f"    Consider retraining with more data and checking annotation quality.\n")
        elif metrics['weighted_mAP'] < 0.7:
            f.write(f"  - WARNING: Model performance is below average (weighted mAP={metrics['weighted_mAP']:.2f}). More training data recommended.\n")
            f.write(f"    Focus on classes with lowest AP values first.\n")
        else:
            f.write(f"  - GOOD: Overall model performance is acceptable (weighted mAP={metrics['weighted_mAP']:.2f}).\n")
            f.write(f"    Can still improve specific classes with targeted data collection.\n")
    
    print(f"Performance report saved to {report_path}")

def main():
    # Configure these parameters
    current_dir = os.getcwd()
    PROJECT_PATH = os.path.dirname(os.path.dirname(current_dir))
    print(f"Project path: {PROJECT_PATH}")
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
    latest_model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')
    model_path = latest_model_path  # Path to your trained model
    test_images_dir = "./images"  # Directory containing test images
    annotations_dir = "./annotations"  # Directory with annotation .txt files (optional)
    output_dir = "test_results"  # Directory to save results
    
    print(f"Using model: {model_path}")
    
    # Test the model
    test_pose_model(
        model_path=model_path,
        test_images_dir=test_images_dir,
        annotations_dir=annotations_dir,
        output_dir=output_dir
    )
    
    print("Testing completed.")


if __name__ == '__main__':
    main()