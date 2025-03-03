# from ultralytics import YOLO
# import os
# from prepare_config import prepare_config_file

# def validate_model():
#     current_dir = os.getcwd()
#     PROJECT_PATH = os.path.dirname(current_dir)  # Project path is one level up

#     # Find the latest training folder dynamically
#     pose_folder = os.path.join(PROJECT_PATH, 'Current trained model/pose')
#     train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

#     # Check if there are train folders
#     if not train_folders:
#         raise FileNotFoundError("No 'train' folders found in the pose directory.")

#     # Determine the latest train folder
#     def extract_suffix(folder_name):
#         return int(folder_name[5:]) if folder_name != "train" else 0

#     latest_train_folder = max(train_folders, key=extract_suffix)

#     # latest_model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')
#     latest_model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'best.pt')
    
#     config_path, project_path = prepare_config_file()

#     print(f"Using model from: {latest_model_path}")
#     print(f"Using config from: {config_path}")

#     # Load the trained YOLOv8 model
#     model = YOLO(latest_model_path)
    
#     # Run validation
#     results = model.val(data=config_path)

#     # Extract per-class mAP scores properly
#     try:
#         per_class_map = results.box.maps  # Correct way to get per-class mAP
#     except AttributeError:
#         print("⚠️ Could not extract per-class mAP scores. Using overall mAP instead.")
#         per_class_map = [results.box.map50] * len(results.names)  # Fallback: Use overall mAP

#     # Display results
#     print("\n📌 Per-Class mAP@50 Scores:")
#     for idx, mAP in enumerate(per_class_map):
#         class_name = results.names[idx]  # Get class name
#         print(f"Class {idx} ({class_name}): {mAP:.3f}")

#     # Identify underperforming and well-performing classes
#     underperforming_classes = [idx for idx, mAP in enumerate(per_class_map) if mAP < 0.70]
#     well_performing_classes = [idx for idx, mAP in enumerate(per_class_map) if mAP >= 0.85]

#     # Display findings
#     print("\n⚠️ Underperforming Classes (mAP < 70%):", [results.names[i] for i in underperforming_classes])
#     print("✅ Well-Performing Classes (mAP > 85%):", [results.names[i] for i in well_performing_classes])

#     # Save results to a file
#     with open("yolo_validation_results.txt", "w") as f:
#         f.write("Per-Class mAP@50 Scores:\n")
#         for idx, mAP in enumerate(per_class_map):
#             f.write(f"Class {idx} ({results.names[idx]}): {mAP:.3f}\n")
#         f.write("\nUnderperforming Classes (mAP < 70%): " + str([results.names[i] for i in underperforming_classes]))
#         f.write("\nWell-Performing Classes (mAP > 85%): " + str([results.names[i] for i in well_performing_classes]))

#     print("\n✅ Validation complete. Results saved to 'yolo_validation_results.txt'.")

# # ✅ Fix for Windows multiprocessing
# if __name__ == "__main__":
#     validate_model()





from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.metrics import confusion_matrix
import json
import logging

# Set up logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
from prepare_config import prepare_config_file

class ModelEvaluator:
    def __init__(self):
        current_dir = os.getcwd()
        self.PROJECT_PATH = os.path.dirname(current_dir)
        self.OUTPUT_DIR = os.path.join(current_dir, "evaluation_results")
        self.model = None
        self.results = None
        self.model_path = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
    def find_latest_model(self):
        # Find the latest training folder dynamically
        pose_folder = os.path.join(self.PROJECT_PATH, 'Current trained model/pose')
        train_folders = [folder for folder in os.listdir(pose_folder) if folder.startswith('train')]

        # Check if there are train folders
        if not train_folders:
            raise FileNotFoundError("No 'train' folders found in the pose directory.")

        # Determine the latest train folder
        def extract_suffix(folder_name):
            return int(folder_name[5:]) if folder_name != "train" else 0

        latest_train_folder = max(train_folders, key=extract_suffix)

        # Get both best and last models for comparison
        best_model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'best.pt')
        last_model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'last.pt')
        
        return {
            'best': best_model_path if os.path.exists(best_model_path) else None,
            'last': last_model_path if os.path.exists(last_model_path) else None,
            'train_folder': latest_train_folder
        }
    
    def load_model(self, model_path):
        print(f"Loading model from: {model_path}")
        self.model_path = model_path
        self.model = YOLO(model_path)
        return self.model
    
    def validate(self, config_path, conf_thresholds=None):
        """Run validation with multiple confidence thresholds if specified"""
        print(f"Using config from: {config_path}")
        
        # Base validation with default threshold
        self.results = self.model.val(data=config_path, verbose=True)
        
        # Fix for analyze_results - store box and pose metrics
        if hasattr(self.results, 'box') and hasattr(self.results, 'pose'):
            # Combine metrics from both for easier access
            self.metrics_data = {
                'box': {
                    'map50': self.results.box.map50,
                    'map': self.results.box.map 
                },
                'pose': {
                    'map50': self.results.pose.map50,
                    'map': self.results.pose.map
                }
            }
            
            # Try to extract precision/recall from results
            if hasattr(self.results, 'results_dict'):
                metrics_dict = self.results.results_dict
                if isinstance(metrics_dict, dict):
                    for metric_type in ['box', 'pose']:
                        for metric in ['p', 'r', 'f1']:
                            key = f'{metric_type}_{metric}'
                            if key in metrics_dict:
                                self.metrics_data[metric_type][metric] = metrics_dict[key]
        
        # Store results for different confidence thresholds if specified
        threshold_results = {}
        if conf_thresholds:
            for conf in conf_thresholds:
                print(f"\nValidating with confidence threshold: {conf}")
                result = self.model.val(data=config_path, conf=conf, verbose=True)
                threshold_results[conf] = result
                
        return self.results, threshold_results
    
    def analyze_results(self, save=True):
        """Analyze validation results and generate comprehensive metrics"""
        if self.results is None:
            raise ValueError("No validation results available. Run validate() first.")
            
        # Initialize metrics_data if not done in validate
        if not hasattr(self, 'metrics_data'):
            self.metrics_data = {
                'box': {},
                'pose': {}
            }
        
        results_dir = None
        if save:
            # Create directory for this evaluation
            model_name = Path(self.model_path).stem
            results_dir = os.path.join(self.OUTPUT_DIR, f"{model_name}_{self.timestamp}")
            os.makedirs(results_dir, exist_ok=True)
            
        # 1. Extract basic metrics
        metrics = {
            'mAP50': float(self.results.box.map50),  # mAP at IoU=0.50
            'mAP50-95': float(self.results.box.map),  # mAP at IoU=0.50:0.95
        }
        
        # Safely extract precision, recall and F1 using results_dict (preferred way)
        if hasattr(self.results, 'results_dict') and isinstance(self.results.results_dict, dict):
            results_dict = self.results.results_dict
            
            # Extract box metrics from results_dict
            if 'metrics/precision(B)' in results_dict:
                metrics['precision'] = float(results_dict['metrics/precision(B)'])
            elif 'metrics/precision' in results_dict:
                metrics['precision'] = float(results_dict['metrics/precision'])
                
            if 'metrics/recall(B)' in results_dict:
                metrics['recall'] = float(results_dict['metrics/recall(B)'])
            elif 'metrics/recall' in results_dict:
                metrics['recall'] = float(results_dict['metrics/recall'])
                
            # Calculate F1 from precision and recall
            if 'precision' in metrics and 'recall' in metrics and metrics['precision'] and metrics['recall']:
                p, r = metrics['precision'], metrics['recall']
                metrics['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
            else:
                metrics['f1'] = None
        else:
            # Fallback to direct attribute access
            try:
                # Try to get precision as scalar
                metrics['precision'] = float(self.results.box.p)
            except (TypeError, ValueError):
                # Handle case where p is an array
                if hasattr(self.results.box, 'p') and hasattr(self.results.box.p, '__len__'):
                    metrics['precision'] = float(np.mean(self.results.box.p))
                else:
                    metrics['precision'] = None
                    
            try:
                # Try to get recall as scalar
                metrics['recall'] = float(self.results.box.r)
            except (TypeError, ValueError):
                # Handle case where r is an array
                if hasattr(self.results.box, 'r') and hasattr(self.results.box.r, '__len__'):
                    metrics['recall'] = float(np.mean(self.results.box.r))
                else:
                    metrics['recall'] = None
                    
            try:
                # Try to get F1 as scalar
                metrics['f1'] = float(self.results.box.f1)
            except (TypeError, ValueError):
                # Handle case where f1 is an array
                if hasattr(self.results.box, 'f1') and hasattr(self.results.box.f1, '__len__'):
                    metrics['f1'] = float(np.mean(self.results.box.f1))
                else:
                    # Calculate F1 if precision and recall are available
                    if metrics['precision'] is not None and metrics['recall'] is not None:
                        p, r = metrics['precision'], metrics['recall']
                        metrics['f1'] = 2 * p * r / (p + r) if (p + r) > 0 else 0
                    else:
                        metrics['f1'] = None
        
        # 2. Per-class metrics
        class_names = self.results.names
        num_classes = len(class_names)
        
        try:
            # For YOLO pose models, choose appropriate metrics source
            if hasattr(self.results, 'pose') and hasattr(self.results.pose, 'maps'):
                print("Using pose metrics for per-class evaluation (better for keypoint models)")
                per_class_maps = self.results.pose.maps  # Per-class mAP@50 for pose
            else:
                per_class_maps = self.results.box.maps  # Per-class mAP@50 for boxes
        except (AttributeError, TypeError):
            print("⚠️ Could not extract per-class mAP scores. Using overall mAP instead.")
            per_class_maps = [float(self.results.box.map50)] * num_classes
            
        # 3. Extract advanced metrics if available
        class_metrics = {}
        for i, class_name in enumerate(class_names):
            # Handle both integer and string class names
            class_key = class_name
            if isinstance(class_name, int):
                class_key = str(class_name)
                
            class_metrics[class_key] = {
                'mAP50': per_class_maps[i],
                'performance_category': 'unknown'
            }
            
            # Categorize performance
            map_value = per_class_maps[i]
            if map_value < 0.5:
                category = 'poor'
            elif map_value < 0.7:
                category = 'needs_improvement'
            elif map_value < 0.85:
                category = 'good'
            else:
                category = 'excellent'
                
            class_metrics[class_key]['performance_category'] = category
        
        # 4. Create performance summary
        performance_categories = {
            'poor': [],
            'needs_improvement': [],
            'good': [],
            'excellent': []
        }
        
        for class_name, metrics_dict in class_metrics.items():
            category = metrics_dict['performance_category']
            # Ensure all values in performance_categories are strings
            performance_categories[category].append(str(class_name))
            
        # Print results
        print("\n== MODEL PERFORMANCE SUMMARY ==")
        print(f"Overall mAP@50: {metrics['mAP50']:.4f}")
        print(f"Overall mAP@50-95: {metrics['mAP50-95']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        print("\n📌 Per-Class mAP@50 Scores:")
        for class_name, class_metric in class_metrics.items():
            print(f"{class_name}: {class_metric['mAP50']:.4f} - {class_metric['performance_category'].replace('_', ' ').title()}")
            
        print("\n🔍 Performance Categories:")
        for category, classes in performance_categories.items():
            if classes:
                print(f"{category.replace('_', ' ').title()}: {', '.join(classes)}")
        
        # Save results
        if save and results_dir:
            # Save to JSON
            json_data = {
                'model_path': self.model_path,
                'timestamp': self.timestamp,
                'overall_metrics': metrics,
                'class_metrics': class_metrics,
                'performance_categories': performance_categories
            }
            
            with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
                json.dump(json_data, f, indent=4)
                
            # Create a CSV for easy tracking across evaluations
            metrics_df = pd.DataFrame({
                'Class': list(class_metrics.keys()),
                'mAP50': [m['mAP50'] for m in class_metrics.values()],
                'Category': [m['performance_category'] for m in class_metrics.values()]
            })
            
            metrics_df.to_csv(os.path.join(results_dir, 'class_metrics.csv'), index=False)
            
            # Create tracking file across all evaluations
            tracking_path = os.path.join(self.OUTPUT_DIR, 'model_history.csv')
            
            history_entry = {
                'Date': self.timestamp,
                'Model': Path(self.model_path).stem,
                'mAP50': metrics['mAP50'],
                'mAP50-95': metrics['mAP50-95'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'Poor_Classes_Count': len(performance_categories['poor']),
                'NeedsImprovement_Classes_Count': len(performance_categories['needs_improvement']),
                'Good_Classes_Count': len(performance_categories['good']),
                'Excellent_Classes_Count': len(performance_categories['excellent'])
            }
            
            # Add or append to history
            if os.path.exists(tracking_path):
                history_df = pd.read_csv(tracking_path)
                history_df = pd.concat([history_df, pd.DataFrame([history_entry])], ignore_index=True)
            else:
                history_df = pd.DataFrame([history_entry])
                
            history_df.to_csv(tracking_path, index=False)
            
            print(f"\n✅ Results saved to: {results_dir}")
            print(f"📊 Model history updated at: {tracking_path}")
            
            # Generate visualizations
            self.generate_visualizations(results_dir, class_metrics)
            
        return metrics, class_metrics, performance_categories
    
    def generate_visualizations(self, output_dir, class_metrics):
        """Generate visualizations of model performance"""
        # 1. Class performance bar chart
        plt.figure(figsize=(12, 8))
        classes = list(class_metrics.keys())
        map_values = [metrics['mAP50'] for metrics in class_metrics.values()]
        
        # Sort by performance
        sorted_indices = np.argsort(map_values)
        sorted_classes = [classes[i] for i in sorted_indices]
        sorted_map_values = [map_values[i] for i in sorted_indices]
        
        # Color by performance category
        colors = []
        for class_name in sorted_classes:
            category = class_metrics[class_name]['performance_category']
            if category == 'poor':
                colors.append('red')
            elif category == 'needs_improvement':
                colors.append('orange')
            elif category == 'good':
                colors.append('lightgreen')
            else:  # excellent
                colors.append('darkgreen')
        
        plt.barh(sorted_classes, sorted_map_values, color=colors)
        plt.xlabel('mAP@50')
        plt.title('Per-Class Performance')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5)
        plt.axvline(x=0.85, color='green', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_performance.png'), dpi=300)
        plt.close()
        
        # 2. Performance distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(map_values, bins=10, kde=True)
        plt.xlabel('mAP@50')
        plt.ylabel('Number of Classes')
        plt.title('Distribution of Class Performance')
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        plt.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5)
        plt.axvline(x=0.85, color='green', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Try to visualize precision-recall curve if available from results
        if hasattr(self.results, 'plot') and callable(getattr(self.results, 'plot', None)):
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                self.results.plot(show=False, save=False, ax=ax)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"Could not plot precision-recall curve: {e}")
    
    def predict_and_analyze(self, data_path, save_dir=None, conf=0.25, show=False):
        """Run prediction on a validation set and analyze challenging cases"""
        if save_dir is None:
            save_dir = os.path.join(self.OUTPUT_DIR, f"predictions_{self.timestamp}")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Run prediction
        results = self.model.predict(
            source=data_path,
            conf=conf,
            save=True,
            save_txt=True,
            save_conf=True,
            project=save_dir,
            name="predictions",
            show=show
        )
        
        print(f"\n✅ Predictions saved to: {os.path.join(save_dir, 'predictions')}")
        return results
    
    def generate_performance_report(self, output_file=None):
        """Generate a comprehensive performance report"""
        if self.results is None:
            raise ValueError("No validation results available. Run validate() first.")
            
        if output_file is None:
            output_file = os.path.join(self.OUTPUT_DIR, f"performance_report_{self.timestamp}.txt")
            
        # Get metrics first - to ensure we have safe values to write
        metrics, class_metrics, performance_categories = self.analyze_results(save=False)
            
        with open(output_file, 'w') as f:
            f.write("=== MODEL PERFORMANCE REPORT ===\n")
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Date: {self.timestamp}\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write(f"mAP@50: {metrics['mAP50']:.4f}\n")
            f.write(f"mAP@50-95: {metrics['mAP50-95']:.4f}\n")
            
            # Safely write precision, recall and F1
            if 'precision' in metrics and metrics['precision'] is not None:
                f.write(f"Precision: {metrics['precision']:.4f}\n")
            else:
                f.write(f"Precision: N/A\n")
                
            if 'recall' in metrics and metrics['recall'] is not None:
                f.write(f"Recall: {metrics['recall']:.4f}\n")
            else:
                f.write(f"Recall: N/A\n")
                
            if 'f1' in metrics and metrics['f1'] is not None:
                f.write(f"F1 Score: {metrics['f1']:.4f}\n")
            else:
                f.write(f"F1 Score: N/A\n")
                
            f.write("\n")
            
            # Add detailed class metrics
            f.write("PER-CLASS METRICS:\n")
            for class_name, class_metric in class_metrics.items():
                f.write(f"{class_name}: {class_metric['mAP50']:.4f} - {class_metric['performance_category'].replace('_', ' ').title()}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            
            # Add recommendations based on results
            if performance_categories['poor'] or performance_categories['needs_improvement']:
                f.write("Classes needing more data or tuning:\n")
                for category in ['poor', 'needs_improvement']:
                    for class_name in performance_categories[category]:
                        f.write(f" - {class_name} (mAP50: {class_metrics[class_name]['mAP50']:.4f})\n")
                        
            f.write("\nSuggested next steps:\n")
            if performance_categories['poor']:
                f.write(" - Add more diverse training data for poor-performing classes\n")
                f.write(" - Check annotation quality for these classes\n")
                f.write(" - Consider class balancing techniques\n")
                
            f.write(" - Monitor performance trends as you add more data\n")
            f.write(" - Consider model ensemble for challenging classes\n")
            
        print(f"\n📝 Performance report generated: {output_file}")
        return output_file


def validate_model():
    """Main function to evaluate model performance"""
    try:
        # Get config file
        config_path, project_path = prepare_config_file()
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Find latest model
        models = evaluator.find_latest_model()
        
        if not models['best'] and not models['last']:
            raise FileNotFoundError("Could not find any trained models.")
            
        # Evaluate best model
        if models['best']:
            model = evaluator.load_model(models['best'])
            print("Evaluating BEST model...")
            results, _ = evaluator.validate(config_path)
            metrics, class_metrics, _ = evaluator.analyze_results()
            evaluator.generate_performance_report()
            
        # Optionally evaluate last model for comparison
        if models['last'] and models['best'] != models['last']:
            print("\nEvaluating LAST model for comparison...")
            model = evaluator.load_model(models['last'])
            results, _ = evaluator.validate(config_path)
            evaluator.analyze_results()
        
        print("\n✅ Model evaluation complete.")
        
    except Exception as e:
        print(f"\n❌ Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide debugging info
        if 'evaluator' in locals() and hasattr(evaluator, 'results'):
            print("\nDebugging information:")
            print(f"Results type: {type(evaluator.results)}")
            print(f"Results attributes: {dir(evaluator.results)}")
            
            if hasattr(evaluator.results, 'results_dict'):
                print(f"Results dict: {evaluator.results.results_dict}")
                
            if hasattr(evaluator.results, 'box'):
                print(f"Box attributes: {dir(evaluator.results.box)}")


# ✅ Fix for Windows multiprocessing
if __name__ == "__main__":
    validate_model()