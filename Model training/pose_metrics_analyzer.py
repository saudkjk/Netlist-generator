from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
import seaborn as sns
from prepare_config import prepare_config_file

class PoseMetricsAnalyzer:
    """
    Specialized analyzer for YOLO pose models with focus on keypoint metrics
    and pose estimation accuracy
    """
    def __init__(self):
        self.model = None
        self.results = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(os.getcwd(), "pose_evaluation")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_model(self, model_path):
        """Load a YOLO pose model"""
        print(f"Loading pose model from: {model_path}")
        self.model = YOLO(model_path)
        return self.model
        
    def run_evaluation(self, data_config):
        """Run validation on the pose model"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print(f"Validating pose model using config: {data_config}")
        self.results = self.model.val(data=data_config, verbose=True)
        return self.results
        
    def analyze_pose_metrics(self):
        """Analyze pose-specific metrics including keypoint detection"""
        if self.results is None:
            raise ValueError("No validation results. Run evaluation first.")
            
        # Create output directory for this run
        results_dir = os.path.join(self.output_dir, f"pose_eval_{self.timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Extract pose-specific metrics
        results_dict = self.results.results_dict if hasattr(self.results, 'results_dict') else {}
        
        # Gather metrics
        pose_metrics = {
            'box_map50': float(self.results.box.map50),
            'box_map': float(self.results.box.map),
            'pose_map50': float(self.results.pose.map50) if hasattr(self.results, 'pose') else None,
            'pose_map': float(self.results.pose.map) if hasattr(self.results, 'pose') else None,
        }
        
        # Extract precision and recall
        for prefix, metric_type in [('metrics/precision(B)', 'box_precision'), 
                                   ('metrics/recall(B)', 'box_recall'),
                                   ('metrics/precision(P)', 'pose_precision'),
                                   ('metrics/recall(P)', 'pose_recall')]:
            if prefix in results_dict:
                pose_metrics[metric_type] = float(results_dict[prefix])
                
        # Per-class metrics for both box and pose
        class_names = self.results.names
        
        # Create a combined dataframe
        metrics_data = []
        
        # Extract per-class metrics
        if hasattr(self.results.box, 'maps') and hasattr(self.results.pose, 'maps'):
            box_maps = self.results.box.maps
            pose_maps = self.results.pose.maps
            
            for i, class_name in enumerate(class_names):
                class_data = {
                    'class': class_name if isinstance(class_name, str) else f"Class {class_name}",
                    'box_map50': box_maps[i],
                    'pose_map50': pose_maps[i],
                }
                metrics_data.append(class_data)
                
        # Convert to DataFrame
        metrics_df = pd.DataFrame(metrics_data)
        
        # Calculate box vs pose performance
        if not metrics_df.empty and 'box_map50' in metrics_df.columns and 'pose_map50' in metrics_df.columns:
            metrics_df['map_difference'] = metrics_df['pose_map50'] - metrics_df['box_map50']
            
        # Save metrics
        metrics_df.to_csv(os.path.join(results_dir, 'pose_class_metrics.csv'), index=False)
        
        # Create visualizations
        self._create_pose_visualizations(metrics_df, results_dir)
        
        # Print summary
        self._print_pose_summary(pose_metrics, metrics_df)
        
        return pose_metrics, metrics_df
        
    def _create_pose_visualizations(self, metrics_df, output_dir):
        """Create pose-specific visualizations"""
        if metrics_df.empty:
            return
            
        # 1. Box vs Pose performance comparison
        if 'box_map50' in metrics_df.columns and 'pose_map50' in metrics_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Prepare data for side-by-side bars
            class_names = metrics_df['class'].tolist()
            box_scores = metrics_df['box_map50'].tolist()
            pose_scores = metrics_df['pose_map50'].tolist()
            
            # Sort by box score for better visualization
            sorted_indices = np.argsort(box_scores)
            class_names = [class_names[i] for i in sorted_indices]
            box_scores = [box_scores[i] for i in sorted_indices]
            pose_scores = [pose_scores[i] for i in sorted_indices]
            
            x = np.arange(len(class_names))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.bar(x - width/2, box_scores, width, label='Box Detection')
            ax.bar(x + width/2, pose_scores, width, label='Pose Estimation')
            
            ax.set_ylabel('mAP@50')
            ax.set_title('Box Detection vs Pose Estimation Performance by Class')
            ax.set_xticks(x)
            ax.set_xticklabels(class_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add baseline
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
            ax.text(0, 0.51, 'Minimum acceptable', color='r', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'box_vs_pose_performance.png'), dpi=300)
            plt.close()
            
            # 2. Map difference plot (pose - box)
            if 'map_difference' in metrics_df.columns:
                plt.figure(figsize=(12, 8))
                
                # Sort by difference 
                metrics_df_sorted = metrics_df.sort_values('map_difference')
                
                # Plot
                plt.barh(metrics_df_sorted['class'], metrics_df_sorted['map_difference'])
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.title('Pose Performance vs Box Performance (Difference)')
                plt.xlabel('mAP Difference (Pose - Box)')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'pose_box_difference.png'), dpi=300)
                plt.close()
                
    def _print_pose_summary(self, pose_metrics, metrics_df):
        """Print a summary of pose evaluation metrics"""
        print("\n=== POSE MODEL PERFORMANCE SUMMARY ===")
        
        print("\nOVERALL METRICS:")
        print(f"Box Detection mAP@50: {pose_metrics.get('box_map50', 'N/A'):.4f}")
        print(f"Pose Estimation mAP@50: {pose_metrics.get('pose_map50', 'N/A'):.4f}")
        print(f"Box Detection mAP@50-95: {pose_metrics.get('box_map', 'N/A'):.4f}")
        print(f"Pose Estimation mAP@50-95: {pose_metrics.get('pose_map', 'N/A'):.4f}")
        
        if 'box_precision' in pose_metrics and 'box_recall' in pose_metrics:
            print(f"Box Precision: {pose_metrics['box_precision']:.4f}")
            print(f"Box Recall: {pose_metrics['box_recall']:.4f}")
            
        if 'pose_precision' in pose_metrics and 'pose_recall' in pose_metrics:
            print(f"Pose Precision: {pose_metrics['pose_precision']:.4f}")
            print(f"Pose Recall: {pose_metrics['pose_recall']:.4f}")
            
        # Analysis summary
        if not metrics_df.empty and 'map_difference' in metrics_df.columns:
            best_pose_class = metrics_df.loc[metrics_df['pose_map50'].idxmax()]
            worst_pose_class = metrics_df.loc[metrics_df['pose_map50'].idxmin()]
            
            print("\nCLASS ANALYSIS:")
            print(f"Best Pose Performance: {best_pose_class['class']} (mAP@50: {best_pose_class['pose_map50']:.4f})")
            print(f"Worst Pose Performance: {worst_pose_class['class']} (mAP@50: {worst_pose_class['pose_map50']:.4f})")
            
            # Classes where pose outperforms box detection
            pose_better = metrics_df[metrics_df['map_difference'] > 0.05]
            box_better = metrics_df[metrics_df['map_difference'] < -0.05]
            
            if not pose_better.empty:
                print("\nClasses where pose estimation is significantly better than box detection:")
                for _, row in pose_better.iterrows():
                    print(f"  - {row['class']}: +{row['map_difference']:.4f} mAP")
                    
            if not box_better.empty:
                print("\nClasses where box detection is significantly better than pose estimation:")
                for _, row in box_better.iterrows():
                    print(f"  - {row['class']}: {row['map_difference']:.4f} mAP")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        if 'pose_map50' in pose_metrics and pose_metrics['pose_map50'] is not None:
            pose_map = pose_metrics['pose_map50']
            if pose_map < 0.7:
                print("- Improve pose keypoint accuracy by adding more diverse training data")
                print("- Focus on challenging angles and occlusion cases")
            elif pose_map < 0.85:
                print("- Current pose estimation is good, consider fine-tuning specific classes")
            else:
                print("- Excellent pose estimation performance")
                
        print("- Monitor both box detection and pose estimation metrics separately")
        print("- For classes with poor pose estimation, check keypoint annotation quality")


def analyze_pose_metrics():
    """Main function to analyze YOLO pose model metrics"""
    try:
        # Prepare config
        config_path, _ = prepare_config_file()
        
        # Find latest pose model
        current_dir = os.getcwd()
        project_path = os.path.dirname(current_dir)
        pose_folder = os.path.join(project_path, 'Current trained model/pose')
        
        # Find latest train folder
        train_folders = [f for f in os.listdir(pose_folder) if f.startswith('train')]
        if not train_folders:
            raise FileNotFoundError("No train folders found")
            
        # Determine latest folder
        def extract_suffix(folder_name):
            return int(folder_name[5:]) if folder_name != "train" else 0
            
        latest_train_folder = max(train_folders, key=extract_suffix)
        model_path = os.path.join(pose_folder, latest_train_folder, 'weights', 'best.pt')
        
        # Run pose metrics analysis
        analyzer = PoseMetricsAnalyzer()
        analyzer.load_model(model_path)
        analyzer.run_evaluation(config_path)
        analyzer.analyze_pose_metrics()
        
        print("\n✅ Pose metrics analysis complete.")
        
    except Exception as e:
        print(f"\n❌ Error during pose metrics analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_pose_metrics()