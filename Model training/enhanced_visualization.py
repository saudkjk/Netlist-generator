import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import glob

class PerformanceTrendsVisualizer:
    """
    Visualizes performance trends across model evaluations to track improvements
    as you add more training data
    """
    def __init__(self, history_file=None):
        self.output_dir = os.path.join(os.getcwd(), "performance_trends")
        os.makedirs(self.output_dir, exist_ok=True)
        
        if history_file is None:
            self.history_file = os.path.join(os.getcwd(), "evaluation_results", "model_history.csv")
        else:
            self.history_file = history_file
            
    def load_history(self):
        """Load model evaluation history from CSV file"""
        if not os.path.exists(self.history_file):
            print(f"History file not found: {self.history_file}")
            return None
            
        try:
            df = pd.read_csv(self.history_file)
            # Convert date to datetime for better plotting
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d_%H%M%S', errors='coerce')
            return df
        except Exception as e:
            print(f"Error loading history: {e}")
            return None
            
    def visualize_performance_trends(self):
        """Create visualizations of model performance trends over time"""
        df = self.load_history()
        if df is None or df.empty:
            print("No history data available for visualization")
            return
            
        # 1. Overall metrics trends
        self._plot_metrics_over_time(df)
        
        # 2. Class distribution trends
        self._plot_class_distribution_trends(df)
        
        # 3. Precision vs Recall trends
        self._plot_precision_recall_trends(df)
        
        print(f"✅ Performance trend visualizations saved to: {self.output_dir}")
        
    def _plot_metrics_over_time(self, df):
        """Plot overall metrics trends over time"""
        metrics = ['mAP50', 'mAP50-95', 'F1']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if not available_metrics:
            return
            
        plt.figure(figsize=(12, 6))
        for metric in available_metrics:
            plt.plot(df['Date'], df[metric], marker='o', label=metric)
            
        plt.title('Model Performance Metrics Over Time')
        plt.xlabel('Evaluation Date')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'metrics_trend.png'), dpi=300)
        plt.close()
        
    def _plot_class_distribution_trends(self, df):
        """Plot trends in class performance categories"""
        category_columns = [col for col in df.columns if col.endswith('_Classes_Count')]
        
        if not category_columns:
            return
            
        # Create stacked bar chart
        plt.figure(figsize=(12, 6))
        
        # Create x positions
        x = np.arange(len(df))
        bottom = np.zeros(len(df))
        
        # Custom colors for categories
        colors = {
            'Poor_Classes_Count': 'salmon',
            'NeedsImprovement_Classes_Count': 'orange',
            'Good_Classes_Count': 'yellowgreen',
            'Excellent_Classes_Count': 'darkgreen'
        }
        
        # Plot each category as stacked bar
        for col in category_columns:
            plt.bar(x, df[col], bottom=bottom, label=col.replace('_Classes_Count', ''), 
                    color=colors.get(col, None))
            bottom += df[col].values
            
        plt.title('Class Performance Distribution Over Time')
        plt.xlabel('Evaluation')
        plt.ylabel('Number of Classes')
        plt.legend()
        
        # Set custom x-axis labels with dates if available
        if 'Date' in df.columns:
            date_labels = df['Date'].dt.strftime('%Y-%m-%d').tolist()
            plt.xticks(x, date_labels, rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'class_distribution_trend.png'), dpi=300)
        plt.close()
        
    def _plot_precision_recall_trends(self, df):
        """Plot precision vs recall trends"""
        if 'Precision' not in df.columns or 'Recall' not in df.columns:
            return
            
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot with size proportional to F1 score
        sizes = df['F1'] * 100 if 'F1' in df.columns else [50] * len(df)
        
        # Create gradient colors based on date
        if 'Date' in df.columns:
            # Normalize dates to 0-1 range for colormap
            dates = df['Date'].astype(np.int64)
            norm_dates = (dates - dates.min()) / (dates.max() - dates.min()) if len(dates) > 1 else [0.5]
            
            scatter = plt.scatter(df['Recall'], df['Precision'], s=sizes, 
                                 c=norm_dates, cmap='viridis', alpha=0.8)
            plt.colorbar(scatter, label='Time (earlier → later)')
        else:
            plt.scatter(df['Recall'], df['Precision'], s=sizes)
        
        # Connect points in chronological order
        plt.plot(df['Recall'], df['Precision'], 'k--', alpha=0.3)
        
        # Plot labels for each evaluation
        for i, row in df.iterrows():
            label = row['Model'] if 'Model' in df.columns else f"Eval {i+1}"
            plt.annotate(label, (row['Recall'], row['Precision']), 
                         xytext=(5, 5), textcoords='offset points')
        
        plt.title('Precision vs Recall Trends')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(linestyle='--', alpha=0.7)
        
        # Draw F1 score isolines
        f1_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
        x = np.linspace(0.01, 1, 100)
        for f1 in f1_levels:
            y = (f1 * x) / (2 * x - f1)
            valid_mask = (y >= 0) & (y <= 1)
            plt.plot(x[valid_mask], y[valid_mask], 'r--', alpha=0.3)
            # Add F1 labels
            idx = len(x[valid_mask]) // 2
            if idx > 0:
                plt.annotate(f'F1={f1}', 
                             (x[valid_mask][idx], y[valid_mask][idx]),
                             color='red', alpha=0.5)
        
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'precision_recall_trend.png'), dpi=300)
        plt.close()
        
    def combine_class_metrics(self):
        """
        Combine per-class metrics across multiple evaluation runs to track
        improvements in specific classes over time
        """
        # Find all class metrics CSV files from evaluation runs
        eval_dir = os.path.join(os.getcwd(), "evaluation_results")
        class_csv_files = []
        
        for root, dirs, files in os.walk(eval_dir):
            for file in files:
                if file == 'class_metrics.csv':
                    # Store tuple of (path, timestamp from parent dir)
                    parent_dir = os.path.basename(root)
                    timestamp = parent_dir.split('_')[-1] if '_' in parent_dir else parent_dir
                    class_csv_files.append((os.path.join(root, file), timestamp))
        
        if not class_csv_files:
            print("No class metrics files found")
            return pd.DataFrame()  # Return empty DataFrame instead of None
            
        # Load and combine all files
        combined_data = []
        for file_path, timestamp in class_csv_files:
            try:
                df = pd.read_csv(file_path)
                df['Timestamp'] = timestamp
                # Convert timestamp to datetime for sorting
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
                combined_data.append(df)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        if not combined_data:
            print("No valid data found in metrics files")
            return pd.DataFrame()  # Return empty DataFrame
            
        # Combine all data
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Check if we have valid data to work with
        if combined_df.empty or 'Class' not in combined_df.columns:
            print("No valid class metrics data found")
            return combined_df
            
        # Drop any rows with NaN values for critical columns
        combined_df = combined_df.dropna(subset=['Class', 'mAP50', 'Timestamp'])
        
        # Create per-class performance trend plots
        classes = combined_df['Class'].unique()
        
        if len(classes) == 0:
            print("No classes found in metrics data")
            return combined_df
            
        # Create directory for class trends
        os.makedirs(os.path.join(self.output_dir, 'class_trends'), exist_ok=True)
        
        try:
            for class_name in classes:
                class_data = combined_df[combined_df['Class'] == class_name].sort_values('Timestamp')
                
                if len(class_data) <= 1:
                    continue  # Skip classes with only one evaluation
                    
                # Plot mAP trend for this class
                plt.figure(figsize=(10, 6))
                plt.plot(class_data['Timestamp'], class_data['mAP50'], marker='o', linestyle='-')
                plt.title(f'Performance Trend: {class_name}')
                plt.xlabel('Date')
                plt.ylabel('mAP50')
                plt.grid(linestyle='--', alpha=0.7)
                plt.xticks(rotation=45)
                
                # Add horizontal lines for performance categories
                plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
                plt.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5)
                plt.axhline(y=0.85, color='g', linestyle='--', alpha=0.5)
                
                plt.tight_layout()
                
                # Make class name safe for filenames
                safe_class_name = str(class_name).replace('/', '_').replace('\\', '_')
                plt.savefig(os.path.join(self.output_dir, 'class_trends', f'{safe_class_name}_trend.png'), dpi=300)
                plt.close()
        except Exception as e:
            print(f"Error creating class trend plots: {e}")
                
        # Create a heatmap showing all classes over time
        self._create_class_performance_heatmap(combined_df)
        
        return combined_df
        
    def _create_class_performance_heatmap(self, combined_df):
        """Create a heatmap showing performance of all classes over time"""
        try:
            # Check for duplicate timestamp+class combinations
            potential_duplicates = combined_df.duplicated(subset=['Timestamp', 'Class'], keep=False)
            if potential_duplicates.any():
                print(f"Warning: Found {potential_duplicates.sum()} duplicate timestamp+class entries.")
                print("Taking the most recent value for each timestamp+class combination.")
                
                # Use the last entry for each timestamp+class combination
                combined_df = combined_df.drop_duplicates(subset=['Timestamp', 'Class'], keep='last')
            
            # First, pivot the data
            pivot_df = combined_df.pivot(index='Timestamp', columns='Class', values='mAP50')
            
            # Sort by timestamp
            pivot_df = pivot_df.sort_index()
            
            # Plot heatmap
            plt.figure(figsize=(12, 8))
            ax = sns.heatmap(pivot_df, cmap='viridis', vmin=0.4, vmax=1.0, 
                            cbar_kws={'label': 'mAP50'})
            
            plt.title('Class Performance Over Time')
            plt.ylabel('Evaluation Date')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, 'class_performance_heatmap.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not create class performance heatmap: {e}")
            print("Skipping heatmap generation but continuing with other visualizations.")
        
    def analyze_performance_gains(self):
        """
        Analyze the performance gains between evaluations to identify which
        classes are improving and which are stagnating
        """
        try:
            combined_df = self.combine_class_metrics()
            
            if combined_df is None or combined_df.empty:
                print("No combined class metrics data available.")
                return
                
            # Get sorted timestamps
            timestamps = sorted(combined_df['Timestamp'].unique())
            
            if len(timestamps) <= 1:
                print("Need at least two evaluations to analyze gains")
                return
                
            # Compare first and last evaluation
            first_eval = combined_df[combined_df['Timestamp'] == timestamps[0]]
            last_eval = combined_df[combined_df['Timestamp'] == timestamps[-1]]
            
            # Check for classes present in both evaluations
            common_classes = set(first_eval['Class']).intersection(set(last_eval['Class']))
            
            if not common_classes:
                print("No common classes found between first and last evaluations")
                return
                
            # Filter for common classes
            first_eval = first_eval[first_eval['Class'].isin(common_classes)]
            last_eval = last_eval[last_eval['Class'].isin(common_classes)]
            
            # Make sure there are no duplicate classes in each evaluation
            first_eval = first_eval.drop_duplicates(subset=['Class'], keep='first')
            last_eval = last_eval.drop_duplicates(subset=['Class'], keep='last')
            
            # Merge to compute differences
            comparison = pd.merge(
                first_eval[['Class', 'mAP50']], 
                last_eval[['Class', 'mAP50']], 
                on='Class', 
                suffixes=('_first', '_last')
            )
            
            # Calculate absolute and relative improvement
            comparison['absolute_improvement'] = comparison['mAP50_last'] - comparison['mAP50_first']
            # Avoid division by zero
            comparison['relative_improvement'] = comparison.apply(
                lambda row: (row['mAP50_last'] / row['mAP50_first'] - 1) * 100 
                if row['mAP50_first'] > 0 else 0, 
                axis=1
            )
            
            # Sort by absolute improvement descending
            comparison = comparison.sort_values('absolute_improvement', ascending=False)
            
            # Plot improvement chart
            plt.figure(figsize=(12, 8))
            
            # Color bars based on improvement (green for improved, red for worse)
            colors = ['green' if x >= 0 else 'red' for x in comparison['absolute_improvement']]
            
            plt.barh(comparison['Class'], comparison['absolute_improvement'], color=colors)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.title('Performance Improvement by Class (First vs Last Evaluation)')
            plt.xlabel('Absolute mAP50 Improvement')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'class_improvement.png'), dpi=300)
            plt.close()
            
            # Calculate overall stats
            improved_classes = comparison[comparison['absolute_improvement'] > 0.05]
            worsened_classes = comparison[comparison['absolute_improvement'] < -0.05]
            stagnant_classes = comparison[
                (comparison['absolute_improvement'] >= -0.05) & 
                (comparison['absolute_improvement'] <= 0.05)
            ]
            
            # Generate summary report
            report_path = os.path.join(self.output_dir, 'improvement_summary.txt')
            with open(report_path, 'w') as f:
                f.write("=== PERFORMANCE IMPROVEMENT SUMMARY ===\n\n")
                f.write(f"First evaluation: {timestamps[0]}\n")
                f.write(f"Latest evaluation: {timestamps[-1]}\n\n")
                
                f.write("OVERALL STATISTICS:\n")
                f.write(f"Total classes analyzed: {len(comparison)}\n")
                f.write(f"Improved classes (>5% mAP): {len(improved_classes)}\n")
                f.write(f"Worsened classes (<-5% mAP): {len(worsened_classes)}\n")
                f.write(f"Stagnant classes (±5% mAP): {len(stagnant_classes)}\n\n")
                
                if not improved_classes.empty:
                    f.write("TOP IMPROVED CLASSES:\n")
                    for i, (_, row) in enumerate(improved_classes.head(5).iterrows()):
                        f.write(f"{i+1}. {row['Class']}: +{row['absolute_improvement']:.4f} mAP ")
                        f.write(f"({row['relative_improvement']:.1f}% relative improvement)\n")
                
                if not worsened_classes.empty:
                    f.write("\nCLASSES NEEDING ATTENTION:\n")
                    for _, row in worsened_classes.iterrows():
                        f.write(f"- {row['Class']}: {row['absolute_improvement']:.4f} mAP ")
                        f.write(f"({row['relative_improvement']:.1f}% relative change)\n")
                    
                f.write("\nRECOMMENDATIONS:\n")
                if len(improved_classes) > 0:
                    f.write("- Continue with similar data collection approach for improved classes\n")
                if len(worsened_classes) > 0:
                    f.write("- Review annotation quality and add more diverse examples for declining classes\n")
                if len(stagnant_classes) > 0:
                    f.write("- Consider class balancing techniques for stagnant classes\n")
                    
            print(f"✅ Performance improvement analysis saved to: {report_path}")
        except Exception as e:
            print(f"Error analyzing performance gains: {e}")
            import traceback
            traceback.print_exc()
        

def visualize_trends():
    """Main function to visualize performance trends"""
    try:
        visualizer = PerformanceTrendsVisualizer()
        visualizer.visualize_performance_trends()
        visualizer.analyze_performance_gains()
        
        print("\n✅ Performance trend visualization complete.")
        
    except Exception as e:
        print(f"\n❌ Error visualizing trends: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    visualize_trends()