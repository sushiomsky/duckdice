#!/usr/bin/env python3
"""
RNG Visualization Tool
Creates comprehensive visualizations of RNG analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


class RNGVisualizer:
    """Visualization tools for RNG analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def plot_distribution(self, save_path: str = None):
        """Plot number distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Number Distribution Analysis', fontsize=16, fontweight='bold')
        
        numbers = self.df['Number'].dropna()
        
        # Histogram
        axes[0, 0].hist(numbers, bins=100, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(numbers.mean(), color='r', linestyle='--', label=f'Mean: {numbers.mean():.2f}')
        axes[0, 0].axvline(5000, color='g', linestyle='--', label='Expected Mean: 5000')
        axes[0, 0].set_xlabel('Number')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Histogram of Numbers')
        axes[0, 0].legend()
        
        # Q-Q plot
        stats.probplot(numbers, dist="uniform", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Uniform Distribution)')
        
        # Box plot
        axes[1, 0].boxplot(numbers, vert=True)
        axes[1, 0].set_ylabel('Number')
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].axhline(5000, color='r', linestyle='--', label='Expected Median')
        axes[1, 0].legend()
        
        # Density plot
        axes[1, 1].hist(numbers, bins=100, density=True, alpha=0.7, label='Observed')
        # Expected uniform density
        axes[1, 1].axhline(1/10000, color='r', linestyle='--', label='Expected Uniform')
        axes[1, 1].set_xlabel('Number')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Density Plot')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_time_series(self, save_path: str = None):
        """Plot time series of numbers"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Numbers over time
        axes[0].plot(self.df.index[:1000], self.df['Number'][:1000], alpha=0.7)
        axes[0].axhline(5000, color='r', linestyle='--', label='Expected Mean')
        axes[0].set_ylabel('Number')
        axes[0].set_title('Numbers Over Time (First 1000 bets)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rolling mean
        rolling_mean = self.df['Number'].rolling(window=100).mean()
        axes[1].plot(self.df.index[:1000], rolling_mean[:1000], label='Rolling Mean (100)')
        axes[1].axhline(5000, color='r', linestyle='--', label='Expected Mean')
        axes[1].fill_between(self.df.index[:1000], 4500, 5500, alpha=0.2, color='red')
        axes[1].set_ylabel('Rolling Mean')
        axes[1].set_title('Rolling Mean (100 bets)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Win rate over time
        win_rate = self.df['Result_Binary'].rolling(window=100).mean()
        axes[2].plot(self.df.index[:1000], win_rate[:1000], label='Win Rate (100)')
        axes[2].axhline(0.12, color='r', linestyle='--', label='Expected (12%)')
        axes[2].set_ylabel('Win Rate')
        axes[2].set_xlabel('Bet Number')
        axes[2].set_title('Rolling Win Rate (100 bets)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_autocorrelation(self, max_lag: int = 100, save_path: str = None):
        """Plot autocorrelation"""
        numbers = self.df['Number'].dropna().values
        
        # Calculate autocorrelation
        autocorr = []
        for lag in range(1, max_lag + 1):
            if len(numbers) > lag:
                corr = np.corrcoef(numbers[:-lag], numbers[lag:])[0, 1]
                autocorr.append(corr)
            else:
                autocorr.append(0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.bar(range(1, max_lag + 1), autocorr, alpha=0.7)
        
        # Significance threshold
        threshold = 2 / np.sqrt(len(numbers))
        ax.axhline(threshold, color='r', linestyle='--', label=f'Significance threshold: ±{threshold:.4f}')
        ax.axhline(-threshold, color='r', linestyle='--')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Autocorrelation Function')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_pattern_heatmap(self, save_path: str = None):
        """Plot heatmap of number patterns"""
        # Create bins
        bins = np.linspace(0, 10000, 21)  # 20 bins
        
        # Current and next number
        current = pd.cut(self.df['Number'], bins=bins, labels=False)
        next_num = pd.cut(self.df['Number'].shift(-1), bins=bins, labels=False)
        
        # Create transition matrix
        transition = pd.crosstab(current, next_num, normalize='index')
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(transition, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Probability'})
        ax.set_xlabel('Next Number Bin')
        ax.set_ylabel('Current Number Bin')
        ax.set_title('Transition Probability Heatmap\n(If RNG is fair, should be uniform)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, importances: Dict, save_path: str = None):
        """Plot feature importance from ML models"""
        if not importances:
            print("No feature importances to plot")
            return
        
        n_models = len(importances)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle('Feature Importance Across Models', fontsize=16, fontweight='bold')
        
        for idx, (model_name, feats) in enumerate(importances.items()):
            # Sort by importance
            sorted_feats = sorted(feats.items(), key=lambda x: x[1], reverse=True)[:15]
            names = [f[0] for f in sorted_feats]
            values = [f[1] for f in sorted_feats]
            
            axes[idx].barh(range(len(names)), values, alpha=0.7)
            axes[idx].set_yticks(range(len(names)))
            axes[idx].set_yticklabels(names)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(model_name)
            axes[idx].invert_yaxis()
            axes[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_ml_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           model_name: str = "Model", save_path: str = None):
        """Plot ML predictions vs actual"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} Predictions Analysis', fontsize=16, fontweight='bold')
        
        # Show only first 500 for clarity
        n_show = min(500, len(y_true))
        
        # Actual vs Predicted (time series)
        axes[0, 0].plot(y_true[:n_show], label='Actual', alpha=0.7)
        axes[0, 0].plot(y_pred[:n_show], label='Predicted', alpha=0.7)
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Number')
        axes[0, 0].set_title(f'Actual vs Predicted (First {n_show} samples)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 1].scatter(y_true, y_pred, alpha=0.3)
        axes[0, 1].plot([0, 10000], [0, 10000], 'r--', label='Perfect prediction')
        axes[0, 1].set_xlabel('Actual')
        axes[0, 1].set_ylabel('Predicted')
        axes[0, 1].set_title('Predicted vs Actual')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(0, color='r', linestyle='--', label=f'Mean: {residuals.mean():.2f}')
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].legend()
        
        # Residuals over samples
        axes[1, 1].scatter(range(n_show), residuals[:n_show], alpha=0.3)
        axes[1, 1].axhline(0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Sample')
        axes[1, 1].set_ylabel('Residual')
        axes[1, 1].set_title('Residuals Over Samples')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_dashboard(self, output_dir: str = "visualizations"):
        """Create all visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        self.plot_distribution(f"{output_dir}/distribution.png")
        self.plot_time_series(f"{output_dir}/time_series.png")
        self.plot_autocorrelation(save_path=f"{output_dir}/autocorrelation.png")
        self.plot_pattern_heatmap(f"{output_dir}/pattern_heatmap.png")
        
        print(f"\n✅ All visualizations saved to {output_dir}/")


if __name__ == "__main__":
    from data_loader import BetHistoryLoader
    
    # Load data
    loader = BetHistoryLoader()
    df = loader.load_all_files()
    df = loader.preprocess_data()
    
    # Create visualizer
    visualizer = RNGVisualizer(df)
    
    # Create all visualizations
    visualizer.create_dashboard()
