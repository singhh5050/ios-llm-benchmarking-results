#!/usr/bin/env python3
"""
iOS LLM Benchmarking Results Analysis
=====================================

This script analyzes performance data for LLM models running on iOS devices.
It generates comprehensive visualizations and summary statistics for:
- Llama 3.2
- Phi3  
- Qwen3

The analysis includes:
- Time to First Token (TTFT) comparisons
- Prefill and decode throughput analysis
- Total latency measurements
- Token generation statistics
- Performance stability analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load and process data from CSV files"""
    data_dir = Path(__file__).parent.parent / 'data'
    figures_dir = Path(__file__).parent.parent / 'figures'
    results_dir = Path(__file__).parent.parent / 'results'
    
    # Create directories if they don't exist
    figures_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    models = {}
    
    # Load each model's data
    model_files = {
        'Llama': 'llama_combined.csv',
        'Phi3': 'phi3_combined.csv', 
        'Qwen': 'qwen_combined.csv'
    }
    
    for model_name, filename in model_files.items():
        filepath = data_dir / filename
        if filepath.exists():
            print(f"Loading {model_name} data from {filename}...")
            df = pd.read_csv(filepath, comment='#')
            
            # Clean and process data
            df = df.dropna()
            
            # Ensure numeric columns are properly typed
            numeric_columns = [
                'Time_to_First_Token_ms', 'Prefill_Latency_ms', 'Decode_Latency_ms',
                'Prefill_Tokens', 'Decode_Tokens', 'Total_Tokens',
                'Prefill_Speed_tps', 'Decode_Speed_tps', 'Total_Latency_ms'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Cap total tokens at 512 to match original analysis
            df['Total_Tokens'] = df['Total_Tokens'].clip(upper=512)
            
            models[model_name] = df
            print(f"  Loaded {len(df)} records")
        else:
            print(f"Warning: {filename} not found, skipping {model_name}")
    
    return models, figures_dir, results_dir

def calculate_summary_stats(models):
    """Calculate summary statistics for all models"""
    summary_data = []
    
    for model_name, df in models.items():
        stats = {
            'Model': model_name,
            'TTFT (ms)': f"{df['Time_to_First_Token_ms'].mean():.1f}±{df['Time_to_First_Token_ms'].std():.0f}",
            'Prefill TPS': f"{df['Prefill_Speed_tps'].mean():.1f}±{df['Prefill_Speed_tps'].std():.0f}",
            'Decode TPS': f"{df['Decode_Speed_tps'].mean():.1f}±{df['Decode_Speed_tps'].std():.1f}",
            'Total Lat (ms)': f"{df['Total_Latency_ms'].mean():.0f}±{df['Total_Latency_ms'].std():.0f}",
            'Total Tokens': f"{df['Total_Tokens'].mean():.1f}±{df['Total_Tokens'].std():.1f}"
        }
        summary_data.append(stats)
    
    return pd.DataFrame(summary_data)

def create_comparison_plots(models, figures_dir):
    """Create comparison plots for all metrics"""
    
    # Set up the plotting style
    plt.rcParams.update({'font.size': 12})
    
    metrics = [
        ('Time_to_First_Token_ms', 'TTFT (ms)', 'Time to First Token Comparison'),
        ('Prefill_Speed_tps', 'Prefill TPS', 'Prefill Throughput Comparison'), 
        ('Decode_Speed_tps', 'Decode TPS', 'Decode Throughput Comparison'),
        ('Total_Latency_ms', 'Total Latency (ms)', 'Total Latency Comparison'),
        ('Total_Tokens', 'Total Tokens', 'Total Tokens Comparison')
    ]
    
    for metric, ylabel, title in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_to_plot = []
        labels = []
        
        for model_name, df in models.items():
            if metric in df.columns:
                data_to_plot.append(df[metric].values)
                labels.append(model_name)
        
        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add mean values as text
            for i, (model_name, df) in enumerate(models.items()):
                if metric in df.columns:
                    mean_val = df[metric].mean()
                    ax.text(i+1, mean_val, f'{mean_val:.1f}', 
                           ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.png'
        plt.savefig(figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Created {filename}")

def create_scatter_plot(models, figures_dir):
    """Create prefill vs decode speed scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['blue', 'green', 'red']
    
    for i, (model_name, df) in enumerate(models.items()):
        if 'Prefill_Speed_tps' in df.columns and 'Decode_Speed_tps' in df.columns:
            ax.scatter(df['Prefill_Speed_tps'], df['Decode_Speed_tps'], 
                      alpha=0.6, label=model_name, color=colors[i % len(colors)])
    
    ax.set_xlabel('Prefill Speed (TPS)')
    ax.set_ylabel('Decode Speed (TPS)')
    ax.set_title('Prefill vs Decode Speed Relationship')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'prefill_vs_decode_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created prefill_vs_decode_scatter.png")

def create_stability_heatmap(models, figures_dir):
    """Create coefficient of variation heatmap"""
    metrics = ['Time_to_First_Token_ms', 'Prefill_Speed_tps', 'Decode_Speed_tps', 
               'Total_Latency_ms', 'Total_Tokens']
    metric_labels = ['TTFT', 'Prefill TPS', 'Decode TPS', 'Total Latency', 'Total Tokens']
    
    cv_data = []
    model_names = []
    
    for model_name, df in models.items():
        model_names.append(model_name)
        cv_row = []
        for metric in metrics:
            if metric in df.columns:
                cv = df[metric].std() / df[metric].mean() if df[metric].mean() != 0 else 0
                cv_row.append(cv)
            else:
                cv_row.append(0)
        cv_data.append(cv_row)
    
    cv_df = pd.DataFrame(cv_data, index=model_names, columns=metric_labels)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(cv_df, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax)
    ax.set_title('Coefficient of Variation Heatmap\n(Lower values indicate more stable performance)')
    ax.set_ylabel('Model')
    ax.set_xlabel('Metric')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'coefficient_variation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created coefficient_variation_heatmap.png")

def create_consistency_plot(models, figures_dir):
    """Create TTFT consistency over time plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['blue', 'green', 'red']
    
    for i, (model_name, df) in enumerate(models.items()):
        if 'Time_to_First_Token_ms' in df.columns:
            # Calculate rolling average (window of 50 samples)
            window_size = min(50, len(df) // 10)
            if window_size > 1:
                rolling_avg = df['Time_to_First_Token_ms'].rolling(window=window_size).mean()
                ax.plot(rolling_avg.index, rolling_avg.values, 
                       label=f'{model_name} (Rolling Avg)', color=colors[i % len(colors)])
    
    ax.set_xlabel('Sample Order')
    ax.set_ylabel('TTFT (ms)')
    ax.set_title('TTFT Consistency Over Time (Rolling Average)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'ttft_consistency_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Created ttft_consistency_over_time.png")

def main():
    """Main analysis function"""
    print("iOS LLM Benchmarking Results Analysis")
    print("=" * 50)
    
    # Load data
    models, figures_dir, results_dir = load_data()
    
    if not models:
        print("No data files found! Please ensure CSV files are in the data/ directory.")
        return
    
    print(f"\nLoaded data for {len(models)} models")
    
    # Calculate summary statistics
    print("\nCalculating summary statistics...")
    summary_df = calculate_summary_stats(models)
    
    # Save summary table
    summary_path = results_dir / 'model_performance_summary_table.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved summary table to {summary_path}")
    
    # Print summary to console
    print("\nPerformance Summary:")
    print(summary_df.to_string(index=False))
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_comparison_plots(models, figures_dir)
    create_scatter_plot(models, figures_dir)
    create_stability_heatmap(models, figures_dir)
    create_consistency_plot(models, figures_dir)
    
    print(f"\nAnalysis complete! Check the following directories:")
    print(f"  - Figures: {figures_dir}")
    print(f"  - Results: {results_dir}")

if __name__ == "__main__":
    main()
