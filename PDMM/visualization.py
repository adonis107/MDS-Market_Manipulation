import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import torch


def plot_lob_snapshot(pipeline, index, levels=10, save_path=None):
    """Visualizes the Order Book shape at a specific index."""
    row = pipeline.raw_df.iloc[index]
    
    bids = [row[f'bid-volume-{i}'] for i in range(1, levels+1)]
    asks = [row[f'ask-volume-{i}'] for i in range(1, levels+1)]
    
    # Levels (1 to 10)
    x = np.arange(1, levels+1)
    
    plt.figure(figsize=(10, 5))
    plt.bar(x, bids, color='green', label='Bid Volume (Buy)', alpha=0.7)
    plt.bar(x, [-a for a in asks], color='red', label='Ask Volume (Sell)', alpha=0.7) # Negative for visual contrast
    
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xlabel("Level (1 = Best Quote)")
    plt.ylabel("Volume (Shares)")
    plt.title(f"LOB Snapshot at Index {index}")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"LOB snapshot plot saved to {save_path}")

    plt.show()


def plot_lob_evolution(pipeline, center_index, offset=10, levels=10, save_path=None):
    """
    Plots LOB snapshots before, during, and after a specific index.
    
    Args:
        center_index: The time index of the detected anomaly.
        offset: Number of time steps to look before/after.
        levels: Number of price levels to display.
    """
    indices = [center_index - offset, center_index, center_index + offset]
    titles = [f"Before (t={center_index - offset})", 
              f"Event (t={center_index})", 
              f"After (t={center_index + offset})"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    x = np.arange(1, levels + 1)
    
    for i, idx in enumerate(indices):
        if 0 <= idx < len(pipeline.raw_df):
            row = pipeline.raw_df.iloc[idx]
            
            # Extract volumes
            bids = [row[f'bid-volume-{l}'] for l in range(1, levels+1)]
            asks = [row[f'ask-volume-{l}'] for l in range(1, levels+1)]
            
            # Plot Bid vs Ask
            axes[i].bar(x, bids, color='green', label='Bid Volume' if i==0 else "", alpha=0.7)
            axes[i].bar(x, [-a for a in asks], color='red', label='Ask Volume' if i==0 else "", alpha=0.7)
            
            axes[i].axhline(0, color='black', linewidth=0.8)
            axes[i].set_title(titles[i])
            axes[i].set_xlabel("Price Level (1=Best)")
            axes[i].grid(True, alpha=0.3)
            
            if i == 0:
                axes[i].set_ylabel("Volume (Shares)")
                axes[i].legend()
        else:
            axes[i].text(0.5, 0.5, "Index Out of Bounds", ha='center')

    plt.suptitle(f"Order Book Dynamics Around Potential Spoofing Event", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"LOB evolution plot saved to {save_path}")

    plt.show()


def plot_sequential_results(results, save_path=None):
    """
    Plots sequential training results across days.
    
    Args:
        results: Dictionary returned by sequential_training_pipeline.
    """
    daily_metrics = pd.DataFrame(results['daily_metrics'])
    
    if len(daily_metrics) == 0:
        print("No daily metrics to plot.")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # AUROC
    axes[0].plot(daily_metrics['day'], daily_metrics['AUROC'], marker='o', color='blue', label='AUROC')
    if results['final_day_morning_metrics']:
        axes[0].axhline(results['final_day_morning_metrics']['AUROC'], color='green', 
                       linestyle='--', label='Final Day Morning')
    if results['final_day_rest_metrics']:
        axes[0].axhline(results['final_day_rest_metrics']['AUROC'], color='red', 
                       linestyle='--', label='Final Day Rest')
    axes[0].set_ylabel('AUROC')
    axes[0].set_title('Sequential Training: AUROC Over Days')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUPRC
    axes[1].plot(daily_metrics['day'], daily_metrics['AUPRC'], marker='s', color='orange', label='AUPRC')
    if results['final_day_morning_metrics']:
        axes[1].axhline(results['final_day_morning_metrics']['AUPRC'], color='green', 
                       linestyle='--', label='Final Day Morning')
    if results['final_day_rest_metrics']:
        axes[1].axhline(results['final_day_rest_metrics']['AUPRC'], color='red', 
                       linestyle='--', label='Final Day Rest')
    axes[1].set_ylabel('AUPRC')
    axes[1].set_title('Sequential Training: AUPRC Over Days')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # F4 Score
    axes[2].plot(daily_metrics['day'], daily_metrics['F4_Score'], marker='^', color='purple', label='F4 Score')
    if results['final_day_morning_metrics']:
        axes[2].axhline(results['final_day_morning_metrics']['F4_Score'], color='green', 
                       linestyle='--', label='Final Day Morning')
    if results['final_day_rest_metrics']:
        axes[2].axhline(results['final_day_rest_metrics']['F4_Score'], color='red', 
                       linestyle='--', label='Final Day Rest')
    axes[2].set_xlabel('Day')
    axes[2].set_ylabel('F4 Score')
    axes[2].set_title('Sequential Training: F4 Score Over Days')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        print(f"Sequential training results plot saved to {save_path}")

    plt.show()


def plot_feature_attribution(attributions, feature_names, title="Feature Importance", top_k=20, save_path=None):
    """
    Aggregates attribution over time (sum of absolute values) and plots top k features.
    Attributions shape: (1, Seq_Len, Feat)
    """
    # Sum absolute attributions over time and batch
    # Ensure result is 1D array to avoid DataFrame errors
    attr_sum = np.sum(np.abs(attributions), axis=(0, 1)).flatten()
    
    # Create DataFrame
    df_attr = pd.DataFrame({
        'Feature': feature_names,
        'Attribution': attr_sum
    }).sort_values(by='Attribution', ascending=False).head(top_k)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_attr, x='Attribution', y='Feature', hue='Feature', palette='viridis', legend=False)
    plt.title(title)
    plt.xlabel("Integrated Gradients Magnitude")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    return df_attr


def plot_global_importance(df_importance, title="Feature Importance", top_k=15):
    """
    Plots a bar chart of the top features.
    df_importance must have columns: ['Feature', 'Importance']
    """
    df_plot = df_importance.sort_values(by='Importance', ascending=False).head(top_k)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_plot, x='Importance', y='Feature', palette='viridis')
    plt.title(title)
    plt.xlabel("Attribution / Impact on Score")
    plt.tight_layout()
    plt.show()


def plot_score_distribution(scores, title="Anomaly Scores", save_path=None):
    """Visualizes the distribution of anomaly scores."""
    plt.figure(figsize=(12, 5))
    
    # Timeline
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.7, linewidth=0.8, color='tab:blue')
    plt.title(f"{title} - Timeline")
    plt.xlabel("Time Step")
    plt.ylabel("Score")
    
    # Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(scores, bins=50, kde=True, color='tab:orange')
    plt.title(f"{title} - Distribution")
    plt.xlabel("Score")
    
    plt.tight_layout()

    if save_path: plt.savefig(save_path)

    plt.show()