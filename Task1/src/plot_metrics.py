# import os
# import json
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set publication-ready aesthetics
# sns.set_theme(style="whitegrid", context="talk")

# def load_metrics(folder_path):
#     """Loads the metrics.json from a given experiment folder into a Pandas DataFrame."""
#     json_path = os.path.join(folder_path, "metrics.json")
#     if not os.path.exists(json_path):
#         print(f"Warning: No metrics.json found in {folder_path}")
#         return None
#     with open(json_path, 'r') as f:
#         data = json.load(f)
#     return pd.DataFrame(data)

# def plot_augmentation_impact(experiments_dict, title="Impact of Data Augmentation on mAP"):
#     """
#     Bar plot comparing the best validation mAP across different augmentation strategies.
#     experiments_dict format: {'Baseline': df1, 'Basic Augs': df2, 'Weather Augs': df3}
#     """
#     names = list(experiments_dict.keys())
#     best_maps = [df['val/mAP_0.50_0.95_all'].max() * 100 for df in experiments_dict.values()]
    
#     plt.figure(figsize=(10, 6))
#     bars = sns.barplot(x=names, y=best_maps, palette="viridis")
#     plt.title(title, pad=20)
#     plt.ylabel("Best Validation mAP (%)")
#     plt.ylim(0, max(best_maps) + 10)
    
#     # Add values on top of bars
#     for bar in bars.containers:
#         bars.bar_label(bar, fmt='%.1f%%', padding=3)
        
#     plt.tight_layout()
#     plt.show()

# def plot_lora_vs_full_convergence(full_df, lora_df, model_name="DETR"):
#     """
#     Plots the validation mAP convergence of Full Fine-Tune vs LoRA over epochs.
#     """
#     plt.figure(figsize=(10, 6))
    
#     plt.plot(full_df['epoch'], full_df['val/mAP_0.50_0.95_all'] * 100, 
#              label='Full Fine-Tune', linewidth=3, marker='o')
#     plt.plot(lora_df['epoch'], lora_df['val/mAP_0.50_0.95_all'] * 100, 
#              label='LoRA (Parameter Efficient)', linewidth=3, marker='s', linestyle='--')
    
#     plt.title(f"{model_name}: Convergence Comparison (Full vs LoRA)", pad=20)
#     plt.xlabel("Epoch")
#     plt.ylabel("Validation mAP (%)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def plot_object_size_performance(experiments_dict, title="Performance by Object Size"):
#     """
#     Grouped bar chart showing mAP for Small, Medium, and Large objects.
#     """
#     data = []
#     for exp_name, df in experiments_dict.items():
#         # Get the row with the best overall mAP
#         best_idx = df['val/mAP_0.50_0.95_all'].idxmax()
#         best_row = df.loc[best_idx]
        
#         data.append({'Experiment': exp_name, 'Size': 'Small', 'mAP': best_row['val/mAP_0.50_0.95_small'] * 100})
#         data.append({'Experiment': exp_name, 'Size': 'Medium', 'mAP': best_row['val/mAP_0.50_0.95_medium'] * 100})
#         data.append({'Experiment': exp_name, 'Size': 'Large', 'mAP': best_row['val/mAP_0.50_0.95_large'] * 100})
        
#     plot_df = pd.DataFrame(data)
    
#     plt.figure(figsize=(12, 7))
#     sns.barplot(data=plot_df, x='Experiment', y='mAP', hue='Size', palette="mako")
#     plt.title(title, pad=20)
#     plt.ylabel("Validation mAP (%)")
#     plt.legend(title="Object Size")
#     plt.tight_layout()
#     plt.show()

# def plot_train_val_convergence(df, title="Training vs Validation Convergence"):
#     """
#     Plots Train vs Validation mAP to analyze overfitting and healthy learning.
#     """
#     plt.figure(figsize=(10, 6))
    
#     plt.plot(df['epoch'], df['train/mAP_0.50_0.95_all'] * 100, label='Train mAP', color='blue', linewidth=2.5)
#     plt.plot(df['epoch'], df['val/mAP_0.50_0.95_all'] * 100, label='Validation mAP', color='orange', linewidth=2.5)
    
#     plt.title(title, pad=20)
#     plt.xlabel("Epoch")
#     plt.ylabel("mAP (%)")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# def plot_detr_dual_axis(df, title="DETR: Validation Loss vs Validation mAP"):
#     """
#     Dual-axis plot showing how loss decreases while mAP increases (or diverges).
#     """
#     if 'val/total_loss' not in df.columns:
#         print("Error: Validation loss not found in this DataFrame.")
#         return
        
#     fig, ax1 = plt.subplots(figsize=(10, 6))

#     color = 'tab:red'
#     ax1.set_xlabel('Epoch')
#     ax1.set_ylabel('Validation Loss', color=color)
#     ax1.plot(df['epoch'], df['val/total_loss'], color=color, linewidth=2.5, marker='o')
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()  
#     color = 'tab:blue'
#     ax2.set_ylabel('Validation mAP (%)', color=color)  
#     ax2.plot(df['epoch'], df['val/mAP_0.50_0.95_all'] * 100, color=color, linewidth=2.5, marker='s')
#     ax2.tick_params(axis='y', labelcolor=color)

#     fig.tight_layout()  
#     plt.title(title, pad=20)
#     plt.show()
    
# def plot_class_performance(experiments_dict, title="Per-Class Performance (Car vs Pedestrian)"):
#     """
#     Bonus Plot: Compares the model's ability to detect Cars versus Pedestrians.
#     """
#     data = []
#     for exp_name, df in experiments_dict.items():
#         best_idx = df['val/mAP_0.50_0.95_all'].idxmax()
#         best_row = df.loc[best_idx]
        
#         data.append({'Experiment': exp_name, 'Class': 'Car', 'mAP': best_row['val/mAP_Car'] * 100})
#         data.append({'Experiment': exp_name, 'Class': 'Pedestrian', 'mAP': best_row['val/mAP_Pedestrian'] * 100})
        
#     plot_df = pd.DataFrame(data)
    
#     plt.figure(figsize=(10, 6))
#     sns.barplot(data=plot_df, x='Experiment', y='mAP', hue='Class', palette="flare")
#     plt.title(title, pad=20)
#     plt.ylabel("Validation mAP (%)")
#     plt.legend(title="Object Class")
#     plt.tight_layout()
#     plt.show()

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Teammate's Custom Palettes
PASTEL_COLORS = ["#A8E6CF", "#FFD3B6", "#FFAAA5", "#D5AAFF", "#CFCFCF"]
LINE_COLORS = {"cyan": "#5BC0DE", "tomato": "#FF7F6E", "green": "#82B366"}
SHORT_DASHES = (0, (2, 2))

def load_metrics(folder_path):
    """Loads the metrics.json from a given experiment folder into a Pandas DataFrame."""
    json_path = os.path.join(folder_path, "metrics.json")
    if not os.path.exists(json_path):
        print(f"Warning: No metrics.json found in {folder_path}")
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def plot_augmentation_impact(experiments_dict, title="Impact of Data Augmentation on mAP"):
    """Bar plot comparing the best validation mAP across different augmentation strategies."""
    names = list(experiments_dict.keys())
    best_maps = [df['val/mAP_0.50_0.95_all'].max() * 100 for df in experiments_dict.values()]
    
    plt.figure(figsize=(11, 5.5))
    bars = sns.barplot(x=names, y=best_maps, palette=PASTEL_COLORS[:len(names)], 
                       edgecolor="black", linewidth=0.8)
    
    # Styling for slide visibility
    plt.title(title, fontsize=15, fontweight="bold", pad=14)
    plt.ylabel("Best Validation mAP (%)", fontsize=13, fontweight="bold")
    plt.ylim(0, max(best_maps) + 10)
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().set_axisbelow(True)
    
    # Value labels
    for bar in bars.containers:
        bars.bar_label(bar, fmt='%.1f%%', padding=3, fontsize=11, fontweight="bold")
        
    plt.tight_layout()
    plt.show()

def plot_lora_vs_full_convergence(full_df, lora_df, model_name="DETR"):
    """Plots the validation mAP convergence of Full Fine-Tune vs LoRA over epochs."""
    plt.figure(figsize=(7, 5.5))
    
    plt.plot(full_df['epoch'], full_df['val/mAP_0.50_0.95_all'] * 100, 
             label='Full Fine-Tune', linewidth=2.3, color=LINE_COLORS['cyan'])
             
    plt.plot(lora_df['epoch'], lora_df['val/mAP_0.50_0.95_all'] * 100, 
             label='LoRA (Parameter Efficient)', linewidth=2.3, color=LINE_COLORS['tomato'], 
             linestyle=SHORT_DASHES)
    
    # Styling
    plt.title(f"{model_name} – Convergence Comparison", fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("Validation mAP (%)", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

def plot_object_size_performance(experiments_dict, title="Performance by Object Size"):
    """Grouped bar chart showing mAP for Small, Medium, and Large objects."""
    data = []
    for exp_name, df in experiments_dict.items():
        best_idx = df['val/mAP_0.50_0.95_all'].idxmax()
        best_row = df.loc[best_idx]
        
        data.append({'Experiment': exp_name, 'Size': 'Small', 'mAP': best_row['val/mAP_0.50_0.95_small'] * 100})
        data.append({'Experiment': exp_name, 'Size': 'Medium', 'mAP': best_row['val/mAP_0.50_0.95_medium'] * 100})
        data.append({'Experiment': exp_name, 'Size': 'Large', 'mAP': best_row['val/mAP_0.50_0.95_large'] * 100})
        
    plot_df = pd.DataFrame(data)
    
    plt.figure(figsize=(11, 5.5))
    sns.barplot(data=plot_df, x='Experiment', y='mAP', hue='Size', 
                palette=PASTEL_COLORS[:3], edgecolor="black", linewidth=0.8)
                
    # Styling
    plt.title(title, fontsize=15, fontweight="bold", pad=14)
    plt.ylabel("Validation mAP (%)", fontsize=13, fontweight="bold")
    plt.xlabel("")
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().set_axisbelow(True)
    plt.legend(title="Object Size", fontsize=11, title_fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

def plot_train_val_convergence(df, title="Training vs Validation Convergence"):
    """Plots Train vs Validation mAP to analyze overfitting and healthy learning."""
    plt.figure(figsize=(7, 5.5))
    
    plt.plot(df['epoch'], df['train/mAP_0.50_0.95_all'] * 100, 
             label='Train mAP', color=LINE_COLORS['cyan'], linewidth=2.3)
             
    plt.plot(df['epoch'], df['val/mAP_0.50_0.95_all'] * 100, 
             label='Validation mAP', color=LINE_COLORS['cyan'], linewidth=2.3, linestyle=SHORT_DASHES)
    
    # Styling
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel("mAP (%)", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

def plot_detr_dual_axis(df, title="DETR: Validation Loss vs Validation mAP"):
    """Dual-axis plot showing how loss decreases while mAP increases (or diverges)."""
    if 'val/total_loss' not in df.columns:
        print("Error: Validation loss not found in this DataFrame.")
        return
        
    fig, ax1 = plt.subplots(figsize=(7, 5.5))

    color_loss = LINE_COLORS['tomato']
    color_map = LINE_COLORS['cyan']
    
    ax1.set_xlabel('Epoch', fontsize=14, fontweight="bold")
    ax1.set_ylabel('Validation Loss', color=color_loss, fontsize=14, fontweight="bold")
    ax1.plot(df['epoch'], df['val/total_loss'], color=color_loss, linewidth=2.3)
    ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=13)
    ax1.tick_params(axis='x', labelsize=13)

    ax2 = ax1.twinx()  
    ax2.set_ylabel('Validation mAP (%)', color=color_map, fontsize=14, fontweight="bold")  
    ax2.plot(df['epoch'], df['val/mAP_0.50_0.95_all'] * 100, color=color_map, linewidth=2.3, linestyle=SHORT_DASHES)
    ax2.tick_params(axis='y', labelcolor=color_map, labelsize=13)

    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    ax1.grid(alpha=0.25)
    
    fig.tight_layout()  
    plt.show()
    
def plot_class_performance(experiments_dict, title="Per-Class Performance (Car vs Pedestrian)"):
    """Bonus Plot: Compares the model's ability to detect Cars versus Pedestrians."""
    data = []
    for exp_name, df in experiments_dict.items():
        best_idx = df['val/mAP_0.50_0.95_all'].idxmax()
        best_row = df.loc[best_idx]
        
        data.append({'Experiment': exp_name, 'Class': 'Car', 'mAP': best_row['val/mAP_Car'] * 100})
        data.append({'Experiment': exp_name, 'Class': 'Pedestrian', 'mAP': best_row['val/mAP_Pedestrian'] * 100})
        
    plot_df = pd.DataFrame(data)
    
    plt.figure(figsize=(11, 5.5))
    # Pick two distinct pastels for the classes
    sns.barplot(data=plot_df, x='Experiment', y='mAP', hue='Class', 
                palette=[PASTEL_COLORS[0], PASTEL_COLORS[2]], edgecolor="black", linewidth=0.8)
                
    # Styling
    plt.title(title, fontsize=15, fontweight="bold", pad=14)
    plt.ylabel("Validation mAP (%)", fontsize=13, fontweight="bold")
    plt.xlabel("")
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.gca().set_axisbelow(True)
    plt.legend(title="Object Class", fontsize=11, title_fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

# --- ADD THESE TO THE BOTTOM OF utils/plot_metrics.py ---

def continuous(series: pd.Series) -> pd.Series:
    """Fill missing values to keep lines continuous."""
    s = pd.to_numeric(series, errors="coerce")
    return s.interpolate(method="linear", limit_direction="both")

def plot_single_train_val(df, metric_suffix, title, ylabel, color_name="cyan"):
    """
    Plots Train vs Val for a single experiment. 
    metric_suffix should be the part after 'train/' or 'val/' (e.g., 'total_loss' or 'mAP_0.50_0.95_all').
    """
    plt.figure(figsize=(7, 5.5))
    
    col_train = f"train/{metric_suffix}"
    col_val = f"val/{metric_suffix}"
    
    y_train = continuous(df[col_train]) if col_train in df.columns else None
    y_val = continuous(df[col_val]) if col_val in df.columns else None
    
    color = LINE_COLORS.get(color_name, LINE_COLORS['cyan'])
    
    if y_train is not None:
        plt.plot(df["epoch"], y_train, label="Train", linewidth=2.3, color=color)
    if y_val is not None:
        plt.plot(df["epoch"], y_val, label="Validation", linewidth=2.3, color=color, linestyle=SHORT_DASHES)
        
    # Styling (slide-friendly)
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

def plot_compare_train_val(df1, df2, label1, label2, metric_suffix, title, ylabel):
    """
    Plots Train vs Val for TWO experiments on the same graph to compare them directly.
    Matches the exact style of the teammate's comparison plot.
    """
    plt.figure(figsize=(7, 5.5))
    
    col_train = f"train/{metric_suffix}"
    col_val = f"val/{metric_suffix}"
    
    # Process Data
    y1_train = continuous(df1[col_train]) if col_train in df1.columns else None
    y1_val = continuous(df1[col_val]) if col_val in df1.columns else None
    y2_train = continuous(df2[col_train]) if col_train in df2.columns else None
    y2_val = continuous(df2[col_val]) if col_val in df2.columns else None
    
    # Plot Experiment 1 (Cyan)
    if y1_train is not None:
        plt.plot(df1["epoch"], y1_train, label=f"Train – {label1}", linewidth=2.3, color=LINE_COLORS["cyan"])
    if y1_val is not None:
        plt.plot(df1["epoch"], y1_val, label=f"Val – {label1}", linewidth=2.3, color=LINE_COLORS["cyan"], linestyle=SHORT_DASHES)

    # Plot Experiment 2 (Tomato)
    if y2_train is not None:
        plt.plot(df2["epoch"], y2_train, label=f"Train – {label2}", linewidth=2.3, color=LINE_COLORS["tomato"])
    if y2_val is not None:
        plt.plot(df2["epoch"], y2_val, label=f"Val – {label2}", linewidth=2.3, color=LINE_COLORS["tomato"], linestyle=SHORT_DASHES)

    # Styling
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

# Make sure your LINE_COLORS at the top of the file has at least 3 colors:
# LINE_COLORS = {"cyan": "#5BC0DE", "tomato": "#FF7F6E", "green": "#82B366", "purple": "#D5AAFF"}

def plot_multiple_train_val(experiments_dict, metric_suffix, title, ylabel):
    """
    Plots Train vs Val for an arbitrary number of experiments on the same graph.
    Places the legend outside the plot to prevent overlapping the lines.
    """
    plt.figure(figsize=(10, 5.5)) # Slightly wider to accommodate the external legend
    
    col_train = f"train/{metric_suffix}"
    col_val = f"val/{metric_suffix}"
    
    # Grab our strict line colors
    color_palette = list(LINE_COLORS.values())
    
    for i, (label, df) in enumerate(experiments_dict.items()):
        color = color_palette[i % len(color_palette)]
        
        y_train = continuous(df[col_train]) if col_train in df.columns else None
        y_val = continuous(df[col_val]) if col_val in df.columns else None
        
        # Plot Solid for Train, Short-Dashed for Val
        if y_train is not None:
            # We scale mAP by 100 if the word 'mAP' is in the ylabel, else plot as-is
            y_t = y_train * 100 if "mAP" in ylabel else y_train
            plt.plot(df["epoch"], y_t, label=f"Train – {label}", linewidth=2.3, color=color)
            
        if y_val is not None:
            y_v = y_val * 100 if "mAP" in ylabel else y_val
            plt.plot(df["epoch"], y_v, label=f"Val – {label}", linewidth=2.3, color=color, linestyle=SHORT_DASHES)

    # Styling (slide-friendly)
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(alpha=0.25)
    
    # Put legend outside the plot box so it doesn't cover the data
    plt.legend(fontsize=11, frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    
    plt.tight_layout()
    plt.show()

def plot_single_experiment_train_val(df, metric_suffix, title, ylabel, base_color="cyan"):
    """
    Plots Train vs Val for ONE single experiment on ONE graph.
    """
    plt.figure(figsize=(7, 5.5))
    
    col_train = f"train/{metric_suffix}"
    col_val = f"val/{metric_suffix}"
    
    y_train = continuous(df[col_train]) if col_train in df.columns else None
    y_val = continuous(df[col_val]) if col_val in df.columns else None
    
    # Get color from teammate's palette (fallback to standard cyan)
    color = LINE_COLORS.get(base_color, "#5BC0DE")
    
    if y_train is not None:
        y_t = y_train * 100 if "mAP" in ylabel else y_train
        plt.plot(df["epoch"], y_t, label="Train", linewidth=2.3, color=color)
        
    if y_val is not None:
        y_v = y_val * 100 if "mAP" in ylabel else y_val
        plt.plot(df["epoch"], y_v, label="Validation", linewidth=2.3, color=color, linestyle=SHORT_DASHES)
        
    # Styling
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(alpha=0.25)
    plt.legend(fontsize=12, frameon=True)
    plt.tight_layout()
    plt.show()

def plot_wandb_style_comparison(experiments_dict, metric_name, title, ylabel):
    """
    WandB Style: Plots a single specific metric (e.g., 'val/mAP_0.50_0.95_all') 
    across multiple experiments on the same graph using solid lines.
    """
    plt.figure(figsize=(8, 5.5))
    
    # Extended palette to match the teammate's aesthetic for multiple lines
    extended_palette = [
        LINE_COLORS.get("cyan", "#5BC0DE"), 
        LINE_COLORS.get("tomato", "#FF7F6E"), 
        LINE_COLORS.get("green", "#82B366"),
        "#D5AAFF", # pastel lavender
        "#FFD3B6"  # pastel peach
    ]
    
    for i, (label, df) in enumerate(experiments_dict.items()):
        color = extended_palette[i % len(extended_palette)]
        
        y_data = continuous(df[metric_name]) if metric_name in df.columns else None
        
        if y_data is not None:
            # Scale to percentage if it's an mAP metric
            y_vals = y_data * 100 if "mAP" in ylabel else y_data
            plt.plot(df["epoch"], y_vals, label=label, linewidth=2.3, color=color)
            
    # Styling
    plt.xlabel("Epoch", fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    plt.title(title, fontsize=16, fontweight="bold", pad=12)
    
    plt.xticks(fontsize=13, fontweight="bold")
    plt.yticks(fontsize=13, fontweight="bold")
    
    plt.grid(alpha=0.25)
    
    # Move legend outside if there are many experiments
    if len(experiments_dict) > 3:
        plt.legend(fontsize=12, frameon=True, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        plt.legend(fontsize=12, frameon=True)
        
    plt.tight_layout()
    plt.show()