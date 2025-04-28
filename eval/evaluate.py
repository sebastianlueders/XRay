import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def evaluate_model(model_folder):
    # Paths
    checkpoint_dir = os.path.join("checkpoints", model_folder)
    plots_dir = os.path.join(checkpoint_dir, "plots")
    metrics_path = os.path.join(checkpoint_dir, "metrics.csv")

    # Make sure plots folder exists
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    # Load metrics
    metrics = pd.read_csv(metrics_path)

    # Plot Losses
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss - {model_folder}")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(plots_dir, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_plot_path)
    print(f"Saved loss plot to {loss_plot_path}")
    plt.close()

    # Plot overfitting detection (gap between train and val loss)
    plt.figure(figsize=(10, 6))
    gap = (metrics['val_loss'] - metrics['train_loss'])
    plt.plot(gap, label='Val Loss - Train Loss (Gap)')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss Gap")
    plt.title(f"Overfitting Detection - {model_folder}")
    plt.legend()
    plt.grid(True)
    gap_plot_path = os.path.join(plots_dir, "loss_gap_curve.png")
    plt.tight_layout()
    plt.savefig(gap_plot_path)
    print(f"Saved gap plot to {gap_plot_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and plot model metrics.")
    parser.add_argument("model_folder", type=str, help="Name of the model folder inside checkpoints/ (e.g., SEResNet_2025-04-27-18-30)")
    args = parser.parse_args()

    evaluate_model(args.model_folder)
