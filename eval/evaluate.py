import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix

# Call like this from the project directory root: python -m eval.evaluate SEResNet_2025-04-27 SEResNet


def evaluate_checkpoint(
    model,
    checkpoint_path,
    test_loader,
    device,
    output_prefix,
    plots_dir,
    checkpoint_dir,
):
    # Load checkpoint trained model
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc=f"Evaluating {output_prefix}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total
    print(f"\n=== Test Results ({output_prefix}) ===")
    print(f"Total Samples: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Confusion Matrix
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Save Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {output_prefix}")
    cm_plot_path = os.path.join(plots_dir, f"confusion_matrix_{output_prefix}.png")
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    print(f"Saved confusion matrix plot to {cm_plot_path}")
    plt.close()

    # Class-wise Accuracy Calc.
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nClass-wise Accuracy:")
    for idx, acc in enumerate(class_acc):
        print(f"Class {idx}: {acc * 100:.2f}%")

    # Save Class-wise Accuracy
    class_acc_df = pd.DataFrame(
        {"Class": np.arange(len(class_acc)), "Accuracy (%)": class_acc * 100}
    )
    class_acc_csv = os.path.join(
        checkpoint_dir, f"class_wise_accuracy_{output_prefix}.csv"
    )
    class_acc_df.to_csv(class_acc_csv, index=False)
    print(f"Saved class-wise accuracy to {class_acc_csv}")


def evaluate_model(model_folder, model_name):
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using {device} device")

    # Import Model (add new logic for each new nn.Module)
    if model_name == "SEResNet":
        from models.SEResNet import SEResNet

        model = SEResNet(input_channels=3, num_classes=7).to(device)
    else:
        raise ValueError(
            f"Model {model_name} is not supported yet. Add it to the script!"
        )

    # Load Test Data
    from train.load import get_dataloaders

    data_dir = "data/FER-2013/"
    _, _, test_loader = get_dataloaders(
        data_dir,
        img_size=48,
        batch_size=64,
        num_workers=2,
        val_split=0.2,
        pin_memory=(device.type == "cuda"),
    )

    # Paths
    checkpoint_dir = os.path.join("checkpoints", model_folder)
    plots_dir = os.path.join(checkpoint_dir, "plots")
    metrics_path = os.path.join(checkpoint_dir, "metrics.csv")
    os.makedirs(plots_dir, exist_ok=True)

    # Paths to Final/Best models
    best_model_path = os.path.join(checkpoint_dir, "Best.pth")
    final_model_path = os.path.join(checkpoint_dir, "Final.pth")

    # Evaluate Best.pth
    if os.path.exists(best_model_path):
        evaluate_checkpoint(
            model,
            best_model_path,
            test_loader,
            device,
            "best",
            plots_dir,
            checkpoint_dir,
        )
    else:
        print(f"WARNING: {best_model_path} not found!")

    # Evaluate Final.pth
    if os.path.exists(final_model_path):
        evaluate_checkpoint(
            model,
            final_model_path,
            test_loader,
            device,
            "final",
            plots_dir,
            checkpoint_dir,
        )
    else:
        print(f"WARNING: {final_model_path} not found!")

    # Plot Losses (if available)
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)

        plt.figure(figsize=(10, 6))
        plt.plot(metrics["train_loss"], label="Train Loss")
        plt.plot(metrics["val_loss"], label="Validation Loss")
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

        plt.figure(figsize=(10, 6))
        gap = metrics["val_loss"] - metrics["train_loss"]
        plt.plot(gap, label="Val Loss - Train Loss (Gap)")
        plt.axhline(0, color="black", linestyle="--")
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
    parser = argparse.ArgumentParser(
        description="Evaluate model checkpoints and plot training metrics."
    )
    parser.add_argument("model_folder", type=str, help="Folder inside checkpoints/")
    parser.add_argument(
        "model_name", type=str, help="Model name to load (e.g., SEResNet)"
    )
    args = parser.parse_args()

    evaluate_model(args.model_folder, args.model_name)
