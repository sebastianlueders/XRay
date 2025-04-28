import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

from torchvision import transforms, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
from train.load import get_dataloaders
from models.SEResNet import SEResNet


if __name__ == "__main__":
    MODEL_NAME = "SEResNet-FER2013"   # Change for each 


    # Set up output directory for saving model results
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join("checkpoints", f"{MODEL_NAME}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving all outputs to: {output_dir}")


    # Setup Training Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using {device} device")


    # Load Data
    data_dir = "data/FER-2013/"  # Change based on dataset

    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir,
        img_size=48,
        batch_size=64,
        num_workers=2,
        val_split=0.2,
        pin_memory=(device.type == "cuda")   # Changes to True when running with CUDA
    )

    # Instantiate model, optimizer, loss and scheduler (for dynamic LR)
    model = SEResNet(input_channels=3, num_classes=7).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    # Training Loop
    num_epochs = 30
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    # Make sure checkpoint directory exists for model output
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, "Best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to {best_model_path}")

    final_model_path = os.path.join(output_dir, "Final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")


    # Save training metrics
    metrics_df = pd.DataFrame({
        "train_loss": train_losses,
        "val_loss": val_losses,
    })

    metrics_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved metrics to {metrics_path}")

