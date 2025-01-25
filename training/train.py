import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import json
import os
import logging
import argparse


class IrisDataset(Dataset):
    """Iris dataset."""

    def __init__(self, data_dir: str, table_name: str, test_size: float, random_state: int):
        # Read data directly in __init__
        train_path = os.path.join(data_dir, table_name)
        try:
            df = pd.read_csv(train_path)
        except FileNotFoundError:
            logging.error(f"Training data file not found at {train_path}")
            raise

        X = df.drop('target', axis=1).values
        y = df['target'].values

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Scale data here
        scaler = MinMaxScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.int64)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.int64)

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def get_val_data(self):
        """Returns the validation data and labels."""
        return self.X_val, self.y_val


class SimpleClassifier(nn.Module):
    """A simple classifier model."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 64)
        self.relu = nn.ReLU()  # Add ReLU activation to introduce non-linearity
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # Apply ReLU activation
        x = self.fc2(x)
        return x


def train(settings_path: str):
    """Trains the model and saves it."""
    with open(settings_path, 'r') as f:
        settings = json.load(f)

    # Logging setup
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Number of epochs from settings: {settings['train']['num_epochs']}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load dataset
    dataset = IrisDataset(
        data_dir=settings["general"]["data_dir"],
        table_name=settings["train"]["table_name"],
        test_size=settings["train"]["test_size"],
        random_state=settings["general"]["random_state"],
    )
    dataloader = DataLoader(dataset, batch_size=settings["train"]["batch_size"], shuffle=True)

    # Model, loss, optimizer
    model = SimpleClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=settings["train"]["learning_rate"])

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = settings["train"]["early_stop_patience"]

    # Training loop
    for epoch in range(settings["train"]["num_epochs"]):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        X_val, y_val = dataset.get_val_data()
        X_val, y_val = X_val.to(device), y_val.to(device)
        with torch.no_grad():
            outputs = model(X_val)
            val_loss = criterion(outputs, y_val).item()

        logging.info(f"Epoch {epoch} completed. Validation Loss: {val_loss:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save model
            model_dir = settings["general"]["models_dir"]
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, settings["train"]["model_name"])
            torch.save(model.state_dict(), model_path)
            logging.info(f"Model saved to {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stop_patience:
                logging.info(f"Early stopping triggered at epoch {epoch}")
                break

        model.train()

    # Evaluate on test set
    model.eval()  # Set model to evaluation mode
    X_test, y_test = dataset.get_val_data()
    X_test, y_test = X_test.to(device), y_test.to(device)
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(y_test.cpu(), predicted.cpu())
        # Removed the line that logs the test accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model on the Iris dataset.")
    parser.add_argument(
        "--settings_path", type=str, default="settings.json", help="Path to the settings JSON file."
    )
    args = parser.parse_args()

    # Load settings
    with open(args.settings_path, 'r') as f:
        settings = json.load(f)

    # Set default values for training parameters in settings
    settings["train"]["batch_size"] = settings["train"].get("batch_size", 32)
    settings["train"]["learning_rate"] = settings["train"].get("learning_rate", 0.001)
    settings["train"]["num_epochs"] = settings["train"].get("num_epochs", 300)
    settings["train"]["early_stop_patience"] = settings["train"].get("early_stop_patience", 5)

    train(args.settings_path)