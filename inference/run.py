import pandas as pd
import json
import os
import argparse
import logging
import sys
import torch
from sklearn.metrics import accuracy_score

sys.path.append('/app')
from training.train import SimpleClassifier  # Import the SimpleClassifier model

def run_inference(settings_path: str, model_name: str, output_folder: str):
    """Runs inference on the test data, calculates the test accuracy, and saves the predictions."""
    logging.info("Starting inference...")

    # Load settings
    try:
        with open(settings_path, 'r') as f:
            settings = json.load(f)
    except FileNotFoundError:
        logging.error(f"Settings file not found at {settings_path}")
        return

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleClassifier().to(device)
    model_path = os.path.join(settings["general"]["models_dir"], model_name)
    logging.info(f"Loading model from: {model_path}")

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # Set the model to evaluation mode
        logging.info("Model loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return

    # Load inference data
    inference_data_path = os.path.join(settings["general"]["data_dir"], settings["inference"]["test_data"])
    logging.info(f"Loading inference data from: {inference_data_path}")

    try:
        df = pd.read_csv(inference_data_path)
        logging.info(f"Inference data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Inference data file not found at {inference_data_path}")
        return

    # Prepare data
    try:
        X = df.drop('target', axis=1).values  # Assuming 'target' column exists
        y = df['target'].values # Extract true labels for evaluation
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        logging.info("Inference data prepared for the model.")
    except KeyError:
        logging.error("Error: 'target' column not found in inference data.")
        return

    # Run inference
    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)

    logging.info("Inference completed.")
    
    # Calculate test accuracy
    y_pred = predicted.cpu().numpy()
    accuracy = accuracy_score(y, y_pred)

    logging.info(f"Test Accuracy: {accuracy:.4f}")

    # Save predictions
    df['predictions'] = predicted.cpu().numpy()
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, settings["inference"]["output_name"])
    logging.info(f"Saving predictions to: {output_file}")

    try:
        df.to_csv(output_file, index=False)
        logging.info(f"Predictions saved successfully to: {output_file}")
    except Exception as e:
        logging.error(f"Error saving predictions: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a trained model.")
    parser.add_argument(
        "--settings_path", type=str, default="settings.json", help="Path to the settings JSON file."
    )
    parser.add_argument(
        "--model_name", type=str, default="iris_model.pth", help="Name of the model file to load."
    )
    parser.add_argument(
        "--output_folder", type=str, default="/app/output", help="Folder to save the predictions."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    run_inference(args.settings_path, args.model_name, args.output_folder)