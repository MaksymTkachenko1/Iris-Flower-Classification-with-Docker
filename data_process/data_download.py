import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import os
import json

def download_and_split_data(settings_path: str):
    """Downloads the Iris dataset, splits it, scales features with MinMaxScaler, and saves to CSV files."""
    try:
        # Load settings
        with open(settings_path, 'r') as f:
            settings = json.load(f)

        # Get data directory from settings
        data_dir = settings["general"]["data_dir"]

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Load Iris dataset
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target

        # Split into training and inference sets
        train_df, inference_df = train_test_split(
            df, test_size=0.3, random_state=settings["general"]["random_state"]
        )

        # Apply MinMaxScaler to features
        scaler = MinMaxScaler()
        feature_columns = [col for col in train_df.columns if col != 'target']

        train_df[feature_columns] = scaler.fit_transform(train_df[feature_columns])
        inference_df[feature_columns] = scaler.transform(inference_df[feature_columns])

        # Construct full paths for saving CSV files
        train_csv_path = os.path.join(data_dir, settings["train"]["table_name"])
        inference_csv_path = os.path.join(data_dir, settings["inference"]["inp_table_name"])

        # Save dataframes to CSV files
        train_df.to_csv(train_csv_path, index=False)
        inference_df.to_csv(inference_csv_path, index=False)

        print(f"Data downloaded, split, and scaled. Training data saved to: {train_csv_path}")
        print(f"Inference data saved to: {inference_csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    SETTINGS_PATH = "settings.json"
    download_and_split_data(SETTINGS_PATH)