import unittest
import tempfile
import os
import json
import shutil
import sys
import torch

# Add training directory to sys.path to import train module
training_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training"))
sys.path.insert(0, training_dir)
from training import train
from training.train import SimpleClassifier, IrisDataset

class TestTraining(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test data and models
        self.test_dir = tempfile.mkdtemp()
        # Create a dummy CSV file with Iris-like data
        self.csv_path = os.path.join(self.test_dir, "iris.csv")
        with open(self.csv_path, "w") as f:
            f.write("sepal_length,sepal_width,petal_length,petal_width,target\n")
            f.write("5.1,3.5,1.4,0.2,0\n")
            f.write("4.9,3.0,1.4,0.2,0\n")
            f.write("6.2,3.4,5.4,2.3,2\n")
            f.write("5.9,3.0,5.1,1.8,2\n")
            f.write("5.5,2.3,4.0,1.3,1\n")
            f.write("6.5,2.8,4.6,1.5,1\n")
        
        # Create a dummy settings file
        self.settings = {
            "general": {
                "data_dir": self.test_dir,
                "models_dir": os.path.join(self.test_dir, "models"),
                "random_state": 42
            },
            "train": {
                "table_name": "iris.csv",
                "test_size": 0.2,
                "batch_size": 2,
                "learning_rate": 0.01,
                "num_epochs": 5,
                "early_stop_patience": 3,
                "model_name": "dummy_model.pth"
            }
        }
        self.settings_path = os.path.join(self.test_dir, "settings.json")
        with open(self.settings_path, "w") as f:
            json.dump(self.settings, f)

    def tearDown(self):
        # Remove temporary directory and all contents
        shutil.rmtree(self.test_dir)

    def test_train_runs_and_saves_model(self):
        # Run training with the dummy settings
        train.train(self.settings_path)
        # Check that the model file is created
        model_path = os.path.join(self.settings["general"]["models_dir"], self.settings["train"]["model_name"])
        self.assertTrue(os.path.exists(model_path), "Model file was not created.")

    def test_model_initialization(self):
        model = SimpleClassifier()
        self.assertIsInstance(model, SimpleClassifier, "Model is not an instance of SimpleClassifier.")

    def test_dataset_loading(self):
        dataset = IrisDataset(
            data_dir=self.settings["general"]["data_dir"],
            table_name=self.settings["train"]["table_name"],
            test_size=self.settings["train"]["test_size"],
            random_state=self.settings["general"]["random_state"]
        )
        self.assertEqual(len(dataset), 4, "Dataset length is not correct.")
        self.assertEqual(len(dataset.get_val_data()[0]), 2, "Validation data length is not correct.")

    def test_train_with_default_parameters(self):
        # Modify settings for default parameters
        self.settings["train"]["num_epochs"] = 1
        with open(self.settings_path, "w") as f:
            json.dump(self.settings, f)
        # Run training with the modified settings
        train.train(self.settings_path)
        # Check that the model file is created
        model_path = os.path.join(self.settings["general"]["models_dir"], self.settings["train"]["model_name"])
        self.assertTrue(os.path.exists(model_path), "Model file was not created with default parameters.")

if __name__ == "__main__":
    unittest.main()
