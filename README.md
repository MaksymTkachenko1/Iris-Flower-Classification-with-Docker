# Iris Flower Classification with Docker

This project demonstrates training and inference of a simple classification model for the Iris flower dataset using Docker. It includes data preprocessing, model training, and inference, all managed within Docker containers.

## Project Summary

The goal of this project is to classify Iris flowers into three species (setosa, versicolor, virginica) based on sepal and petal measurements. We achieve this by training a neural network model using PyTorch. The project leverages Docker to ensure a consistent and reproducible environment for development, training, and inference.

## Project Structure

```
Homework_Maksym_Tkachenko_MLE/
├── data_process/
│   └── data_download.py        # Downloads, preprocesses, and splits the Iris dataset
├── inference/
│   └── run.py                  # Inference script
│   └── Dockerfile              # Dockerfile for inference image
├── training/
│   └── train.py                # Training script, including the model definition
│   └── Dockerfile              # Dockerfile for training image
├── .gitignore                  # Specifies files and directories to be ignored by Git
├── README.md                   # This file
├── __init__.py                 
├── requirements.txt            # Lists project dependencies
├── settings.json               # Configuration file for training and inference
```

**Note**: The directories `data/`, `models/`, and `output/` are not present in the initial repository structure. They are created during the data processing, training and inference, respectively.

## Prerequisites

*   [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running.

## Dataset

The Iris dataset is a classic dataset in machine learning and statistics. It contains 150 samples of Iris flowers, with 50 samples for each of the three species:

*   Iris setosa
*   Iris versicolor
*   Iris virginica

Each sample has four features:

*   Sepal length (cm)
*   Sepal width (cm)
*   Petal length (cm)
*   Petal width (cm)

The dataset is publicly available and is downloaded within the project using the `data_download.py` script.

The `data_download.py` script also preprocesses the dataset. It converts categorical labels to numerical representations (0, 1, 2) and scales the numerical features using `MinMaxScaler` to a range between 0 and 1. This preprocessing step ensures that the features are on a similar scale, which can improve the performance of the model.

## Model

The model used for classification is a simple fully connected neural network defined in `train.py`. It consists of:

*   An input layer with 4 neurons (corresponding to the 4 features).
*   A hidden layer with 64 neurons and ReLU activation.
*   An output layer with 3 neurons (corresponding to the 3 classes) and no activation (as we're using CrossEntropyLoss which implicitly applies Softmax).

**Key Hyperparameters** (configurable in `settings.json`):

*   `learning_rate`: 0.001
*   `epochs`: 300
*   `batch_size`: 32

These hyperparameters were chosen based on experimentation and can be further tuned for better performance.

## Results

The model was trained on the training set and achieved the following performance on the inference set:

*   **Accuracy:** The model achieved a high accuracy during testing, often reaching between 95% and 100% on the inference set. The model was trained for ≈180 epochs (EarlyStopping). During the training process, the model's performance was monitored, and the best model was saved for inference.

## Steps to Reproduce

**Note:** These commands are intended to be run in a **PowerShell** terminal.

### 1. Clone the Repository

```powershell
git clone https://github.com/MaksymTkachenko1/Homework_Maksym_Tkachenko_MLE.git
```

### 2. Navigate to the Project Directory

```powershell
cd Homework_Maksym_Tkachenko_MLE
```

### 3. Build the Training Image

```powershell
docker build -f ./training/Dockerfile -t training_image .
```

### 4. Run data_download.py inside the training image to get data

To run the `data_download.py` script and populate local machine with the necessary data, use the following command. This command mounts the data directory from current working directory to the /app/data directory inside the Docker container, ensuring that the processed data is saved to local machine.

```powershell
docker run -v ${PWD}/data:/app/data training_image python /app/data_process/data_download.py
```

This step will run `data_download.py` to get, preprocess and prepare the data in newly created `data` directory. It will also scale numerical features using `MinMaxScaler`. The processed data will be split into `iris_train.csv` and `iris_inference.csv`.

**Note**: Although `data_download.py` is included in the training image build process, we run this script independently at this step to explicitly verify that the data exists and is prepared correctly before training. This also ensures that the data is available on local machine.

### 5. Run Training

```powershell
docker run -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models training_image
```

This will train the model and save it as `iris_model.pth` in the newly created `models` directory.

### 6. Build the Inference Image

```powershell
docker build -f ./inference/Dockerfile -t inference_image .
```

### 7. Run Inference

```powershell
docker run -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/output:/app/output inference_image
```

This will generate predictions and save them to `predictions.csv` in the newly created `output` directory. It will also calculate and print the accuracy of the model on the inference dataset.

## Settings

The `settings.json` file contains configuration parameters for training and inference. You can modify these settings to change the model name, data paths, training hyperparameters, etc.

## Notes

*   The `data_download.py` script downloads the Iris dataset, preprocesses it (including scaling numerical features using `MinMaxScaler`), and splits it into training and inference sets.
*   The trained model is saved in the `models` directory.
*   The inference results (predictions and accuracy) are saved in the `output` directory.
*   The `requirements.txt` file lists the Python dependencies for this project.

## Created by

Maksym Tkachenko