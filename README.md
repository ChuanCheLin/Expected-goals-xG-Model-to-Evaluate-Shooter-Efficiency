# Expected-goals-xG-Model-to-Evaluate-Shooter-Efficiency

Welcome to the GitHub repository for the Expected Goals (xG) Model to Evaluate Shooter Efficiency project! This project applies machine learning techniques to predict the probability of a shot resulting in a goal in football, based on various factors such as shot location and method of assist. Below is a brief introduction to the contents of this repository:

## Directory and File Overview

- `/data`: Contains all the necessary data files except for `events.csv`, which is too large for direct upload. Please unzip `archive.zip` into this directory to obtain `events.csv`.

- `/models`: This directory stores the `.pkl` files for all the trained models that can be used for prediction without the need to retrain.

- `archive.zip`: A compressed file containing `events.csv`. After cloning the repository, you need to manually unzip this file into the `data` directory to fully set up the project.

- `model.py`: Contains the Python modules for model training and evaluation, including functions for training models with cross-validation and computing evaluation metrics.

- `preprocessing.py`: A Python script that outlines the data preprocessing steps, including data cleaning, merging, and encoding necessary for preparing the data for model training.

- `model_analysis_<model>.ipynb`: Jupyter notebooks dedicated to the analysis of different trained models (Logistic Regression (LR), Random Forest (RF), Gradient Boosting (GB)).

- `model_training_<model>_<metric>.ipynb`: These notebooks contain the process of training different machine learning models and evaluating them based on various metrics (accuracy, AUC-ROC).

## How to Set Up

1. Clone the repository to your local machine.
2. Navigate to the `/data` directory.
3. Unzip the `archive.zip` file to extract the `events.csv` file.
4. Place the `events.csv` file into the `/data` directory.

By following these steps, you'll have the complete dataset needed to run the analysis. Each Jupyter notebook in the repository is named systematically to reflect whether it's for model training or analysis, the machine learning model used, and the metric focused on in the evaluation.

## Note

The `events.csv` file is not directly included due to its size. It's essential to unzip `archive.zip` following the instructions above to ensure the data is correctly placed for the scripts and notebooks to function as intended.
