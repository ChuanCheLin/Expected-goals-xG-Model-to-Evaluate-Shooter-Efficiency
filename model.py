import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import time

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
file_handler = logging.FileHandler('log.txt')
stream_handler = logging.StreamHandler()

# Set level and format for handlers
file_handler.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add handlers to the logger
# Check if handlers are already present to avoid duplicates
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def train_model(X_train, y_train, model=LogisticRegression(max_iter=1000), parameters=None, cv=5):
    """
    Trains a machine learning model using GridSearchCV for hyperparameter tuning and logs the duration of the training.
    
    Parameters:
    - X_train: Feature matrix for training data.
    - y_train: Target vector for training data.
    - model: The machine learning model to train. Default is LogisticRegression.
    - parameters: Dictionary of parameters for GridSearchCV. Default is None.
    - cv: Number of cross-validation folds for GridSearchCV. Default is 5.
    
    Returns:
    - best_model: The best model from GridSearchCV.
    """
    if parameters is None:
        # default params for Logistic Regression
        parameters = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'saga']}
    
    logger.info("Starting model training.")
    start_time = time.time()
    
    grid_search = GridSearchCV(model, parameters, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    end_time = time.time()
    training_time = end_time - start_time
    best_model = grid_search.best_estimator_
    
    logger.info(f"Model training completed in {training_time:.2f} seconds.")
    logger.info(f"Best parameters: {grid_search.best_params_}. Best score: {grid_search.best_score_}.")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained machine learning model on the test set.

    Parameters:
    - model: The trained machine learning model.
    - X_test: Feature matrix for testing data.
    - y_test: Target vector for testing data.

    Returns:
    - None
    """
    # Get model predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    
    # Generate a classification report
    report = classification_report(y_test, predictions)
    
    # Calculate precision-recall pairs for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    
    # Calculate the AUPRC
    auprc = auc(recall, precision)
    
    # Calculate the ROC AUC
    roc_auc = roc_auc_score(y_test, probabilities)
    
    # Logging the evaluation metrics
    logger.info(f"Model evaluation completed. Accuracy: {accuracy}.")
    logger.info(f"Classification Report: \n{report}")
    logger.info(f"Area Under the Precision-Recall Curve (AUPRC): {auprc:.4f}")
    logger.info(f"Receiver Operating Characteristic Area Under the Curve (ROC AUC): {roc_auc:.4f}")

def save_model(model, filename='trained_model.pkl'):
    """
    Saves the trained model to a file.
    
    Parameters:
    - model: The trained machine learning model.
    - filename: Name of the file to save the model. Default is 'trained_model.pkl'.
    
    Returns:
    - None
    """
    joblib.dump(model, filename)
    logger.info(f"Model saved to {filename}.")

def load_model(filename='trained_model.pkl'):
    """
    Loads a trained machine learning model from a file.
    
    Parameters:
    - filename: Name of the file from which to load the model. Default is 'trained_model.pkl'.
    
    Returns:
    - model: The loaded machine learning model.
    """
    model = joblib.load(filename)
    logger.info(f"Model loaded from {filename}.")
    return model
