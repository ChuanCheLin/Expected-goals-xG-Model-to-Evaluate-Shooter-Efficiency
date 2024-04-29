import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import numpy as np
import logging

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers and set level
file_handler = logging.FileHandler('preprocessing_log.txt')
stream_handler = logging.StreamHandler()
file_handler.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)

# Create a formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Check if handlers are already present to avoid duplicates
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

def load_and_merge_data(events_file_path: str, ginf_file_path: str) -> pd.DataFrame:
    """
    Loads the event and game information datasets and merges them on the unique game identifier.
    
    Parameters:
    - events_file_path: str, the path to the events.csv file.
    - ginf_file_path: str, the path to the ginf.csv file.
    
    Returns:
    - pd.DataFrame: The merged dataset containing both event and game information.
    """
    logger.info("Starting to load datasets.")
    
    # Load datasets
    events_df = pd.read_csv(events_file_path)
    logger.info(f"Loaded events data from {events_file_path}.")
    
    ginf_df = pd.read_csv(ginf_file_path)
    logger.info(f"Loaded game information data from {ginf_file_path}.")
    
    # Merge datasets on unique game identifier
    full_df = pd.merge(events_df, ginf_df, on='id_odsp')
    logger.info("Merged events and game information datasets.")
    
    return full_df

def preprocess_data(df: pd.DataFrame, scale: bool = False) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """
    Preprocesses the football events data for the xG model, applying meaningful naming for one-hot encoding and optional scaling.
    
    Parameters:
    - df: pd.DataFrame, the merged dataset containing both event and game information.
    - scale: bool, whether to apply feature scaling. Default is False.
    
    Returns:
    - Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]: The preprocessed and optionally scaled training and testing feature matrices and target vectors.
    """
    logger.info("Starting preprocessing of data.")
    
    # Remove duplicates
    df = df.drop_duplicates()
    logger.info("Removed duplicate records.")
    
    # Fill missing values for categorical data
    categorical_features = ['location', 'bodypart', 'assist_method', 'situation', 'fast_break']
    for column in categorical_features:
        df[column] = df[column].fillna(df[column].mode()[0])
    logger.info("Filled missing values for categorical data.")
    
    # Mapping dictionaries for categorical features
    location_mapping = {
        1: "Attacking_half", 2: "Defensive_half", 3: "Centre_of_the_box",
        4: "Left_wing", 5: "Right_wing", 6: "Difficult_angle_and_long_range",
        7: "Difficult_angle_on_the_left", 8: "Difficult_angle_on_the_right",
        9: "Left_side_of_the_box", 10: "Left_side_of_the_six_yard_box",
        11: "Right_side_of_the_box", 12: "Right_side_of_the_six_yard_box",
        13: "Very_close_range", 14: "Penalty_spot", 15: "Outside_the_box",
        16: "Long_range", 17: "More_than_35_yards", 18: "More_than_40_yards",
        19: "Not_recorded"
    }
    bodypart_mapping = {1: "right_foot", 2: "left_foot", 3: "head"}
    assist_method_mapping = {0: "None", 1: "Pass", 2: "Cross", 3: "Headed_pass", 4: "Through_ball"}
    situation_mapping = {1: "Open_play", 2: "Set_piece", 3: "Corner", 4: "Free_kick"}
    
    # Replace categorical features with mapped values before one-hot encoding
    df.replace({"location": location_mapping, "bodypart": bodypart_mapping, 
                "assist_method": assist_method_mapping, "situation": situation_mapping}, inplace=True)
    
    # Select features and target
    features = categorical_features
    target = 'is_goal'
    X = pd.get_dummies(df[features], prefix_sep='_')
    y = df[target]
    logger.info("Selected features and target. Applied meaningful one-hot encoding.")
    
    # Splitting the data into training and testing sets with stratification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    logger.info("Split data into training and testing sets with similar distribution for the target variable.")
    
    # Scaling features if scale parameter is True
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Applied feature scaling.")
        return X_train_scaled, X_test_scaled, y_train, y_test
    else:
        logger.info("Returned data without scaling.")
        return X_train, X_test, y_train, y_test
