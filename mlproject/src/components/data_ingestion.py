
import os
import sys
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.pipeline.exception import CustomException
from src.pipeline.logger import logger

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.hyperparameter import ModelTuner


@dataclass
class DataIngestionConfig:

    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "artifacts",
        "data.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logger.info("Entered Data Ingestion Method")

        try:

            # Load dataset
            df = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",
                header=None
            )

            logger.info("Dataset loaded successfully")

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            # Save raw dataset
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logger.info("Raw Dataset Saved")

            # Split dataset into training and testing
            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42
            )

            logger.info("Train Test Split Completed")

            # Save training dataset
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save testing dataset
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info("Train and Test Files Saved")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)


# MAIN EXECUTION PIPELINE

if __name__ == "__main__":

    logger.info("Application Started")

    # STEP 1: DATA INGESTION

    obj = DataIngestion()

    logger.info("Data Ingestion Object Created")

    train_data, test_data = (
        obj.initiate_data_ingestion()
    )

    logger.info("Data Ingestion Completed")

    print("Data Ingestion Completed Successfully")

    # STEP 2: DATA TRANSFORMATION

    data_transformation = DataTransformation()

    logger.info("Data Transformation Object Created")

    train_arr, test_arr, preprocessor_path = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    logger.info("Data Transformation Completed")

    print("Data Transformation Completed Successfully")

    print("Training Data Shape:", train_arr.shape)
    print("Testing Data Shape:", test_arr.shape)

    # STEP 3: MODEL TRAINING

    model_trainer = ModelTrainer()

    logger.info("Model Trainer Object Created")

    print("\nStarting Model Training...")

    best_model_name, best_model_score = (
        model_trainer.initiate_model_trainer(
            train_arr,
            test_arr,
            preprocessor_path
        )
    )

    logger.info("Model Training Completed")

    print("\nModel Training Completed Successfully")

    print("Best Model:", best_model_name)

    print(
        "Best Model CV F1 Score:",
        round(best_model_score, 4)
    )

    # STEP 4: HYPERPARAMETER TUNING

    model_tuner = ModelTuner()

    logger.info("Model Tuner Object Created")

    print("\nStarting Hyperparameter Tuning...")

    tuned_model_name, best_params, tuned_test_f1 = (
        model_tuner.inistiate_model_tuning(
            train_arr,
            test_arr,
            best_model_name
        )
    )

    logger.info("Hyperparameter Tuning Completed")

    # STEP 5: DISPLAY TUNING RESULTS

    print("\nHyperparameter Tuning Completed Successfully")

    print("Tuned Model:", tuned_model_name)

    print("Best Hyperparameters:", best_params)

    print(
        "Tuned Model Test F1 Score:",
        round(tuned_test_f1, 4)
    )

    logger.info("Entire ML Pipeline Executed Successfully")

    print("\nEntire ML Pipeline Executed Successfully")
