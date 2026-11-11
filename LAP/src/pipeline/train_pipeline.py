from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.hyperparameter import ModelTuner

from src.logger import logger
from src.exception import CustomException


class TrainPipeline:

    def __init__(self):
        pass

    def start_training_pipeline(self):

        logger.info(
            "Training Pipeline Started"
        )

        # Data Ingestion
        ingestion = DataIngestion()

        train_path, test_path = (
            ingestion.initiate_data_ingestion()
        )

        logger.info(
            "Data Ingestion Completed"
        )

        # Data Transformation
        transformation = DataTransformation()

        train_arr, test_arr, _ = (
            transformation.initiate_data_transformation(
                train_path,
                test_path
            )
        )

        logger.info(
            "Data Transformation Completed"
        )

        # Model Training
        trainer = ModelTrainer()

        trainer.initiate_model_trainer(
            train_arr,
            test_arr
        )

        logger.info(
            "Model Training Completed"
        )

        # Hyperparameter Tuning
        tuner = ModelTuner()

        tuner.initiate_model_tuning(
            train_arr,
            test_arr
        )

        logger.info(
            "Model Tuning Completed"
        )

        print(
            "\nLoan Approval Training Pipeline Completed Successfully"
        )


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.start_training_pipeline()