import os
import sys

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            ".."
        )
    )
)

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.pipeline.logger import logging
from src.pipeline.exception import CustomException


class TrainPipeline:

    def __init__(self):
        pass

    def start_training_pipeline(self):

        try:

            logging.info(
                "Training Pipeline Started"
            )

            # Data Ingestion

            ingestion = DataIngestion()

            train_data_path, test_data_path = (
                ingestion.initiate_data_ingestion()
            )

            logging.info(
                "Data Ingestion Completed"
            )

            # Data Transformation

            data_transformation = (
                DataTransformation()
            )

            (
                train_arr,
                test_arr,
                preprocessor_path
            ) = (
                data_transformation
                .initiate_data_transformation(
                    train_data_path,
                    test_data_path
                )
            )

            logging.info(
                "Data Transformation Completed"
            )

            # Model Training

            model_trainer = ModelTrainer()

            model_score = (
                model_trainer
                .initiate_model_trainer(
                    train_arr,
                    test_arr,
                    preprocessor_path
                )
            )

            logging.info(
                "Model Training Completed"
            )

            logging.info(
                f"Best Model Score: {model_score}"
            )

            return model_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = TrainPipeline()

    pipeline.start_training_pipeline()

    print(
        "\nTraining Pipeline Completed Successfully!"
    )