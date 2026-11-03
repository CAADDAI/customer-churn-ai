import os
import sys

import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.pipeline.exception import CustomException
from src.pipeline.logger import logger

from src.components.data_transformation import DataTransformation


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

        logger.info(
            "Entered the Data Ingestion Method"
        )

        try:

            df = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",
                header=None
            )

            logger.info(
                "Dataset loaded successfully as DataFrame"
            )

            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logger.info(
                "Raw dataset saved successfully"
            )

            logger.info(
                "Train-test split initiated"
            )

            train_set, test_set = train_test_split(
                df,
                test_size=0.20,
                random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logger.info(
                "Train dataset saved successfully"
            )

            logger.info(
                "Test dataset saved successfully"
            )

            logger.info(
                "Data ingestion completed successfully"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    try:

        logger.info(
            "Application Started"
        )

        obj = DataIngestion()

        logger.info(
            "Data Ingestion Object Created"
        )

        train_data, test_data = (
            obj.initiate_data_ingestion()
        )

        logger.info(
            "Train and Test Paths Returned"
        )

        data_transformation = DataTransformation()

        logger.info(
            "Data Transformation Object Created"
        )

        train_arr, test_arr, _ = (
            data_transformation.initiate_data_transformation(
                train_data,
                test_data
            )
        )

        logger.info(
            "Data Transformation Completed"
        )

        print("Pipeline Executed Successfully")

    except Exception as e:
        logger.error(str(e))
        raise