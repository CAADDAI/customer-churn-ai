import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            numerical_columns = [
                "Age",
                "Experience",
                "Income",
                "Family",
                "CCAvg",
                "Mortgage",
                "Education",
                "Securities.Account",
                "CD.Account",
                "Online",
                "CreditCard"
            ]

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            logger.info(
                "Numerical Pipeline Completed"
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(
                "Train and Test Data Read Successfully"
            )

            target_column_name = "Personal.Loan"

            drop_columns = [
                "ID",
                "ZIP.Code",
                target_column_name
            ]

            # Training Features and Target
            input_feature_train_df = train_df.drop(
                columns=drop_columns
            )

            target_feature_train_df = train_df[
                target_column_name
            ]

            # Testing Features and Target
            input_feature_test_df = test_df.drop(
                columns=drop_columns
            )

            target_feature_test_df = test_df[
                target_column_name
            ]

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logger.info(
                "Preprocessor Saved Successfully"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()

    train_data, test_data = (
        ingestion.initiate_data_ingestion()
    )

    transformation = DataTransformation()

    train_arr, test_arr, preprocessor = (
        transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    print("Transformation Successful")
    print("Train Shape:", train_arr.shape)
    print("Test Shape :", test_arr.shape)
    print("Preprocessor:", preprocessor)