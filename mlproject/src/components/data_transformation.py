import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.pipeline.exception import CustomException
from src.pipeline.logger import logger
from src.pipeline.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = (
            DataTransformationConfig()
        )

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                1,
                2,
                7,
                10,
                13,
                14
            ]

            categorical_columns = [
                0,
                3,
                4,
                5,
                6,
                8,
                9,
                11,
                12
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

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            logger.info(
                "Preprocessor object created successfully"
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

            logger.info(
                "Data Transformation Started"
            )

            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)

            print("Train Columns:", train_df.columns)

            logger.info(
                f"Train Columns: {list(train_df.columns)}"
            )

            logger.info(
                "Train and test datasets loaded successfully"
            )

            train_df.replace(
                "?",
                np.nan,
                inplace=True
            )

            test_df.replace(
                "?",
                np.nan,
                inplace=True
            )

            logger.info(
                "Missing values replaced successfully"
            )


            target_column_name = "15"

            X_train = train_df.drop(
                columns=[target_column_name]
            )

            y_train = train_df[
                target_column_name
            ]

            X_test = test_df.drop(
                columns=[target_column_name]
            )

            y_test = test_df[
                target_column_name
            ]

            y_train = y_train.map(
                {
                    "+": 1,
                    "-": 0
                }
            )

            y_test = y_test.map(
                {
                    "+": 1,
                    "-": 0
                }
            )

            logger.info(
                "Target variable transformed successfully"
            )

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            logger.info(
                "Applying preprocessing object"
            )

            X_train_arr = (
                preprocessing_obj.fit_transform(
                    X_train
                )
            )

            X_test_arr = (
                preprocessing_obj.transform(
                    X_test
                )
            )

            logger.info(
                "Transformation completed successfully"
            )

            train_arr = np.c_[
                X_train_arr,
                np.array(y_train)
            ]

            test_arr = np.c_[
                X_test_arr,
                np.array(y_test)
            ]

            logger.info(
                "Saving Preprocessor Object"
            )

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logger.info(
                "Preprocessor Object Saved Successfully"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)
        