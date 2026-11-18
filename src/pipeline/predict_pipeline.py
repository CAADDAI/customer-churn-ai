import os
import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_object


class PredictPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "artifacts",
            "tuned_model.pkl"
        )

        self.preprocessor_path = os.path.join(
            "artifacts",
            "preprocessor.pkl"
        )

        logger.info(
            "Predict Pipeline Initialized"
        )

    def predict(self, features):

        try:

            logger.info(
                "Loading model and preprocessor"
            )

            model = load_object(
                self.model_path
            )

            preprocessor = load_object(
                self.preprocessor_path
            )

            logger.info(
                "Transforming input features"
            )

            transformed_features = (
                preprocessor.transform(
                    features
                )
            )

            logger.info(
                "Generating prediction"
            )

            prediction = model.predict(
                transformed_features
            )

            prediction_probability = (
                model.predict_proba(
                    transformed_features
                )
            )

            confidence = round(
                prediction_probability.max() * 100,
                2
            )

            logger.info(
                f"Prediction: {prediction[0]}"
            )

            logger.info(
                f"Confidence: {confidence}%"
            )

            return (
                int(prediction[0]),
                confidence
            )

        except Exception as e:

            logger.error(
                f"Prediction Error: {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )


class CustomData:

    def __init__(
        self,
        CreditScore,
        Geography,
        Gender,
        Age,
        Tenure,
        Balance,
        NumOfProducts,
        HasCrCard,
        IsActiveMember,
        EstimatedSalary
    ):

        self.CreditScore = CreditScore
        self.Geography = Geography
        self.Gender = Gender
        self.Age = Age
        self.Tenure = Tenure
        self.Balance = Balance
        self.NumOfProducts = NumOfProducts
        self.HasCrCard = HasCrCard
        self.IsActiveMember = IsActiveMember
        self.EstimatedSalary = EstimatedSalary

    def get_data_as_dataframe(self):

        try:

            custom_data_dict = {

                "CreditScore":
                [self.CreditScore],

                "Geography":
                [self.Geography],

                "Gender":
                [self.Gender],

                "Age":
                [self.Age],

                "Tenure":
                [self.Tenure],

                "Balance":
                [self.Balance],

                "NumOfProducts":
                [self.NumOfProducts],

                "HasCrCard":
                [self.HasCrCard],

                "IsActiveMember":
                [self.IsActiveMember],

                "EstimatedSalary":
                [self.EstimatedSalary]

            }

            data_frame = pd.DataFrame(
                custom_data_dict
            )

            logger.info(
                "Input DataFrame Created Successfully"
            )

            return data_frame

        except Exception as e:

            logger.error(
                f"DataFrame Creation Failed: {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )