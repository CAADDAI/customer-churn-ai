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

        logger.info("Predict Pipeline Initialized")

    def predict(self, features):

        try:

            logger.info("Loading model...")
            model = load_object(self.model_path)

            logger.info("Loading preprocessor...")
            preprocessor = load_object(
                self.preprocessor_path
            )

            logger.info("Transforming input data...")
            scaled_data = preprocessor.transform(
                features
            )

            logger.info("Making prediction...")
            prediction = model.predict(
                scaled_data
            )

            logger.info(
                "Generating probability score..."
            )

            probability = model.predict_proba(
                scaled_data
            )

            confidence = round(
                probability[0].max() * 100,
                2
            )

            logger.info(
                f"Prediction: {prediction[0]}"
            )

            logger.info(
                f"Confidence Score: {confidence}%"
            )

            logger.info(
                f"Probabilities: {probability[0]}"
            )

            return prediction, confidence

        except Exception as e:

            logger.error(
                f"Prediction Failed: {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )


class CustomData:

    def __init__(
        self,
        Age,
        Experience,
        Income,
        Family,
        CCAvg,
        Education,
        Mortgage,
        Securities_Account,
        CD_Account,
        Online,
        CreditCard
    ):

        self.Age = Age
        self.Experience = Experience
        self.Income = Income
        self.Family = Family
        self.CCAvg = CCAvg
        self.Education = Education
        self.Mortgage = Mortgage
        self.Securities_Account = Securities_Account
        self.CD_Account = CD_Account
        self.Online = Online
        self.CreditCard = CreditCard

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "Age": [self.Age],

                "Experience": [
                    self.Experience
                ],

                "Income": [
                    self.Income
                ],

                "Family": [
                    self.Family
                ],

                "CCAvg": [
                    self.CCAvg
                ],

                "Education": [
                    self.Education
                ],

                "Mortgage": [
                    self.Mortgage
                ],

                "Securities.Account": [
                    self.Securities_Account
                ],

                "CD.Account": [
                    self.CD_Account
                ],

                "Online": [
                    self.Online
                ],

                "CreditCard": [
                    self.CreditCard
                ]
            }

            df = pd.DataFrame(
                custom_data_input_dict
            )

            logger.info(
                "Input DataFrame Created Successfully"
            )

            logger.info(
                f"\n{df}"
            )

            return df

        except Exception as e:

            logger.error(
                f"DataFrame Creation Failed: {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    sample_data = CustomData(
        Age=35,
        Experience=10,
        Income=120,
        Family=2,
        CCAvg=2.5,
        Education=2,
        Mortgage=200,
        Securities_Account=1,
        CD_Account=1,
        Online=1,
        CreditCard=1
    )

    df = sample_data.get_data_as_dataframe()

    predictor = PredictPipeline()

    prediction, confidence = (
        predictor.predict(df)
    )

    print(
        f"\nPrediction: {prediction[0]}"
    )

    print(
        f"Confidence: {confidence}%"
    )

    if prediction[0] == 1:

        print(
            "✅ Loan Approved"
        )

    else:

        print(
            "❌ Loan Rejected"
        )