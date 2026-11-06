import os
import sys

import pandas as pd

from src.pipeline.exception import CustomException
from src.pipeline.utils import load_object


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        try:

            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            model = load_object(
                file_path=model_path
            )

            preprocessor = load_object(
                file_path=preprocessor_path
            )

            data_scaled = preprocessor.transform(
                features
            )

            prediction = model.predict(
                data_scaled
            )

            return prediction

        except Exception as e:

            raise CustomException(
                e,
                sys
            )


class CustomData:

    def __init__(
        self,
        A1,
        A2,
        A3,
        A4,
        A5,
        A6,
        A7,
        A8,
        A9,
        A10,
        A11,
        A12,
        A13,
        A14,
        A15
    ):

        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A4 = A4
        self.A5 = A5
        self.A6 = A6
        self.A7 = A7
        self.A8 = A8
        self.A9 = A9
        self.A10 = A10
        self.A11 = A11
        self.A12 = A12
        self.A13 = A13
        self.A14 = A14
        self.A15 = A15

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {

                "0": [self.A1],
                "1": [float(self.A2)],
                "2": [float(self.A3)],
                "3": [self.A4],
                "4": [self.A5],
                "5": [self.A6],
                "6": [self.A7],
                "7": [float(self.A8)],
                "8": [self.A9],
                "9": [self.A10],
                "10": [float(self.A11)],
                "11": [self.A12],
                "12": [self.A13],
                "13": [float(self.A14)],
                "14": [float(self.A15)]

            }

            df = pd.DataFrame(
                custom_data_input_dict
            )

            return df

        except Exception as e:

            raise CustomException(
                e,
                sys
            )