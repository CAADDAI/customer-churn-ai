import os
import sys
import pickle

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logger


def save_object(file_path, obj):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info(
            f"Object saved successfully at {file_path}"
        )

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):

    try:

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    X_train,
    y_train,
    X_test,
    y_test,
    models
):

    try:

        report = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            score = accuracy_score(
                y_test,
                y_pred
            )

            report[model_name] = score

        return report

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    logger.info("Utils file is working")
    print("Utils module loaded successfully")
