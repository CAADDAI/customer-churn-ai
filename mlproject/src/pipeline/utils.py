
import os
import sys
import joblib

from src.pipeline.exception import CustomException

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        return joblib.load(file_path)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(
    x_train,
    y_train,
    x_test,
    y_test,
    models
):
    try:
        report = {}

        for model_name, model in models.items():

            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)
            train_accuracy = accuracy_score(
                y_train, y_train_pred
            )

            test_accuracy = accuracy_score(
                y_test, y_test_pred
            )

            test_precision = precision_score(
                y_test,
                y_test_pred,
                zero_division=0
            )

            test_recall = recall_score(
                y_test,
                y_test_pred,
                zero_division=0
            )

            test_f1 = f1_score(
                y_test,
                y_test_pred,
                zero_division=0
            )

            report[model_name] = {
                "Train Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Precision": test_precision,
                "Recall": test_recall,
                "F1 Score": test_f1
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
