import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class HyperparameterConfig:
    tuned_model_file_path = os.path.join(
        "artifacts",
        "tuned_model.pkl"
    )


class ModelTuner:

    def __init__(self):
        self.hyperparameter_config = HyperparameterConfig()

    def initiate_model_tuning(
        self,
        train_array,
        test_array
    ):

        try:

            logger.info(
                "Starting Random Forest Hyperparameter Tuning"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            model = RandomForestClassifier(
                class_weight="balanced",
                random_state=42
            )

            param_grid = {

                "n_estimators": [
                    100,
                    200,
                    300
                ],

                "max_depth": [
                    5,
                    10,
                    20,
                    None
                ],

                "min_samples_split": [
                    2,
                    5,
                    10
                ],

                "min_samples_leaf": [
                    1,
                    2,
                    4
                ]
            }

            logger.info(
                "Starting GridSearchCV"
            )

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring="f1",
                n_jobs=-1,
                verbose=2
            )

            grid_search.fit(
                X_train,
                y_train
            )

            best_model = (
                grid_search.best_estimator_
            )

            best_params = (
                grid_search.best_params_
            )

            best_score = (
                grid_search.best_score_
            )

            logger.info(
                f"Best Parameters : {best_params}"
            )

            logger.info(
                f"Best Cross Validation F1 Score : {best_score:.4f}"
            )

            # Test Evaluation

            y_pred = best_model.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred
            )

            recall = recall_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred
            )

            roc_auc = roc_auc_score(
                y_test,
                y_pred
            )

            cm = confusion_matrix(
                y_test,
                y_pred
            )

            save_object(
                file_path=self.hyperparameter_config.tuned_model_file_path,
                obj=best_model
            )

            logger.info(
                "Tuned Model Saved Successfully"
            )

            print("\n" + "=" * 60)
            print("HYPERPARAMETER TUNING RESULTS")
            print("=" * 60)

            print("\nBest Parameters:")
            print(best_params)

            print(
                f"\nBest Cross Validation F1 Score: {best_score:.4f}"
            )

            print("\n" + "=" * 60)
            print("TEST SET RESULTS")
            print("=" * 60)

            print(f"Accuracy : {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall   : {recall:.4f}")
            print(f"F1 Score : {f1:.4f}")
            print(f"ROC AUC  : {roc_auc:.4f}")

            print("\nCONFUSION MATRIX")
            print("=" * 60)
            print(cm)

            print("\nCLASSIFICATION REPORT")
            print("=" * 60)

            print(
                classification_report(
                    y_test,
                    y_pred
                )
            )

            return (
                best_model,
                best_params,
                best_score
            )

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)
