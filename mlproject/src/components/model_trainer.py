
import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import (
    cross_val_score,
    StratifiedKFold
)

from src.pipeline.exception import CustomException
from src.pipeline.logger import logger
from src.pipeline.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        preprocessor_path
    ):

        try:

            logger.info("Entered Model Trainer")

            logger.info(
                "Splitting training and testing input data"
            )

            # Separate features and target

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define machine learning models

            models = {

                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),

                "Decision Tree": DecisionTreeClassifier(
                    random_state=42
                ),

                "Random Forest": RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

                "KNN": KNeighborsClassifier(
                    n_neighbors=5
                )

            }

            # Cross-validation

            logger.info(
                "Model cross-validation started"
            )

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            model_report = {}

            for model_name, model in models.items():

                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="f1"
                )

                model_report[model_name] = scores.mean()

                logger.info(
                    f"{model_name} CV F1 Score: {scores.mean():.4f}"
                )

            # Identify the best model

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]

            best_model = models[best_model_name]

            logger.info(
                f"Best Model Found: {best_model_name}"
            )

            logger.info(
                f"Best Model CV F1 Score: {best_model_score:.4f}"
            )

            # Evaluate the selected model

            evaluation_report = evaluate_models(
                X_train,
                y_train,
                X_test,
                y_test,
                {
                    best_model_name: best_model
                }
            )

            logger.info(
                f"Evaluation Report: {evaluation_report}"
            )

            # Save the trained model
            # evaluate_models() has already fitted the model.

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logger.info(
                "Best Model Saved Successfully"
            )

            return (
                best_model_name,
                best_model_score
            )

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)
