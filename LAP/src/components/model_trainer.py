import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logger.info(
                "Splitting Training and Testing Arrays"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "AdaBoost": AdaBoostClassifier()
            }

            best_model_score = 0
            best_model_name = None
            best_model = None

            for model_name, model in models.items():

                logger.info(
                    f"Training {model_name}"
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                score = accuracy_score(
                    y_test,
                    y_pred
                )

                logger.info(
                    f"{model_name} Accuracy : {score}"
                )

                print(
                    f"{model_name} Accuracy : {score:.4f}"
                )

                if score > best_model_score:

                    best_model_score = score
                    best_model_name = model_name
                    best_model = model

            logger.info(
                f"Best Model : {best_model_name}"
            )

            logger.info(
                f"Best Accuracy : {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logger.info(
                "Best model saved successfully"
            )

            print(
                f"\nBest Model : {best_model_name}"
            )

            print(
                f"Best Accuracy : {best_model_score:.4f}"
            )

            return best_model_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    transformation = DataTransformation()

    train_arr, test_arr, _ = (
        transformation.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    trainer = ModelTrainer()

    trainer.initiate_model_trainer(
        train_arr,
        test_arr
    )