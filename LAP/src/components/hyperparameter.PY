import os
import sys

from dataclasses import dataclass

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

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
                "Starting Hyperparameter Tuning"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            model = GradientBoostingClassifier(
                random_state=42
            )

            param_grid = {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }

            logger.info(
                "Grid Search Started"
            )

            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=5,
                scoring="accuracy",
                n_jobs=-1,
                verbose=1
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
                f"Best CV Score : {best_score:.4f}"
            )

            save_object(
                file_path=self.hyperparameter_config.tuned_model_file_path,
                obj=best_model
            )

            logger.info(
                "Tuned Model Saved Successfully"
            )

            print("\n========== HYPERPARAMETER TUNING ==========")
            print("Best Parameters:")
            print(best_params)

            print(
                f"\nBest Cross Validation Score: {best_score:.4f}"
            )

            return (
                best_model,
                best_params,
                best_score
            )

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)


if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()

    train_path, test_path = (
        ingestion.initiate_data_ingestion()
    )

    transformer = DataTransformation()

    train_arr, test_arr, _ = (
        transformer.initiate_data_transformation(
            train_path,
            test_path
        )
    )

    tuner = ModelTuner()

    tuner.initiate_model_tuning(
        train_arr,
        test_arr
    )