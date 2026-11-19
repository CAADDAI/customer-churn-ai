
import os
import sys

from dataclasses import dataclass

from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    ParameterGrid
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from src.pipeline.exception import CustomException
from src.pipeline.logger import logger
from src.pipeline.utils import save_object


@dataclass
class ModelTuningConfig:

    tuned_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTuner:

    def __init__(self):

        self.model_tuning_config = ModelTuningConfig()

    def initiate_model_tuning(
        self,
        train_array,
        test_array,
        best_model_name
    ):

        try:

            logger.info("Starting Hyperparameter Tuning")

            # Separate training and testing data

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            # Define models and hyperparameters

            models = {

                "Logistic Regression": {

                    "model": LogisticRegression(
                        max_iter=1000,
                        random_state=42
                    ),

                    "params": {

                        "C": [0.01, 0.1, 1, 10, 100],

                        "solver": ["liblinear"],

                        "class_weight": [None, "balanced"]

                    }
                },

                "Decision Tree": {

                    "model": DecisionTreeClassifier(
                        random_state=42
                    ),

                    "params": {

                        "max_depth": [3, 5, 10, 15, 20, None],

                        "min_samples_split": [2, 5, 10, 20],

                        "min_samples_leaf": [1, 2, 4, 8],

                        "class_weight": [None, "balanced"]

                    }
                },

                "Random Forest": {

                    "model": RandomForestClassifier(
                        random_state=42,
                        n_jobs=1
                    ),

                    "params": {

                        "n_estimators": [100, 200, 300, 500],

                        "max_depth": [5, 10, 15, 20, None],

                        "min_samples_split": [2, 5, 10],

                        "min_samples_leaf": [1, 2, 4],

                        "class_weight": [None, "balanced"]

                    }
                },

                "KNN": {

                    "model": KNeighborsClassifier(),

                    "params": {

                        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],

                        "weights": ["uniform", "distance"],

                        "p": [1, 2]

                    }
                }

            }

            # Check selected model

            if best_model_name not in models:

                raise ValueError(
                    f"Model {best_model_name} not found"
                )

            model_config = models[best_model_name]

            model = model_config["model"]

            params = model_config["params"]

            logger.info(
                f"Tuning started for {best_model_name}"
            )

            # Stratified cross-validation

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            # Calculate available parameter combinations

            total_combinations = len(
                list(ParameterGrid(params))
            )

            n_iter = min(20, total_combinations)

            logger.info(
                f"Randomized Search Iterations: {n_iter}"
            )

            # Randomized Search CV

            random_search = RandomizedSearchCV(

                estimator=model,

                param_distributions=params,

                n_iter=n_iter,

                scoring="f1",

                cv=cv,

                verbose=1,

                random_state=42,

                n_jobs=-1,

                refit=True,

                error_score="raise"

            )

            # Fit tuning model

            logger.info("Hyperparameter Search Started")

            random_search.fit(
                X_train,
                y_train
            )

            best_model = random_search.best_estimator_

            best_params = random_search.best_params_

            best_cv_score = random_search.best_score_

            logger.info(
                f"Best Parameters: {best_params}"
            )

            logger.info(
                f"Best CV F1 Score: {best_cv_score:.4f}"
            )

            # Evaluate tuned model on test data

            y_pred = best_model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            logger.info(
                f"Test Accuracy: {accuracy:.4f}"
            )

            logger.info(
                f"Test Precision: {precision:.4f}"
            )

            logger.info(
                f"Test Recall: {recall:.4f}"
            )

            logger.info(
                f"Test F1 Score: {f1:.4f}"
            )

            logger.info(
                "Classification Report:\n%s",
                classification_report(
                    y_test,
                    y_pred,
                    zero_division=0
                )
            )

            # Save tuned model

            save_object(

                file_path=self.model_tuning_config.tuned_model_file_path,

                obj=best_model

            )

            logger.info(
                "Tuned Model Saved Successfully"
            )

            return (

                best_model_name,

                best_params,

                f1

            )

        except Exception as e:

            logger.error(str(e))

            raise CustomException(e, sys)
