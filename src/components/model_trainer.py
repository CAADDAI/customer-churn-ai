import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
                "Splitting train and test arrays"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Logistic Regression":
                LogisticRegression(
                    class_weight="balanced",
                    random_state=42,
                    max_iter=1000
                ),

                "Decision Tree":
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=42
                ),

                "Random Forest":
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42
                ),

                "KNeighborsClassifier":
                KNeighborsClassifier(
                    n_neighbors=5
                ),

                "Gaussian Naive Bayes":
                GaussianNB()
            }

            best_model = None
            best_model_name = None
            best_f1_score = 0

            for model_name, model in models.items():

                logger.info(
                    f"Training {model_name}"
                )

                model.fit(
                    X_train,
                    y_train
                )

                y_pred = model.predict(
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

                print("\n" + "=" * 60)
                print(f"Model : {model_name}")
                print("=" * 60)
                print(f"Accuracy : {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall   : {recall:.4f}")
                print(f"F1 Score : {f1:.4f}")
                print(f"ROC AUC  : {roc_auc:.4f}")

                logger.info(
                    f"{model_name} | "
                    f"Accuracy={accuracy:.4f}, "
                    f"Precision={precision:.4f}, "
                    f"Recall={recall:.4f}, "
                    f"F1={f1:.4f}, "
                    f"ROC_AUC={roc_auc:.4f}"
                )

                if f1 > best_f1_score:

                    best_f1_score = f1
                    best_model_name = model_name
                    best_model = model

            logger.info(
                f"Best Model: {best_model_name}"
            )

            logger.info(
                f"Best F1 Score: {best_f1_score:.4f}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logger.info(
                "Best Model Saved Successfully"
            )

            print("\n" + "=" * 60)
            print(f"BEST MODEL : {best_model_name}")
            print(f"BEST F1 SCORE : {best_f1_score:.4f}")
            print("=" * 60)

            # -----------------------------------
            # Confusion Matrix
            # -----------------------------------

            y_pred_best = best_model.predict(
                X_test
            )

            cm = confusion_matrix(
                y_test,
                y_pred_best
            )

            print("\nCONFUSION MATRIX")
            print("=" * 60)
            print(cm)

            # -----------------------------------
            # Classification Report
            # -----------------------------------

            print("\nCLASSIFICATION REPORT")
            print("=" * 60)

            print(
                classification_report(
                    y_test,
                    y_pred_best
                )
            )

            return best_f1_score

        except Exception as e:
            raise CustomException(e, sys)