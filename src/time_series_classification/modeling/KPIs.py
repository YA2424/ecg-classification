from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics as m


def get_metrics(y_test: List[np.array, pd.Series, list], y_pred: np.array) -> Dict:
    metrics = {}
    metrics["accuracy"] = m.accuracy_score(y_test, y_pred)
    metrics["precision"] = m.precision_score(y_test, y_pred, average="weighted")
    metrics["recall"] = m.recall_score(y_test, y_pred, average="weighted")
    metrics["f1"] = m.f1_score(y_test, y_pred, average="weighted")
    metrics["balanced_accuracy"] = m.balanced_accuracy_score(y_test, y_pred)
    return metrics


def get_classification_report(
    y_test: List[np.array, pd.Series, list], y_pred: List[np.array, pd.Series, list]
):
    report = pd.DataFrame(m.classification_report(y_test, y_pred, output_dict=True)).T
    report.to_csv("reports/classification_report.csv")


def get_confusion_matrix_plot(
    y_test: List[np.array, pd.Series, list], y_pred: List[np.array, pd.Series, list]
):
    confusion_matrix = pd.DataFrame(m.confusion_matrix(y_test, y_pred))
    sns.heatmap(confusion_matrix, annot=True, fmt="d")
    plt.savefig("reports/confusion_matrix.png")
