import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)


def compute_all_metrics(y_true, y_pred):
    metrics = {}
    metrics["mse"] = mean_squared_error(y_true, y_pred)
    metrics["rmse"] = np.sqrt(metrics["mse"])
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["mape"] = mean_absolute_percentage_error(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)
    return metrics


def compute_multitask_metrics(y_true, y_pred, mean, std):
    y_true = y_true * std + mean
    y_pred = y_pred * std + mean
    y_true, y_pred = y_true.T, y_pred.T

    names = ["Carbon", "Population", "Light"]
    metrics = {}
    for i, name in enumerate(names):
        m = compute_all_metrics(y_true[i], y_pred[i])
        for k, v in m.items():
            metrics[f"{k}_{name}"] = v
    return metrics