from __future__ import annotations
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from numpy.typing import ArrayLike
from joblib import dump


def train_regression_model(X_train: ArrayLike, y_train: ArrayLike) -> LinearRegression:
    """
    Train a regression model using the provided training data.

    This function takes training data consisting of a feature matrix 'X_train' and
    corresponding target values 'y_train', and trains a linear regression model.
    The trained model is returned for further use.

    Args:
        X_train (array-like): Training feature matrix.
        y_train (array-like): Target values for training.

    Returns:
        sklearn.linear_model.LinearRegression: Trained linear regression model.

    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def save_regression_model(model: LinearRegression, filename: str = "linear_regression_model.joblib"):
    """
    Serialize and save the regression model.

    This function takes a trained regression 'model' and file name 'filename' that has a default value.
    It uses Joblib 'dump' to save the model using the provided file name.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model to be evaluated.
        filename (str): Name of the file that is used to store the model.

    """
    
    dump(model, filename)

def evaluate_regression_model(model: LinearRegression, X_test: ArrayLike, y_test: ArrayLike):
    """
    Evaluate the performance of a regression model on test data.

    This function takes a trained regression 'model', test feature matrix 'X_test',
    and corresponding test target values 'y_test'. It calculates Mean Squared Error (MSE)
    and prints it in terminal.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained regression model to be evaluated.
        X_test (array-like): Test feature matrix.
        y_test (array-like): Validation target values.

    """
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")

def save_initial_datasets(X: ArrayLike, y: ArrayLike):
    """
    Serialize and save datasets.

    This function takes entire feature matrix 'X', and corresponding target values 'y'.
    It uses Joblib 'dump' to save both arrays as predefined files.

    Args:
        X (array-like): Test feature matrix.
        y (array-like): Validation target values.

    """
    X_filename = "X.joblib"
    y_filename = "y.joblib"
    
    dump(X, X_filename)
    dump(y, y_filename)

def save_model_metrics(metrics: dict, filename: str = "metrics.joblib"):
    """
    Serialize and save model evaluation metrics for use in the Streamlit app.

    Args:
        metrics (dict): Dictionary with model names as keys and metric details as values.
        filename (str): Name of the file used to store the metrics.
    """
    dump(metrics, filename)

def _build_model_registry() -> dict[str, tuple[str, object]]:
    """Return configured models with their output filenames."""
    return {
        "Linear Regression": ("linear_regression_model.joblib", LinearRegression()),
        "Ridge Regression": ("ridge_model.joblib", Ridge()),
        "Polynomial (degree 2)": (
            "polynomial_model.joblib",
            Pipeline([
                ("poly", PolynomialFeatures(degree=2)),
                ("lr", LinearRegression()),
            ]),
        ),
    }

def _extract_model_parameters(model: object, model_name: str) -> dict:
    """Extract coefficient information for display in the Streamlit sidebar."""
    if model_name == "Polynomial (degree 2)":
        linear_model = model.named_steps["lr"]
        return {
            "coefficients": linear_model.coef_.tolist(),
            "intercept": float(linear_model.intercept_),
        }

    return {
        "coefficient": float(model.coef_[0]),
        "intercept": float(model.intercept_),
    }

def train_and_evaluate_models(
    X_train: ArrayLike,
    X_test: ArrayLike,
    y_train: ArrayLike,
    y_test: ArrayLike,
) -> dict:
    """
    Train multiple regression models, evaluate them, and save each trained model.

    Returns:
        dict: Metrics for each trained model.
    """
    metrics = {}

    for model_name, (filename, model) in _build_model_registry().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        model_metrics = {
            "mse": float(mean_squared_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
            "filename": filename,
        }
        model_metrics.update(_extract_model_parameters(model, model_name))

        save_regression_model(model, filename)
        metrics[model_name] = model_metrics

        print(f"{model_name} - MSE: {model_metrics['mse']:.2f}, R²: {model_metrics['r2']:.4f}")

    return metrics

if __name__ == '__main__':
    # Generate a dataset
    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

    # Scale feature X to range -3..3
    X = np.interp(X, (X.min(), X.max()), (-3, 3))

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = train_regression_model(X_train, y_train)

    # Evaluate the model
    evaluate_regression_model(model, X_test, y_test)

    # Train, compare, and save multiple models with metrics
    metrics = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    save_model_metrics(metrics)

    # Save datasets
    save_initial_datasets(X, y)