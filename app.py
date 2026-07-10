from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import pandas as pd

@st.cache_resource
def load_model(filename: str):
    """Load and cache a serialized regression model."""
    return load(filename)

@st.cache_data
def load_metrics() -> dict:
    """Load and cache model evaluation metrics."""
    return load("metrics.joblib")

@st.cache_data
def load_datasets() -> tuple[np.ndarray, np.ndarray]:
    """Load and cache the original feature and target datasets."""
    return load("X.joblib"), load("y.joblib")

def load_and_predict(X: ArrayLike, filename: str = "linear_regression_model.joblib") -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.

    This function takes a file name 'filename' that has a default value.
    It uses Joblib 'load' to load the model using the provided file name.
    When the model is loaded, call its `predict` method on provied data.

    Args:
        X (array-like): User provided data used for prediction.
        filename (str): Name of the file that is used to store the model.

    Returns:
        np.ndarray: Predicted value.
    """
    model = load_model(filename)
    y = model.predict(X)

    return y

def _render_metrics_sidebar(metrics: dict) -> tuple[str, dict]:
    """Render model selection and metrics panel in the sidebar."""
    st.sidebar.header("Model Metrics")

    model_names = list(metrics.keys())
    selected_model = st.sidebar.selectbox("Select model", model_names)
    model_info = metrics[selected_model]

    st.sidebar.metric("MSE", f"{model_info['mse']:.2f}")
    st.sidebar.metric("R²", f"{model_info['r2']:.4f}")

    if "coefficient" in model_info:
        st.sidebar.write(f"**Coefficient:** {model_info['coefficient']:.4f}")
    else:
        st.sidebar.write(f"**Coefficients:** {model_info['coefficients']}")

    st.sidebar.write(f"**Intercept:** {model_info['intercept']:.4f}")

    with st.sidebar.expander("Compare all models"):
        comparison_df = pd.DataFrame([
            {
                "Model": name,
                "MSE": info["mse"],
                "R²": info["r2"],
            }
            for name, info in metrics.items()
        ]).sort_values("MSE")
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)

    return selected_model, model_info

def create_streamlit_app():
    """
    Creates a Streamlit web application for making predictions with a simple regression model.

    This function sets up a Streamlit app with a user interface for inputting a single feature 
    value and making predictions using a pre-trained regression model. The app includes:
    
    - A title displayed at the top of the app.
    - A slider for the user to select an input feature value within a specified range (-3.0 to 3.0).
    - Live predictions that update when the slider value changes.
    - A sidebar with model metrics and model comparison.
    - A chart that compares predicted and actual values and shows the regression line.

    Note: This function does not return any value. It directly manipulates the Streamlit app's UI by 
    writing content and rendering UI elements.
    """
    st.title("Regression Model Prediction")

    metrics = load_metrics()
    selected_model, model_info = _render_metrics_sidebar(metrics)

    st.subheader(f"Prediction: {selected_model}")
    input_feature = st.slider("Select a feature value", -3.0, 3.0, 0.0)
    X, y = load_datasets()
    actual_value = y[_index_of_closest(X, input_feature)]
    prediction = load_and_predict([[input_feature]], model_info["filename"])
    st.write(f"Predicted value: {prediction[0]:.2f}")
    st.write(f"Actual value: {actual_value:.2f}")
    visualize_difference(input_feature, prediction[0], model_info["filename"], selected_model)

def visualize_difference(
    input_feature: float,
    prediction: ArrayLike,
    model_filename: str,
    model_name: str,
):
    """
    Deserialize and load the initial datasets. Calculate the difference between actual data
    in the 'y' dataset and the predicted value for a given 'input_feature'.

    Visualize the difference by plotting the entire 'X' & 'y' as a Scatter plot. Then add
    a blue dot that represents the actual target value, and a red dot that represents the predicted target value for the given 'input_feature'.
    Add a dashed line connects these points, highlighting the difference between them, which is annotated on the plot.

    Args:
        input_feature (float): User provided data used for prediction.
        prediction (array-like): Predicted value.
        model_filename (str): Filename of the selected serialized model.
        model_name (str): Display name of the selected model.

    """
    X, y = load_datasets()
    model = load_model(model_filename)

    actual_target = y[_index_of_closest(X, input_feature)]

    # Calculate difference
    difference = actual_target - prediction

    # Visualization
    fig = plt.figure(figsize=(6, 4))

    x_values = np.asarray(X).flatten()
    x_line = np.linspace(x_values.min(), x_values.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    plt.scatter(x_values, y, c='grey', label='Dataset', alpha=0.6)
    plt.plot(x_line, y_line, color='green', linewidth=2, label='Regression line')
    plt.scatter(input_feature, actual_target, c='blue', label='Actual target', s=100, zorder=5)
    plt.scatter(input_feature, prediction, c='red', label='Predicted target', s=100, zorder=5)

    plt.legend()
    plt.title(f'Actual vs Predicted Target Value ({model_name})')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.grid(True)

    plt.plot([input_feature, input_feature], [actual_target, prediction], 'k--')

    mid_y = (actual_target + prediction) / 2
    plt.annotate(
        f'Difference: {difference:.2f}',
        xy=(input_feature, mid_y),
        xytext=(15, 0),
        textcoords='offset points',
    )

    st.pyplot(fig)
    plt.close(fig)

# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    This function takes an array-like object `X` and a float `k`, and returns the index of the 
    element in `X` that is closest to `k`. The function first converts `X` into a NumPy array 
    (if it isn't one already) to ensure compatibility with NumPy operations. It then calculates 
    the absolute difference between each element in `X` and `k`, identifies the minimum value 
    among these differences, and returns the index of this minimum difference.

    Args:
        X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple, 
      or any object that can be converted to a NumPy array.
        k (float): The target value to which the closest element in `X` is sought.

    Returns:
        int: The index of the element in `X` that is closest to the value `k`.
    Returns:
        int: Index for the closest value to k in X.
    Finds the index of the element in `X` that is closest to the value `k`.

    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == '__main__':
    create_streamlit_app()
