import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.title("✈ Flight Delay Prediction App")

# Upload dataset
uploaded_file = st.file_uploader("Upload Flight Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    # Check if required column exists
    if 'arrival_delay_min' in df.columns:

        X = df.drop(['arrival_delay_min'], axis=1)
        y = df['arrival_delay_min']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("Model Trained Successfully")

        # Predictions
        predictions = model.predict(X_test)

        result = pd.DataFrame({
            "Actual Delay": y_test,
            "Predicted Delay": predictions
        })

        st.subheader("Prediction Results")
        st.write(result.head())

        # Evaluation Metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        st.subheader("Model Evaluation Metrics")
        st.write("Mean Absolute Error:", mae)
        st.write("Mean Squared Error:", mse)
        st.write("Root Mean Squared Error:", rmse)
        st.write("R² Score:", r2)

        # Visualization
        st.subheader("Actual vs Predicted Delay")

        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions)
        ax.set_xlabel("Actual Delay")
        ax.set_ylabel("Predicted Delay")
        ax.set_title("Actual vs Predicted Flight Delay")

        st.pyplot(fig)

    else:
        st.error("Dataset must contain 'arrival_delay_min' column")
