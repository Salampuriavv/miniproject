import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from pickle import load
from sklearn.preprocessing import MinMaxScaler

# Load leakage data
df_leak = pd.read_csv('leak_values_random.csv', header=None)

# Load pressure data
df_pressure = pd.read_csv('pressure_response_random.csv', header=None)

# Split data into features (X) and target (y)
X = df_pressure.values
y = df_leak.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Load scaler_x
scaler_x = MinMaxScaler()
scaler_x.fit(X_train)

# Load scaler_y
scaler_y = MinMaxScaler()
scaler_y.fit(y_train)

# Load the trained model
json_file = open('model_sigmoid.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model_sigmoid.h5")

# Define the Streamlit app
def main():
    st.title("Leak Detection Model Results")

    # Load and transform the test data
    X_test_scaled = scaler_x.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test)

    # Predictions on test data
    y_pred = loaded_model.predict(X_test_scaled)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_test_scaled, y_pred)
    st.subheader(f"Mean Absolute Error (MAE): {mae}")

    # Display Scatter plot of True vs. Predicted Values
    st.subheader("Scatter plot of True vs. Predicted Values")
    fig, axes = plt.subplots(figsize=(12, 8))
    plt.scatter(y_test_scaled, y_pred_inverse, alpha=0.5)
    plt.title('True vs. Predicted Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    st.pyplot(fig)

    # Display Histogram of Residuals
    residuals = y_test_scaled - y_pred
    st.subheader("Histogram of Residuals")
    fig, axes = plt.subplots(figsize=(12, 8))
    plt.hist(residuals, bins=50)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    st.pyplot(fig)

    # Display other relevant graphs or metrics as needed

if __name__ == "__main__":
    main()
