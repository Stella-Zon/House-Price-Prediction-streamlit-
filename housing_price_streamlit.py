import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the California housing dataset
california = fetch_california_housing()
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.Series(california.target, name='MedHouseVal')

# Train a simple linear regression model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Define the Streamlit app
st.title("California Housing Price Prediction")

st.image('housingprice.jpeg', caption='Sunrise by the mountains')

st.write("""
This app predicts the **California Housing Prices** using machine learning!
""")

# Create sliders for user input
MedInc = st.slider('Median income in block group (MedInc)', float(X.MedInc.min()), float(X.MedInc.max()), float(X.MedInc.mean()))
HouseAge = st.slider('Median house age in block group (HouseAge)', float(X.HouseAge.min()), float(X.HouseAge.max()), float(X.HouseAge.mean()))
AveRooms = st.slider('Average number of rooms per household (AveRooms)', float(X.AveRooms.min()), float(X.AveRooms.max()), float(X.AveRooms.mean()))
AveBedrms = st.slider('Average number of bedrooms per household (AveBedrms)', float(X.AveBedrms.min()), float(X.AveBedrms.max()), float(X.AveBedrms.mean()))
Population = st.slider('Block group population (Population)', float(X.Population.min()), float(X.Population.max()), float(X.Population.mean()))
AveOccup = st.slider('Average number of household members (AveOccup)', float(X.AveOccup.min()), float(X.AveOccup.max()), float(X.AveOccup.mean()))
Latitude = st.slider('Block group latitude (Latitude)', float(X.Latitude.min()), float(X.Latitude.max()), float(X.Latitude.mean()))
Longitude = st.slider('Block group longitude (Longitude)', float(X.Longitude.min()), float(X.Longitude.max()), float(X.Longitude.mean()))

# Create a dataframe with the user input
input_data = pd.DataFrame({
    'MedInc': [MedInc],
    'HouseAge': [HouseAge],
    'AveRooms': [AveRooms],
    'AveBedrms': [AveBedrms],
    'Population': [Population],
    'AveOccup': [AveOccup],
    'Latitude': [Latitude],
    'Longitude': [Longitude]
})

# Make predictions
prediction = model.predict(input_data)

# Display the prediction
st.subheader("Predicted Median House Value (in $100,000s):")
st.write(prediction[0])
