# Import essential libraries
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime  # Import datetime to display the current date

# Load dataset
train = pd.read_csv('train.csv')

# Select relevant features for prediction
X = train[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = train['SalePrice']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI starts here
# ---- Display Current Date ----
current_date = datetime.now().strftime("%Y-%m-%d")  # Format the date
st.write(f"### Date: {current_date}")  # Display date at the top

st.title('House Price Prediction App with Graphical Representation and Sidebar')
st.write("Predict the price of a house based on its square footage, number of bedrooms, and bathrooms")

# ---- Sidebar for User Input ----
st.sidebar.header('Input Features')

# Sidebar sliders for user input
square_footage = st.sidebar.slider('Square Footage', min_value=334, max_value=5642, step=100, value=1500)
bedrooms = st.sidebar.slider('Number of Bedrooms', min_value=0, max_value=8, step=1, value=3)
bathrooms = st.sidebar.slider('Number of Bathrooms', min_value=0, max_value=3, step=1, value=2)

# Prepare user input for prediction
input_data = np.array([[square_footage, bedrooms, bathrooms]])

# Make prediction
predicted_price = model.predict(input_data)

# Display prediction
st.write(f"The predicted price of the house is: ${predicted_price[0]:,.2f}")

# Optional: Display performance on test data
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"Model RMSE on test data: {rmse:,.2f}")

# ---- Graphical Representations ----

# Scatter Plot: GrLivArea vs. SalePrice
st.subheader("Scatter Plot: Square Footage vs. SalePrice")
fig, ax = plt.subplots()
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train, ax=ax)
ax.set_title('Square Footage vs. House Price')
st.pyplot(fig)

# Scatter Plot: BedroomAbvGr vs. SalePrice
st.subheader("Scatter Plot: Number of Bedrooms vs. SalePrice")
fig, ax = plt.subplots()
sns.scatterplot(x='BedroomAbvGr', y='SalePrice', data=train, ax=ax)
ax.set_title('Number of Bedrooms vs. House Price')
st.pyplot(fig)

# Scatter Plot: FullBath vs. SalePrice
st.subheader("Scatter Plot: Number of Bathrooms vs. SalePrice")
fig, ax = plt.subplots()
sns.scatterplot(x='FullBath', y='SalePrice', data=train, ax=ax)
ax.set_title('Number of Bathrooms vs. House Price')
st.pyplot(fig)

# Distribution of Predicted Prices
st.subheader("Distribution of Predicted Prices")
fig, ax = plt.subplots()
sns.histplot(y_pred, bins=20, kde=True, ax=ax)
ax.set_title('Distribution of Predicted Prices')
st.pyplot(fig)

# ---- Add Description at the Bottom of Sidebar ----
st.sidebar.markdown("---")  # Add a separator line
st.sidebar.write("""
### Project Description:
This application uses a **Linear Regression** model to predict house prices based on square footage, number of bedrooms, and number of bathrooms. 
The model is trained on historical data and provides real-time price predictions based on user inputs.

#### Features:
- **Square Footage**: The living area of the house.
- **Number of Bedrooms**: Total number of bedrooms in the house.
- **Number of Bathrooms**: Total number of full bathrooms in the house.

Built using Python libraries such as `scikit-learn`, `matplotlib`, `seaborn`, and `Streamlit` for the frontend UI.
                 
                 By Engineer Shahan Nafees
""")
