import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
with open('/Users/normanangel/Library/CloudStorage/OneDrive-StrathmoreUniversity/MSc. DSA/Module 4/Applied Machine Learning/Gradient_Boosting_Classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define the function for churn prediction
def churn_prediction(input_data):
    # Convert input data to numpy array and reshape it
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)  # Convert input to float
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data_reshaped)

    # Return the prediction result
    if prediction[0] == 0:
        return "The Customer has not churned"
    else:
        return "The customer churned"

# Define the main function to create the web app
def main():
    st.title('Churn Prediction Web App')

    # User Input
    Age = st.text_input('Age of the customer')
    Annual_Income = st.text_input('Customer Annual Income')
    Service_Opted = st.text_input('Service Opted by the customer')
    Hotel_Booked = st.text_input('Has Customer booked hotel')
    Account_Synced = st.text_input('Account Synced')
    Frequent_Flyer_No = st.text_input('Frequent Flyer No')
    Frequent_Flyer_Yes = st.text_input('Frequent Flyer Yes')

    # Code for Prediction
    prediction = ''

    # Creating a button for prediction
    if st.button('Churn Prediction Result'):
        # Call the churn_prediction function with user input
        prediction = churn_prediction([Age, Annual_Income, Service_Opted, Hotel_Booked, Account_Synced, Frequent_Flyer_No, Frequent_Flyer_Yes])

    # Display prediction result
    st.success(prediction)

if __name__ == '__main__':
    main()
