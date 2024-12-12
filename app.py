import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data and train model
location = 'Seat_Cockpit_Combined_2.csv'
df = pd.read_csv(location)

# Prepare features and target
X = df[['Loudness', 'Sharpness', 'OA SPL']]  # Only keep Loudness, Sharpness, and OA SPL
y = df['Subjective Rating']

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Train the RandomForest model
RF = RandomForestClassifier(random_state=42, class_weight='balanced_subsample', criterion='entropy')
RF.fit(X_train, y_train)

# Streamlit app
def predict_rating(loudness, sharpness, oa_spl):
    features = pd.DataFrame([[loudness, sharpness, oa_spl]], columns=['Loudness', 'Sharpness', 'OA SPL'])
    prediction = RF.predict(features)
    return prediction[0]

# Create the Streamlit interface
st.title("Subjective Rating Prediction")
st.write("Enter the values for Loudness, Sharpness, and OA SPL to predict the Subjective Rating:")

# Input fields
loudness = st.number_input("Loudness", min_value=float(X['Loudness'].min()), max_value=float(X['Loudness'].max()))
sharpness = st.number_input("Sharpness", min_value=float(X['Sharpness'].min()), max_value=float(X['Sharpness'].max()))
oa_spl = st.number_input("OA SPL", min_value=float(X['OA SPL'].min()), max_value=float(X['OA SPL'].max()))

# Predict button
if st.button("Predict"):
    prediction = predict_rating(loudness, sharpness, oa_spl)
    
    # Set text color based on the prediction
    if prediction > 5:
        color = "red"
    else:
        color = "green"
    
    # Display the prediction with color
    st.markdown(f"<h3 style='color:{color};'>The predicted Subjective Rating is: {prediction}</h3>", unsafe_allow_html=True)



