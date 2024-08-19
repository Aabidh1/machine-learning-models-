from flask import Flask, request, render_template
import pickle
import numpy as np
import sklearn  # This will import the sklearn library
# Load the trained classifier from the pickle file
with open("model/svm_classifier.pkl", "rb") as file:
    classifier = pickle.load(file)

app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = float(request.form['age'])

    # Make a prediction
    new_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
    prediction = classifier.predict(new_data)

    # Convert prediction to human-readable form
    prediction_text = "Yes" if prediction[0] == 1 else "No"

    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
