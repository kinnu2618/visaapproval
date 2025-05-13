import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('lgbm.pkl', 'rb'))

# Encoding dictionaries
continent_dict = {
    "Africa": 0, "Asia": 1, "Europe": 2,
    "North America": 3, "Oceania": 4, "South America": 5
}

education_dict = {
    "Bachelor's": 0, "Doctorate": 1, "High School": 2, "Master's": 3
}

experience_dict = {"No": 0, "Yes": 1}
training_dict = {"No": 0, "Yes": 1}

region_dict = {
    "Island": 0, "Midwest": 1, "Northeast": 2,
    "South": 3, "West": 4
}

wage_unit_dict = {"Hour": 0, "Month": 1, "Week": 2, "Year": 3}
full_time_dict = {"No": 0, "Yes": 1}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        form = request.form

        # Convert input values using mapping
        data = [
            continent_dict[form['continent']],
            education_dict[form['education']],
            experience_dict[form['experience']],
            training_dict[form['training']],
            int(form['num_employees']),
            int(form['established']),
            region_dict[form['region']],
            float(form['wage']),
            wage_unit_dict[form['unit']],
            full_time_dict[form['full_time']]
        ]

        prediction = model.predict([data])[0]
        result = "Certified" if prediction == 0 else "Denied"

        return render_template('home.html', prediction_text=f"Visa Case Status: {result}")
    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
