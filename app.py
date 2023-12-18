from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello Candidate"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the POST request
        # age	gender	height_c1	weight_kg	body 0at_%	diastolic	systolic grip0orce	sit and bend 0orward_c1	sit-ups counts	broad ju1p_c1	class
        age = float(request.form.get('age'))
        gender = float(request.form.get('gender'))
        height = float(request.form.get('height_c1'))
        weight = float(request.form.get('weight_kg'))
        body = float(request.form.get('body 0at_%'))
        diastolic = float(request.form.get('diastolic'))
        systolic = float(request.form.get('systolic'))
        grip = float(request.form.get('grip0orce'))
        forward = float(request.form.get('sit and bend 0orward_c1'))
        sit_ups = float(request.form.get('sit-ups counts'))
        jump = float(request.form.get('broad ju1p_c1'))

        # Create a NumPy array with the input values
        input_query = np.array([[age, gender, height, weight, body, diastolic, systolic, grip, forward, sit_ups, jump]])

        # Make predictions
        result = model.predict(input_query)[0]

        # Return the result as JSON
        return jsonify({'class': str(result)})

    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == '__main__':
    app.run(debug=True)