# heart-disease-prediction-using-machine-learning
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('heart_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/find', methods=['POST'])
def predict():
    age = int(request.form['age'])
    cp = int(request.form['cp'])
    bp = int(request.form['bp'])
    chol = int(request.form['chol'])
    max_hr = int(request.form['max_hr'])
    st_depression = float(request.form['st_depression'])
    vessels = int(request.form['vessels'])
    thal = int(request.form['thal'])

    # Arrange input data as expected by the model
    input_data = np.array([[age, cp, bp, chol, max_hr, st_depression, vessels, thal]])

    # Make prediction
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = 'Presence of Heart Disease'
    else:
        result = 'Absence of Heart Disease'

    return render_template('result.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
    import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('heart.csv')

# Select features and target
X = data[['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca', 'thal']]
y = data['target']  # 1: presence, 0: absence

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('heart_model.pkl', 'wb'))

print("Model trained and saved as 'heart_model.pkl'")
<!DOCTYPE html>
<html>
<head>
    <title>Predict Heart Disease</title>
    <style>
        body { display: flex; font-family: Arial; }
        .left { background-color: #fff176; width: 50%; text-align: center; padding: 50px; }
        .right { background-color: #b2ebf2; width: 50%; padding: 50px; }
    </style>
</head>
<body>
    <div class="left">
        <h1>Heart Disease Prediction</h1>
        <h2>Welcome Prathap</h2>
        <img src="https://img.icons8.com/color/200/heart-with-pulse.png" alt="Heart Image"/>
    </div>

    <div class="right">
        <h3>Fill the form below to check if you have heart disease or not!</h3>
        <form action="/find" method="post">
            <input type="number" name="age" placeholder="Enter Age" required><br><br>
            <label>Chest Pain Type:</label><br>
            <input type="radio" name="cp" value="0" required> 0
            <input type="radio" name="cp" value="1"> 1
            <input type="radio" name="cp" value="2"> 2
            <input type="radio" name="cp" value="3"> 3
            <br><br>
            <input type="number" name="bp" placeholder="Enter BP" required><br><br>
            <input type="number" name="chol" placeholder="Enter Cholesterol" required><br><br>
            <input type="number" name="max_hr" placeholder="Enter Max Heart Rate" required><br><br>
            <input type="number" step="0.1" name="st_depression" placeholder="Enter ST depression" required><br><br>
            <input type="number" name="vessels" placeholder="Enter No. of vessels fluro" required><br><br>
            <input type="number" name="thal" placeholder="Thallium" required><br><br>
            <button type="submit">Predict</button>
        </form>
    </div>
</body>
</html>
<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body style="text-align: center; font-family: Arial;">
    <h1>{{ prediction_text }}</h1>
    <a href="/">Back to Home</a>
</body>
</html>

