import numpy as np
from flask import Flask, request, render_template, url_for
import joblib
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
model = joblib.load('corona_model')
# print("probability",model.predict_proba([[101.9389794,0,96,1,-1]])[0][1])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    print("input fetures",input_features)
    value = np.array(input_features)
    output = model.predict_proba([value])[0][1]  
    print(output)
    return render_template('index.html', Prediction_text=f"Patients probabilty of infection is {round(output*100)}%.")


if __name__ == "__main__":
    app.run()
