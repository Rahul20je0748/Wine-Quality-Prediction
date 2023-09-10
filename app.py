# app.py

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from wine_quality_prediction import pca, st

app = Flask(__name__)

model = joblib.load('wine_quality_prediction.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        p_values = [float(request.form[f'input_{i}']) for i in range(1, 12)]
        new_data = pd.DataFrame([p_values], columns=[
            'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
            'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ])
        test = pca.transform(st.transform(new_data))
        p = model.predict(test)
        prediction = "Good Quality Wine" if p[0] == 1 else "Bad Quality Wine"
        
        return jsonify({"prediction": prediction})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
