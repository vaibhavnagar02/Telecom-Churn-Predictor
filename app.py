from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the trained model
with open('stack33_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file and file.filename.endswith('.xlsx'):
        # Read the Excel file
        df = pd.read_excel(file)

        # Preprocessing
        # Fill missing values with median or some other appropriate method
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Predict
        predictions = model.predict(df)
        df['Churn_Prediction'] = predictions

        return df.to_json(orient='records', lines=True)
    else:
        return jsonify({"error": "Invalid file format. Please upload an Excel file."})

if __name__ == '__main__':
    app.run(debug=True)