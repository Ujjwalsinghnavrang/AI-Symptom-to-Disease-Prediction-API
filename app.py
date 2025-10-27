from flask import Flask, request, jsonify
import joblib
import json

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.joblib')

# Load the mapping of labels to diseases
with open('mapping.json') as f:
    label_mapping = json.load(f)

# Reverse the mapping for easy lookup
reverse_mapping = {v: k for k, v in label_mapping.items()}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Make prediction
    prediction = model.predict([text])
    predicted_label = int(prediction[0])
    disease_name = reverse_mapping.get(predicted_label, "Unknown condition")
    
    return jsonify({'prediction': predicted_label, 'disease': disease_name})

if __name__ == '__main__':
    app.run(debug=True)