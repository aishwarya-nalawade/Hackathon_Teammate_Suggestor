import pickle
import numpy as np
import pandas as pd
import cohere
from flask import Flask, render_template, request, jsonify

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Initialize cohere client
co = cohere.Client("7WAkMVy3p8OC0Jk4xmaXVt6VMbl8BJyDYHEQRi2c")

app = Flask(__name__)

# Load the dataset
dataset = 'Team_mates.xlsx'
df = pd.read_excel(dataset)

# Get text embeddings
def get_embeddings(texts, model='embed-english-v2.0'):
    output = co.embed(model=model, texts=texts)
    return output.embeddings

# Function to find similar individuals
def find_similar_people(description, k):
    X_test = get_embeddings([description])
    distances, indices = model.kneighbors(X_test, n_neighbors=k)
    similar_people = df.iloc[indices[0]][['ID', 'Name', 'Description']]
    return similar_people


'''@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    description = request.form['description']
    k = 5  # Number of similar individuals to find
    similar_people = find_similar_people(description, k)
    return render_template('result.html', data=similar_people.to_html(index=False))'''

# ... (existing code) ...

# Remove this line as we are not using 'df' directly in this file
# df = pd.read_excel(dataset)

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        data = request.json
        description = data.get('description')
        print(description)
        k = 5  # Number of similar individuals to find
        similar_people = find_similar_people(description, k)
        print(similar_people)
        return jsonify(similar_people.to_dict(orient='records'))
       # return jsonify(similar_people)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

