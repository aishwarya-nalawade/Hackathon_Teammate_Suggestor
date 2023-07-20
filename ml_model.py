import numpy as np
import pandas as pd
import cohere

dataset = 'Team_mates.xlsx'
df = pd.read_excel(dataset)  # Replace 'your_dataframe.csv' with your actual dataframe filename

#print(df.head())

co = cohere.Client("7WAkMVy3p8OC0Jk4xmaXVt6VMbl8BJyDYHEQRi2c")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import pickle

# # Preprocess and vectorize the textual data
# corpus = df['Description'].tolist()  # Assuming 'df' is your dataframe
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(corpus)
# Vectorize the descriptions
# vectorizer = TfidfVectorizer()
# X = vectorizer.fit_transform(df['Description'])

# Get text embeddings

def get_embeddings(texts,model='embed-english-v2.0'):

  output = co.embed(

                model=model,

                texts=texts)

  return output.embeddings



# Embed the dataset

df['Description_embeddings'] = get_embeddings(df['Description'].tolist())

# df.head()

X = df['Description_embeddings']
# Set the target labels
Y = df['Name']
embeddings_array = np.array(df['Description_embeddings'].to_list())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings_array, Y, test_size=0.2, random_state=42)

# Train the KNN model
k = 5  # Number of neighbors to consider
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Pickling - Serialize the model and scaler
pickle.dump(knn, open('model.pkl', 'wb')) # Serialize the model
print("Model saved")

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Predict similar individuals
def find_similar_people(description, k):
    X_test = get_embeddings([description])
    #distances, indices = knn.kneighbors(X_test, n_neighbors=k)
    distances, indices = model.kneighbors(X_test, n_neighbors=k)
    similar_people = df.iloc[indices[0]][['ID', 'Name', 'Description']]
    return similar_people

# Example usage
description = "I am well versed in machine learning with strong base in Python, Ruby on Rails along with UI/UX experience. I am also well versed in databases like MySQL, SQLLite."
k = 5  # Number of similar individuals to find
similar_people = find_similar_people(description, k)
print(similar_people)