from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Load the data
file_path = 'datasets/All_Streaming_Shows.csv'
df = pd.read_csv(file_path)

# Data Preprocessing
df['Genre'] = df['Genre'].fillna('')
df['Content Rating'] = df['Content Rating'].fillna('')
df['features'] = df['Genre'] + ' ' + df['Content Rating'] + ' ' + df['IMDB Rating'].astype(str)

# Vectorize the features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommendation function
def get_recommendations(title):
    if title not in df['Series Title'].values:
        return None
    
    idx = df[df['Series Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    series_indices = [i[0] for i in sim_scores]
    return df['Series Title'].iloc[series_indices].tolist()

# Flask route for recommendation
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    series_title = data.get('title', None)
    
    if not series_title:
        return jsonify({'error': 'No series title provided'}), 400
    
    recommendations = get_recommendations(series_title)
    
    if recommendations is None:
        return jsonify({'error': 'Series not found'}), 404
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
