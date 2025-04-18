from flask import Flask, request, jsonify
import pickle
import requests
import pandas as pd

app = Flask(__name__)
CLIENT_ID = "1b51b0e34b4a4da2a78dd5bd9d1d7e02"
CLIENT_SECRET = "3450c693ec4541e980faa7bac14844c7"
# Load the model and data
with open('music_recommender.pkl', 'rb') as f:
    data = pickle.load(f)
    cosine_sim = data['cosine_sim']
    df = data['df']

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def get_recommendations(title, cosine_sim=cosine_sim, df=df):
    try:
        idx = df[df['song_name'] == title].index[0]
    except IndexError:
        return pd.DataFrame()  # Return empty if song not found
    
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    song_indices = [i[0] for i in sim_scores]
    
    return df[['song_name', 'singer', 'duration', 'popularity', 'Stream']].iloc[song_indices]

# @app.route('/recommend', methods=['GET'])
# def recommend():
#     song_name = request.args.get('song')
#     if not song_name:
#         return jsonify({'error': 'Please provide a song name'}), 400
    
#     recommendations = get_recommendations(song_name)
#     if recommendations.empty:
#         return jsonify({'error': 'Song not found in database'}), 404
    
#     return jsonify(recommendations.to_dict('records'))

@app.route('/recommend', methods=['GET'])
def recommend():
    song_name = request.args.get('song')
    if not song_name:
        return jsonify({'error': 'Please provide a song name'}), 400

    recommendations = get_recommendations(song_name)
    print("Columns in recommendations:", recommendations.columns)
    if recommendations.empty:
        return jsonify({'error': 'Song not found in database'}), 404

    # Add Spotify data to each recommended song
    access_token = "BQDKUxaFBKXsrVuC27lCIe2ioo26Gko0rpQq6hWmhEnx7jMNHxVkcvgagwp6wINRmtwAVlm6T5xr7TX6WfBodi4vQul2FpZHt0Tqv3PchSaHSqrdC06c2FwUmIvqMaocPwT3OL-JeeveKmgOFCJqcEaU_kuLCDbQEtRsONpwNIWOEtIpdLRpG6n_yqlt99L2MD4GDHxVQS2832VnkxFgKilMOR8oBwMKiL4zKnoIn74Oc85WwK55MhN6FR8VT3GTGo25Ei8O"  # Use valid token from OAuth
    enriched_results = []

    for _, row in recommendations.iterrows():
        track_info = search_spotify_track(row['song_name'], access_token)
        enriched_results.append({
            'track_name': row['song_name'],
            'artist': row.get('artist', ''),
            'url': track_info.get('url', None),
            'id': track_info.get('id', None)
        })

    return jsonify(enriched_results)

def search_spotify_track(track_name, access_token):
    url = "https://api.spotify.com/v1/search"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        "q": track_name,
        "type": "track",
        "limit": 1
    }

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return {}

    items = response.json().get('tracks', {}).get('items', [])
    if not items:
        return {}

    track = items[0]
    return {
        'url': track['external_urls']['spotify'],
        'id': track['id']
    }

if __name__ == '__main__':
    app.run(debug=True)